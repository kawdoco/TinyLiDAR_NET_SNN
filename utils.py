# -*- coding: utf-8 -*-

import os
from typing import Callable
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from typing import Iterable
import warnings

from spikingjelly.activation_based.neuron import IFNode


class STEye(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super().__init__()
        self.root = root

        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # self.data = np.load(
        #     os.path.join(root, f"{'train'if self.train else'test'}_data.npy")
        # )
        # self.targets = np.load(
        #     os.path.join(root, f"{'train'if self.train else'test'}_targets.npy")
        # )
        self.data = np.load(os.path.join(root, "train_data.npy"))
        self.targets = np.load(os.path.join(root, "train_targets.npy"))
        N = int(len(self.data) * 0.5 )
        if self.train:
            self.data = self.data[:N]
            self.targets = self.targets[:N]
        else:
            self.data = self.data[N:]
            self.targets = self.targets[N:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)

        return data, target


def img_one_hot(x: Tensor, grid_size: int) -> Tensor:
    x = torch.stack(
        [
            torch.cat([torch.arange(grid_size), torch.arange(grid_size - 1).flip(0)])[
                grid_size - 1 - D : 2 * grid_size - 1 - D
            ]
            for D in x
        ]
    )
    # x=x/grid_size
    x = x.float().softmax(-1)

    # x = torch.tensor([[1 / (1 + np.abs(D - i)) for i in range(grid_size)] for D in x])
    # x = torch.nn.functional.one_hot(x.long(), grid_size).float()

    # X,Y=x
    # x = torch.zeros(2, grid_size)
    # x[0, X] = .5
    # if X>0:
    #     x[0, X-1] = .25
    # if X<grid_size-1:
    #     x[0, X+1] = .25
    # x[1, Y] = .5
    # if Y>0:
    #     x[1, Y-1] = .25
    # if Y<grid_size-1:
    #     x[1, Y+1] = .25

    # x = x.flatten()
    return x


def poisson_encode(x: Tensor, T: int) -> Tensor:
    x = torch.stack([torch.rand_like(x) < x for _ in range(T)]).float()
    return x


def list_convertable(parent: nn.Module, convertable: Iterable[nn.Module]):
    l = []
    for name, child in parent.named_children():
        if isinstance(child, tuple(convertable)):
            l.append({"parent": parent, "name": name, "child": child})
        elif len(list(child.children())) > 0:
            l.extend(list_convertable(child, convertable))
    return l


@torch.no_grad()
def fake_quant(model: nn.Module, bits: int = 8) -> nn.Module:
    weight_max = 2 ** (bits - 1) - 1
    connections = [nn.Linear, nn.Conv2d]

    for module in list_convertable(model, connections):
        weight = module["child"].weight
        scale = weight_max / weight.abs().max()
        weight = (weight * scale).round() / scale
        module["child"].weight.copy_(weight)

    return model


@torch.no_grad()
def convert_to_quant(model: nn.Module, bits: int = 8) -> nn.Module:
    connections = [nn.Linear, nn.Conv2d]
    neurons = [IFNode]
    module_list = list_convertable(model, connections + neurons)
    for connection, neuron in zip(module_list, module_list[1:]):
        if isinstance(connection["child"], tuple(connections)) and isinstance(
            neuron["child"], tuple(neurons)
        ):
            weight_max = 2 ** (bits - 1) - 1
            weight_scale = weight_max / connection["child"].weight.abs().max()
            threshold_max = 2**bits - 1
            threshold_scale = threshold_max / torch.tensor(neuron["child"].v_threshold)
            scale = min(weight_scale, threshold_scale)
            connection["child"].weight.copy_(
                (connection["child"].weight * scale).round()
            )
            neuron["child"].v_threshold = (neuron["child"].v_threshold * scale).round()

    return model


@torch.no_grad()
def convert_to_quant_th(model: nn.Module, bits: int = 8) -> nn.Module:
    connections = [nn.Linear, nn.Conv2d]
    neurons = [IFNode]
    module_list = list_convertable(model, connections + neurons)
    for connection, neuron in zip(module_list, module_list[1:]):
        if isinstance(connection["child"], tuple(connections)) and isinstance(
            neuron["child"], tuple(neurons)
        ):
            threshold_max = 2 ** (bits - 1)
            scale = threshold_max / torch.tensor(neuron["child"].v_threshold)
            if max(connection["child"].weight * scale).round() >= threshold_max:
                warnings.warn("weight overflow")
            connection["child"].weight.copy_(
                (connection["child"].weight * scale).round()
            )
            neuron["child"].v_threshold = (neuron["child"].v_threshold * scale).round()

    return model


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = (x >= 0).float()
        return x

    @staticmethod
    def backward(ctx, grad_x):
        return grad_x


class SG_Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = (x >= 0).float()
        return x

    @staticmethod
    def backward(ctx, grad_x):
        sg = grad_x.sigmoid()
        grad_x = sg * (1 - sg)
        return grad_x


class IF(nn.Module):
    def __init__(self,log=False):
        super().__init__()
        self.v = 0.0
        self.th = 1.0
        self.v_seq = []
        self.log = log

    def forward(self, x):
        self.v = self.v + x
        self.v = torch.clamp(self.v, min=0)
        x = STE.apply(self.v - self.th)
        self.v = self.v * (1 - x)
        if self.log:
            self.v_seq += [self.v.clone().detach()]
        return x

    def reset(self):
        self.v = 0.0
        self.v_seq = []


class Loop(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, log=False):
        self.model[-1].reset()
        self.model[-1].log = log
        y=[]
        for xt in x:
            y += [self.model(xt)]
        return torch.stack(y)