import torch
from torch import nn
from torch.autograd import Function
from typing import *


class Reversal(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, lmbda: float) -> torch.Tensor:
        '''Return identity on forward pass'''
        ctx.lmbda = lmbda
        return input
    
    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor) -> torch.Tensor:
        '''Multiply the grad outputs by -lambda on backwards pass'''
        return -ctx.lmbda * grad_outputs, None


class GRL(nn.Module):
    '''Module wrapped around custom Reversal function'''
    def __init__(self, lmbda: float) -> None:
        super().__init__()
        self.lmbda = lmbda
        self.fn = Reversal.apply

    def forward(self, inputs) -> Tuple[torch.Tensor]:
        return self.fn(inputs, self.lmbda)