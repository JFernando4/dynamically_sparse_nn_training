
from torch.nn import Module, GroupNorm, Parameter, init
from torch import Tensor, Size
import torch

from typing import Union, List

_shape_t = Union[int, List[int], Size]


class ShiftedGroupNorm(Module):

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))

        self.group_norm_without_affine = GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=False, **factory_kwargs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the same operation as GroupNorm but instead of doing
                normalized_activations * gamma + beta
        it does:
                normalized_activations * (1 + gamma) + beta
        """
        x = self.group_norm_without_affine(x)
        x = x * (1 + self.weight) + self.bias
        return x
