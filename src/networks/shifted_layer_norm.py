
from torch.nn import Module, LayerNorm, Parameter, init
from torch import Tensor, Size
import torch

import numbers
from typing import Union, List

_shape_t = Union[int, List[int], Size]


class ShiftedLayerNorm(Module):

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))

        self.ln_without_affine = LayerNorm(normalized_shape, eps, elementwise_affine=False, **factory_kwargs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the same operation as LayerNorm but instead of doing
                normalized_activations * gamma + beta
        it does:
                normalized_activations * (1 + gamma) + beta
        """
        x = self.ln_without_affine(x)
        x = x * (1 + self.weight) + self.bias
        return x
