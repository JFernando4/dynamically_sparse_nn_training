"""
Modification of pytorch's SGD optimizer that instead of decaying weights towards zero, they're decayed towards their
initial values.

"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Tuple

_foreach_supported_types = [torch.Tensor, torch.nn.parameter.Parameter]


class _RequiredParameter:
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


def _use_grad_for_differentiable(func):
    def _use_grad(self, *args, **kwargs):
        prev_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(self.defaults['differentiable'])
            ret = func(self, *args, **kwargs)
        finally:
            torch.set_grad_enabled(prev_grad)
        return ret
    return _use_grad


def _default_to_fused_or_foreach(params: List[torch.Tensor],
                                 differentiable: bool,
                                 use_fused: bool = False) -> Tuple[bool, bool]:
    if torch.jit.is_scripting() or differentiable:
        return False, False
    fused = use_fused and all(
        p is None or (type(p) in _foreach_supported_types and p.is_cuda and torch.is_floating_point(p)) for p in params
    )
    foreach = not fused and all(
        p is None or (type(p) in _foreach_supported_types and p.is_cuda) for p in params
    )
    return fused, foreach


class SGDL2Init(Optimizer):

    def __init__(self, params, l2_init_flags: List[bool], reg_flags: List[bool] = None, lr=required, momentum=0,
                 dampening=0, weight_decay=0, nesterov=False, *, maximize: bool = False,
                 foreach: Optional[bool] = None, differentiable: bool = False):
        """
        Args:
            l2_init_flags: list[bool] of flags indicating which parameters to apply l2 init
            reg_flags: list[bool] of flags indicating which parameters to apply l2 regularization
            Other arguments are from torch's SGD optimizer class
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable)

        params = list(params)
        self.original_params = [p.detach().clone() for p in params]
        self.l2_init_flags = l2_init_flags
        self.reg_flags = reg_flags if reg_flags is not None else [True for _ in params]
        assert len(self.original_params) == len(self.l2_init_flags) and len(self.reg_flags) == len(self.original_params)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        has_sparse_grad = False

        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state['momentum_buffer'])

        return has_sparse_grad


    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            has_sparse_grad = self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

            sgd(params_with_grad,
                d_p_list,
                self.original_params,
                self.l2_init_flags,
                self.reg_flags,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        original_params: List[Tensor],
        l2_init_flags: List[bool],
        reg_flags:List[bool],
        momentum_buffer_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: Optional[bool] = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # why must we be explicit about an if statement for torch.jit.is_scripting here?
        # because JIT can't handle Optionals nor fancy conditionals when scripting
        if not torch.jit.is_scripting():
            _, foreach = _default_to_fused_or_foreach(params, differentiable=False, use_fused=False)
        else:
            foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    _single_tensor_sgd(params,
                       d_p_list,
                       original_params,
                       l2_init_flags,
                       reg_flags,
                       momentum_buffer_list,
                       weight_decay=weight_decay,
                       momentum=momentum,
                       lr=lr,
                       dampening=dampening,
                       nesterov=nesterov,
                       has_sparse_grad=has_sparse_grad,
                       maximize=maximize)


def _single_tensor_sgd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       original_params: List[Tensor],
                       l2_init_flags: List[bool],
                       reg_flags:List[bool],
                       momentum_buffer_list: List[Optional[Tensor]],
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool):

    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]

        if weight_decay != 0:
            if reg_flags[i]:
                if l2_init_flags[i]:
                    d_p = d_p.add(param - original_params[i], alpha=weight_decay)
                else:
                    d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)
