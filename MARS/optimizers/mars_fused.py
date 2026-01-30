# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import math
import torch
from torch.optim.optimizer import Optimizer
import os
import numpy as np
import math
# from megatron.optimizer.l2_norm import l2_norm
from typing import cast, List, Optional, Tuple, Union
from torch import Tensor

from torch.optim.optimizer import (
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _disable_dynamo_if_unsupported,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _stack_if_compiling,
    _use_grad_for_differentiable,
    _view_as_real,
    DeviceDict,
    Optimizer,
)

# TODO: when mars_type == "mars-shampoo"
# TODO: _fused_mars_single_tensor: when mars_type == "mars-lion"/"mars-shampoo"/"mars-sgd"; currently c_t is not clipped
# Thanks to Adan: https://github.com/sail-sg/Adan
# For fused_mars: first python setup.py install

def exists(val):
    return val is not None

@torch.compile
def NewtonSchulz(M, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = M.bfloat16() / (M.norm() + eps)
    if M.size(0) > M.size(1):
        X = X.T 
    for _ in range(steps):
        A = X @ X.T 
        B = A @ X 
        X = a * X + b * B + c * A @ B 
    if M.size(0) > M.size(1):
        X = X.T 
    return X.to(M.dtype)

class MultiTensorApply(object):
    available = False
    warned = False

    def __init__(self, chunk_size):
        try:
            MultiTensorApply.available = True
            self.chunk_size = chunk_size
        except ImportError as err:
            MultiTensorApply.available = False
            MultiTensorApply.import_err = err

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)

def _single_tensor_mars(
    params: List[Tensor],
    grads: List[Tensor],
    last_grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: Union[Tensor, float],
    beta2: Union[Tensor, float],
    lr: Union[Tensor, float],
    weight_decay: float,
    gamma: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
    optimize_1d: bool,
    lr_1d_factor: float,
    betas_1d: Tuple[float, float],
    weight_decay_1d: float,
    mars_type: str,
):
    assert grad_scale is None and found_inf is None
    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)
        assert isinstance(beta1, float)
        assert isinstance(beta2, float)
    beta1_1d, beta2_1d = betas_1d
    # optimize_1d: use MARS for 1d para, not: use AdamW for 1d para
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        is_grad_2d = (len(grad.shape) == 2)
        last_grad = last_grads[i] if not maximize else -last_grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]


        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            last_grad = torch.view_as_real(last_grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        if optimize_1d or is_grad_2d:
            param.mul_(1 - lr * weight_decay)
        else:
            param.mul_(1 - lr * lr_1d_factor * weight_decay_1d)

        device_beta1 = beta1 if optimize_1d or is_grad_2d else beta1_1d
        device_beta2 = beta2 if optimize_1d or is_grad_2d else beta2_1d

        # Decay the first and second moment running average coefficient
        if optimize_1d or is_grad_2d:
            c_t = (grad - last_grad).mul(gamma * (device_beta1 / (1. - device_beta1))).add(grad)
            c_t_norm = torch.norm(c_t)
            if c_t_norm > 1.:
                c_t = c_t / c_t_norm
            exp_avg.lerp_(c_t, 1 - device_beta1)
        else:
            exp_avg.lerp_(grad, 1 - device_beta1)
            
        if mars_type == "mars-adamw" or (mars_type == "mars-shampoo" and not is_grad_2d) or not (optimize_1d or is_grad_2d):
            if optimize_1d or is_grad_2d:
                exp_avg_sq.mul_(device_beta2).addcmul_(c_t, c_t, value=1 - device_beta2)
            else:
                exp_avg_sq.mul_(device_beta2).addcmul_(grad, grad, value=1 - device_beta2)

            if capturable or differentiable:
                step = step_t

                bias_correction1 = 1 - device_beta1**step
                bias_correction2 = 1 - device_beta2**step

                step_size = lr / bias_correction1
                step_size_neg = step_size.neg()

                bias_correction2_sqrt = bias_correction2.sqrt()

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    if differentiable:
                        max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                    else:
                        max_exp_avg_sq = max_exp_avg_sqs[i]

                    max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                    # Uses the max. for normalizing running avg. of gradient
                    # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                    # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                    denom = (
                        max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)
                    ).add_(eps / step_size_neg)
                else:
                    denom = (
                        exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)
                    ).add_(eps / step_size_neg)

                param.addcdiv_(exp_avg, denom)
            else:
                step = _get_value(step_t)

                bias_correction1 = 1 - device_beta1**step
                bias_correction2 = 1 - device_beta2**step

                step_size = lr / bias_correction1

                bias_correction2_sqrt = bias_correction2**0.5

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                param.addcdiv_(exp_avg, denom, value=-step_size)

            # Lastly, switch back to complex view
            if amsgrad and torch.is_complex(params[i]):
                max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])
        elif mars_type == "mars-lion":
            param.add_(-lr * exp_avg.sign())
        elif mars_type == "mars-shampoo" and is_grad_2d:
            factor = max(1, grad.size(0)/grad.size(1))**0.5
            param.add_(NewtonSchulz(exp_avg.mul(1./(1.-device_beta1)), eps=eps).mul(factor).mul(-lr))
        elif mars_type == "mars-sgd":
            exp_avg.mul_(beta1).add_(c_t)
            param.add_(-lr * exp_avg)

def _multi_tensor_mars(
    params: List[Tensor],
    grads: List[Tensor],
    last_grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    lr: Union[Tensor, float],
    weight_decay: float,
    beta1: Union[Tensor, float],
    beta2: Union[Tensor, float],
    eps: float,
    amsgrad: bool,
    gamma: float,
    *,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
    optimize_1d: bool,
    lr_1d_factor: float,
    betas_1d: Tuple[float, float],
    weight_decay_1d: float,
    mars_type: str,
):
    assert grad_scale is None and found_inf is None
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError(
            "lr as a Tensor is not supported for capturable=False and foreach=True"
        )

    if isinstance(beta1, Tensor):
        if not capturable:
            raise ValueError(
                "beta1 as a Tensor is not supported for capturable=False and foreach=True"
            )
        if beta1.numel() != 1:
            raise ValueError("Tensor beta1 must be 1-element")

    if isinstance(beta2, Tensor):
        if not capturable:
            raise ValueError(
                "beta2 as a Tensor is not supported for capturable=False and foreach=True"
            )
        if beta2.numel() != 1:
            raise ValueError("Tensor beta2 must be 1-element")

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch.compiler.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    assert not differentiable, "_foreach ops don't support autograd"

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, last_grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )

    # We only shuffle around the beta when it is a Tensor and on CUDA, otherwise, we prefer
    # treating it as a scalar.
    beta1_dict: Optional[DeviceDict] = (  # type: ignore[attr-defined]
        {beta1.device: beta1}
        if isinstance(beta1, Tensor) and str(beta1.device) != "cpu"
        else None
    )

    for (
        device_params_,
        device_grads_,
        device_last_grads_,
        device_exp_avgs_,
        device_exp_avg_sqs_,
        device_max_exp_avg_sqs_,
        device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast(List[Tensor], device_params_)
        device_grads = cast(List[Tensor], device_grads_)
        device_last_grads = cast(List[Tensor], device_last_grads_)
        device_exp_avgs = cast(List[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(List[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(List[Tensor], device_state_steps_)

        device = device_params[0].device
        if beta1_dict is not None and device not in beta1_dict:
            beta1_dict[device] = beta1.to(device=device, non_blocking=True)  # type: ignore[union-attr]

        device_beta1 = beta1_dict[device] if beta1_dict else beta1

        if has_complex:
            if amsgrad:
                device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)
                _view_as_real(
                    device_params,
                    device_grads,
                    device_last_grads,
                    device_exp_avgs,
                    device_exp_avg_sqs,
                    device_max_exp_avg_sqs,
                )
            else:
                _view_as_real(
                    device_params, device_grads, device_last_grads, device_exp_avgs, device_exp_avg_sqs
                )

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]
            device_last_grads = torch._foreach_neg(device_last_grads)  # type: ignore[assignment]
        
        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch.compiler.is_compiling() and device_state_steps[0].is_cpu:
            torch._foreach_add_(
                device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(device_state_steps, 1)

        if optimize_1d:
            # Perform stepweight decay
            if weight_decay != 0:
                torch._foreach_mul_(device_params, 1 - lr * weight_decay)

            # Decay the first and second moment running average coefficient
            c_t = torch._foreach_sub(device_grads, device_last_grads)
            torch._foreach_mul_(c_t, gamma * (device_beta1 / (1. - device_beta1)))
            torch._foreach_add_(c_t, device_grads)
            # if c_t_norm > 1. for some c_t, then clip c_t to c_t / c_t_norm
            c_t_norm = torch._foreach_norm(c_t)
            c_t = [c_t[i] / c_t_norm[i] if c_t_norm[i] > 1.0 else c_t[i] for i in range(len(c_t))]
            
            torch._foreach_lerp_(device_exp_avgs, c_t, 1 - device_beta1)

        
            if mars_type == "mars-adamw":
                torch._foreach_mul_(device_exp_avg_sqs, beta2)

                if isinstance(beta2, torch.Tensor):
                    scaled_device_grads = torch._foreach_mul(device_grads, 1 - beta2)  # type: ignore[assignment]
                    value = 1.0
                else:
                    scaled_device_grads = device_grads  # type: ignore[assignment]
                    value = 1 - beta2

                torch._foreach_addcmul_(
                    device_exp_avg_sqs, scaled_device_grads, device_grads, value
                )

                # Delete the local intermediate(s) since they won't be used anymore to save on peak memory
                del device_grads
                del scaled_device_grads

                bias_correction1 = [
                    1 - beta1 ** _get_value(step) for step in device_state_steps
                ]
                bias_correction2 = [
                    1 - beta2 ** _get_value(step) for step in device_state_steps
                ]

                step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])

                bias_correction2_sqrt = [
                    bc**0.5 for bc in bias_correction2  # type: ignore[arg-type]
                ]

                if amsgrad:
                    device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)

                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                    # Use the max. for normalizing running avg. of gradient
                    exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
                else:
                    exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

                torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
                torch._foreach_add_(exp_avg_sq_sqrt, eps)
                torch._foreach_addcdiv_(
                    device_params,
                    device_exp_avgs,
                    exp_avg_sq_sqrt,
                    step_size,  # type: ignore[arg-type]
                )
            elif mars_type == "mars-lion":
                signs = torch._foreach_sign(device_exp_avgs)
                torch._foreach_add_(device_params, torch._foreach_mul(signs, -lr))
                del signs
            elif mars_type == "mars-sgd":
                torch._foreach_add_(device_params, torch._foreach_mul(device_exp_avgs, -lr))
        else:
            is_param_2d = [len(device_params[i].shape) == 2 for i in range(len(device_params))]
            is_param_2d_float = [torch.tensor(float(is_param_2d[i])).to(device_params.device) for i in range(len(device_params))]
            is_param_not_2d_float = [torch.tensor(float(not is_param_2d[i])).to(device_params.device) for i in range(len(device_params))]

            element_lr = torch._foreach_add(torch._foreach_mul(is_param_not_2d_float, lr * lr_1d_factor), lr)
            if weight_decay != 0:
                lr_wd_2d = torch._foreach_mul(is_param_2d_float, lr * weight_decay)
                lr_wd_1d = torch._foreach_mul(is_param_not_2d_float, lr * lr_1d_factor * weight_decay_1d)
                lr_wd = torch._foreach_add(lr_wd_2d, lr_wd_1d)
                del lr_wd_2d
                del lr_wd_1d
                torch._foreach_mul_(device_params, 1 - lr_wd)
                del lr_wd

            # Decay the first and second moment running average coefficient
            c_t = torch._foreach_sub(device_grads, device_last_grads)
            torch._foreach_mul_(c_t, gamma * (device_beta1 / (1. - device_beta1)))
            torch._foreach_add_(c_t, device_grads)
            # if c_t_norm > 1. for some c_t, then clip c_t to c_t / c_t_norm
            c_t_norm = torch._foreach_norm(c_t)
            c_t = [c_t[i] / c_t_norm[i] if c_t_norm[i] > 1.0 else c_t[i] for i in range(len(c_t))]
            
            c_t = [c_t[i] if is_param_2d[i] else device_grads[i] for i in range(len(c_t))]
            
            beta_1_2d = torch._foreach_mul(is_param_2d_float, device_beta1)
            beta_1_1d = torch._foreach_mul(is_param_not_2d_float, betas_1d[0])
            beta_1 = torch._foreach_add(beta_1_2d, beta_1_1d)
            # Decay the first and second moment running average coefficient
            torch._foreach_mul_(device_exp_avgs, beta_1)
            torch._foreach_add_(device_exp_avgs, torch._foreach_mul(c_t, 1 - beta_1))
            del beta_1_2d
            del beta_1_1d  
                        
            if mars_type == "mars-adamw":
                beta_2_2d = torch._foreach_mul(is_param_2d_float, beta2)
                beta_2_1d = torch._foreach_mul(is_param_not_2d_float, betas_1d[1])
                beta_2 = torch._foreach_add(beta_2_2d, beta_2_1d)
                torch._foreach_mul_(device_exp_avg_sqs, beta_2)
                torch._foreach_add_(device_exp_avg_sqs, torch._foreach_mul(c_t, 1 - beta_2))

                # Delete the local intermediate(s) since they won't be used anymore to save on peak memory
                del device_grads
                del beta_2_2d
                del beta_2_1d

                bias_correction1 = [
                    1 - beta_1[i] ** _get_value(device_state_steps[i]) for i in range(len(device_state_steps))
                ]
                bias_correction2 = [
                    1 - beta_2[i] ** _get_value(device_state_steps[i]) for i in range(len(device_state_steps))
                ]
                del beta_2
                step_size = _stack_if_compiling([(element_lr[i] / bias_correction1[i]) * -1 for i in range(len(bias_correction1))])

                bias_correction2_sqrt = [
                    bc**0.5 for bc in bias_correction2  # type: ignore[arg-type]
                ]

                if amsgrad:
                    device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)

                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                    # Use the max. for normalizing running avg. of gradient
                    exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
                else:
                    exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

                torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
                torch._foreach_add_(exp_avg_sq_sqrt, eps)
                torch._foreach_addcdiv_(
                    device_params,
                    device_exp_avgs,
                    exp_avg_sq_sqrt,
                    step_size,  # type: ignore[arg-type]
                )
            elif mars_type == "mars-lion":
                signs = torch._foreach_sign(device_exp_avgs)
                
                torch._foreach_add_(device_params, torch._foreach_mul(signs, -element_lr))
                del signs
            elif mars_type == "mars-sgd":
                element_lr = torch._foreach_add(torch._foreach_mul(is_param_not_2d_float, lr * lr_1d_factor), lr)
                torch._foreach_add_(device_params, torch._foreach_mul(device_exp_avgs, -element_lr)) 
              
            del beta_1              
            del element_lr

def _fused_mars_single_tensor(
    params: List[Tensor],
    grads: List[Tensor],
    last_grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    lr: Union[Tensor, float],
    weight_decay: float,
    beta1: Union[Tensor, float],
    beta2: Union[Tensor, float],
    eps: float,
    amsgrad: bool,
    gamma: float,
    *,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
    optimize_1d: bool,
    lr_1d_factor: float,
    betas_1d: Tuple[float, float],
    weight_decay_1d: float,
    mars_type: str,
):
    for i, param in enumerate(params):
        p_data_fp32 = param.data.float()
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        use_mars = optimize_1d or (len(grad.shape) == 2)
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]
        else:
            max_exp_avg_sq = exp_avg_sqs[i]
        last_grad = last_grads[i]
        step_t = state_steps[i] + 1
        if use_mars:
            beta1_, beta2_ = beta1, beta2
            c_t_ = (grad - last_grad).mul(gamma * (beta1 / (1. - beta1))).add(grad)
            c_t_norm = torch.norm(c_t_)
            c_t_factor = 1. / c_t_norm if c_t_norm > 1. else 1.
            grad = c_t_ * c_t_factor
            LR = lr
            WD = weight_decay
        else:
            beta1_, beta2_ = betas_1d
            LR = lr * lr_1d_factor
            WD = weight_decay_1d
        bias_correction1 = 1 - beta1_**step_t
        bias_correction2 = 1 - beta2_**step_t

        # optimize_1d or is_grad_2d -> use_mars (deprecated)
        # c_t_ -> grad
        # LR, WD combined, gamma deprecated
        with torch.cuda.device(param.device):
            import fused_mars
            fused_mars.mars_single_tensor(
                p_data_fp32,
                grad,
                exp_avg,
                exp_avg_sq, beta1_, beta2_, 
                bias_correction1, bias_correction2, 
                LR, WD, eps, amsgrad, max_exp_avg_sq
            )
        state_steps[i] = step_t
        
import fused_mars
def _fused_mars_multi_tensor(
    params: List[Tensor],
    grads: List[Tensor],
    last_grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    lr: Union[Tensor, float],
    weight_decay: float,
    beta1: Union[Tensor, float],
    beta2: Union[Tensor, float],
    eps: float,
    amsgrad: bool,
    gamma: float,
    *,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
    optimize_1d: bool,
    lr_1d_factor: float,
    betas_1d: Tuple[float, float],
    weight_decay_1d: float,
    mars_type: str,
):
    # here is_grad_2d == optimize_1d || is_grad_2d
    # is_grad_2d = [torch.tensor((len(grad.shape) == 2) or optimize_1d, dtype=params[0].dtype, device=params[0].device) for grad in grads]
    is_grad_2d = []
    # c_t_list = []
    var_reduction_factor = gamma * (beta1 / (1. - beta1))
    for i, param in enumerate(params):
        step_t = state_steps[i] + 1
        state_steps[i] = step_t
        grad_ = grads[i]

        if len(param.shape) == 2 or optimize_1d:
            # is_grad_2d.append(param * 0 + 1)
            is_grad_2d.append(torch.ones_like(param, dtype=param.dtype, device=param.device))
            # c_t_ = (grad_ - last_grads[i]).mul(var_reduction_factor).add(grad_)
            # c_t_norm = torch.norm(c_t_)
            # c_t_factor = 1. / c_t_norm if c_t_norm > 1. else 1.
            # # c_t_ = c_t_ * c_t_factor
            # grads[i] = c_t_ * c_t_factor
            grad_.add_(grad_ - last_grads[i], alpha=var_reduction_factor)
            grad_factor = torch.min(1. / torch.norm(grad_), torch.tensor(1.0, device=grad_.device))
            grad_.mul_(grad_factor)
        else:
            # is_grad_2d.append(param * 0 + 1)
            is_grad_2d.append(torch.zeros_like(param, dtype=param.dtype, device=param.device))
            # c_t_ = grad_
        # c_t_list.append(c_t_)
    
    if amsgrad:
        max_exp_avg_sq = max_exp_avg_sqs
    else:
        max_exp_avg_sq = exp_avg_sqs
    bias_correction1 = 1 - beta1**step_t
    bias_correction2 = math.sqrt(1 - beta2**step_t) # here use sqrt for bias_correction2
    beta_1_1d, beta_2_1d = betas_1d
    bc_1_1d = 1 - beta_1_1d**step_t
    bc_2_1d = math.sqrt(1 - beta_2_1d**step_t)
    multi_tensor_applier = MultiTensorApply(2048 * 32)
    _dummy_overflow_buf = torch.cuda.IntTensor([0])
    multi_tensor_applier(
        fused_mars.mars_multi_tensor, _dummy_overflow_buf,
        # [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sq, c_t_list, is_grad_2d],
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sq, is_grad_2d],
        beta1, beta2, 
        bias_correction1, bias_correction2, 
        lr, weight_decay, eps, amsgrad, gamma,
        optimize_1d, lr_1d_factor, beta_1_1d, beta_2_1d, 
        bc_1_1d, bc_2_1d, weight_decay_1d)
    for item in is_grad_2d:
        del item
    del is_grad_2d
    torch.cuda.empty_cache()

class MARS(Optimizer):
    def __init__(self, params, lr=3e-3, betas=(0.95, 0.99), eps=1e-8, weight_decay=0., amsgrad=False, gamma=0.025, 
                 is_approx=True, mars_type="mars-adamw", optimize_1d=False, lr_1d=3e-3, betas_1d=(0.9, 0.95), weight_decay_1d=0.1,
                 *, maximize: bool = False, foreach: Optional[bool] = None, capturable: bool = False, differentiable: bool = False,
                fused: Optional[bool] = None,):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        assert mars_type in ["mars-adamw", "mars-lion", "mars-shampoo", "mars-sgd"], "MARS type not supported"
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, 
                        mars_type=mars_type, gamma=gamma, 
                        optimize_1d=optimize_1d, weight_decay_1d=weight_decay_1d,
                        foreach=foreach,
                        maximize=maximize,
                        capturable=capturable,
                        differentiable=differentiable,
                        fused=fused,)
        super(MARS, self).__init__(params, defaults)
        self.eps = eps
        self.lr = lr
        self.weight_decay=weight_decay
        self.amsgrad = amsgrad
        self.step_num = 0
        self.is_approx = is_approx
        self.gamma = gamma
        self.mars_type = mars_type
        self.optimize_1d = optimize_1d
        self.lr_1d_factor = lr_1d / lr
        self.weight_decay_1d = weight_decay_1d
        self.betas_1d = betas_1d
        if fused:
            _check_fused_available()

    @torch.no_grad()
    def update_last_grad(self):
        if not self.is_approx:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    if "last_grad" not in state:
                        state["last_grad"] = torch.zeros_like(p)
                    state["last_grad"].zero_().add_(state["previous_grad"], alpha=1.0)
    @torch.no_grad()
    def update_previous_grad(self):
        if not self.is_approx:
            for group in self.param_groups:
                #print ("para name", len(group['params']), len(group['names']), group['names'])
                for p in group['params']:
                    # import pdb
                    # pdb.set_trace()
                    if p.grad is None:
                        print (p, "grad is none")
                        continue
                    state = self.state[p]
                    if "previous_grad" not in state:
                        state['previous_grad'] = torch.zeros_like(p)
                    state['previous_grad'].zero_().add_(p.grad, alpha=1.0)

    def __setstate__(self, state):
        super(MARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        last_grads,
        amsgrad,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                if group["fused"]:
                    _device_dtype_check_for_fused(p)
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = (
                    torch.zeros(
                        (),
                        dtype=_get_scalar_dtype(is_fused=group["fused"]),
                        device=p.device,
                    )
                    if group["capturable"] or group["fused"]
                    else torch.tensor(0.0, dtype=_get_scalar_dtype())
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state['last_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            last_grads.append(state["last_grad"])

            if group["amsgrad"]:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])
            if group["differentiable"] and state["step"].requires_grad:
                raise RuntimeError(
                    "`requires_grad` is not supported for `step` in differentiable mode"
                )

            # Foreach without capturable does not support a tensor lr
            if (
                group["foreach"]
                and isinstance(group["lr"], Tensor)
                and not group["capturable"]
            ):
                raise RuntimeError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )

            state_steps.append(state["step"])
        return has_complex
    
    @_use_grad_for_differentiable
    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError('FusedAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments.')
        
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()
        real_update = 0
        real_update_wo_lr = 0
        gamma = self.gamma
        
        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            last_grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            max_exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []
            amsgrad: bool = group["amsgrad"]
            beta1, beta2 = cast(Tuple[float, float], group["betas"])

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                last_grads,
                amsgrad,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )
            mars(
                params_with_grad,
                grads,
                last_grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                gamma=gamma,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                has_complex=has_complex,
                mars_type=self.mars_type,
                optimize_1d=self.optimize_1d,
                lr_1d_factor=self.lr_1d_factor,
                betas_1d=self.betas_1d,
                weight_decay_1d=group["weight_decay_1d"],
            )
            if self.is_approx:
                for i in range(len(params_with_grad)):
                    p = params_with_grad[i]
                    state = self.state[p]
                    state['last_grad'] = grads[i]
        self.step_num += 1
        return loss




@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_mars)
def mars(
    params: List[Tensor],
    grads: List[Tensor],
    last_grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    gamma: float,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    mars_type: str,
    optimize_1d: bool,
    lr_1d_factor: float,
    betas_1d: Tuple[float, float],
    weight_decay_1d: float,
):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    if not torch.compiler.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    torch_jit_is_scripting = torch.jit.is_scripting()
    if foreach and torch_jit_is_scripting:
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch_jit_is_scripting:
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if fused and not torch_jit_is_scripting and mars_type == "mars-adamw":
        if foreach:
            func = _fused_mars_multi_tensor
        else:
            func = _fused_mars_single_tensor
    elif not optimize_1d or mars_type == "mars-shampoo":
        func = _single_tensor_mars
    elif foreach and not torch_jit_is_scripting:
        func = _multi_tensor_mars
    else:
        func = _single_tensor_mars

    func(
        params,
        grads,
        last_grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        gamma=gamma,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        found_inf=found_inf,
        grad_scale=grad_scale,
        has_complex=has_complex,
        optimize_1d=optimize_1d,
        lr_1d_factor=lr_1d_factor,
        betas_1d=betas_1d,
        weight_decay_1d=weight_decay_1d,
        mars_type=mars_type
    )
        
def _check_fused_available():
    try:
        import fused_mars
    except ImportError as exc:
        if torch.cuda.is_available():
            # The module should be available but isn't. Try to
            # help the user in this case.
            raise ImportError((
                str(exc)
                + (
                    '\nThis could be caused by not having compiled '
                    'the CUDA extension during package installation. '
                    'Please try to re-install the package with '
                    'the environment flag `FORCE_CUDA=1` set.'
                )
            ))
        else:
            raise ImportError(
                str(exc) + '\nFused MARS does not support CPU.')