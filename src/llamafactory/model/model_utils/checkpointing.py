# Copyright 2024 HuggingFace Inc., Daniel Han-Chen & the Unsloth team and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers and PEFT library,
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/modeling_utils.py
# https://github.com/huggingface/peft/blob/v0.10.0/src/peft/utils/other.py
# and the Unsloth library.
# https://github.com/unslothai/unsloth/blob/July-2024/unsloth/models/_utils.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from functools import WRAPPER_ASSIGNMENTS, partial, wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch

from ...extras import logging
from ...extras.constants import LAYERNORM_NAMES


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def get_unsloth_gradient_checkpointing_func() -> Callable:
    class UnslothGradientCheckpointing(torch.autograd.Function):
        r"""
        Saves VRAM by smartly offloading to RAM.
        """

        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(
            ctx: "torch.autograd.Function",
            forward_function: "torch.Module",
            hidden_states: "torch.Tensor",
            *args: Union["torch.Tensor", Any],
        ) -> "torch.Tensor":
            saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
            with torch.no_grad():
                output = forward_function(hidden_states, *args)

            ctx.save_for_backward(saved_hidden_states)
            ctx.forward_function = forward_function
            ctx.args = args
            return output

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx: "torch.autograd.Function", grad_output: "torch.Tensor") -> "torch.Tensor":
            (hidden_states,) = ctx.saved_tensors
            hidden_states = hidden_states.to("cuda", non_blocking=True).detach()
            hidden_states.requires_grad_(True)
            with torch.enable_grad():
                (output,) = ctx.forward_function(hidden_states, *ctx.args)

            torch.autograd.backward(output, grad_output)
            return (None, hidden_states.grad) + (None,) * len(ctx.args)

    return UnslothGradientCheckpointing.apply

def get_custom_gradient_checkpointing_func(gradient_checkpointing_func: Callable) -> Callable:
    r"""
    Only applies gradient checkpointing to trainable layers.

    Fix:
    - checkpoint may not accept **kwargs
    - more importantly: Tensor kwargs MUST be passed as checkpoint inputs,
      otherwise reentrant checkpoint can capture outer graph and cause
      "backward through graph a second time".
    """

    @wraps(gradient_checkpointing_func, assigned=WRAPPER_ASSIGNMENTS + ("__self__",))
    def custom_gradient_checkpointing_func(func: Callable, *args: Union["torch.Tensor", Any], **kwargs):
        module: "torch.nn.Module" = func.__self__

        # 1) 只对可训练层启用：把浮点 Tensor 输入标记为需要梯度（保持你原逻辑）
        if any(p.requires_grad for p in module.parameters()):
            for a in args:
                if torch.is_tensor(a) and torch.is_floating_point(a):
                    a.requires_grad_(True)

        # 2) 将 kwargs 分成：Tensor kwargs vs 非 Tensor kwargs
        if kwargs:
            tensor_kw_keys = []
            tensor_kw_vals = []
            static_kwargs = {}
            for k, v in kwargs.items():
                if torch.is_tensor(v):
                    tensor_kw_keys.append(k)
                    tensor_kw_vals.append(v)
                else:
                    static_kwargs[k] = v

            # 同样对 tensor kwargs 做 requires_grad_(True)（如果浮点且该层可训练）
            if any(p.requires_grad for p in module.parameters()):
                for v in tensor_kw_vals:
                    if torch.is_floating_point(v):
                        v.requires_grad_(True)

            n_tkw = len(tensor_kw_keys)

            def wrapped(*inputs):
                # inputs = 原 args + tensor_kw_vals
                if n_tkw > 0:
                    base_inputs = inputs[:-n_tkw]
                    tkw_inputs = inputs[-n_tkw:]
                    dyn_kwargs = dict(static_kwargs)
                    for k, v in zip(tensor_kw_keys, tkw_inputs):
                        dyn_kwargs[k] = v
                    return func(*base_inputs, **dyn_kwargs)
                else:
                    return func(*inputs, **static_kwargs)

            # 关键：把 tensor kwargs 当作“显式输入”交给 checkpoint
            return gradient_checkpointing_func(wrapped, *args, *tensor_kw_vals)

        # 3) 没有 kwargs：正常走
        return gradient_checkpointing_func(func, *args)

    return custom_gradient_checkpointing_func
# def get_custom_gradient_checkpointing_func(gradient_checkpointing_func: Callable) -> Callable:
#     r"""
#     Only applies gradient checkpointing to trainable layers.
#     """

#     @wraps(gradient_checkpointing_func, assigned=WRAPPER_ASSIGNMENTS + ("__self__",))
#     def custom_gradient_checkpointing_func(func: Callable, *args: Union["torch.Tensor", Any], **kwargs):
#         module: "torch.nn.Module" = func.__self__

#         if any(param.requires_grad for param in module.parameters()):
#             for arg in args:
#                 if torch.is_tensor(arg) and torch.is_floating_point(arg):
#                     arg.requires_grad_(True)

#         return gradient_checkpointing_func(func, *args, **kwargs)

#     return custom_gradient_checkpointing_func


def _gradient_checkpointing_enable(
    self: "PreTrainedModel",
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None,
    use_unsloth_gc: bool = False,
) -> None:
    r"""
    Activates gradient checkpointing for the current model.

    Modification of the original method to enable gradient checkpointing for block-wise optimizer.
    """
    from torch.utils.checkpoint import checkpoint

    if not self.supports_gradient_checkpointing:
        raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": True}

    if use_unsloth_gc:
        gradient_checkpointing_func = get_unsloth_gradient_checkpointing_func()
    else:
        gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)

    gradient_checkpointing_func = get_custom_gradient_checkpointing_func(gradient_checkpointing_func)
    if "value" in inspect.signature(self._set_gradient_checkpointing).parameters:  # old GC format
        self.apply(partial(self._set_gradient_checkpointing, value=True))
        self.enable_input_require_grads()
        logger.warning_once("You are using the old GC format, some features (e.g. BAdam) will be invalid.")
    else:  # have already enabled input require gradients
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)


def _fp32_forward_post_hook(
    module: "torch.nn.Module", args: Tuple["torch.Tensor"], output: "torch.Tensor"
) -> "torch.Tensor":
    return output.to(torch.float32)


def prepare_model_for_training(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    r"""
    Includes:
        (1) cast the layernorm in fp32
        (2) make output embedding layer require grads
        (3) add the upcasting of the lm_head in fp32
    """
    if model_args.upcast_layernorm:
        logger.info_rank0("Upcasting layernorm weights in float32.")
        for name, param in model.named_parameters():
            if param.ndim == 1 and any(ln_name in name for ln_name in LAYERNORM_NAMES):
                param.data = param.data.to(torch.float32)

    if not model_args.disable_gradient_checkpointing:
        if not getattr(model, "supports_gradient_checkpointing", False):
            logger.warning_rank0("Current model does not support gradient checkpointing.")
        else:
            # use_reentrant=False might increase VRAM usage (have not been empirically verified yet)
            # According to: https://github.com/huggingface/transformers/issues/28339
            gradient_checkpointing_enable = partial(
                _gradient_checkpointing_enable, use_unsloth_gc=model_args.use_unsloth_gc
            )
            model.gradient_checkpointing_enable = MethodType(gradient_checkpointing_enable, model)
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": model_args.use_reentrant_gc}
            )
            setattr(model.config, "use_cache", False)  # turn off when gradient checkpointing is enabled
            logger.info_rank0("Gradient checkpointing enabled.")

    if model_args.upcast_lmhead_output:
        output_layer = model.get_output_embeddings()
        if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
            logger.info_rank0("Upcasting lm_head outputs in float32.")
            output_layer.register_forward_hook(_fp32_forward_post_hook)
