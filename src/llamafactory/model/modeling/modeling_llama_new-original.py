# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig
import wandb

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "LlamaConfig"

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaLinearScalingRotaryEmbedding` is deprecated an will be removed in v4.46. Please use "
            "`LlamaRotaryEmbedding`, which now also does linear scaling (simply pass the model config to __init__)."
        )
        kwargs["rope_type"] = "linear"
        super().__init__(*args, **kwargs)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaDynamicNTKScalingRotaryEmbedding` is deprecated an will be removed in v4.46. Please use "
            "`LlamaRotaryEmbedding`, which now also does dynamic ntk scaling (simply pass the model config to "
            "__init__)."
        )
        kwargs["rope_type"] = "dynamic"
        super().__init__(*args, **kwargs)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]
        self.down_proj._is_mlp_output = True  # 标记为MLP输出层
    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.o_proj._is_attention_output = True  # 标记为注意力输出层
        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        ponder_gate: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# class LlamaFlashAttention2(LlamaAttention):
#     """
#     Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
#     untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
#     flash attention and deal with padding tokens in case the input contains any of them.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
#         # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
#         # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
#         self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
#         ponder_gate: Optional[torch.Tensor] = None,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         if isinstance(past_key_value, StaticCache):
#             raise ValueError(
#                 "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
#                 "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
#             )

#         output_attentions = False

#         bsz, q_len, _ = hidden_states.size()

#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         # Flash attention requires the input to have the shape
#         # batch_size x seq_length x head_dim x hidden_dim
#         # therefore we just need to keep the original shape
#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#         if position_embeddings is None:
#             logger.warning_once(
#                 "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
#                 "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
#                 "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
#                 "removed and `position_embeddings` will be mandatory."
#             )
#             cos, sin = self.rotary_emb(value_states, position_ids)
#         else:
#             cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

#         # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
#         # to be able to avoid many of these transpose/reshape/view.
#         query_states = query_states.transpose(1, 2)
#         key_states = key_states.transpose(1, 2)
#         value_states = value_states.transpose(1, 2)

#         dropout_rate = self.attention_dropout if self.training else 0.0

#         # In PEFT, usually we cast the layer norms in float32 for training stability reasons
#         # therefore the input hidden states gets silently casted in float32. Hence, we need
#         # cast them back in the correct dtype just to be sure everything works as expected.
#         # This might slowdown training & inference so it is recommended to not cast the LayerNorms
#         # in fp32. (LlamaRMSNorm handles it correctly)

#         input_dtype = query_states.dtype
#         if input_dtype == torch.float32:
#             if torch.is_autocast_enabled():
#                 target_dtype = torch.get_autocast_gpu_dtype()
#             # Handle the case where the model is quantized
#             elif hasattr(self.config, "_pre_quantization_dtype"):
#                 target_dtype = self.config._pre_quantization_dtype
#             else:
#                 target_dtype = self.q_proj.weight.dtype

#             logger.warning_once(
#                 f"The input hidden states seems to be silently casted in float32, this might be related to"
#                 f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
#                 f" {target_dtype}."
#             )

#             query_states = query_states.to(target_dtype)
#             key_states = key_states.to(target_dtype)
#             value_states = value_states.to(target_dtype)

#         attn_output = _flash_attention_forward(
#             query_states,
#             key_states,
#             value_states,
#             attention_mask,
#             q_len,
#             position_ids=position_ids,
#             dropout=dropout_rate,
#             sliding_window=getattr(self, "sliding_window", None),
#             use_top_left_mask=self._flash_attn_uses_top_left_mask,
#             is_causal=self.is_causal,
#         )

#         attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
#         attn_output = self.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value

class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module with Ponder mechanism injected.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ponder_gate: Optional[torch.Tensor] = None,  # <--- 新增参数
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        
        # RoPE 仍然在原始维度上进行
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # Cache 更新使用的是原始的 key/value (未扩展)，这符合 NeoX 的逻辑
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim].
        # 此时形状变为 [B, L, H, D]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        
        # ### PONDER MODIFICATION START ###
        # 在这里进行维度扩展。此时数据形状为 [Batch, Seq_Len, Num_Heads, Head_Dim]
        
        d_orig = query_states.size(-1) # 记录原始 head_dim
        # 默认 scale
        softmax_scale = 1.0 / math.sqrt(d_orig) 

        if ponder_gate is not None:
            # 1. 准备参数
            # 注意：Llama 支持 GQA，Query 和 Key 的 Head 数可能不同，需分别获取
            B, L, H_q, D = query_states.shape
            H_k = key_states.shape[2] 
            
            eps = 1e-12
            thr = 1e-4
            pad_w = 8
            d_aug = d_orig + pad_w
            
            # 更新 softmax_scale 以匹配扩展后的维度
            softmax_scale = 1.0 / math.sqrt(float(d_aug))

            # 2. 处理 Gate
            if ponder_gate.dim() == 2: # [B, L]
                g = ponder_gate
            else:
                raise ValueError(f"Unsupported ponder_gate shape: {tuple(ponder_gate.shape)}; expect [B,L].")

            G = torch.log(g.clamp_min(eps)) # [B, L]
            G_hard = torch.where(g <= thr, torch.full_like(G, -1e4), G)
            
            # 3. 构造 Extra Tensor
            # NeoX 是 [B, H, L, pad_w]，Llama 这里是 [B, L, H, pad_w]
            # 所以 G_exp 需要在 dim=2 (Heads) 维度上进行 expand
            G_exp = G_hard.unsqueeze(2).expand(B, L, H_k) # [B, L, H_k]

            # 使用 new_zeros 确保 device 和 dtype 一致
            q_extra = query_states.new_zeros(B, L, H_q, pad_w)
            q_extra[..., 0] = math.sqrt(float(d_aug)) # 第一维设为 sqrt(d_aug)

            k_extra = key_states.new_zeros(B, L, H_k, pad_w)
            k_extra[..., 0] = G_exp # 第一维设为 G

            v_extra = value_states.new_zeros(B, L, H_k, pad_w) # 全 0

            # 4. 拼接
            query_states = torch.cat([query_states, q_extra], dim=-1) # [B, L, H_q, d_aug]
            key_states   = torch.cat([key_states,   k_extra], dim=-1) # [B, L, H_k, d_aug]
            value_states = torch.cat([value_states, v_extra], dim=-1) # [B, L, H_k, d_aug]
            
        # ### PONDER MODIFICATION END ###

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
            softmax_scale=softmax_scale, # <--- 显式传入修改后的 scale
        )

        # ### PONDER MODIFICATION START ###
        # 切片还原维度
        if ponder_gate is not None:
            # attn_output: [B, L, H, d_aug] -> [B, L, H, d_orig]
            attn_output = attn_output[..., :d_orig]
        # ### PONDER MODIFICATION END ###

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        ponder_gate: Optional[torch.FloatTensor] = None,  # [B, L_total, 1] 或 None
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            ponder_gate=ponder_gate,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    # def _init_weights(self, module):
    #     std = self.config.initializer_range
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 使用截断正态分布初始化权重，截断在3σ处，方差为2/(5*hidden_size)
            std = math.sqrt(2.0 / (5 * self.config.hidden_size))
            
            # 对于注意力层和MLP层的输出投影，根据网络深度进行额外缩放
            if hasattr(module, '_is_attention_output') or hasattr(module, '_is_mlp_output'):
                std = std / math.sqrt(2.0 * self.config.num_hidden_layers)
                print(f"small output std: {std}")
                
            nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-3*std, b=3*std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 使用截断正态分布初始化嵌入层权重
            std = math.sqrt(2.0 / (5 * self.config.hidden_size))
            nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-3*std, b=3*std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.checkpoint_num_layers = config.checkpoint_num_layers

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        ponder_gate: Optional[torch.FloatTensor] = None,  # [B, L_total, 1] 或 None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training and i < self.checkpoint_num_layers:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    ponder_gate=ponder_gate,          # <--- 透传
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    ponder_gate=ponder_gate,          # <--- 透传
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.ponder_head = torch.nn.Linear(config.hidden_size, self.config.more_iterations + 1, bias=True)
        # TODO: 初始化 ponder_head，可以试试 uniform 的情况
        # 注意：输出维度是 K+1，包含步骤0（阶段0，不ponder）到步骤K（ponder K次）

        # Minimum weight penalty loss scheduler parameters
        self.min_weight_penalty_lambda_start = float(getattr(self.config, "min_weight_penalty_lambda_start", 0.0))
        self.min_weight_penalty_lambda_max = float(getattr(self.config, "min_weight_penalty_lambda_max", 0.0))
        self.min_weight_penalty_warmup_steps = int(getattr(self.config, "min_weight_penalty_warmup_steps", 1000))
        self.min_weight_penalty_peak_steps = int(getattr(self.config, "min_weight_penalty_peak_steps", 4000))
        # Min weight penalty method: "accuracy" or "delta_loss"
        self.min_weight_penalty_method = getattr(self.config, "min_weight_penalty_method", "accuracy")
        self._internal_global_step = 0
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # Helper function to prepare inputs for stages (adapted from GPT-NeoX)
    def _prepare_inputs_for_stages(
        self,
        current_stages_list: List[torch.Tensor],
        batch_size: int, seq_len_orig: int, hidden_size: int, dev: torch.device,
        orig_pos_ids: torch.Tensor, orig_attn_mask_2d: torch.Tensor,
        scale_emb_val: Optional[torch.Tensor],
        config_obj,
        w_var: Optional[torch.Tensor] = None,     # [B, L_orig, K] 或 None
        force_eval_prune: Optional[bool] = None,  # None=遵循config；True=强制剪枝；False=强制不剪
    ):
        B, L, H = current_stages_list[0].shape
        num_stages_to_interleave = len(current_stages_list)
        interleaved_len = num_stages_to_interleave * seq_len_orig

        loop_base_embeds = torch.empty((batch_size, interleaved_len, hidden_size), dtype=current_stages_list[0].dtype, device=dev)
        loop_pos_ids = torch.empty((batch_size, interleaved_len), dtype=orig_pos_ids.dtype, device=dev)
        loop_attn_mask = torch.empty((batch_size, interleaved_len), dtype=orig_attn_mask_2d.dtype, device=dev)
        loop_input_embeds_final = torch.empty_like(loop_base_embeds)

        for stage_idx in range(num_stages_to_interleave):
            s_start = stage_idx  # offset for interleaving
            loop_base_embeds[:, s_start::num_stages_to_interleave, :] = current_stages_list[stage_idx]
            loop_pos_ids[:, s_start::num_stages_to_interleave] = orig_pos_ids
            loop_attn_mask[:, s_start::num_stages_to_interleave] = orig_attn_mask_2d

            current_base_slice = current_stages_list[stage_idx]
            loop_input_embeds_final[:, s_start::num_stages_to_interleave, :] = current_base_slice

        total_len = loop_input_embeds_final.size(1)  # = num_stages_to_interleave * L_orig
        ponder_w = loop_input_embeds_final.new_ones(B, total_len)

        if w_var is not None:
            if num_stages_to_interleave > 0:
                ponder_w[:, 0::num_stages_to_interleave] = w_var[..., 0]  # [B, L_orig]
            for i in range(1, num_stages_to_interleave):
                wi = w_var[..., i]  # [B, L_orig]
                ponder_w[:, i::num_stages_to_interleave] = wi

        do_prune = (not self.training) and getattr(self.config, "eval_prune_by_gate", True)
        if force_eval_prune is not None:
            do_prune = (not self.training) and bool(force_eval_prune)

        if do_prune:
            thr = float(getattr(self.config, "ponder_gate_eval_thr", 1e-4))

            # ==========================================================
            # ==== [CHANGED 1] 修 bug：hard_keep 应该是对每个位置的 mask，
            #              不能写 ponder_w[..., 0] 只取第0个位置
            # ==========================================================
            hard_keep = (ponder_w >= thr)  # [B, L*S] True=保留, False=剪掉

            # ==========================================================
            # ==== [CHANGED 2] shape-robust：确保 hard_keep 和 loop_attn_mask
            #              在 dim=1 上严格一致，避免偶现 2048 vs 4
            # ==========================================================
            if hard_keep.dim() == 1:
                hard_keep = hard_keep.unsqueeze(1)  # [B,1]

            # 如果某些分支让 hard_keep 变成 [B,S]（你遇到的 4），强制扩展到 [B, L*S]
            if hard_keep.shape[1] != loop_attn_mask.shape[1]:
                # 典型情况：hard_keep=[B,S]，S=num_stages_to_interleave；需要扩展到每个 stage 的 token
                if hard_keep.shape[1] == num_stages_to_interleave:
                    hard_keep = hard_keep.unsqueeze(-1).expand(B, num_stages_to_interleave, seq_len_orig).reshape(B, interleaved_len)
                else:
                    # 最保守的策略：直接报更清晰的错 + 打印形状，方便定位
                    if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
                        print(f"[ERR] hard_keep shape mismatch: hard_keep={tuple(hard_keep.shape)}, "
                            f"loop_attn_mask={tuple(loop_attn_mask.shape)}, S={num_stages_to_interleave}, L={seq_len_orig}")
                    raise RuntimeError(
                        f"hard_keep shape {tuple(hard_keep.shape)} cannot be aligned to loop_attn_mask {tuple(loop_attn_mask.shape)}"
                    )

            # ==========================================================
            # ==== [CHANGED 3] 可选 debug：只在 rank0 打印一次 shape 和剪枝比例
            # ==========================================================
            if getattr(self.config, "debug_eval_prune_shape", False):
                if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
                    keep_ratio = hard_keep.to(torch.float32).mean().item()
                    print(f"[DBG] do_prune={do_prune} thr={thr} "
                        f"loop_attn_mask={tuple(loop_attn_mask.shape)} ponder_w={tuple(ponder_w.shape)} "
                        f"hard_keep={tuple(hard_keep.shape)} keep_ratio={keep_ratio:.4f}")

            # 只把 mask 位置置 0（padding）；不把 ponder_w 硬化为 0/1，也不改 embed
            loop_attn_mask = loop_attn_mask * hard_keep.to(loop_attn_mask.dtype)

        return loop_input_embeds_final, loop_pos_ids, loop_attn_mask, ponder_w

    # Helper function to extract hidden states for interpolation/refinement (adapted from GPT-NeoX)
    def _extract_hidden_for_computation(self, model_outputs, selector_slice, config_obj): # config_obj is self.config
        # LlamaModelOutputWithPast has 'last_hidden_state' as the first element if not return_dict,
        # or as an attribute if return_dict=True.
        last_h_state = model_outputs.last_hidden_state if hasattr(model_outputs, 'last_hidden_state') else model_outputs[0]
        return last_h_state[:, selector_slice, :]

    # Helper function to compute interpolated embeddings (adapted from GPT-NeoX)
    def _compute_interpolated_embeds(self, weight: torch.Tensor, hidden: Union[torch.Tensor, Tuple[torch.Tensor, ...]], use_topk: bool = True) -> torch.Tensor:
        softmax_temp = getattr(self.config, 'softmax_temperature', 1.0)
        
        if use_topk:
            top_k_val = getattr(self.config, 'interpolation_topk', 100)
            logits = self.lm_head(hidden) # Use Llama's lm_head
            logits_scaled = logits / softmax_temp
            
            actual_top_k = min(top_k_val, logits_scaled.size(-1))
            if actual_top_k <= 0: # Avoid error if vocab_size is 0 or top_k is 0 or negative
                 # Return a zero tensor with the expected shape: (batch_size, sequence_length, hidden_size)
                return torch.zeros_like(hidden)


            topk_values, topk_indices = torch.topk(logits_scaled, k=actual_top_k, dim=-1)
            probs_topk = torch.softmax(topk_values, dim=-1)
            
            # Weight is self.model.embed_tokens.weight, shape (vocab_size, hidden_size)
            embedding_topk = weight[topk_indices] # Shape: (batch, seq_len, actual_top_k, hidden_size)
            interpolated_embeds = torch.einsum("bsk,bskh->bsh", probs_topk, embedding_topk) # k = actual_top_k
            return interpolated_embeds
        else: # use_topk is False (full softmax)
            logits = self.lm_head(hidden) / softmax_temp # Use Llama's lm_head
            probs = torch.softmax(logits, dim=-1)
            interpolated_embeds = torch.matmul(probs, weight) # weight is self.model.embed_tokens.weight
            return interpolated_embeds

    def _compute_ponder_weights(self, hs_stage0: torch.Tensor, return_logits: bool = False):
        K = self.config.more_iterations
        num_steps = K + 1

        logits_raw = self.ponder_head(hs_stage0)  # [B, L, K+1]
        s = torch.softmax(logits_raw, dim=-1)
        w = torch.flip(torch.cumsum(torch.flip(s, dims=[-1]), dim=-1), dims=[-1])

        idx = torch.arange(0, num_steps, device=hs_stage0.device, dtype=hs_stage0.dtype).view(1, 1, num_steps)
        expected = torch.sum(idx * s, dim=-1)

        ent = -torch.sum(s * torch.clamp(torch.log(s + 1e-12), min=-50), dim=-1)
        s_mean = s.mean(dim=(0, 1))
        ent_mean = -torch.sum(s_mean * torch.clamp(torch.log(s_mean + 1e-12), min=-50))

        if return_logits:
            return s, w, logits_raw, expected, ent, ent_mean
        else:
            return s, w, expected, ent, ent_mean

    def _compute_lambda_min_weight_penalty(self, global_step: int) -> float:
        """
        根据 global_step 计算当前的 lambda_min_weight_penalty 值。
        调度策略：
        - 前 min_weight_penalty_warmup_steps 步：保持 min_weight_penalty_lambda_start
        - min_weight_penalty_warmup_steps 到 min_weight_penalty_peak_steps：从 min_weight_penalty_lambda_start 线性增长到 min_weight_penalty_lambda_max
        - min_weight_penalty_peak_steps 之后：保持 min_weight_penalty_lambda_max
        """
        if global_step < self.min_weight_penalty_warmup_steps:
            return self.min_weight_penalty_lambda_start
        elif global_step >= self.min_weight_penalty_peak_steps:
            return self.min_weight_penalty_lambda_max
        else:
            # 线性插值：从 min_weight_penalty_warmup_steps 到 min_weight_penalty_peak_steps
            # 防止除零错误：如果 peak_steps <= warmup_steps，直接返回 lambda_max
            if self.min_weight_penalty_peak_steps <= self.min_weight_penalty_warmup_steps:
                return self.min_weight_penalty_lambda_max
            progress = (global_step - self.min_weight_penalty_warmup_steps) / (
                self.min_weight_penalty_peak_steps - self.min_weight_penalty_warmup_steps
            )
            return self.min_weight_penalty_lambda_start + progress * (
                self.min_weight_penalty_lambda_max - self.min_weight_penalty_lambda_start
            )

    def _accumulate_ponder_losses(self, entropy, expected_steps, ent_mean, w_var, global_step: Optional[int] = None, step_accuracies: Optional[torch.Tensor] = None, step_penalty_ratios: Optional[torch.Tensor] = None, step_ce_losses: Optional[torch.Tensor] = None):
        """
        把五类loss（熵、步数惩罚、diverse loss、weight_dist loss、min_weight_penalty loss）聚合为标量；CE主损在forward末尾已有。
        参数:
            entropy: [B, L] - 每个token的熵
            expected_steps: [B, L] - 每个token的期望步数
            ent_mean: 标量 - s_mean分布的熵（在_compute_ponder_weights中已计算）
            w_var: [B, L, K+1] - 权重分布
            global_step: 当前全局步数
            step_accuracies: [K+1] - 每个ponder步骤的准确率（用于wandb记录）
            step_penalty_ratios: [K+1] - 每个ponder步骤的惩罚比例（基于softmax概率，用于惩罚，方法1）
            step_ce_losses: [K+1] - 每个ponder步骤的CE loss（用于方法2），step_ce_losses[0]就是不ponder的CE loss
        返回：(ponder_aux_loss, entropy_mean, entropy_loss, cost_mean, cost_loss, diverse_mean, diverse_loss, weight_dist_mean, weight_dist_loss, min_weight_penalty_loss, current_lambda_entropy, current_lambda_ponder_cost, current_lambda_diverse, current_lambda_weight_dist, current_lambda_min_weight_penalty, penalty_ratios)
        """
        # [B, L] -> 标量
        entropy_mean = entropy.mean()
        cost_mean = expected_steps.mean()  # 期望步数越小越好

        # 计算当前的 lambda_entropy、lambda_ponder_cost、lambda_diverse、lambda_weight_dist 和 lambda_min_weight_penalty
        if global_step is None:
            global_step = getattr(self, "_internal_global_step", 0)
        current_lambda_min_weight_penalty = self._compute_lambda_min_weight_penalty(global_step)

        ponder_aux = 0.0
        # Compute minimum weight penalty loss
        # 惩罚每个阶段（w1, w2, w3, w4, w5等）中最低的一定比例的w值
        # 惩罚比例根据每个ponder步骤的预测准确率计算
        min_weight_penalty_loss = torch.tensor(0.0, device=entropy.device, dtype=entropy.dtype)
        penalty_ratios = {}  # 存储每个步骤的惩罚比例，用于wandb记录
        
        # 即使lambda为0，也要计算并记录penalty_ratios用于wandb监控
        if w_var is not None:
            # w_var shape: [B, L, K+1]
            B, L, K_plus_1 = w_var.shape
            K = K_plus_1 - 1  # K是more_iterations，不包括w0
            
            # 对每个阶段k（w1, w2, w3, w4, w5等），跳过w0（k=0），因为w0永远等于1，不需要惩罚
            # 先计算penalty_ratios（无论lambda是否为0，都要记录）
            for k in range(1, K_plus_1):
                # 使用传入的penalty_ratio（已经在前面计算好了）
                # step_penalty_ratios的索引对应关系：
                # - 方法1（accuracy）：step_penalty_ratios[0]是w0的，用于惩罚w1；step_penalty_ratios[1]是w1的，用于惩罚w2
                # - 方法2（delta_loss）：step_penalty_ratios[0]是w1的（delta1），用于惩罚w1；step_penalty_ratios[1]是w2的（delta2），用于惩罚w2
                if step_penalty_ratios is not None:
                    if self.min_weight_penalty_method == "delta_loss":
                        # 方法2：step_penalty_ratios[0]对应w1，step_penalty_ratios[1]对应w2，以此类推
                        step_idx = k - 1  # w_k对应step_penalty_ratios[k-1]
                    else:
                        # 方法1：step_penalty_ratios[0]对应w0（用于惩罚w1），step_penalty_ratios[1]对应w1（用于惩罚w2）
                        step_idx = k - 1  # w_k使用步骤k-1的惩罚比例
                    
                    if len(step_penalty_ratios) > step_idx:
                        penalty_ratio = float(step_penalty_ratios[step_idx].clamp(0.0, 1.0).item())
                    else:
                        # 如果没有惩罚比例，回退到原来的等差数列计算
                        penalty_ratio = (k - 1) / K_plus_1
                else:
                    # 如果没有惩罚比例，回退到原来的等差数列计算
                    penalty_ratio = (k - 1) / K_plus_1
                
                # 存储惩罚比例用于wandb记录
                penalty_ratios[f'w{k}'] = penalty_ratio
            
            # 只有当lambda > 0时才计算loss
            if current_lambda_min_weight_penalty > 0:
                total_min_mean_sum = torch.tensor(0.0, device=w_var.device, dtype=w_var.dtype)
                
                # 对每个阶段k（w1, w2, w3, w4, w5等），跳过w0（k=0），因为w0永远等于1，不需要惩罚
                prev_penalty_ratio = None
                for k in range(1, K_plus_1):
                    w_k = w_var[..., k]  # [B, L]
                    w_k_flat = w_k.flatten()  # [B*L]
                    
                    # 获取当前阶段的penalty_ratio
                    current_penalty_ratio = penalty_ratios.get(f'w{k}', (k - 1) / K_plus_1)
                    
                    # w1的penalty_ratio保持不变，w2及以后使用相对于前一个阶段的增量
                    if k == 1:
                        penalty_ratio = current_penalty_ratio
                    else:
                        # penalty_ratio = max(penalty_ratio(wk) - penalty_ratio(w(k-1)), 0)
                        penalty_ratio = max(current_penalty_ratio - prev_penalty_ratio, 0.0)
                    
                    # 保存当前penalty_ratio作为下一个阶段的prev_penalty_ratio
                    prev_penalty_ratio = current_penalty_ratio
                    
                    # 计算最低的指定比例的值
                    num_elements = w_k_flat.numel()
                    num_min = max(1, int(num_elements * penalty_ratio))
                    
                    # 获取最低的num_min个值
                    min_values, _ = torch.topk(w_k_flat, num_min, largest=False)
                    
                    # 计算该阶段最低值的平均值
                    min_mean = min_values.mean()
                    
                    # 累加每个阶段的平均值
                    total_min_mean_sum = total_min_mean_sum + min_mean
                
                # 乘以lambda（使用scheduler计算的值）
                min_weight_penalty_loss = current_lambda_min_weight_penalty * total_min_mean_sum
                ponder_aux = ponder_aux + min_weight_penalty_loss

        return (
            ponder_aux,
            min_weight_penalty_loss.detach(),
            current_lambda_min_weight_penalty,
            penalty_ratios,
        )

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0, # Retained from original Llama, used in non-recurrent path
        global_step: Optional[int] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""       
        if not hasattr(self, "_wandb_inited"):
            try:
                if wandb.run is None:  # 只在没 init 的时候 init
                    wandb.init(project="Ponder2-A", reinit=False)
                self._wandb_inited = True
                # wandb.define_metric("refine/step")  # x 轴
                # wandb.define_metric("refine/*", step_metric="refine/step")
                self._wandb_refine_defined = True
            except Exception as e:
                print(f"[WARN] wandb.init failed: {e}")
                self._wandb_inited = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # user_requested_output_hidden_states is for the final model pass in recurrent logic
        user_requested_output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.training:
            if global_step is None:
                global_step = self._internal_global_step
                self._internal_global_step += 1
            self._wandb_log_step = int(global_step)
        
        if getattr(self.config, "recurrent_model", False):
            # ---- Recurrent model logic (adapted from GPT-NeoX) ----
            if inputs_embeds is None:
                if input_ids is None: raise ValueError("input_ids must be provided for recurrent model if inputs_embeds is None")
                initial_embeds_raw = self.model.embed_tokens(input_ids)
            else:
                initial_embeds_raw = inputs_embeds

            B, L_orig, H = initial_embeds_raw.shape
            device = initial_embeds_raw.device

            if position_ids is None:
                original_repeated_position_ids = torch.arange(0, L_orig, dtype=torch.long, device=device).unsqueeze(0).expand(B, L_orig)
            else:
                original_repeated_position_ids = position_ids
            
            # Prepare 2D attention mask (B, L_orig)
            if attention_mask is None:
                original_attention_mask_2d = torch.ones((B, L_orig), dtype=torch.long, device=device)
            else:
                if attention_mask.ndim == 4: # e.g., (B, 1, L_query, L_key)
                    if attention_mask.shape[1] == 1 and attention_mask.shape[2] == L_orig and attention_mask.shape[3] == L_orig: # Standard causal mask shape
                        original_attention_mask_2d = torch.ones((B, L_orig), dtype=attention_mask.dtype, device=device) # Assume all non-padding if complex
                    elif attention_mask.shape[1] == 1 and attention_mask.shape[2] == 1: # (B, 1, 1, L_key) used by Llama for KV cache
                         original_attention_mask_2d = attention_mask[:, 0, 0, :L_orig] if L_orig <= attention_mask.shape[3] else attention_mask[:,0,0,:]
                    else: # Fallback or specific logic for other 4D masks
                        # This part might need refinement based on specific 4D mask structures encountered
                        original_attention_mask_2d = attention_mask.squeeze(1).any(dim=1).long() # A general heuristic
                        if original_attention_mask_2d.shape[1] != L_orig: # Ensure correct length
                             original_attention_mask_2d = torch.ones((B, L_orig), dtype=torch.long, device=device)


                elif attention_mask.ndim == 2: # (B, L_orig)
                    original_attention_mask_2d = attention_mask
                else:
                    raise ValueError(f"Unsupported attention_mask dimension for recurrent path: {attention_mask.ndim}. Expected 2D or 4D.")
            
            inputs_embeds0 = initial_embeds_raw
            weight_for_interpolation = self.model.embed_tokens.weight # This is W_e
            embed_scale = None

            if getattr(self.config, 'scale_embeds', False):
                if self.config.interpolation == True:
                    embed_scale = torch.sqrt(torch.tensor(H, dtype=initial_embeds_raw.dtype, device=device))
                else:
                    embed_scale = torch.sqrt(2.5*torch.tensor(H, dtype=initial_embeds_raw.dtype, device=device))
                inputs_embeds0 = initial_embeds_raw * embed_scale
                weight_for_interpolation = self.model.embed_tokens.weight * embed_scale
            
            scaled_inputs_embeds0_for_orig_slots = inputs_embeds0
            
            num_initial_interpolation_stages = getattr(self.config, 'more_iterations', 0)
            all_embeddings_stages: List[torch.Tensor] = [scaled_inputs_embeds0_for_orig_slots]

            # ---- 在 interpolation 循环之前，先计算 w_var ----
            # 这样第0轮就能使用 w_var，阶段0的 ponder_gate 就能使用 w_var[..., 0]
            if num_initial_interpolation_stages > 0:
                # 先运行一次 gpt_neox，只使用阶段0的输入，获取阶段0的 hidden states
                pre_iter_stages = [scaled_inputs_embeds0_for_orig_slots]
                pre_iter_input_embeds, pre_iter_pos_ids, pre_iter_attn_mask, pre_iter_ponder_w = self._prepare_inputs_for_stages(
                    pre_iter_stages, B, L_orig, H, device,
                    original_repeated_position_ids, original_attention_mask_2d,
                    embed_scale, self.config,
                    w_var=None  # 此时还没有 w_var，阶段0的 ponder_gate 会是 1.0
                )
                
                pre_iter_model_outputs = self.model(
                    inputs_embeds=pre_iter_input_embeds,
                    attention_mask=pre_iter_attn_mask,
                    position_ids=pre_iter_pos_ids,
                    past_key_values=None, use_cache=False,
                    output_attentions=False, output_hidden_states=user_requested_output_hidden_states,
                    return_dict=True, cache_position=None,
                    ponder_gate=pre_iter_ponder_w,  # 使用 prepare_gpt_inputs_for_stages 返回的 ponder_w
                )
                
                # 提取阶段0的 hidden states
                hs_E0_pre = self._extract_hidden_for_computation(
                    pre_iter_model_outputs, slice(0, None, 1), self.config
                )  # [B, L, H]
                
                # 计算 s_var 和 w_var（包含步骤0到K，共K+1个）
                s_var, w_var, expected_steps_var, entropy_var, ent_mean_var = self._compute_ponder_weights(hs_E0_pre)
            else:
                # 如果 K == 0，不需要计算 w_var
                s_var = None
                w_var = None
                expected_steps_var = None
                entropy_var = None
                ent_mean_var = None
            
            # ---- Initial Interpolation Loop ----
            # Intermediate model calls do not need full output_hidden_states unless _compute_interpolated_embeds changes
            # For simplicity and matching NeoX, we pass user_requested_output_hidden_states, but it might be optimized
            for iter_k_idx in range(num_initial_interpolation_stages):
                current_stages_for_iter_pass = list(all_embeddings_stages)
                num_stages_in_current_iter_input = len(current_stages_for_iter_pass)

                iter_input_embeds, iter_pos_ids, iter_attn_mask, iter_ponder_w = self._prepare_inputs_for_stages(
                    current_stages_for_iter_pass, B, L_orig, H, device,
                    original_repeated_position_ids, original_attention_mask_2d,
                    embed_scale, self.config,
                    w_var=w_var   # 现在第0轮也能使用 w_var 了
                )
                
                # LlamaModel does not use head_mask. past_key_values and use_cache are None/False for these internal loops.
                iter_model_outputs = self.model(
                    inputs_embeds=iter_input_embeds,
                    attention_mask=iter_attn_mask, 
                    position_ids=iter_pos_ids, 
                    past_key_values=None, use_cache=False, # No KV cache in interpolation/refinement
                    output_attentions=False, # No attentions needed for interpolation
                    output_hidden_states=False, # Need last_hidden_state for _extract_hidden_for_computation
                    return_dict=True, cache_position=None,
                    ponder_gate=iter_ponder_w,     # <--- 新增
                )
                
                interp_use_topk = getattr(self.config, 'interpolation_use_topk', False)
                
                computed_embeddings_from_hs_slices = []
                for stage_slice_idx in range(num_stages_in_current_iter_input):
                    current_slice_selector = slice(stage_slice_idx, None, num_stages_in_current_iter_input)
                    hidden_input_for_this_slice = self._extract_hidden_for_computation(
                        iter_model_outputs, current_slice_selector, self.config
                    )
                    if self.config.interpolation == True:
                        embedding = self._compute_interpolated_embeds(
                            weight_for_interpolation, hidden_input_for_this_slice, use_topk=interp_use_topk
                        )
                    else:
                        embedding = hidden_input_for_this_slice
                    computed_embeddings_from_hs_slices.append(embedding)

                # Logic for updating all_embeddings_stages (same as in GPT-NeoX)
                if iter_k_idx == 0:
                    # 第0轮：使用之前计算的 w_var，不需要重新计算
                    new_stage = computed_embeddings_from_hs_slices[0]  # from E_orig HS
                    staged = [all_embeddings_stages[0], new_stage]
                    # First iteration: computed_embeddings_from_hs_slices[0] (from E_orig's HS) is the first new interpolated stage (E_interp_0)
                    all_embeddings_stages.append(staged[1])
                else:
                    # Subsequent iterations:
                    # Refine existing interpolated stages E_interp_0 to E_interp_{iter_k_idx-1}
                    # all_embeddings_stages[j+1] (which is E_interp_j) is refined by
                    # computed_embeddings_from_hs_slices[j] (which is from E_interp_{j-1}'s HS, or E_orig's HS if j=0)
                    for j in range(iter_k_idx): # j from 0 to iter_k_idx-1
                        all_embeddings_stages[j+1] = computed_embeddings_from_hs_slices[j]
                    
                    # Add the new interpolated stage E_interp_{iter_k_idx}
                    # This is computed_embeddings_from_hs_slices[iter_k_idx] (from E_interp_{iter_k_idx-1}'s HS)
                    all_embeddings_stages.append(computed_embeddings_from_hs_slices[iter_k_idx])

            # ---- Refinement Loop ----
            if self.training:
                num_refinement_steps = getattr(self.config, "training_refinement_steps", 5)
                if getattr(self.config, "vary_refine_steps", False) and global_step is not None:
                    num_refinement_steps = torch.randint(self.config.more_iterations+1, self.config.more_iterations+4, (1,), device=device).item()  # avoid CPU sync from default CPU tensor
            else:
                if getattr(self.config, "vary_refine_steps", False):
                    num_refinement_steps = torch.randint(self.config.more_iterations+1, self.config.more_iterations+4, (1,), device=device).item()
                else:
                    num_refinement_steps = getattr(self.config, "eval_refinement_steps", 10)

            if num_initial_interpolation_stages > 0 and num_refinement_steps > 0:
                # ========= [OPT] 1) 这些 config 值放到循环外，避免每一步 getattr =========
                interp_use_topk = bool(getattr(self.config, "interpolation_use_topk", False))
                do_interp = bool(getattr(self.config, "interpolation", False))

                damping_alpha = float(getattr(self.config, "damping_alpha", 0.1))
                last_n_hard   = int(getattr(self.config, "last_n_steps_update_w", 1))
                damping_alpha = 0.0 if damping_alpha < 0.0 else (1.0 if damping_alpha > 1.0 else damping_alpha)
                last_n_hard   = 0 if last_n_hard < 0 else last_n_hard
                hard_tail_start = max(0, num_refinement_steps - last_n_hard)

                # ========= [OPT] 2) wandb define_metric 只做一次 + rank0 =========
                if getattr(self, "_wandb_refine_defined", False) is False:
                    try:
                        is_rank0 = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
                        if is_rank0 and (wandb.run is not None):
                            wandb.define_metric("refine/step")
                            wandb.define_metric("refine/*", step_metric="refine/step")
                        self._wandb_refine_defined = True
                    except Exception:
                        self._wandb_refine_defined = True  # 防止反复尝试
                
                for ref_idx in range(num_refinement_steps):
                    current_stages_for_refinement_pass = list(all_embeddings_stages)
                    num_stages_in_refinement_input = len(current_stages_for_refinement_pass)
                    S_ref = num_stages_in_refinement_input  # just alias

                    ref_input_embeds, ref_pos_ids, ref_attn_mask, ref_ponder_w = self._prepare_inputs_for_stages(
                        current_stages_for_refinement_pass, B, L_orig, H, device,
                        original_repeated_position_ids, original_attention_mask_2d,
                        embed_scale, self.config,
                        w_var=w_var,
                    )

                    ref_model_outputs = self.model(
                        inputs_embeds=ref_input_embeds,
                        attention_mask=ref_attn_mask,
                        position_ids=ref_pos_ids,
                        past_key_values=None, use_cache=False,
                        output_attentions=False, 
                        output_hidden_states=False, # Need last_hidden_state
                        return_dict=True, cache_position=None,
                        ponder_gate=ref_ponder_w,
                    )

                    # ---- update w/s (same behavior) ----
                    hs_E0_ref = self._extract_hidden_for_computation(
                        ref_model_outputs, slice(0, None, S_ref), self.config
                    )
                    s_new, w_new, expected_steps_new, entropy_new, ent_mean_new = self._compute_ponder_weights(hs_E0_ref)

                    use_hard = (ref_idx >= hard_tail_start)
                    if (s_var is None) or (w_var is None) or use_hard or (damping_alpha >= 1.0):
                        s_var, w_var = s_new, w_new
                        expected_steps_var, entropy_var, ent_mean_var = expected_steps_new, entropy_new, ent_mean_new
                    elif damping_alpha <= 0.0:
                        pass
                    else:
                        a = damping_alpha
                        one_minus_a = 1.0 - a
                        s_var = one_minus_a * s_var + a * s_new
                        w_var = one_minus_a * w_var + a * w_new
                        expected_steps_var = one_minus_a * expected_steps_var + a * expected_steps_new
                        entropy_var        = one_minus_a * entropy_var        + a * entropy_new
                        ent_mean_var       = one_minus_a * ent_mean_var       + a * ent_mean_new

                    # ========= [OPT] 3) 只在最后一轮做 MSE 统计，其他轮不 clone 不算 =========
                    is_last = (ref_idx == num_refinement_steps - 1)
                    mse_sum = 0.0
                    rel_sum = 0.0
                    mse_cnt = 0

                    # refine stages: i=0..K-1 refine all_embeddings_stages[i+1]
                    for i in range(num_initial_interpolation_stages):
                        # selector for hidden slice of stage i
                        refine_source_selector = slice(i, None, S_ref)
                        hidden_input_for_refine_fn = self._extract_hidden_for_computation(
                            ref_model_outputs, refine_source_selector, self.config
                        )

                        if do_interp:
                            refined_embeds_for_stage = self._compute_interpolated_embeds(
                                weight_for_interpolation, hidden_input_for_refine_fn, use_topk=interp_use_topk
                            )
                        else:
                            refined_embeds_for_stage = hidden_input_for_refine_fn

                        if is_last:
                            # [OPT] 不 clone list；对“当前 stage”的上一轮值做 detach 引用，然后立即算 diff
                            prev = all_embeddings_stages[i + 1].detach()
                            diff = refined_embeds_for_stage - prev
                            mse = (diff * diff).mean(dtype=torch.float32)  # fp32 accumulate 更稳定
                            denom = (prev * prev).mean(dtype=torch.float32)
                            rel = mse / denom.clamp_min(1e-9)

                            mse_sum += mse
                            rel_sum += rel
                            mse_cnt += 1

                        all_embeddings_stages[i + 1] = refined_embeds_for_stage

                    # ========= [OPT] 4) wandb log：只在最后一轮 log MSE；其他轮只 log gate 概览（可选） =========
                    is_rank0 = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
                    if is_rank0 and (wandb.run is not None):
                        refine_step_global = ref_idx
                        if hasattr(self, "_wandb_log_step"):
                            refine_step_global += self._wandb_log_step * num_refinement_steps

                        log_payload = {"refine/step": refine_step_global}

                        if is_last:
                            mse_mean_current = (mse_sum / max(1, mse_cnt)).item()
                            rel_mse_mean_current = (rel_sum / max(1, mse_cnt)).item()
                            log_payload["refine/mse_mean/last"] = mse_mean_current
                            log_payload["refine/rel_mse_mean/last"] = rel_mse_mean_current

                        # gate stats（这些很便宜，但也会有同步；你可以只在 last 记录）
                        if is_last:
                            w_stage_mean = w_var.mean(dim=(0, 1)).to(torch.float32)  # [K+1]
                            for k in range(w_stage_mean.numel()):
                                log_payload[f"refine/w_stage_mean/{k}"] = w_stage_mean[k].item()
                                w_k = w_var[..., k]
                                log_payload[f"refine/gate/{k}<1e-4"] = (w_k < 1e-4).to(torch.float32).mean().item()

                        wandb.log(log_payload, commit=False)

            # ---- Final pass after all interpolations and refinements ----
            final_stages_for_pass = list(all_embeddings_stages)
            num_total_stages_final = len(final_stages_for_pass)
            S = num_total_stages_final
            K = S - 1
            ponder_aux_loss = torch.tensor(0.0, device=device, dtype=initial_embeds_raw.dtype)


            final_interleaved_input_embeds, final_interleaved_pos_ids, final_interleaved_attn_mask, final_gate = \
                self._prepare_inputs_for_stages(
                    final_stages_for_pass, B, L_orig, H, device,
                    original_repeated_position_ids, original_attention_mask_2d,
                    embed_scale, self.config, w_var=w_var, force_eval_prune=False)
            
            final_pass_model_outputs = self.model(
                inputs_embeds=final_interleaved_input_embeds,
                attention_mask=final_interleaved_attn_mask, 
                position_ids=final_interleaved_pos_ids, 
                past_key_values=past_key_values, # Use user's past_key_values for generation
                use_cache=use_cache,             # User's cache preference
                output_attentions=output_attentions,
                output_hidden_states=user_requested_output_hidden_states, # User's preference
                return_dict=True, 
                cache_position=cache_position,   # User's cache_position
                ponder_gate=final_gate,
            )
            
            all_hidden_states_from_final_pass = final_pass_model_outputs.last_hidden_state
            hidden_states_final = torch.zeros_like(initial_embeds_raw)

            for i in range(0, S):
                hidden_states_final += all_hidden_states_from_final_pass[:, i::S, :] * s_var[..., i].unsqueeze(-1)

            final_output_logits = self.lm_head(hidden_states_final)
            
            step_accuracies = None
            step_penalty_ratios = None  # 基于softmax概率的惩罚比例（方法1：accuracy）
            step_ce_losses = None  # 每个步骤的CE loss（方法2：delta_loss），step_ce_losses[0]就是不ponder的CE loss
            if self.training and labels is not None and w_var is not None and S > 0:
                if final_output_logits.shape[1] == L_orig:
                    shift_labels = labels[:, 1:].contiguous()  # [B, L-1]
                    ignore_index = -100
                    mask = (shift_labels != ignore_index)  # 用于忽略padding tokens
                    
                    step_accuracies = []
                    # 根据方法类型分别计算
                    if self.min_weight_penalty_method == "accuracy":
                        # 方法1：基于softmax概率计算惩罚比例
                        step_penalty_ratios = []
                        # 对每个步骤i（从0到S-1，对应w0到wK），计算准确率和惩罚比例
                        for i in range(0, S):  # 包括i=0（w0）
                            # 获取步骤i对应的hidden states：使用步骤0到步骤i的加权求和
                            hs_step_i = torch.zeros_like(initial_embeds_raw)
                            for j in range(0, i + 1):  # 从步骤0累加到步骤i
                                hs_step_i += all_hidden_states_from_final_pass[:, j::S, :] * s_var[..., j].unsqueeze(-1)
                            
                            # 计算logits
                            logits_step_i = self.lm_head(hs_step_i)  # [B, L, vocab_size]
                            
                            if logits_step_i.shape[1] == L_orig:
                                shift_logits = logits_step_i[:, :-1, :].contiguous()  # [B, L-1, vocab_size]
                                
                                # 计算softmax概率
                                probs = torch.softmax(shift_logits, dim=-1)  # [B, L-1, vocab_size]
                                
                                # 获取该步骤的预测
                                step_preds = torch.argmax(shift_logits, dim=-1)  # [B, L-1]
                                
                                # 计算准确率：该步骤的预测与真实labels的一致性
                                if mask.sum() > 0:
                                    correct = (step_preds == shift_labels) & mask  # [B, L-1]
                                    accuracy = correct.float().sum() / mask.float().sum()
                                    step_accuracies.append(accuracy.detach())
                                    
                                    # 计算惩罚比例：基于softmax概率
                                    # 对于预测正确的token，使用其softmax概率作为权重
                                    # 对于预测错误的token，权重为0
                                    # 惩罚比例 = sum(预测正确的token的softmax概率) / 总token数
                                    batch_indices = torch.arange(shift_logits.shape[0], device=shift_logits.device).unsqueeze(1).expand(-1, shift_logits.shape[1])  # [B, L-1]
                                    seq_indices = torch.arange(shift_logits.shape[1], device=shift_logits.device).unsqueeze(0).expand(shift_logits.shape[0], -1)  # [B, L-1]
                                    pred_probs = probs[batch_indices, seq_indices, step_preds]  # [B, L-1] - 每个位置预测的softmax概率
                                    
                                    # 只对预测正确的token使用softmax概率，预测错误的token权重为0
                                    weighted_correct = correct.float() * pred_probs  # [B, L-1]
                                    
                                    # 惩罚比例 = sum(预测正确的token的softmax概率) / 总token数（考虑mask）
                                    penalty_ratio = weighted_correct[mask].sum() / mask.float().sum()
                                    step_penalty_ratios.append(penalty_ratio.detach())
                                else:
                                    step_accuracies.append(torch.tensor(0.0, device=device))
                                    step_penalty_ratios.append(torch.tensor(0.0, device=device))
                        
                        # 转换为tensor
                        if len(step_accuracies) > 0:
                            step_accuracies = torch.stack(step_accuracies)  # [S] = [K+1]，包含w0到wK
                            step_penalty_ratios = torch.stack(step_penalty_ratios)  # [S] = [K+1]，包含w0到wK
                        else:
                            step_accuracies = None
                            step_penalty_ratios = None
                    
                    elif self.min_weight_penalty_method == "delta_loss":
                        # 方法2：基于delta loss计算惩罚比例
                        # 对每个token先计算delta loss，然后过激活函数，然后求平均值为penalty_ratio
                        step_per_token_ce_losses = []  # 存储每个步骤的per-token CE loss
                        step_penalty_ratios = []
                        # 对每个步骤i（从0到S-1，对应w0到wK），计算准确率和per-token CE loss
                        for i in range(0, S):  # 包括i=0（w0）
                            # 获取步骤i对应的hidden states：使用步骤0到步骤i的加权求和
                            hs_step_i = torch.zeros_like(initial_embeds_raw)
                            for j in range(0, i + 1):  # 从步骤0累加到步骤i
                                hs_step_i += all_hidden_states_from_final_pass[:, j::S, :] * s_var[..., j].unsqueeze(-1)
                            
                            # 计算logits
                            logits_step_i = self.lm_head(hs_step_i)  # [B, L, vocab_size]
                            
                            if logits_step_i.shape[1] == L_orig:
                                shift_logits = logits_step_i[:, :-1, :].contiguous()  # [B, L-1, vocab_size]
                                
                                # 获取该步骤的预测
                                step_preds = torch.argmax(shift_logits, dim=-1)  # [B, L-1]
                                
                                # 计算准确率：该步骤的预测与真实labels的一致性（batch平均）
                                if mask.sum() > 0:
                                    correct = (step_preds == shift_labels) & mask  # [B, L-1]
                                    accuracy = correct.float().sum() / mask.float().sum()
                                    step_accuracies.append(accuracy.detach())
                                    
                                    # 计算该步骤的per-token CE loss（每个token的CE loss，不reduce）
                                    loss_fct = CrossEntropyLoss(reduction='none')  # 不reduce，返回每个token的loss
                                    per_token_ce_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))  # [B*(L-1)]
                                    per_token_ce_loss = per_token_ce_loss.view(shift_logits.shape[0], shift_logits.shape[1])  # [B, L-1]
                                    step_per_token_ce_losses.append(per_token_ce_loss.detach())
                                else:
                                    step_accuracies.append(torch.tensor(0.0, device=device))
                                    step_per_token_ce_losses.append(torch.zeros((shift_logits.shape[0], shift_logits.shape[1]), device=device))
                        
                        # 计算penalty_ratio：对每个token先计算delta loss，然后过激活函数，然后求平均
                        if len(step_accuracies) > 0 and len(step_per_token_ce_losses) > 0:
                            step_accuracies = torch.stack(step_accuracies)  # [S] = [K+1]，包含w0到wK
                            
                            # 计算每个步骤的delta loss和penalty_ratio
                            # step_per_token_ce_losses[0]是不ponder的per-token CE loss
                            # delta1 = step_per_token_ce_losses[1] - step_per_token_ce_losses[0]，用于惩罚w1
                            # delta2 = step_per_token_ce_losses[2] - step_per_token_ce_losses[1]，用于惩罚w2
                            for i in range(1, S):  # 从1开始，因为w0不惩罚
                                # 计算每个token的delta loss：当前步骤的per-token CE loss - 上一步的per-token CE loss
                                current_per_token_ce_loss = step_per_token_ce_losses[i]  # [B, L-1]
                                prev_per_token_ce_loss = step_per_token_ce_losses[i - 1]  # [B, L-1]
                                per_token_delta_loss = current_per_token_ce_loss - prev_per_token_ce_loss  # [B, L-1]
                                
                                # 对delta_loss应用min(x, 0)，将上限设为0
                                per_token_delta_loss = torch.clamp(per_token_delta_loss, max=0.0)  # [B, L-1]
                                
                                # 对每个token的delta loss应用sigmoid激活函数
                                per_token_penalty_ratio = torch.sigmoid(50.0 * (per_token_delta_loss))  # [B, L-1]
                                
                                # 求平均值作为penalty_ratio（只考虑mask的位置）
                                penalty_ratio = per_token_penalty_ratio[mask].sum() / mask.float().sum()
                                step_penalty_ratios.append(penalty_ratio.detach())
                            
                            # 转换为tensor
                            if len(step_penalty_ratios) > 0:
                                step_penalty_ratios = torch.stack(step_penalty_ratios)  # [K]，包含w1到wK的penalty_ratio
                            else:
                                step_penalty_ratios = None
                        else:
                            step_accuracies = None
                            step_penalty_ratios = None
                    elif self.min_weight_penalty_method == "ce_loss":
                        # 方法3：基于每一步自身的 CE loss 计算惩罚比例
                        # 逻辑：
                        # 1) 直接与真实 label 计算 per-token CE loss
                        # 2) per-token 过 sigmoid(10*(ce-0.5))
                        # 3) 对 mask 求平均得到 mean_sigmoid
                        # 4) 惩罚系数 = 1 - mean_sigmoid
                        #
                        # 输出 step_penalty_ratios: [K]，对应惩罚 w1..wK（w0不惩罚）
                        step_penalty_ratios = []

                        # loss_fct 只创建一次
                        loss_fct = CrossEntropyLoss(reduction="none")

                        for i in range(0, S):  # i=0..K
                            # 获取步骤i对应的hidden states：使用步骤0到步骤i的加权求和
                            hs_step_i = torch.zeros_like(initial_embeds_raw)
                            for j in range(0, i + 1):
                                hs_step_i += all_hidden_states_from_final_pass[:, j::S, :] * s_var[..., j].unsqueeze(-1)

                            # 计算logits
                            logits_step_i = self.lm_head(hs_step_i)  # [B, L, vocab_size]

                            if logits_step_i.shape[1] == L_orig:
                                shift_logits = logits_step_i[:, :-1, :].contiguous()  # [B, L-1, vocab_size]

                                # 计算预测准确率（沿用你前面逻辑，便于 wandb 对齐）
                                step_preds = torch.argmax(shift_logits, dim=-1)  # [B, L-1]
                                if mask.sum() > 0:
                                    correct = (step_preds == shift_labels) & mask
                                    accuracy = correct.float().sum() / mask.float().sum()
                                    step_accuracies.append(accuracy.detach())
                                else:
                                    step_accuracies.append(torch.tensor(0.0, device=device))

                                # 计算 per-token CE loss: [B, L-1]
                                per_token_ce = loss_fct(
                                    shift_logits.view(-1, self.config.vocab_size),
                                    shift_labels.view(-1),
                                ).view(shift_logits.shape[0], shift_logits.shape[1])

                                if mask.sum() > 0:
                                    # sigmoid(10*(ce-0.5))
                                    per_token_score = torch.sigmoid(10.0 * (per_token_ce - 0.5))  # [B, L-1]
                                    mean_score = per_token_score[mask].sum() / mask.float().sum()
                                    penalty_ratio = 1.0 - mean_score  # 惩罚系数
                                else:
                                    penalty_ratio = torch.tensor(0.0, device=device)

                                # w0 不惩罚，所以从 i>=1 才 append
                                if i >= 1:
                                    step_penalty_ratios.append(penalty_ratio.detach())

                        # 转换为 tensor
                        if len(step_accuracies) > 0:
                            step_accuracies = torch.stack(step_accuracies)  # [S] = [K+1]
                        else:
                            step_accuracies = None

                        if len(step_penalty_ratios) > 0:
                            step_penalty_ratios = torch.stack(step_penalty_ratios)  # [K]，对应 w1..wK
                        else:
                            step_penalty_ratios = None

                else:
                    step_accuracies = None
                    step_penalty_ratios = None
                    step_ce_losses = None

            # 训练时计算ponder_aux_loss
            if self.training and ent_mean_var is not None:
                ponder_aux_loss, min_weight_penalty_loss, current_lambda_min_weight_penalty, penalty_ratios = self._accumulate_ponder_losses(
                    entropy_var, expected_steps_var, ent_mean_var, w_var, global_step, step_accuracies, step_penalty_ratios, step_ce_losses
                )
            elif self.training:
                # 如果ent_mean_var为None（即more_iterations == 0），不计算ponder loss
                ponder_aux_loss = torch.tensor(0.0, device=device, dtype=initial_embeds_raw.dtype)
                min_weight_penalty_loss = torch.tensor(0.0, device=device)
                current_lambda_min_weight_penalty = 0.0
                penalty_ratios = None

            loss = None
            if labels is not None:
                if final_output_logits is None:
                    raise ValueError("final_output_logits is None, cannot compute loss.")
                # self.loss_function expects logits and labels.
                # final_output_logits shape: (batch_size, L_orig, vocab_size)
                # labels shape: (batch_size, L_orig)

                # shift_logits = final_output_logits[:, :-1, :].contiguous()
                # shift_labels = labels[:, 1:].contiguous()
                # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                # base_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                #                     shift_labels.view(-1))
                # loss_kwargs.pop("num_items_in_batch", None)
                # loss_kwargs.pop("average_tokens_across_devices", None)  # 兼容其它版本
                base_loss = self.loss_function(logits=final_output_logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)
                # base_loss = self.loss_function(logits=final_output_logits, labels=labels, vocab_size=self.config.vocab_size)
                # 训练时加上ponder_aux_loss
                # if self.training:
                #     loss = base_loss + ponder_aux_loss
                #     try:
                #         log_dict = {
                #             "train/base_loss": float(base_loss.detach().cpu()),
                #             "train/min_weight_penalty_loss": float(min_weight_penalty_loss),
                #             "train/lambda_min_weight_penalty": float(current_lambda_min_weight_penalty),
                #         }
                #         # 添加每个步骤的惩罚比例
                #         if penalty_ratios is not None:
                #             for step_name, ratio in penalty_ratios.items():
                #                 log_dict[f"train/penalty_ratio_{step_name}"] = float(ratio)
                #         # 添加每个步骤的准确率
                #         if step_accuracies is not None:
                #             for i, acc in enumerate(step_accuracies):
                #                 # i=0对应w0，i=1对应w1，以此类推
                #                 log_dict[f"train/step_accuracy_w{i}"] = float(acc.detach().cpu())
                #         wandb.log(log_dict, commit=False)
                #     except Exception:
                #         pass
                # else:
                #     loss = base_loss
                if self.training:
                    # -------- [ADD] 让 aux loss 和 base_loss 使用同一归一化尺度 --------
                    aux_to_add = ponder_aux_loss
                    try:
                        # 计算本卡有效 token 数（与 ignore_index 对齐）
                        shift_labels = labels[:, 1:].contiguous()
                        ignore_index = -100
                        local_num_tokens = (shift_labels != ignore_index).sum().clamp_min(1)

                        # Trainer/DS 可能传入 num_items_in_batch（通常是“全局有效 token 数”或其变体）
                        num_items = loss_kwargs.get("num_items_in_batch", None)
                        if num_items is not None:
                            denom = torch.as_tensor(num_items, device=base_loss.device, dtype=base_loss.dtype).clamp_min(1.0)

                            # base_loss 由 loss_function 产生，通常等价于 sum_ce / denom
                            # 但 aux_loss 你通常是 mean over tokens（或其它局部归一化）
                            # => 把 aux 乘 local_num_tokens/denom，使它变到同一尺度
                            scale = (local_num_tokens.to(base_loss.dtype) / denom).detach()
                            aux_to_add = ponder_aux_loss * scale
                        # else: 没有 num_items_in_batch 就不缩放，维持原语义
                    except Exception:
                        aux_to_add = ponder_aux_loss
                    # -------- [ADD END] --------

                    loss = base_loss + aux_to_add

                    try:
                        log_dict = {
                            "train/base_loss": float(base_loss.detach().cpu()),
                            "train/min_weight_penalty_loss": float(min_weight_penalty_loss),
                            "train/lambda_min_weight_penalty": float(current_lambda_min_weight_penalty),
                        }
                        # （可选，但强烈建议）把缩放后的 aux 也记录一下，方便你确认尺度是否对齐
                        log_dict["train/ponder_aux_loss_scaled"] = float(aux_to_add.detach().cpu())
                        log_dict["train/ponder_aux_loss_raw"] = float(ponder_aux_loss.detach().cpu())
                        if loss_kwargs.get("num_items_in_batch", None) is not None:
                            log_dict["train/num_items_in_batch"] = float(loss_kwargs["num_items_in_batch"])

                        if penalty_ratios is not None:
                            for step_name, ratio in penalty_ratios.items():
                                log_dict[f"train/penalty_ratio_{step_name}"] = float(ratio)
                        if step_accuracies is not None:
                            for i, acc in enumerate(step_accuracies):
                                log_dict[f"train/step_accuracy_w{i}"] = float(acc.detach().cpu())
                        wandb.log(log_dict, commit=False)
                    except Exception:
                        pass
                else:
                    loss = base_loss


            final_hidden_states_to_return = final_pass_model_outputs.hidden_states if user_requested_output_hidden_states else None
            final_attentions_to_return = final_pass_model_outputs.attentions if output_attentions else None
            final_past_key_values = final_pass_model_outputs.past_key_values if use_cache else None

            if not return_dict:
                output_items_tuple = [final_output_logits] 
                if final_past_key_values is not None: output_items_tuple.append(final_past_key_values)
                if final_hidden_states_to_return is not None: output_items_tuple.append(final_hidden_states_to_return)
                if final_attentions_to_return is not None: output_items_tuple.append(final_attentions_to_return)
                final_tuple = tuple(output_items_tuple)
                return ((loss,) + final_tuple) if loss is not None else final_tuple

            # if self.training and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            #     if global_step is not None and global_step < 2:
            #         ni = loss_kwargs.get("num_items_in_batch", None)
            #         print(f"[DBG-SCALE] step={global_step} base={base_loss.item():.6f} "
            #             f"aux_raw={ponder_aux_loss.item():.6f} aux_add={aux_to_add.item():.6f} "
            #             f"num_items={ni}", flush=True)
            # print(f"... local_tokens={int(local_num_tokens.item())} ga={self.config.gradient_accumulation_steps if hasattr(self.config,'gradient_accumulation_steps') else 'NA'} ...")

            return CausalLMOutputWithPast(
                loss=loss,
                logits=final_output_logits, 
                past_key_values=final_past_key_values,
                hidden_states=final_hidden_states_to_return,
                attentions=final_attentions_to_return,
            )

@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
