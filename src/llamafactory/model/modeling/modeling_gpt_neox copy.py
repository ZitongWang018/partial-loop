# coding=utf-8
# Copyright 2022 EleutherAI The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch GPTNeoX model."""

from typing import Optional, Tuple, Union, List

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    get_torch_version,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
import math

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "trl-internal-testing/tiny-random-GPTNeoXForCausalLM"
_REAL_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-neox-20b"
_CONFIG_FOR_DOC = "GPTNeoXConfig"


class GPTNeoXPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTNeoXConfig
    base_model_prefix = "gpt_neox"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTNeoXLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 使用截断正态分布初始化权重，截断在3σ处，方差为2/(5*hidden_size)
            std = math.sqrt(2.0 / (5 * self.config.hidden_size))
            
            # 对于注意力层和MLP层的输出投影，根据网络深度进行额外缩放
            if hasattr(module, '_is_attention_output') or hasattr(module, '_is_mlp_output'):
                std = std / math.sqrt(2.0 * self.config.num_hidden_layers * (self.config.more_iterations + 1))
                # std = 2 / self.config.num_hidden_layers / math.sqrt(self.config.hidden_size)/(self.config.more_iterations + 1)
            # nn.init.normal_(module.weight.data, mean=0.0, std=std, a=-3*std, b=3*std)                           
            nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-3*std, b=3*std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 使用截断正态分布初始化嵌入层权重
            std = math.sqrt(2.0 / (5 * self.config.hidden_size))
            # nn.init.normal_(module.weight.data, mean=0.0, std=std, a=-3*std, b=3*std)
            nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-3*std, b=3*std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GPTNeoXAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them"
            )
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self.rope_theta = config.rotary_emb_base
        self._init_bias(config.max_position_embeddings)

        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)
        self.rotary_emb = GPTNeoXRotaryEmbedding(config=self.config)

        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.norm_factor = self.head_size**-0.5
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.dense._is_attention_output = True  # 标记为注意力输出层
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.is_causal = True
        self.layer_idx = layer_idx

    def _init_bias(self, max_positions, device=None):
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        if device is not None:
            self.bias = self.bias.to(device)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        padding_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        # Apply attention-specific projections and rope
        query, key, value, present = self._attn_projections_and_rope(
            hidden_states=hidden_states,
            position_ids=position_ids,
            layer_past=layer_past,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _attn_projections_and_rope(
        self,
        hidden_states: torch.FloatTensor,
        position_ids: torch.LongTensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value, position_ids)
        else:
            cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if layer_past is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "partial_rotation_size": self.rotary_ndims,
                "cache_position": cache_position,
            }
            key, value = layer_past.update(key, value, self.layer_idx, cache_kwargs)

        return query, key, value, layer_past

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        # dynamically increase the causal mask with the key length, if needed.
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length, device=key.device)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=self.norm_factor,
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn_scores = attn_scores + causal_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


class GPTNeoXFlashAttention2(GPTNeoXAttention):
    """
    GPTNeoX flash attention module. This module inherits from `GPTNeoXAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        # Apply attention-specific projections and rope
        query, key, value, present = self._attn_projections_and_rope(
            hidden_states=hidden_states,
            position_ids=position_ids,
            layer_past=layer_past,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        query_length = query.shape[-2]

        # GPT-neo-X casts query and key in fp32 to apply rotary embedding in full precision
        target_dtype = value.dtype
        if query.dtype != target_dtype:
            query = query.to(target_dtype)
        if key.dtype != target_dtype:
            key = key.to(target_dtype)

        # Permute to get the expected shape for Flash Attention
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 / bfloat16 just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        input_dtype = query.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.query_key_value.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query = query.to(target_dtype)
            key = key.to(target_dtype)
            value = value.to(target_dtype)

        attention_dropout = self.config.attention_dropout if self.training else 0.0

        # Compute attention
        attn_weights = _flash_attention_forward(
            query,
            key,
            value,
            attention_mask,
            query_length,
            dropout=attention_dropout,
            softmax_scale=self.norm_factor,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        # Reshape outputs
        attn_output = attn_weights.reshape(
            attn_weights.shape[0], attn_weights.shape[1], self.num_attention_heads * self.head_size
        )
        attn_output = self.dense(attn_output)

        outputs = (attn_output, layer_past)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class GPTNeoXSdpaAttention(GPTNeoXAttention):
    """
    GPTNeoX attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `GPTNeoXAttention` as the weights of the module stays untouched. The only changes are on the forward pass
    to adapt to the SDPA API.
    """

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()`. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        if output_attentions or head_mask is not None:
            logger.warning_once(
                "`GPTNeoXSdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "`output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, but "
                "specifying the manual implementation will be required from Transformers version v5.0.0 onwards. "
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        # Apply attention-specific projections and rope
        query, key, value, present = self._attn_projections_and_rope(
            hidden_states=hidden_states,
            position_ids=position_ids,
            layer_past=layer_past,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]

        # GPT-neo-X casts query and key in fp32 to apply rotary embedding in full precision
        target_dtype = value.dtype
        if query.dtype != target_dtype:
            query = query.to(target_dtype)
        if key.dtype != target_dtype:
            key = key.to(target_dtype)

        # Avoid torch==2.1.2 specific bug for the memory-efficient backend in SDPA
        if self.require_contiguous_qkv and query.device.type == "cuda" and attention_mask is not None:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape outputs
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.dense(attn_output)

        return attn_output, present, None


def attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(~ltor_mask, torch.finfo(attention_scores.dtype).min)
    return attention_scores


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->GPTNeoX
class GPTNeoXRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[GPTNeoXConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`GPTNeoXRotaryEmbedding` can now be fully parameterized by passing the model config through the "
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


# Copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->GPTNeoX
class GPTNeoXLinearScalingRotaryEmbedding(GPTNeoXRotaryEmbedding):
    """GPTNeoXRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`GPTNeoXLinearScalingRotaryEmbedding` is deprecated an will be removed in v4.46. Please use "
            "`GPTNeoXRotaryEmbedding`, which now also does linear scaling (simply pass the model config to __init__)."
        )
        kwargs["rope_type"] = "linear"
        super().__init__(*args, **kwargs)


# Copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->GPTNeoX
class GPTNeoXDynamicNTKScalingRotaryEmbedding(GPTNeoXRotaryEmbedding):
    """GPTNeoXRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`GPTNeoXDynamicNTKScalingRotaryEmbedding` is deprecated an will be removed in v4.46. Please use "
            "`GPTNeoXRotaryEmbedding`, which now also does dynamic ntk scaling (simply pass the model config to "
            "__init__)."
        )
        kwargs["rope_type"] = "dynamic"
        super().__init__(*args, **kwargs)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
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


class GPTNeoXMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense_4h_to_h._is_mlp_output = True  # 标记为MLP输出层
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


GPT_NEOX_ATTENTION_CLASSES = {
    "eager": GPTNeoXAttention,
    "flash_attention_2": GPTNeoXFlashAttention2,
    "sdpa": GPTNeoXSdpaAttention,
}


class GPTNeoXLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        self.attention = GPT_NEOX_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.mlp = GPTNeoXMLP(config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        attention_layer_outputs = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        attn_output = attention_layer_outputs[0]  # output_attn: attn_output, present, (attn_weights)
        attn_output = self.post_attention_dropout(attn_output)
        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output

        if use_cache:
            outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs


GPT_NEOX_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~GPTNeoXConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPT_NEOX_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
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
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare GPTNeoX Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_NEOX_START_DOCSTRING,
)
class GPTNeoXModel(GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([GPTNeoXLayer(config, i) for i in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = GPTNeoXRotaryEmbedding(config=config)

        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing = False
        self.checkpoint_num_layers = config.checkpoint_num_layers

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value):
        self.embed_in = value

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.FloatTensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

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

        seq_length = inputs_embeds.shape[1]
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        hidden_states = self.emb_dropout(inputs_embeds)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        next_decoder_cache = None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, layer in enumerate(
            self.layers,
        ):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training and i < self.checkpoint_num_layers:
                outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    head_mask[i],
                    use_cache,
                    None,
                    output_attentions,
                    cache_position,
                    position_embeddings,
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    layer_past=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            hidden_states = outputs[0]
            if use_cache is True:
                next_decoder_cache = outputs[1]
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.final_layer_norm(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
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
    # Copied from transformers.models.llama.modeling_llama.LlamaModel._prepare_4d_causal_attention_mask_with_cache_position
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


@add_start_docstrings(
    """GPTNeoX Model with a `language modeling` head on top for CLM fine-tuning.""", GPT_NEOX_START_DOCSTRING
)
class GPTNeoXForCausalLM(GPTNeoXPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if hasattr(config, 'recurrent_model') and config.recurrent_model:
            if not hasattr(config, 'more_iterations') or not isinstance(config.more_iterations, int) or config.more_iterations <= 0:
                # If more_iterations is 0, no extra iteration embeddings are needed.
                # If > 0, we need embeddings for stages 1 to N.
                if config.more_iterations > 0:
                     self.iteration_embeddings = nn.Embedding(config.more_iterations, config.hidden_size)
                else: # more_iterations == 0
                    self.iteration_embeddings = None # Or an empty module, to avoid attribute errors
            else: # more_iterations > 0
                 self.iteration_embeddings = nn.Embedding(config.more_iterations, config.hidden_size)

        self.post_init()        

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def compute_interpolated_embeds(self, weight: torch.Tensor, hidden: Union[torch.Tensor, Tuple[torch.Tensor, ...]], use_topk: bool = True) -> torch.Tensor:
        # 保留用户原有的 compute_interpolated_embeds 实现细节，仅做微小修正以处理可能的边界或配置缺失
        # 确保 hidden_layer_num 和 softmax_temperature, interpolation_topk 是从 config 读取
        softmax_temp = getattr(self.config, 'softmax_temperature', 1.0)
        hidden_layer_step = getattr(self.config, 'hidden_layer_num', 1)
        if hidden_layer_step <= 0: hidden_layer_step = 1 # Ensure positive step

        if use_topk:
            top_k_val = getattr(self.config, 'interpolation_topk', 100)
            if isinstance(hidden, torch.Tensor):
                logits = self.embed_out(hidden)
                logits_scaled = logits / softmax_temp
                
                actual_top_k = min(top_k_val, logits_scaled.size(-1)) # Ensure top_k is not larger than vocab_size
                if actual_top_k == 0: # Avoid error if vocab_size is 0 for some reason, or top_k is 0
                    return torch.zeros_like(hidden)


                topk_values, topk_indices = torch.topk(logits_scaled, k=actual_top_k, dim=-1)
                probs_topk = torch.softmax(topk_values, dim=-1)
                embedding_topk = weight[topk_indices]
                interpolated_embeds = torch.einsum("bsl,bslh->bsh", probs_topk, embedding_topk)
                return interpolated_embeds
            else: # hidden is a tuple
                num_hidden_entries = len(hidden)
                selected_layer_outputs = []
                if num_hidden_entries > 1: 
                    for i in range(num_hidden_entries - 1, 0, -hidden_layer_step):
                        selected_layer_outputs.append(hidden[i])
                
                if not selected_layer_outputs: 
                    if num_hidden_entries > 0 and isinstance(hidden[-1], torch.Tensor): 
                         selected_layer_outputs = [hidden[-1]]
                    else: 
                        if num_hidden_entries > 0 and isinstance(hidden[0], torch.Tensor): 
                             example_tensor = hidden[0]
                             return torch.zeros_like(example_tensor) 
                        else:
                             raise ValueError("Cannot compute interpolated embeds: 'hidden' tuple has no processable layers and shape cannot be determined.")
                list_interpolated = []
                for layer_output in selected_layer_outputs:
                    logits = self.embed_out(layer_output)
                    logits_scaled = logits / softmax_temp
                    actual_top_k = min(top_k_val, logits_scaled.size(-1))
                    if actual_top_k == 0: continue 
                    topk_values, topk_indices = torch.topk(logits_scaled, k=actual_top_k, dim=-1)
                    probs_topk = torch.softmax(topk_values, dim=-1)
                    embedding_topk = weight[topk_indices]
                    interpolated_layer = torch.einsum("bsl,bslh->bsh", probs_topk, embedding_topk)
                    list_interpolated.append(interpolated_layer)
                
                if not list_interpolated:
                    if num_hidden_entries > 0 and isinstance(hidden[0], torch.Tensor):
                         return torch.zeros_like(hidden[0]) 
                    else:
                         raise ValueError("Cannot compute final interpolated embedding: list_interpolated is empty and shape cannot be determined.")
                final_interpolated = sum(list_interpolated) / len(list_interpolated)
                return final_interpolated
        else: # use_topk is False (full softmax)
            if isinstance(hidden, torch.Tensor):
                logits = self.embed_out(hidden) / softmax_temp
                probs = torch.softmax(logits, dim=-1)
            else: # hidden is a tuple
                num_hidden_entries = len(hidden)
                selected_layer_outputs = []
                if num_hidden_entries > 1:
                    for i in range(num_hidden_entries - 1, 0, -hidden_layer_step):
                        selected_layer_outputs.append(hidden[i])
                if not selected_layer_outputs: 
                    if num_hidden_entries > 0 and isinstance(hidden[-1], torch.Tensor):
                        selected_layer_outputs = [hidden[-1]]
                    else:
                        if num_hidden_entries > 0 and isinstance(hidden[0], torch.Tensor):
                             return torch.zeros_like(hidden[0])
                        else:
                             raise ValueError("Cannot compute interpolated embeds (no topk): 'hidden' tuple has no processable layers and shape cannot be determined.")
                sum_probs = None
                for layer_output in selected_layer_outputs:
                    current_logits = self.embed_out(layer_output) / softmax_temp
                    current_probs = torch.softmax(current_logits, dim=-1)
                    if sum_probs is None:
                        sum_probs = current_probs
                    else:
                        sum_probs += current_probs
                if sum_probs is None: 
                    if num_hidden_entries > 0 and isinstance(hidden[0], torch.Tensor):
                         return torch.zeros_like(hidden[0])
                    else:
                         raise ValueError("Sum of probabilities is None in no-topk multi-layer interpolation and shape cannot be determined.")
                probs = sum_probs 
            interpolated_embeds = torch.matmul(probs, weight)
            return interpolated_embeds

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        high_entropy_mask: Optional[torch.LongTensor] = None, 
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.FloatTensor]]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.
            head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
                Contains precomputed key and value hidden states of the attention blocks.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the left-to-right language modeling loss (next word prediction).
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            cache_position (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the cache.

        Returns:
            Returns a [`CausalLMOutputWithPast`] or a tuple of tensors if `return_dict=False`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if not hasattr(self.config, 'recurrent_model') or not self.config.recurrent_model:
            # Standard forward pass (remains unchanged)
            outputs = self.gpt_neox(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                head_mask=head_mask, inputs_embeds=inputs_embeds, past_key_values=past_key_values,
                use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                return_dict=True, cache_position=cache_position,
            )
            hidden_states_from_neox = outputs.last_hidden_state 
            logits = self.embed_out(hidden_states_from_neox)
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            if not return_dict:
                output_items = [logits]
                if outputs.past_key_values is not None: output_items.append(outputs.past_key_values)
                if outputs.hidden_states is not None: output_items.append(outputs.hidden_states)
                if outputs.attentions is not None: output_items.append(outputs.attentions)
                output_tuple = tuple(output_items)
                return ((loss,) + output_tuple) if loss is not None else output_tuple
            return CausalLMOutputWithPast(
                loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            )

        # ---- Recurrent model logic with Iteration Embeddings (only for iter_stage > 0) ----
        if inputs_embeds is None:
            if input_ids is None: raise ValueError("input_ids must be provided for recurrent model")
            initial_embeds_raw = self.gpt_neox.embed_in(input_ids)
        else:
            initial_embeds_raw = inputs_embeds

        B, L_orig, H = initial_embeds_raw.shape
        device = initial_embeds_raw.device

        if position_ids is None:
            original_repeated_position_ids = torch.arange(0, L_orig, dtype=torch.long, device=device).unsqueeze(0).expand(B, L_orig)
        else:
            original_repeated_position_ids = position_ids

        if attention_mask is None:
            original_attention_mask_2d = torch.ones((B, L_orig), dtype=torch.long, device=device)
        else:
            original_attention_mask_2d = attention_mask if attention_mask.ndim == 2 else attention_mask.squeeze()


        inputs_embeds0 = initial_embeds_raw 
        weight_for_interpolation = self.gpt_neox.embed_in.weight
        if hasattr(self.config, 'scale_embeds') and self.config.scale_embeds:
            embed_scale = torch.sqrt(torch.tensor(H, dtype=initial_embeds_raw.dtype, device=device))
            inputs_embeds0 = initial_embeds_raw * embed_scale
            weight_for_interpolation = self.gpt_neox.embed_in.weight * embed_scale
        
        scaled_inputs_embeds0_for_orig_slots = inputs_embeds0 
        if hasattr(self.config, 'mutiply_iterations') and self.config.mutiply_iterations:
            factor = self.config.mutiply_iterations
            if isinstance(factor, bool) and factor:
                 if hasattr(self.config, 'more_iterations') and isinstance(self.config.more_iterations, int) and self.config.more_iterations > 0:
                    scaled_inputs_embeds0_for_orig_slots = inputs_embeds0 * self.config.more_iterations
            elif isinstance(factor, (int, float)) and factor > 0 :
                scaled_inputs_embeds0_for_orig_slots = inputs_embeds0 * factor
        
        num_interpolation_iterations = self.config.more_iterations
        all_embeddings_stages: List[torch.Tensor] = [scaled_inputs_embeds0_for_orig_slots] 

        for iter_k_idx in range(num_interpolation_iterations): 
            current_num_stages_in_loop_input = len(all_embeddings_stages) 
            
            current_loop_base_embeds = torch.empty((B, current_num_stages_in_loop_input * L_orig, H), dtype=inputs_embeds0.dtype, device=device)
            current_loop_pos_ids_for_rope = torch.empty((B, current_num_stages_in_loop_input * L_orig), dtype=original_repeated_position_ids.dtype, device=device)
            current_loop_attn_mask_2d = torch.empty((B, current_num_stages_in_loop_input * L_orig), dtype=original_attention_mask_2d.dtype, device=device)
            
            # Prepare final input by adding iteration embeddings selectively
            current_loop_input_embeds_final = torch.empty_like(current_loop_base_embeds)

            for stage_content_idx in range(current_num_stages_in_loop_input): # stage_content_idx is 0 for original, 1 for interp1, etc.
                s_start = stage_content_idx
                s_end = current_num_stages_in_loop_input * L_orig
                
                current_loop_base_embeds[:, s_start::current_num_stages_in_loop_input, :] = all_embeddings_stages[stage_content_idx]
                current_loop_pos_ids_for_rope[:, s_start::current_num_stages_in_loop_input] = original_repeated_position_ids
                current_loop_attn_mask_2d[:, s_start::current_num_stages_in_loop_input] = original_attention_mask_2d
                
                # Add iteration embedding ONLY if stage_content_idx > 0 (i.e., for interpolated stages)
                # and self.iteration_embeddings exists (i.e. config.more_iterations > 0)
                base_slice = all_embeddings_stages[stage_content_idx]
                if stage_content_idx > 0 and self.iteration_embeddings is not None:
                    # Iteration IDs for lookup are 0 to N-1 for stages 1 to N
                    iter_id_for_lookup = torch.tensor(stage_content_idx - 1, device=device, dtype=torch.long)
                    iter_emb_to_add = self.iteration_embeddings(iter_id_for_lookup) * embed_scale # Shape [H]
                    current_loop_input_embeds_final[:, s_start::current_num_stages_in_loop_input, :] = base_slice + iter_emb_to_add.unsqueeze(0).unsqueeze(0)
                else:
                    current_loop_input_embeds_final[:, s_start::current_num_stages_in_loop_input, :] = base_slice
            
            iter_gpt_neox_outputs = self.gpt_neox(
                inputs_embeds=current_loop_input_embeds_final,
                attention_mask=current_loop_attn_mask_2d, 
                position_ids=current_loop_pos_ids_for_rope, 
                head_mask=head_mask, past_key_values=None, use_cache=False,
                output_attentions=False, output_hidden_states=True, return_dict=True, cache_position=None,
            )
            
            hidden_for_interp_source_selector = slice(current_num_stages_in_loop_input - 1, None, current_num_stages_in_loop_input)
            
            use_multilayer_hidden_for_interp = (
                hasattr(self.config, 'hidden_layer_num') and self.config.hidden_layer_num is not None and 
                self.config.hidden_layer_num > 0 and iter_gpt_neox_outputs.hidden_states is not None and
                len(iter_gpt_neox_outputs.hidden_states) > 1 
            )
            if use_multilayer_hidden_for_interp:
                source_hidden_states_tuple = tuple(
                    h_layer[:, hidden_for_interp_source_selector, :] for h_layer in iter_gpt_neox_outputs.hidden_states
                )
                hidden_input_for_compute_fn = source_hidden_states_tuple
            else:
                last_h_state = iter_gpt_neox_outputs.last_hidden_state
                hidden_input_for_compute_fn = last_h_state[:, hidden_for_interp_source_selector, :]
            
            interp_use_topk = getattr(self.config, 'interpolation_use_topk', False)
            new_interpolated_embeds = self.compute_interpolated_embeds(
                weight_for_interpolation, hidden_input_for_compute_fn, use_topk=interp_use_topk
            )
            all_embeddings_stages.append(new_interpolated_embeds)

        # ---- Final pass after all interpolations ----
        num_total_stages_final = len(all_embeddings_stages) 
        L_final_interleaved = num_total_stages_final * L_orig

        final_interleaved_base_embeds = torch.empty((B, L_final_interleaved, H), dtype=inputs_embeds0.dtype, device=device)
        final_interleaved_pos_ids_for_rope = torch.empty((B, L_final_interleaved), dtype=original_repeated_position_ids.dtype, device=device)
        final_interleaved_attn_mask_2d = torch.empty((B, L_final_interleaved), dtype=original_attention_mask_2d.dtype, device=device)
        
        final_interleaved_input_embeds = torch.empty_like(final_interleaved_base_embeds)
        final_pass_cache_position = None

        for stage_idx in range(num_total_stages_final): # stage_idx is 0 for original, 1 for interp1, etc.
            s_start = stage_idx
            s_end = L_final_interleaved # Not used in current slicing

            base_slice = all_embeddings_stages[stage_idx]
            final_interleaved_base_embeds[:, s_start::num_total_stages_final, :] = base_slice
            final_interleaved_pos_ids_for_rope[:, s_start::num_total_stages_final] = original_repeated_position_ids
            final_interleaved_attn_mask_2d[:, s_start::num_total_stages_final] = original_attention_mask_2d
            
            if stage_idx > 0 and self.iteration_embeddings is not None:
                iter_id_for_lookup = torch.tensor(stage_idx - 1, device=device, dtype=torch.long)
                iter_emb_to_add = self.iteration_embeddings(iter_id_for_lookup) * embed_scale
                final_interleaved_input_embeds[:, s_start::num_total_stages_final, :] = base_slice + iter_emb_to_add.unsqueeze(0).unsqueeze(0)
            else:
                final_interleaved_input_embeds[:, s_start::num_total_stages_final, :] = base_slice
        
        final_pass_gpt_neox_outputs = self.gpt_neox(
            inputs_embeds=final_interleaved_input_embeds,
            attention_mask=final_interleaved_attn_mask_2d, 
            position_ids=final_interleaved_pos_ids_for_rope, 
            head_mask=head_mask, past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, 
            return_dict=True, cache_position=final_pass_cache_position,
        )

        logits_all_final_sequence = self.embed_out(final_pass_gpt_neox_outputs.last_hidden_state)
            
        loss = None
        if labels is not None:
            logits_from_last_interp_stage = logits_all_final_sequence[:, num_interpolation_iterations::num_total_stages_final, :]
            shift_logits = logits_from_last_interp_stage[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
                
        if not return_dict:
            output_items_tuple = [logits_all_final_sequence]
            if final_pass_gpt_neox_outputs.past_key_values is not None: output_items_tuple.append(final_pass_gpt_neox_outputs.past_key_values)
            if final_pass_gpt_neox_outputs.hidden_states is not None: output_items_tuple.append(final_pass_gpt_neox_outputs.hidden_states)
            if final_pass_gpt_neox_outputs.attentions is not None: output_items_tuple.append(final_pass_gpt_neox_outputs.attentions)
            final_tuple = tuple(output_items_tuple)
            return ((loss,) + final_tuple) if loss is not None else final_tuple
            
        return CausalLMOutputWithPast(
            loss=loss, logits=logits_all_final_sequence,
            past_key_values=final_pass_gpt_neox_outputs.past_key_values,
            hidden_states=final_pass_gpt_neox_outputs.hidden_states,
            attentions=final_pass_gpt_neox_outputs.attentions,
        )
        # 第一次forward获取hidden states
        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=self.config.output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_layer_num = self.config.hidden_layer_num
        if self.config.output_hidden_states:
            hidden_states = outputs['hidden_states'][hidden_layer_num]
        else:
            hidden_states = outputs['last_hidden_state']
        
        # 如果启用uniform_real_time模式，实时计算high_entropy_mask
        if self.config.uniform_real_time and high_entropy_mask is None:
            # print("uniform_real_time mode")
            # 计算每个位置的logits和熵
            logits = self.embed_out(hidden_states)  # [batch_size, seq_len, vocab_size]
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-3), dim=-1)  # [batch_size, seq_len]
            
            # 创建排除位置的掩码
            batch_size, seq_len = input_ids.shape
            excluded_positions = torch.zeros_like(input_ids, dtype=torch.bool)
            
            # 排除每个序列的开头5个和结尾3个token
            excluded_positions[:, :5] = True
            excluded_positions[:, -3:] = True
            
            # 排除EOS token (0) 周围的位置
            eos_positions = (input_ids == 0)
            for i in range(-3, 5):  # 前3个和后5个
                if i < 0:
                    shifted = torch.roll(eos_positions, i, dims=1)
                    shifted[:, i:] = False
                else:
                    shifted = torch.roll(eos_positions, i, dims=1)
                    shifted[:, :i] = False
                excluded_positions |= shifted
            
            # 将排除位置的熵设为极小值
            entropy = entropy.masked_fill(excluded_positions, float('-inf'))
            
            # 为每个序列选择目标数量的高熵位置
            target_count = int(seq_len * 0.02)+1
            _, top_indices = torch.topk(entropy, k=target_count, dim=1)
            
            # 创建high_entropy_mask
            high_entropy_mask = torch.zeros_like(input_ids)
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, target_count)
            high_entropy_mask[batch_indices, top_indices] = 1
            del logits, probs, entropy, _
            # torch.cuda.empty_cache()
        hidden_states = outputs[0]
        # 清理不需要的张量
        del outputs
        # torch.cuda.empty_cache()

        # 准备第二次forward的inputs_embeds
        if inputs_embeds is None:
            inputs_embeds = self.gpt_neox.embed_in(input_ids)
        
        batch_size = inputs_embeds.size(0)
        seq_len = inputs_embeds.size(1)
        
        # 将high_entropy_mask转换为每个batch的位置列表
        high_entropy_positions = []
        for i in range(batch_size):
            positions = torch.nonzero(high_entropy_mask[i]).squeeze(-1).tolist()
            # 确保positions是列表，即使为空或只有一个位置
            if isinstance(positions, int):
                positions = [positions]
            elif not positions:  # 处理全为0的情况
                positions = []
            high_entropy_positions.append(positions)
        
        # 找出最大的新序列长度(用于padding)
        max_new_seq_len = seq_len
        for positions in high_entropy_positions:
            if self.config.add_ponderer_token:
                max_new_seq_len = max(max_new_seq_len, seq_len + 2*len(positions))
            else:
                max_new_seq_len = max(max_new_seq_len, seq_len + len(positions))
        
        # 收集所有需要计算的位置
        batch_indices = []
        position_indices = []
        for i in range(batch_size):
            indices = high_entropy_positions[i]
            if len(indices) > 0:
                batch_indices.extend([i] * len(indices))  # 重复batch索引
                position_indices.extend(indices)  # 对应的位置索引

        if batch_indices:  # 只在有需要计算的位置时执行
            # 一次性获取所有需要的hidden states
            selected_hidden = hidden_states[batch_indices, position_indices]
            
            # 一次性计算所有logits和probabilities
            selected_logits = self.embed_out(selected_hidden)
            all_probs = torch.softmax(selected_logits/self.config.softmax_temperature, dim=-1)
        else:
            all_probs = []

        # 批量计算所有interpolated_embeds
        all_interpolated_embeds = torch.matmul(all_probs, self.gpt_neox.embed_in.weight)
        if self.config.is_normalize_hidden_states:
            all_interpolated_embeds = self.normalize_hidden_states(all_interpolated_embeds, target_std=std)
        del all_probs,selected_logits,selected_hidden,hidden_states
        # torch.cuda.empty_cache()
        # 在循环中使用预计算的结果
        embed_idx = 0
        new_inputs_embeds = []
        new_labels = [] if labels is not None else None
        new_attention_mask = []

        # 获取ponderer token的embedding
        if self.config.add_ponderer_token:
            ponderer_embedding = self.gpt_neox.embed_in.weight[50277].unsqueeze(0) # 获取最后一个token的embedding

        for i in range(batch_size):
            indices = high_entropy_positions[i]
            current_embeds = []
            current_labels = [] if labels is not None else None
            current_mask = []
            
            last_idx = 0
            for pos in indices:
                # 添加原始序列到当前高熵位置前
                current_embeds.append(inputs_embeds[i, last_idx:pos])
                current_mask.extend([1] * (pos - last_idx))
                
                if labels is not None:
                    curr_labels = labels[i, last_idx:pos].clone()
                    current_labels.append(curr_labels)
                
                # 按顺序添加: ponderer token -> 高熵位置token -> interpolated embedding
                if self.config.add_ponderer_token:
                    current_embeds.append(ponderer_embedding)
                    current_mask.append(1)
                    if labels is not None:
                        current_labels.append(labels[i, pos:pos+1])
                
                # 添加高熵位置的原始token
                current_embeds.append(inputs_embeds[i, pos:pos+1])
                current_mask.append(1)
                if labels is not None:
                    current_labels.append(torch.tensor([-100], device=labels.device))
                
                # 添加interpolated embedding
                current_embeds.append(all_interpolated_embeds[embed_idx:embed_idx+1])
                current_mask.append(1)
                embed_idx += 1
                
                if labels is not None:
                    if self.training and self.config.add_loss_for_ponderer:
                        current_labels.append(torch.tensor([labels[i, pos+1]], device=labels.device))
                    else:
                        current_labels.append(torch.tensor([-100], device=labels.device))
                
                last_idx = pos + 1
            
            # 添加剩余的序列
            if last_idx < seq_len:
                current_embeds.append(inputs_embeds[i, last_idx:])
                current_mask.extend([1] * (seq_len - last_idx))
                if labels is not None:
                    current_labels.append(labels[i, last_idx:])
            
            # 连接当前batch的所有部分
            batch_embeds = torch.cat(current_embeds, dim=0)
            
            # 确保current_mask长度与batch_embeds长度匹配
            current_seq_len = batch_embeds.size(0)
            
            # Padding处理
            pad_length = max_new_seq_len - current_seq_len
            if pad_length > 0:
                # Pad embeddings
                pad_embeds = torch.zeros(pad_length, batch_embeds.size(-1), device=batch_embeds.device, dtype=batch_embeds.dtype)
                batch_embeds = torch.cat([batch_embeds, pad_embeds], dim=0)
                # Pad attention mask
                current_mask.extend([0] * pad_length)
                # Pad labels if needed
                if labels is not None:
                    batch_labels = torch.cat(current_labels, dim=0)
                    pad_labels = torch.full((pad_length,), -100, device=batch_labels.device)
                    batch_labels = torch.cat([batch_labels, pad_labels], dim=0)
                    new_labels.append(batch_labels)
            else:
                if labels is not None:
                    batch_labels = torch.cat(current_labels, dim=0)
                    new_labels.append(batch_labels)
            
            # 确保attention mask长度正确
            assert len(current_mask) == max_new_seq_len, f"Attention mask length {len(current_mask)} does not match sequence length {max_new_seq_len}"
            
            new_inputs_embeds.append(batch_embeds)
            new_attention_mask.append(current_mask)
        
        # 将所有batch的结果堆叠在一起
        new_inputs_embeds = torch.stack(new_inputs_embeds)
        new_attention_mask = torch.tensor(new_attention_mask, device=new_inputs_embeds.device)
        if labels is not None:
            new_labels = torch.stack(new_labels)

        if not self.config.vary_position:
            # 创建新的position_ids，保持原始位置编码
            new_position_ids = []
            for i in range(batch_size):
                indices = high_entropy_positions[i]
                current_positions = []
                last_idx = 0
                
                for pos in indices:
                    # 添加原始序列到当前高熵位置的position ids
                    if position_ids is not None:
                        current_positions.extend(position_ids[i, last_idx:pos].tolist())
                    else:
                        current_positions.extend(range(last_idx, pos))
                    
                    # 为插入的ponderer token使用对应高熵位置的position id
                    current_positions.append(pos)
                    
                    # 为插入的高熵位置token使用对应高熵位置的position id
                    current_positions.append(pos)
                    
                    # 为插入的interpolated_embed使用对应高熵位置的position id
                    current_positions.append(pos)
                    
                    last_idx = pos + 1
                
                # 添加剩余序列的position ids
                if last_idx < seq_len:
                    if position_ids is not None:
                        current_positions.extend(position_ids[i, last_idx:].tolist())
                    else:
                        current_positions.extend(range(last_idx, seq_len))
                
                # Padding处理
                pad_length = max_new_seq_len - len(current_positions)
                if pad_length > 0:
                    current_positions.extend([0] * pad_length)  # 用0填充position ids
                
                new_position_ids.append(current_positions)
            
            new_position_ids = torch.tensor(new_position_ids, device=new_inputs_embeds.device)
        if self.config.vary_position:
            new_position_ids = None

        final_outputs = self.gpt_neox(
            input_ids=None,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,  # 使用新的position_ids
            head_mask=head_mask,
            inputs_embeds=iter_inputs_embeds,  # 使用最后一次迭代的inputs_embeds
            past_key_values=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        del iter_inputs_embeds,new_position_ids,new_attention_mask
        # torch.cuda.empty_cache()
        hidden_states = final_outputs[0]
        logits = self.embed_out(hidden_states)
        del hidden_states#,final_outputs
        # torch.cuda.empty_cache()
        loss = None
        if labels is not None:
            # 计算loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = new_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + final_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=final_outputs.past_key_values,
            hidden_states=final_outputs.hidden_states,
            attentions=final_outputs.attentions,
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past


@add_start_docstrings(
    """
    The GPTNeoX Model transformer with a sequence classification head on top for extractive question-answering tasks like
    SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    GPT_NEOX_START_DOCSTRING,
)
class GPTNeoXForSequenceClassification(GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.gpt_neox = GPTNeoXModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.FloatTensor]]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

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
                logger.warning_once(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GPTNeoXForTokenClassification(GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.gpt_neox = GPTNeoXModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint="LarsJonasson/pythia-410m-deduped-sft-swedish",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_loss=0.25,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor]]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The GPT-NeoX Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    GPT_NEOX_START_DOCSTRING,
)
class GPTNeoXForQuestionAnswering(GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.gpt_neox = GPTNeoXModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
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

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )