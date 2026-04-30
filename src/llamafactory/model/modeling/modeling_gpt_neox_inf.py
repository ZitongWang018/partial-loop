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
import torch.distributed as dist
import time
import wandb

def ddp_sync_int(x: int, device) -> int:
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([x], device=device, dtype=torch.int64)
        dist.broadcast(t, src=0)
        return int(t.item())
    return x

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
        ponder_gate: Optional[torch.Tensor] = None,
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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.
        ponder_gate: Optional[torch.Tensor] = None,
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

        # 形状: 当前是 [B, nH, L, d]
        d_orig = query.size(-1)
        B, H, L = query.size(0), query.size(1), query.size(2)
        if ponder_gate is not None:
            # g -> G=log(g)，按 key（列）生效
            eps = 1e-12
            thr = 1e-4
            if ponder_gate.dim() == 2:
                g = ponder_gate
            # elif ponder_gate.dim() == 3 and ponder_gate.shape[1] == 1:  # [B, L, 1]
            #     g = ponder_gate[:, 0, :] 
            # elif ponder_gate.dim() == 3 and ponder_gate.shape[1] == L:  # [B, L, L]
            #     g = ponder_gate[:, 0, :] 
            else:
                raise ValueError(f"Unsupported ponder_gate shape: {tuple(ponder_gate.shape)}; "
                         f"expect [B,L].")
            G = torch.log(g.clamp_min(eps))  # [B, L] dtype 与 value 相同
            G_hard = torch.where(g <= thr, torch.full_like(G, -1e4), G)
            G_exp = G_hard.unsqueeze(1).expand(B, H, L)  # [B, H, L]

            # 选择拓展宽度 8，保证 head_dim % 8 == 0
            pad_w = 8
            d_aug = d_orig + pad_w

            # 构造 Q/K/V 拓展分量；注意广播到 per-head
            # Q 的 extra: 第一维恒为 sqrt(d_aug)，其他 7 维为 0
            q_extra = query.new_zeros(B, H, L, pad_w)
            q_extra[..., 0] = math.sqrt(float(d_aug))
            # K 的 extra: 第一维为 G（按 key/列）；broadcast 到 head 维
            # 先 [B, L, 1] -> [B, 1, L, 1] -> expand 到 H
            k_extra = key.new_zeros(B, H, L, pad_w)
            k_extra[..., 0] = G_exp
            # V 的 extra: 全 0（避免影响输出特征），pad_w 维
            v_extra = value.new_zeros(B, H, L, pad_w)

            query = torch.cat([query, q_extra], dim=-1)  # [B,H,L,d_aug]
            key   = torch.cat([key,   k_extra], dim=-1)
            value = torch.cat([value, v_extra], dim=-1)
            softmax_scale = 1.0 / math.sqrt(float(d_aug))
        else:
            d_aug = d_orig
            softmax_scale = self.norm_factor

        # Permute to get the expected shape for Flash Attention
        query = query.permute(0, 2, 1, 3)  # [B,L,H,d_aug]
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
            # softmax_scale=self.norm_factor,
            softmax_scale=softmax_scale,  # <--- 用 d_aug 对应的 scale
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        # Reshape outputs
        # attn_output = attn_weights.reshape(
        #     attn_weights.shape[0], attn_weights.shape[1], self.num_attention_heads * self.head_size
        # )
        # 如果做了维度拓展，先切回原始 d
        if ponder_gate is not None:
            attn_output = attn_weights[..., :d_orig]
            attn_output = attn_output.reshape(B, L, self.num_attention_heads * d_orig)
        else:
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
        # ponder_gate: Optional[torch.Tensor] = None,  # [B, L_total]
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
        # print(f"[DEBUG] layer {layer_idx} attention impl:", type(self.attention).__name__)

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
        ponder_gate: Optional[torch.FloatTensor] = None,  # [B, L_total, 1] 或 None
    ):
        # x_ln = self.input_layernorm(hidden_states)
        # if ponder_weights is not None:
        #     x_ln = x_ln * ponder_weights             # [B,L_total,H] * [B,L_total,1] 广播
        attention_layer_outputs = self.attention(
            self.input_layernorm(hidden_states),
            # x_ln,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            ponder_gate=ponder_gate,
        )
        attn_output = attention_layer_outputs[0]  # output_attn: attn_output, present, (attn_weights)
        attn_output = self.post_attention_dropout(attn_output)
        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            # mlp_in = self.post_attention_layernorm(hidden_states)
            # if ponder_weights is not None:
            #     mlp_in = mlp_in * ponder_weights
            # mlp_output = self.mlp(mlp_in)
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            mlp_in = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(mlp_in)
            # mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
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

        # 在 GPTNeoXPreTrainedModel.__init__ 或 GPTNeoXModel.__init__ 一开始：
        # if getattr(config, "_attn_implementation", None) != "fa2":
        #     config._attn_implementation = "fa2"

        # # 某些 transformers 版本还看这个字段
        # if hasattr(config, "attn_implementation"):
        #     config.attn_implementation = "flash_attention_2"

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
        ponder_gate: Optional[torch.FloatTensor] = None,  # [B, L_total, 1] 或 None
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
                    ponder_gate=ponder_gate,          # <--- 透传
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
                    ponder_gate=ponder_gate,          # <--- 透传
                )
            hidden_states = outputs[0]
            if use_cache is True:
                next_decoder_cache = outputs[1]
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.final_layer_norm(hidden_states)
        # if ponder_weights is not None:
        #     hidden_states = hidden_states * ponder_weights
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
def print_top_k_from_logits_in_debug(scaled_logits_tensor: torch.Tensor,  k: int = 5):
    """
    辅助函数，用于在调试 `compute_interpolated_embeds` 时打印 top-k token 及其概率。
    假定在 `compute_interpolated_embeds` 函数内部被调用。
    此时，输入的 `scaled_logits_tensor` 应该是 `self.embed_out(hidden) / self.config.softmax_temperature` 的结果。

    Args:
        scaled_logits_tensor (torch.Tensor): 模型输出的、已经经过 temperature 缩放的 logits，
                                            形状通常为 [B, L, vocab_size] 或 [L, vocab_size]。
        current_tokenizer: 当前使用的 tokenizer 实例。
        k (int): 要显示的 top-k token 的数量。
    """
    from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    )
    current_tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/public/data/sxsong/1.4b_wd01_addtoken/checkpoint-40000",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if not isinstance(scaled_logits_tensor, torch.Tensor):
        print("错误: scaled_logits_tensor 必须是 torch.Tensor 类型。")
        return
    if not hasattr(current_tokenizer, 'decode'):
        print("错误: current_tokenizer 必须拥有 'decode' 方法。")
        return

    # 我们关注序列中最后一个位置的下一个token的概率
    logits_for_last_token_scaled = None
    if scaled_logits_tensor.ndim == 3:  # Batch, Seq_Len, Vocab_Size
        # 调试时，通常关注第一个样本的最后一个token
        logits_for_last_token_scaled = scaled_logits_tensor[0, -1, :]
    elif scaled_logits_tensor.ndim == 2:  # Seq_Len, Vocab_Size (假设 batch_size=1 且已被压缩)
        logits_for_last_token_scaled = scaled_logits_tensor[-1, :]
    elif scaled_logits_tensor.ndim == 1: # Vocab_Size (假设 batch_size=1, seq_len=1 且已被压缩)
        logits_for_last_token_scaled = scaled_logits_tensor
        print("dim1")
    else:
        print(f"错误: scaled_logits_tensor 的维度 ({scaled_logits_tensor.ndim}) 不符合预期。应为 1, 2, 或 3。")
        return

    if logits_for_last_token_scaled is None:
        print("错误: 未能从 scaled_logits_tensor 中提取有效的 logits。")
        return

    probs = torch.softmax(logits_for_last_token_scaled, dim=-1)

    if probs.numel() == 0:
        print("错误: softmax 后的 probs 张量为空。")
        return

    effective_k = min(k, probs.size(-1))
    if effective_k <= 0:
        print("错误: 词汇表大小为 0 或调整后的 k <= 0。")
        return
        
    top_k_probs, top_k_indices = torch.topk(probs, k=effective_k)

    print(f"\n--- Debug (compute_interpolated_embeds iteration): Top {effective_k} 候选 Token 概率 (来自最后一个位置的 scaled_logits) ---")
    for i in range(effective_k):
        token_id = top_k_indices[i].item()
        token_str = current_tokenizer.decode([token_id], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        prob_val = top_k_probs[i].item()
        print(f"  Token: '{token_str}' (ID: {token_id}), 概率: {prob_val:.4f}")
    print("-------------------------------------------------------------------------------------------------------\n")

class GPTNeoXForCausalLM(GPTNeoXPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.ponder_head = torch.nn.Linear(config.hidden_size, self.config.more_iterations + 1, bias=True)
        # TODO: 初始化 ponder_head，可以试试 uniform 的情况
        # 注意：输出维度是 K+1，包含步骤0（阶段0，不ponder）到步骤K（ponder K次）

        # Ponder entropy scheduler parameters
        self.ponder_ent_lambda_start = float(getattr(self.config, "ponder_ent_lambda_start", 0.0))
        self.ponder_ent_lambda_max = float(getattr(self.config, "ponder_ent_lambda_max", 0.2))
        self.ponder_ent_warmup_steps = int(getattr(self.config, "ponder_ent_warmup_steps", 1000))
        self.ponder_ent_peak_steps = int(getattr(self.config, "ponder_ent_peak_steps", 4000))
        # Ponder cost scheduler parameters
        self.ponder_cost_lambda_start = float(getattr(self.config, "ponder_cost_lambda_start", 0.0))
        self.ponder_cost_lambda_max = float(getattr(self.config, "ponder_cost_lambda_max", 0.1))
        self.ponder_cost_warmup_steps = int(getattr(self.config, "ponder_cost_warmup_steps", 1000))
        self.ponder_cost_peak_steps = int(getattr(self.config, "ponder_cost_peak_steps", 4000))
        # Diverse loss scheduler parameters
        self.diverse_lambda_start = float(getattr(self.config, "diverse_lambda_start", 0.0))
        self.diverse_lambda_max = float(getattr(self.config, "diverse_lambda_max", 0.1))
        self.diverse_warmup_steps = int(getattr(self.config, "diverse_warmup_steps", 1000))
        self.diverse_peak_steps = int(getattr(self.config, "diverse_peak_steps", 4000))
        # Weight distribution loss scheduler parameters
        self.weight_dist_lambda_start = float(getattr(self.config, "weight_dist_lambda_start", 0.0))
        self.weight_dist_lambda_max = float(getattr(self.config, "weight_dist_lambda_max", 0.1))
        self.weight_dist_warmup_steps = int(getattr(self.config, "weight_dist_warmup_steps", 1000))
        self.weight_dist_peak_steps = int(getattr(self.config, "weight_dist_peak_steps", 4000))
        # Minimum weight penalty loss scheduler parameters
        self.min_weight_penalty_lambda_start = float(getattr(self.config, "min_weight_penalty_lambda_start", 0.0))
        self.min_weight_penalty_lambda_max = float(getattr(self.config, "min_weight_penalty_lambda_max", 0.0))
        self.min_weight_penalty_warmup_steps = int(getattr(self.config, "min_weight_penalty_warmup_steps", 1000))
        self.min_weight_penalty_peak_steps = int(getattr(self.config, "min_weight_penalty_peak_steps", 4000))
        # Min weight penalty method: "accuracy" or "delta_loss"
        self.min_weight_penalty_method = getattr(self.config, "min_weight_penalty_method", "accuracy")
        self.sigma_slope = float(getattr(self.config, "sigma_slope", 1.0))
        self._internal_global_step = 0

        self.post_init()        

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings
        # --- Helper function to prepare inputs for GPT-NeoX ---

    def build_key_bias_mask(self, B, L_total, device, key_visible_2d, dtype, g_key_2d):
        """
        key_visible_2d: [B, L_total]  1=可见, 0=padding
        g_key_2d      : [B, L_total]  每个 key 的 gate \in (0,1]
        返回: attn_bias [B, 1, L_total, L_total] = causal_bias + pad_bias + log_gate_bias
        说明:
        - causal_bias: 下三角允许，上三角(未来)加 -inf
        - pad_bias: key padding 位置加 -inf，按 [B,1,1,L] 广播到 [B,1,L,L]
        - log_gate_bias: log(g) 作为加性偏置，按 [B,1,1,L] 广播到 [B,1,L,L]
        """
        # -- 0) 基本量
        finfo = torch.finfo(dtype)
        mask_value = torch.tensor(finfo.min, dtype=dtype, device=device)

        # -- 1) 因果遮挡: 直接用 tril 造 [1,1,L,L] 的下三角
        causal_allow = torch.tril(torch.ones((L_total, L_total), dtype=torch.bool, device=device))
        causal_bias = (~causal_allow).to(dtype).view(1, 1, L_total, L_total) * mask_value  # [1,1,L,L]

        # -- 2) padding 偏置: [B,1,1,L]
        pad_bias = (1.0 - key_visible_2d.to(dtype)).view(B, 1, 1, L_total) * mask_value

        # -- 3) log(gate) 偏置: [B,1,1,L]
        eps = 1e-6
        log_g = torch.log(g_key_2d.clamp_min(eps)).to(dtype).view(B, 1, 1, L_total)

        # -- 4) 合成（广播）：(B,1,1,L) + (1,1,L,L) → (B,1,L,L)
        attn_bias = causal_bias + pad_bias + log_g
        return attn_bias  # [B,1,L,L]

    # def prepare_gpt_inputs_for_stages(
    #     self,
    #     current_stages_list: List[torch.Tensor],
    #     batch_size: int, seq_len_orig: int, hidden_size: int, dev: torch.device,
    #     orig_pos_ids: torch.Tensor, orig_attn_mask_2d: torch.Tensor,
    #     scale_emb_val: Optional[torch.Tensor],
    #     config_obj,
    #     w_var: Optional[torch.Tensor] = None,     # [B, L_orig, K] 或 None
    #     force_eval_prune: Optional[bool] = None,  # <--- 新增：None=遵循config；True=强制剪枝；False=强制不剪
    # ):
    #     B, L, H = current_stages_list[0].shape
    #     num_stages_to_interleave = len(current_stages_list)
    #     interleaved_len = num_stages_to_interleave * seq_len_orig

    #     loop_base_embeds = torch.empty((batch_size, interleaved_len, hidden_size), dtype=current_stages_list[0].dtype, device=dev)
    #     loop_pos_ids = torch.empty((batch_size, interleaved_len), dtype=orig_pos_ids.dtype, device=dev)
    #     loop_attn_mask = torch.empty((batch_size, interleaved_len), dtype=orig_attn_mask_2d.dtype, device=dev)
    #     loop_input_embeds_final = torch.empty_like(loop_base_embeds)

    #     for stage_idx in range(num_stages_to_interleave):
    #         s_start = stage_idx # Defines the offset for interleaving

    #         loop_base_embeds[:, s_start::num_stages_to_interleave, :] = current_stages_list[stage_idx]
    #         loop_pos_ids[:, s_start::num_stages_to_interleave] = orig_pos_ids
    #         loop_attn_mask[:, s_start::num_stages_to_interleave] = orig_attn_mask_2d
            
    #         current_base_slice = current_stages_list[stage_idx]
    #         loop_input_embeds_final[:, s_start::num_stages_to_interleave, :] = current_base_slice

    #     total_len = loop_input_embeds_final.size(1)  # = num_stages_to_interleave * L_orig
    #     ponder_w = loop_input_embeds_final.new_ones(B, total_len)
    #     if w_var is not None:
    #         # 阶段0也使用 w_var[..., 0]（步骤0，阶段0，不ponder）
    #         if num_stages_to_interleave > 0:
    #             ponder_w[:, 0::num_stages_to_interleave] = w_var[..., 0]  # [B, L_orig]
    #         # 阶段i（i>=1）使用 w_var[..., i]（步骤i，阶段i，ponder i次）
    #         for i in range(1, num_stages_to_interleave):
    #             wi = w_var[..., i]      # [B, L_orig] - 注意：索引从 i-1 改为 i
    #             ponder_w[:, i::num_stages_to_interleave] = wi

    #     do_prune = (not self.training) and getattr(self.config, "eval_prune_by_gate", True)
    #     if force_eval_prune is not None:
    #         do_prune = (not self.training) and bool(force_eval_prune)

    #     if do_prune:
    #         thr = float(getattr(self.config, "ponder_gate_eval_thr", 1e-4))
    #         hard_keep = (ponder_w[..., 0] >= thr)  # [B, L*S] True=保留, False=剪掉

    #         # 只把 mask 位置置 0（padding）；不把 ponder_w 硬化为 0/1，也不改 embed（你目前的需求）
    #         loop_attn_mask = loop_attn_mask * hard_keep.to(loop_attn_mask.dtype)
    #     # return loop_input_embeds_final, loop_pos_ids, key_bias_4d
    #     return loop_input_embeds_final, loop_pos_ids, loop_attn_mask, ponder_w

    # # --- Helper function to extract hidden states for interpolation/refinement ---
    def extract_hidden_for_computation(self, gpt_outputs, selector_slice, config_obj):
        last_h_state = gpt_outputs.last_hidden_state
        return last_h_state[:, selector_slice, :]

    # ============================================================
    # [ADDED] Eval-only mismatch metrics helpers
    # ============================================================
    def _rank0_print(self, msg: str):
        try:
            import torch.distributed as dist
            if (not dist.is_initialized()) or (dist.get_rank() == 0):
                print(msg)
        except Exception:
            print(msg)

    @torch.no_grad()
    def _dbg_w_metrics(self, w_a: torch.Tensor, w_b: torch.Tensor, thr: float = 1e-4):
        w_mse = torch.mean((w_a - w_b) ** 2)

        ma = (w_a < thr)
        mb = (w_b < thr)
        mask_hamming = (ma ^ mb).float().mean()

        inter = (ma & mb).sum().float()
        union = (ma | mb).sum().float()
        mask_jaccard = inter / (union + 1e-12)
        return w_mse, mask_hamming, mask_jaccard

    @torch.no_grad()
    def _dbg_logits_metrics(self, logits_a: torch.Tensor, logits_b: torch.Tensor, attn_mask_2d: torch.Tensor):
        mask = attn_mask_2d.to(dtype=torch.bool)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits_a.device), torch.tensor(0.0, device=logits_a.device)

        logits_mse = ((logits_a - logits_b)[mask].float() ** 2).mean()

        top1_a = torch.argmax(logits_a, dim=-1)
        top1_b = torch.argmax(logits_b, dim=-1)
        top1_agree = (top1_a[mask] == top1_b[mask]).float().mean()
        return logits_mse, top1_agree

    
    def prepare_gpt_inputs_for_stages(
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

    def compute_interpolated_embeds(self, weight: torch.Tensor, hidden: Union[torch.Tensor, Tuple[torch.Tensor, ...]], use_topk: bool = True) -> torch.Tensor:
        # 保留用户原有的 compute_interpolated_embeds 实现细节，仅做微小修正以处理可能的边界或配置缺失
        # 确保 hidden_layer_num 和 softmax_temperature, interpolation_topk 是从 config 读取
        softmax_temp = getattr(self.config, 'softmax_temperature', 1.0)
        hidden_layer_step = getattr(self.config, 'hidden_layer_num', 1)
        if hidden_layer_step <= 0: hidden_layer_step = 1 # Ensure positive step

        if use_topk:
            top_k_val = getattr(self.config, 'interpolation_topk', 100)
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
        else: # use_topk is False (full softmax)
            logits = self.embed_out(hidden) / softmax_temp
            probs = torch.softmax(logits, dim=-1)
            interpolated_embeds = torch.matmul(probs, weight)
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


    # def _compute_ponder_weights(self, hs_stage0: torch.Tensor):
    #     """
    #     hs_stage0: [B, L, H]  —— 来自阶段0（真实token对应阶段）的 hidden states
    #     返回:
    #     s: [B, L, K+1]       —— softmax 后的步长分布，s[..., 0]对应步骤0（阶段0），s[..., i]对应步骤i（i>=1）
    #     w: [B, L, K+1]       —— 逆向积分得到的权重（w_i = sum_{j=i..K} s_j），w[..., 0]对应步骤0，w[..., i]对应步骤i
    #     expected_steps: [B, L] —— 可选指标：期望步数 = sum_i i * s_i（从0计步，即步骤0对应i=0）
    #     entropy: [B, L]      —— 分布熵
    #     """
    #     K = self.config.more_iterations
    #     # K+1 个步骤：步骤0（阶段0，不ponder）到步骤K（ponder K次）
    #     num_steps = K + 1
        
    #     logits_raw = self.ponder_head(hs_stage0)  # [B, L, K+1]
    #     s = torch.softmax(logits_raw, dim=-1)  # [B, L, K+1]
    #     # 逆向积分：w_i = s_i + s_{i+1} + ... + s_K
    #     # 用flip+cumsum实现
    #     w_rev = torch.flip(torch.cumsum(torch.flip(s, dims=[-1]), dim=-1), dims=[-1])  # [B, L, K+1]
    #     w = w_rev

    #     # 期望步数（从0开始）：E[step] = sum_i i * s_i，其中i从0到K
    #     idx = torch.arange(0, num_steps, device=hs_stage0.device, dtype=hs_stage0.dtype).view(1, 1, num_steps)
    #     expected = torch.sum(idx * s, dim=-1)  # [B, L]

    #     # 熵：H(s) = -sum s log s（避免log(0)）
    #     ent = -torch.sum(s * torch.clamp(torch.log(s + 1e-12), min=-50), dim=-1)  # [B, L]
        
    #     # 计算diverse loss需要的ent_mean: s_mean分布的熵
    #     s_mean = s.mean(dim=(0, 1))  # [K+1]
    #     ent_mean = -torch.sum(s_mean * torch.clamp(torch.log(s_mean + 1e-12), min=-50))  # 标量

    #     return s, w, expected, ent, ent_mean

    def _compute_lambda_entropy(self, global_step: int) -> float:
        """
        根据 global_step 计算当前的 lambda_entropy 值。
        调度策略：
        - 前 warmup_steps 步：保持 lambda_start
        - warmup_steps 到 peak_steps：从 lambda_start 线性增长到 lambda_max
        - peak_steps 之后：保持 lambda_max
        """
        if global_step < self.ponder_ent_warmup_steps:
            return self.ponder_ent_lambda_start
        elif global_step >= self.ponder_ent_peak_steps:
            return self.ponder_ent_lambda_max
        else:
            # 线性插值：从 warmup_steps 到 peak_steps
            # 防止除零错误：如果 peak_steps <= warmup_steps，直接返回 lambda_max
            if self.ponder_ent_peak_steps <= self.ponder_ent_warmup_steps:
                return self.ponder_ent_lambda_max
            progress = (global_step - self.ponder_ent_warmup_steps) / (
                self.ponder_ent_peak_steps - self.ponder_ent_warmup_steps
            )
            return self.ponder_ent_lambda_start + progress * (
                self.ponder_ent_lambda_max - self.ponder_ent_lambda_start
            )

    def _compute_lambda_ponder_cost(self, global_step: int) -> float:
        """
        根据 global_step 计算当前的 lambda_ponder_cost 值。
        调度策略：
        - 前 ponder_cost_warmup_steps 步：保持 ponder_cost_lambda_start
        - ponder_cost_warmup_steps 到 ponder_cost_peak_steps：从 ponder_cost_lambda_start 线性增长到 ponder_cost_lambda_max
        - ponder_cost_peak_steps 之后：保持 ponder_cost_lambda_max
        """
        if global_step < self.ponder_cost_warmup_steps:
            return self.ponder_cost_lambda_start
        elif global_step >= self.ponder_cost_peak_steps:
            return self.ponder_cost_lambda_max
        else:
            # 线性插值：从 ponder_cost_warmup_steps 到 ponder_cost_peak_steps
            # 防止除零错误：如果 peak_steps <= warmup_steps，直接返回 lambda_max
            if self.ponder_cost_peak_steps <= self.ponder_cost_warmup_steps:
                return self.ponder_cost_lambda_max
            progress = (global_step - self.ponder_cost_warmup_steps) / (
                self.ponder_cost_peak_steps - self.ponder_cost_warmup_steps
            )
            return self.ponder_cost_lambda_start + progress * (
                self.ponder_cost_lambda_max - self.ponder_cost_lambda_start
            )

    def _compute_lambda_diverse(self, global_step: int) -> float:
        """
        根据 global_step 计算当前的 lambda_diverse 值。
        调度策略：
        - 前 diverse_warmup_steps 步：保持 diverse_lambda_start
        - diverse_warmup_steps 到 diverse_peak_steps：从 diverse_lambda_start 线性增长到 diverse_lambda_max
        - diverse_peak_steps 之后：保持 diverse_lambda_max
        """
        if global_step < self.diverse_warmup_steps:
            return self.diverse_lambda_start
        elif global_step >= self.diverse_peak_steps:
            return self.diverse_lambda_max
        else:
            # 线性插值：从 diverse_warmup_steps 到 diverse_peak_steps
            # 防止除零错误：如果 peak_steps <= warmup_steps，直接返回 lambda_max
            if self.diverse_peak_steps <= self.diverse_warmup_steps:
                return self.diverse_lambda_max
            progress = (global_step - self.diverse_warmup_steps) / (
                self.diverse_peak_steps - self.diverse_warmup_steps
            )
            return self.diverse_lambda_start + progress * (
                self.diverse_lambda_max - self.diverse_lambda_start
            )

    def _compute_lambda_weight_dist(self, global_step: int) -> float:
        """
        根据 global_step 计算当前的 lambda_weight_dist 值。
        调度策略：
        - 前 weight_dist_warmup_steps 步：保持 weight_dist_lambda_start
        - weight_dist_warmup_steps 到 weight_dist_peak_steps：从 weight_dist_lambda_start 线性增长到 weight_dist_lambda_max
        - weight_dist_peak_steps 之后：保持 weight_dist_lambda_max
        """
        if global_step < self.weight_dist_warmup_steps:
            return self.weight_dist_lambda_start
        elif global_step >= self.weight_dist_peak_steps:
            return self.weight_dist_lambda_max
        else:
            # 线性插值：从 weight_dist_warmup_steps 到 weight_dist_peak_steps
            # 防止除零错误：如果 peak_steps <= warmup_steps，直接返回 lambda_max
            if self.weight_dist_peak_steps <= self.weight_dist_warmup_steps:
                return self.weight_dist_lambda_max
            progress = (global_step - self.weight_dist_warmup_steps) / (
                self.weight_dist_peak_steps - self.weight_dist_warmup_steps
            )
            return self.weight_dist_lambda_start + progress * (
                self.weight_dist_lambda_max - self.weight_dist_lambda_start
            )

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
        current_lambda_entropy = self._compute_lambda_entropy(global_step)
        current_lambda_ponder_cost = self._compute_lambda_ponder_cost(global_step)
        current_lambda_diverse = self._compute_lambda_diverse(global_step)
        current_lambda_weight_dist = self._compute_lambda_weight_dist(global_step)
        current_lambda_min_weight_penalty = self._compute_lambda_min_weight_penalty(global_step)

        ponder_aux = 0.0
        entropy_loss = current_lambda_entropy * entropy_mean
        ponder_aux = ponder_aux + entropy_loss

        # Use scheduled lambda_ponder_cost for cost loss
        cost_loss = current_lambda_ponder_cost * cost_mean
        ponder_aux = ponder_aux + cost_loss

        # Use scheduled lambda_diverse for diverse loss
        # 由于ent_mean是正值（负熵），高熵时值更大，我们做减法来奖励高熵（鼓励多样性）
        diverse_mean = ent_mean  # 直接使用传入的ent_mean，避免重复计算
        diverse_loss = current_lambda_diverse * diverse_mean
        ponder_aux = ponder_aux - diverse_loss  # 减法：奖励多样性

        # Use scheduled lambda_weight_dist for weight distribution loss
        # 计算 w_mean 和 w_target（等差数列）
        if w_var is not None and current_lambda_weight_dist > 0:
            w_mean = w_var.mean(dim=(0, 1))  # [K+1]
            K_plus_1 = w_mean.shape[0]
            # 生成等差数列：从 1.0 开始，公差为 1/(K+1)，不包含 0
            # 如果 K+1=5，公差 = 1/5 = 0.2，则 w_target = [1.0, 0.8, 0.6, 0.4, 0.2]
            if K_plus_1 > 1:
                step = 1.0 / K_plus_1  # 公差
                indices = torch.arange(0, K_plus_1, device=w_mean.device, dtype=w_mean.dtype)
                w_target = 1.0 - indices * step  # w_target[i] = 1.0 - i * (1/(K+1))
            else:
                w_target = torch.tensor([1.0], device=w_mean.device, dtype=w_mean.dtype)
            # 计算 weight_dist_loss
            weight_dist_mean = ((w_mean - w_target) ** 2).mean()
            weight_dist_loss = current_lambda_weight_dist * weight_dist_mean
            ponder_aux = ponder_aux + weight_dist_loss
        else:
            weight_dist_mean = torch.tensor(0.0, device=entropy.device, dtype=entropy.dtype)
            weight_dist_loss = torch.tensor(0.0, device=entropy.device, dtype=entropy.dtype)

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
            entropy_mean.detach(),
            entropy_loss.detach(),
            cost_mean.detach(),
            cost_loss.detach(),
            diverse_mean.detach(),
            diverse_loss.detach(),
            weight_dist_mean.detach(),
            weight_dist_loss.detach(),
            min_weight_penalty_loss.detach(),
            current_lambda_entropy,
            current_lambda_ponder_cost,
            current_lambda_diverse,
            current_lambda_weight_dist,
            current_lambda_min_weight_penalty,
            penalty_ratios,
        )


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
        global_step: Optional[int] = None,
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        user_requested_output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # For interpolation and refinement, we need hidden states.
        if self.training:
            if global_step is None:
                global_step = self._internal_global_step
                self._internal_global_step += 1
            self._wandb_log_step = int(global_step)

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
            # Ensure attention_mask is 2D: (batch_size, sequence_length)
            if attention_mask.ndim == 4: # Usually (B, 1, L, L_kv)
                 original_attention_mask_2d = attention_mask[:, 0, 0, :L_orig] # A heuristic, might need adjustment
            elif attention_mask.ndim == 3: # Usually (B, L, L_kv)
                 original_attention_mask_2d = attention_mask[:, 0, :L_orig] # A heuristic
            elif attention_mask.ndim == 2:
                 original_attention_mask_2d = attention_mask
            else:
                raise ValueError(f"Unsupported attention_mask dimension: {attention_mask.ndim}")


        inputs_embeds0 = initial_embeds_raw 
        weight_for_interpolation = self.gpt_neox.embed_in.weight
        embed_scale = None
        if hasattr(self.config, 'scale_embeds') and self.config.scale_embeds:
            if self.config.interpolation == True:
                embed_scale = torch.sqrt(torch.tensor(H, dtype=initial_embeds_raw.dtype, device=device))
            else:
                embed_scale = torch.sqrt(2.5*torch.tensor(H, dtype=initial_embeds_raw.dtype, device=device))
            inputs_embeds0 = initial_embeds_raw * embed_scale
            weight_for_interpolation = self.gpt_neox.embed_in.weight * embed_scale
        
        scaled_inputs_embeds0_for_orig_slots = inputs_embeds0 
        
        num_initial_interpolation_stages = self.config.more_iterations
        all_embeddings_stages: List[torch.Tensor] = [scaled_inputs_embeds0_for_orig_slots] 

        # ---- 在 interpolation 循环之前，先计算 w_var ----
        # 这样第0轮就能使用 w_var，阶段0的 ponder_gate 就能使用 w_var[..., 0]
        if num_initial_interpolation_stages > 0:
            # 先运行一次 gpt_neox，只使用阶段0的输入，获取阶段0的 hidden states
            pre_iter_stages = [scaled_inputs_embeds0_for_orig_slots]
            pre_iter_input_embeds, pre_iter_pos_ids, pre_iter_attn_mask, pre_iter_ponder_w = self.prepare_gpt_inputs_for_stages(
                pre_iter_stages, B, L_orig, H, device,
                original_repeated_position_ids, original_attention_mask_2d,
                embed_scale, self.config,
                w_var=None  # 此时还没有 w_var，阶段0的 ponder_gate 会是 1.0
            )
            
            pre_iter_gpt_neox_outputs = self.gpt_neox(
                inputs_embeds=pre_iter_input_embeds,
                attention_mask=pre_iter_attn_mask,
                position_ids=pre_iter_pos_ids,
                head_mask=head_mask, past_key_values=None, use_cache=False,
                output_attentions=False, output_hidden_states=user_requested_output_hidden_states,
                return_dict=True, cache_position=None,
                ponder_gate=pre_iter_ponder_w,  # 使用 prepare_gpt_inputs_for_stages 返回的 ponder_w
            )
            
            # 提取阶段0的 hidden states
            hs_E0_pre = self.extract_hidden_for_computation(
                pre_iter_gpt_neox_outputs, slice(0, None, 1), self.config
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
        for iter_k_idx in range(num_initial_interpolation_stages): 
            current_stages_for_iter_pass = list(all_embeddings_stages)
            num_stages_in_current_iter_input = len(current_stages_for_iter_pass)

            iter_input_embeds, iter_pos_ids, iter_attn_mask, iter_ponder_w = self.prepare_gpt_inputs_for_stages(
                current_stages_for_iter_pass, B, L_orig, H, device,
                original_repeated_position_ids, original_attention_mask_2d,
                embed_scale, self.config,
                w_var=w_var   # 现在第0轮也能使用 w_var 了
            )
            
            iter_gpt_neox_outputs = self.gpt_neox(
                inputs_embeds=iter_input_embeds,
                attention_mask=iter_attn_mask, 
                position_ids=iter_pos_ids, 
                head_mask=head_mask, past_key_values=None, use_cache=False,
                output_attentions=False, output_hidden_states=user_requested_output_hidden_states, 
                return_dict=True, cache_position=None,
                ponder_gate=iter_ponder_w,     # <--- 新增
            )
            
            interp_use_topk = getattr(self.config, 'interpolation_use_topk', False)
            
            computed_embeddings_from_hs_slices = []
            for stage_slice_idx in range(num_stages_in_current_iter_input):
                current_slice_selector = slice(stage_slice_idx, None, num_stages_in_current_iter_input)
                hidden_input_for_this_slice = self.extract_hidden_for_computation(
                    iter_gpt_neox_outputs, current_slice_selector, self.config
                )
                if self.config.interpolation == True:
                    embedding = self.compute_interpolated_embeds(
                        weight_for_interpolation, hidden_input_for_this_slice, use_topk=interp_use_topk
                    )
                else:
                    embedding = hidden_input_for_this_slice
                computed_embeddings_from_hs_slices.append(embedding)

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
            num_refinement_steps = getattr(self.config, 'training_refinement_steps', 5)
            if getattr(self.config, "vary_refine_steps", False) and global_step is not None:
                # # --- 关键：恢复并强制使用随机同步机制 ---
                # # 保存RNG状态，以避免影响dropout等其他随机过程
                # rng_state = torch.get_rng_state()
                # # 使用global_step作为种子，确保所有DDP进程在同一步使用相同的随机数
                # torch.manual_seed(global_step)
                        
                # # 在课程学习开始前，num_refinement_steps会保持其默认值(2)，无需额外处理
                if (torch.distributed.is_available() and torch.distributed.is_initialized()):
                    if torch.distributed.get_rank() == 0:
                        sampled = int(torch.randint(4, 7, (1,), device=device).item())
                    else:
                        sampled = 0
                    num_refinement_steps = ddp_sync_int(sampled, device)
                else:
                    num_refinement_steps = int(torch.randint(4, 7, (1,), device=device).item())

                # # --- 关键：恢复RNG状态 ---
                # torch.set_rng_state(rng_state)

        else:
            if getattr(self.config, "vary_refine_steps", False):
                if (torch.distributed.is_available() and torch.distributed.is_initialized()):
                    if torch.distributed.get_rank() == 0:
                        sampled = int(torch.randint(4, 7, (1,), device=device).item())
                    else:
                        sampled = 0
                    num_refinement_steps = ddp_sync_int(sampled, device)
                else:
                    num_refinement_steps = int(torch.randint(4, 7, (1,), device=device).item())           
            else:
                num_refinement_steps = getattr(self.config, 'eval_refinement_steps', 10)
            # print(f"num_refinement_steps: {num_refinement_steps}")

        # ---- Refinement Loop (Two-Phase, Scheme 1) ----
        # ============================================================
        # [CHANGED] Refinement (optional) + Always-run Final Pass
        # ============================================================

        # ============================================================
        # Joint Refinement with Damping (LOGIT or W)
        # (w 和 embedding 每一步同步更新)
        # ============================================================

        hs_E0_last = None
        S_current = len(all_embeddings_stages)

        do_refine = (num_initial_interpolation_stages > 0 and num_refinement_steps > 0)

        if do_refine:

            # ------------------------------
            # debug switches
            # ------------------------------
            debug_refine = bool(getattr(self.config, "debug_refine", True))
            debug_rerun_after_embed = bool(getattr(self.config, "debug_refine_rerun_after_embed", True))
            debug_rerun_every = int(getattr(self.config, "debug_refine_rerun_every", 1))
            debug_rerun_every = max(1, debug_rerun_every)

            def _rank0_print(msg: str):
                if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
                    print(msg)

            def _safe_mean(x: torch.Tensor) -> float:
                return float(x.detach().float().mean().cpu())

            def _mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return torch.mean((a - b) ** 2)

            # ============================================================
            # refine begin
            # ============================================================
            if debug_refine:
                _rank0_print(f"[REFINE] joint_refine_steps={num_refinement_steps}")

            # ------------------------------------------------------------
            # damping hyper-params
            # ------------------------------------------------------------
            damping_alpha = float(getattr(self.config, "damping_alpha", 0.1))
            last_n_hard   = int(getattr(self.config, "last_n_steps_update_w", 1))
            damping_alpha = max(0.0, min(1.0, damping_alpha))
            last_n_hard   = max(0, last_n_hard)

            # ------------------------------------------------------------
            # damping mode
            #   "logit": EMA on logits -> softmax -> w (recommended for alignment)
            #   "w":     EMA on w      -> reconstruct s from w (often better for training)
            # ------------------------------------------------------------
            # damping_mode = str(getattr(self.config, "damping_mode", "w")).lower().strip()
            # if damping_mode not in ("logit", "w"):
            #     if debug_refine:
            #         _rank0_print(f"[WARN] unknown damping_mode={damping_mode}, fallback to 'logit'")
            #     damping_mode = "logit"

            # ------------------------------------------------------------
            # STATE vars
            # ------------------------------------------------------------
            logits_var = None  # used only in logit damping

            # ------------------------------------------------------------
            # debug STATE
            # ------------------------------------------------------------
            hsE0_prev = None
            w_prev = None
            s_prev = None

            # -------------------------
            # before for-loop (state vars)
            # -------------------------
            damping_mode = str(getattr(self.config, "damping_mode", "s"))  # "logit" | "w" | "s"
            logits_state = None   # only for logit damping
            s_state      = None   # only for s damping
            w_state      = None   # only for w damping (optional)

            def _tail_cdf_from_s(s: torch.Tensor) -> torch.Tensor:
                return torch.flip(torch.cumsum(torch.flip(s, dims=[-1]), dim=-1), dims=[-1])

            def _stats_from_s(s: torch.Tensor):
                idx = torch.arange(0, s.shape[-1], device=s.device, dtype=s.dtype).view(1, 1, -1)
                expected = torch.sum(idx * s, dim=-1)
                entropy  = -torch.sum(s * torch.log(s + 1e-12), dim=-1)
                s_mean   = s.mean(dim=(0, 1))
                ent_mean = -torch.sum(s_mean * torch.log(s_mean + 1e-12))
                return expected, entropy, ent_mean

            for ref_idx in range(num_refinement_steps):

                current_stages = list(all_embeddings_stages)
                S = len(current_stages)

                # ========================================================
                # 1) forward with CURRENT embeddings + CURRENT w_var
                # ========================================================
                ref_input_embeds, ref_pos_ids, ref_attn_mask, ref_ponder_w = self.prepare_gpt_inputs_for_stages(
                    current_stages, B, L_orig, H, device,
                    original_repeated_position_ids, original_attention_mask_2d,
                    embed_scale, self.config,
                    w_var=w_var,
                )

                outputs = self.gpt_neox(
                    inputs_embeds=ref_input_embeds,
                    attention_mask=ref_attn_mask,
                    position_ids=ref_pos_ids,
                    head_mask=head_mask,
                    past_key_values=None,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=user_requested_output_hidden_states,
                    return_dict=True,
                    cache_position=None,
                    ponder_gate=ref_ponder_w,
                )

                # stage-0 hidden (用于更新 w)
                hs_E0 = self.extract_hidden_for_computation(outputs, slice(0, None, S), self.config)
                hs_E0_last = hs_E0

                # debug snapshot (给你后面的 ALIGN eval 用)
                self._last_hs_E0_ref = hs_E0.detach()
                if w_var is not None:
                    self._last_w_var = w_var.detach()

                # ========================================================
                # 1.5) DEBUG: hs_E0 / w / s 是否在变？
                # ========================================================
                if debug_refine:
                    if hsE0_prev is None:
                        hsE0_mse_prev = torch.tensor(0.0, device=device)
                    else:
                        hsE0_mse_prev = _mse(hs_E0, hsE0_prev)

                    if (w_prev is None) or (w_var is None):
                        w_mse_prev = torch.tensor(0.0, device=device)
                    else:
                        w_mse_prev = _mse(w_var, w_prev)

                    if (s_prev is None) or (s_var is None):
                        s_mse_prev = torch.tensor(0.0, device=device)
                    else:
                        s_mse_prev = _mse(s_var, s_prev)

                    _rank0_print(
                        f"[REFINE][{ref_idx}][PRE] "
                        f"hsE0_mse_to_prev={hsE0_mse_prev.item():.3e} "
                        f"w_mse_to_prev={w_mse_prev.item():.3e} "
                        f"s_mse_to_prev={s_mse_prev.item():.3e}"
                    )

                    hsE0_prev = hs_E0.detach()
                    if w_var is not None:
                        w_prev = w_var.detach()
                    if s_var is not None:
                        s_prev = s_var.detach()

                # ========================================================
                # 2) recompute fresh ponder weights + logits from hs_E0
                # ========================================================
                (
                    s_new,
                    w_new,
                    logits_new,
                    expected_new,
                    entropy_new,
                    ent_mean_new,
                ) = self._compute_ponder_weights(hs_E0, return_logits=True)

                # ========================================================
                # 2.5) DEBUG: logits_new / w_new 的幅度
                # ========================================================
                if debug_refine:
                    _rank0_print(
                        f"[REFINE][{ref_idx}][WNEW] "
                        f"w_new_mean={_safe_mean(w_new):.3e} "
                        f"s_new_sum_mean={_safe_mean(s_new.sum(dim=-1)):.3e} "
                        f"logits_new_mean={_safe_mean(logits_new):.3e} "
                        f"logits_new_std={float(logits_new.detach().float().std().cpu()):.3e}"
                    )

                
                # -------------------------
                # inside loop after (s_new, w_new, logits_new, ...)
                # -------------------------
                hard_tail_start = max(0, num_refinement_steps - last_n_hard)
                use_hard = (ref_idx >= hard_tail_start)

                def _tail_cdf_from_s(s: torch.Tensor) -> torch.Tensor:
                    return torch.flip(torch.cumsum(torch.flip(s, dims=[-1]), dim=-1), dims=[-1])

                def _stats_from_s(s: torch.Tensor):
                    idx = torch.arange(0, s.shape[-1], device=s.device, dtype=s.dtype).view(1, 1, -1)
                    expected = torch.sum(idx * s, dim=-1)
                    entropy  = -torch.sum(s * torch.log(s + 1e-12), dim=-1)
                    s_mean   = s.mean(dim=(0, 1))
                    ent_mean = -torch.sum(s_mean * torch.log(s_mean + 1e-12))
                    return expected, entropy, ent_mean

                def _project_tail_cdf_light(w: torch.Tensor) -> torch.Tensor:
                    # minimal projection for tail-CDF: [0,1], w0=1, monotone non-increasing
                    w = w.clamp(0.0, 1.0)
                    w[..., 0] = 1.0
                    # enforce w_k >= w_{k+1} via right-to-left cummax
                    w = torch.flip(torch.cummax(torch.flip(w, dims=[-1]), dim=-1)[0], dims=[-1])
                    return w

                damping_mode = str(getattr(self.config, "damping_mode", "s_log"))
                eps = float(getattr(self.config, "damping_eps", 1e-12))

                # -------------------- LOGIT damping --------------------
                if damping_mode == "logit":
                    if logits_state is None:
                        logits_state = logits_new.detach()
                        init_state = True
                    elif use_hard or damping_alpha >= 1.0:
                        logits_state = logits_new.detach()
                        init_state = True
                    elif damping_alpha <= 0.0:
                        init_state = False
                    else:
                        a = damping_alpha
                        logits_state = (1.0 - a) * logits_state + a * logits_new
                        init_state = False

                    s_var = torch.softmax(logits_state, dim=-1)
                    w_var = _tail_cdf_from_s(s_var)
                    expected_steps_var, entropy_var, ent_mean_var = _stats_from_s(s_var)

                # -------------------- W damping (RAW, light projection, no w->s->w) --------------------
                elif damping_mode == "w_raw":
                    if w_state is None:
                        w_state = w_new.detach()
                        init_state = True
                    elif use_hard or damping_alpha >= 1.0:
                        w_state = w_new.detach()
                        init_state = True
                    elif damping_alpha <= 0.0:
                        init_state = False
                    else:
                        a = damping_alpha
                        w_state = (1.0 - a) * w_state + a * w_new
                        init_state = False

                    w_var = _project_tail_cdf_light(w_state)

                    # derive s only for stats (optional)
                    s_var = torch.zeros_like(w_var)
                    s_var[..., :-1] = (w_var[..., :-1] - w_var[..., 1:]).clamp_min(0.0)
                    s_var[..., -1]  = w_var[..., -1].clamp_min(0.0)
                    s_var = s_var / s_var.sum(dim=-1, keepdim=True).clamp_min(eps)
                    expected_steps_var, entropy_var, ent_mean_var = _stats_from_s(s_var)

                # -------------------- W damping (PROJECTED, with w->s->w rebuild) --------------------
                elif damping_mode == "w_proj":
                    if w_state is None:
                        w_state = w_new.detach()
                        init_state = True
                    elif use_hard or damping_alpha >= 1.0:
                        w_state = w_new.detach()
                        init_state = True
                    elif damping_alpha <= 0.0:
                        init_state = False
                    else:
                        a = damping_alpha
                        w_state = (1.0 - a) * w_state + a * w_new
                        init_state = False

                    w_tmp = _project_tail_cdf_light(w_state)

                    # rebuild s then rebuild w to ensure exact consistency
                    s_tmp = torch.zeros_like(w_tmp)
                    s_tmp[..., :-1] = (w_tmp[..., :-1] - w_tmp[..., 1:]).clamp_min(0.0)
                    s_tmp[..., -1]  = w_tmp[..., -1].clamp_min(0.0)
                    s_tmp = s_tmp / s_tmp.sum(dim=-1, keepdim=True).clamp_min(eps)

                    s_var = s_tmp
                    w_var = _tail_cdf_from_s(s_var)
                    expected_steps_var, entropy_var, ent_mean_var = _stats_from_s(s_var)

                # -------------------- S damping (EMA in simplex) --------------------
                elif damping_mode == "s":
                    if s_state is None:
                        s_state = s_new.detach()
                        init_state = True
                    elif use_hard or damping_alpha >= 1.0:
                        s_state = s_new.detach()
                        init_state = True
                    elif damping_alpha <= 0.0:
                        init_state = False
                    else:
                        a = damping_alpha
                        s_state = (1.0 - a) * s_state + a * s_new
                        init_state = False

                    s_state = s_state.clamp_min(0.0)
                    s_state = s_state / s_state.sum(dim=-1, keepdim=True).clamp_min(eps)

                    s_var = s_state
                    w_var = _tail_cdf_from_s(s_var)
                    expected_steps_var, entropy_var, ent_mean_var = _stats_from_s(s_var)

                # -------------------- S damping in log-space (trick) --------------------
                elif damping_mode == "s_log":
                    ls_new = torch.log(s_new.clamp_min(eps))
                    if s_state is None:
                        s_state = ls_new.detach()          # reuse s_state as log-s state
                        init_state = True
                    elif use_hard or damping_alpha >= 1.0:
                        s_state = ls_new.detach()
                        init_state = True
                    elif damping_alpha <= 0.0:
                        init_state = False
                    else:
                        a = damping_alpha
                        s_state = (1.0 - a) * s_state + a * ls_new
                        init_state = False

                    # softmax(log s) = normalize(exp(log s)) = normalize(s), but smoother under EMA
                    s_var = torch.softmax(s_state, dim=-1)
                    w_var = _tail_cdf_from_s(s_var)
                    expected_steps_var, entropy_var, ent_mean_var = _stats_from_s(s_var)

                else:
                    raise ValueError(f"Unknown damping_mode={damping_mode}")


                # -------------------------
                # MUST-HAVE debug: prove w_var equals / differs from w_new
                # -------------------------
                if debug_refine:
                    w_mse_new_var = torch.mean((w_var - w_new) ** 2)
                    _rank0_print(
                        f"[REFINE][{ref_idx}][POST-W] mode={damping_mode} "
                        f"init_state={init_state} use_hard={use_hard} "
                        f"w_new_mean={_safe_mean(w_new):.3e} w_var_mean={_safe_mean(w_var):.3e} "
                        f"w_mse(new,var)={w_mse_new_var.item():.3e}"
                    )



                # ========================================================
                # 5) update embeddings USING UPDATED w_var
                # ========================================================
                interp_use_topk = getattr(self.config, "interpolation_use_topk", False)
                prev_embeddings = [x.detach().clone() for x in all_embeddings_stages[1:]]

                mse_list = []
                for i in range(num_initial_interpolation_stages):
                    sel = slice(i, None, S)
                    hs = self.extract_hidden_for_computation(outputs, sel, self.config)

                    if self.config.interpolation:
                        new_embed = self.compute_interpolated_embeds(
                            weight_for_interpolation, hs, use_topk=interp_use_topk
                        )
                    else:
                        new_embed = hs

                    mse = torch.mean((new_embed - prev_embeddings[i]) ** 2)
                    mse_list.append(mse.item())
                    all_embeddings_stages[i + 1] = new_embed

                # ========================================================
                # 5.5) DEBUG: embedding 更新后，hs_E0 会不会变？
                # ========================================================
                if debug_refine and debug_rerun_after_embed and ((ref_idx % debug_rerun_every) == 0):
                    with torch.no_grad():
                        stages_after = list(all_embeddings_stages)
                        S2 = len(stages_after)
                        rerun_inp, rerun_pos, rerun_mask, rerun_gate = self.prepare_gpt_inputs_for_stages(
                            stages_after, B, L_orig, H, device,
                            original_repeated_position_ids, original_attention_mask_2d,
                            embed_scale, self.config,
                            w_var=w_var,
                        )
                        rerun_out = self.gpt_neox(
                            inputs_embeds=rerun_inp,
                            attention_mask=rerun_mask,
                            position_ids=rerun_pos,
                            head_mask=head_mask,
                            past_key_values=None,
                            use_cache=False,
                            output_attentions=False,
                            output_hidden_states=False,
                            return_dict=True,
                            cache_position=None,
                            ponder_gate=rerun_gate,
                        )
                        hsE0_after = self.extract_hidden_for_computation(rerun_out, slice(0, None, S2), self.config)
                        hsE0_after_mse = _mse(hsE0_after, hs_E0)

                    _rank0_print(
                        f"[REFINE][{ref_idx}][RERUN] hsE0_mse(after_embed_update)={hsE0_after_mse.item():.3e}"
                    )

                # ========================================================
                # 6) logging (your original line)
                # ========================================================
                if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                    print(
                        f"[REFINE][{ref_idx}] "
                        f"mode={damping_mode} "
                        f"emb_mse_mean={sum(mse_list)/max(1,len(mse_list)):.3e} "
                        f"w_mean={w_var.mean().item():.3e}"
                    )


        # ---------- defaults / safety ----------
        # hs_E0_last = None  # [ADD] always define, even if no refine
        # S_current = len(all_embeddings_stages)  # stage count after interpolation

        # do_refine = (num_initial_interpolation_stages > 0 and num_refinement_steps > 0)

        # if do_refine:
        #     phase1_steps = int(getattr(self.config, "phase1_refine_steps", num_refinement_steps))
        #     phase2_steps = int(getattr(self.config, "phase2_refine_steps", 0))
        #     phase1_steps = max(0, phase1_steps)
        #     phase2_steps = max(0, phase2_steps)

        #     if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        #         print(f"[REFINE] phase1_steps={phase1_steps}, phase2_steps={phase2_steps}")

        #     def _run_one_refine_step(all_embeddings_stages, w_var, tag: str):
        #         current_stages = list(all_embeddings_stages)
        #         S = len(current_stages)

        #         ref_input_embeds, ref_pos_ids, ref_attn_mask, ref_ponder_w = \
        #             self.prepare_gpt_inputs_for_stages(
        #                 current_stages, B, L_orig, H, device,
        #                 original_repeated_position_ids, original_attention_mask_2d,
        #                 embed_scale, self.config,
        #                 w_var=w_var,
        #             )

        #         outputs = self.gpt_neox(
        #             inputs_embeds=ref_input_embeds,
        #             attention_mask=ref_attn_mask,
        #             position_ids=ref_pos_ids,
        #             head_mask=head_mask,
        #             past_key_values=None,
        #             use_cache=False,
        #             output_attentions=False,
        #             output_hidden_states=user_requested_output_hidden_states,
        #             return_dict=True,
        #             cache_position=None,
        #             ponder_gate=ref_ponder_w,
        #         )

        #         hs_E0 = self.extract_hidden_for_computation(outputs, slice(0, None, S), self.config)

        #         # ===== [ADD] store last hs for ALIGN debug =====
        #         self._last_hs_E0_ref = hs_E0.detach()
        #         if w_var is not None:
        #             self._last_w_var = w_var.detach()

        #         interp_use_topk = getattr(self.config, "interpolation_use_topk", False)
        #         prev_embeddings = [x.detach().clone() for x in all_embeddings_stages[1:]]

        #         mse_list = []
        #         for i in range(num_initial_interpolation_stages):
        #             sel = slice(i, None, S)
        #             hs = self.extract_hidden_for_computation(outputs, sel, self.config)

        #             if self.config.interpolation:
        #                 new_embed = self.compute_interpolated_embeds(
        #                     weight_for_interpolation, hs, use_topk=interp_use_topk
        #                 )
        #             else:
        #                 new_embed = hs

        #             mse = torch.mean((new_embed - prev_embeddings[i]) ** 2)
        #             mse_list.append(mse.item())
        #             all_embeddings_stages[i + 1] = new_embed

        #         if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        #             print(f"[REFINE][{tag}] emb_mse_mean={sum(mse_list)/max(1,len(mse_list)):.3e}")

        #         return hs_E0

        #     # ---------------- Phase 1: refine with w0 frozen ----------------
        #     for t in range(phase1_steps):
        #         hs_E0_last = _run_one_refine_step(all_embeddings_stages, w_var, tag=f"P1-{t}")

        #     # [ADD] phase1_steps=0 兜底：没有 refine 时也必须能 recompute w
        #     if hs_E0_last is None:
        #         # 用当前 stages 跑一次 "不更新 embedding" 的 forward 来取 hs_E0
        #         # 最小开销：只跑一遍 refine forward，但不写回 all_embeddings_stages
        #         current_stages = list(all_embeddings_stages)
        #         S_tmp = len(current_stages)
        #         tmp_inp, tmp_pos, tmp_mask, tmp_gate = self.prepare_gpt_inputs_for_stages(
        #             current_stages, B, L_orig, H, device,
        #             original_repeated_position_ids, original_attention_mask_2d,
        #             embed_scale, self.config,
        #             w_var=w_var,
        #         )
        #         tmp_out = self.gpt_neox(
        #             inputs_embeds=tmp_inp,
        #             attention_mask=tmp_mask,
        #             position_ids=tmp_pos,
        #             head_mask=head_mask,
        #             past_key_values=None,
        #             use_cache=False,
        #             output_attentions=False,
        #             output_hidden_states=user_requested_output_hidden_states,
        #             return_dict=True,
        #             cache_position=None,
        #             ponder_gate=tmp_gate,
        #         )
        #         hs_E0_last = self.extract_hidden_for_computation(tmp_out, slice(0, None, S_tmp), self.config)

        #     # ---------------- Recompute w AFTER Phase 1 ----------------
        #     # ===== [ADD] snapshot old w for ALIGN debug =====
        #     if w_var is not None:
        #         self._last_w_var_before_recomp = w_var.detach()

        #     s_re, w_re, expected_re, ent_re, ent_mean_re = self._compute_ponder_weights(hs_E0_last)

        #     # [ADD] print w_jump
        #     if w_var is not None:
        #         w_jump = torch.mean((w_re - w_var) ** 2).item()
        #     else:
        #         w_jump = float("nan")

        #     if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        #         print(f"[REFINE][RECOMP] w_jump_mse={w_jump:.3e}")

        #     # s_var = s_re
        #     # w_var = w_re
        #     # expected_steps_var = expected_re
        #     # entropy_var = ent_re
        #     # ent_mean_var = ent_mean_re
        #     # ===== [DAMPING] blend w/s instead of hard replace =====
        #     alpha = float(getattr(self.config, "recomp_damping_alpha", 0.2))  # 0~1
        #     alpha = max(0.0, min(1.0, alpha))

        #     if alpha >= 1.0 or (w_var is None) or (s_var is None):
        #         # hard replace (old behavior)
        #         s_var = s_re
        #         w_var = w_re
        #     else:
        #         # EMA in *w-space* + keep s consistent via EMA too
        #         w_var = (1.0 - alpha) * w_var + alpha * w_re
        #         s_var = (1.0 - alpha) * s_var + alpha * s_re

        #         # optional: renormalize s_var to sum=1 (avoid drift)
        #         s_sum = s_var.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        #         s_var = s_var / s_sum

        #     # stats (recompute to match current s_var/w_var)
        #     expected_steps_var = expected_re  # 可先偷懒用 expected_re
        #     entropy_var = ent_re
        #     ent_mean_var = ent_mean_re
        #     # ===== [DAMPING END] =====

        #     # ===== [ADD] snapshot new w for ALIGN debug =====
        #     self._last_w_var_after_recomp = w_var.detach()

        #     # ---------------- Phase 2: refine with w_recomp frozen ----------------
        #     for t in range(phase2_steps):
        #         hs_E0_last = _run_one_refine_step(all_embeddings_stages, w_var, tag=f"P2-{t}")

        #     # ============================================================
        #     # [ADD][POST-P2 ALIGN DEBUG] recompute w_post and approx w_inf
        #     # ============================================================
        #     # ============================================================
        #     # [CHANGED][POST-P2 ALIGN DEBUG]
        #     # IMPORTANT: Under ZeRO-3, any extra self.gpt_neox forward must run on ALL ranks,
        #     # otherwise it will deadlock on collectives.
        #     # ============================================================
        #     try:
        #         with torch.no_grad():
        #             # ---- 1) recompute w from LAST hs_E0 after Phase2 (pure tensor ops, safe) ----
        #             s_post, w_post, expected_post, ent_post, ent_mean_post = self._compute_ponder_weights(hs_E0_last)
        #             w_post_mse_vs_current = torch.mean((w_post - w_var) ** 2)

        #             # ---- 2) approximate inference-time w: stage0-only forward (MUST RUN ON ALL RANKS) ----
        #             stage0_only = [all_embeddings_stages[0]]  # [B, L, H]
        #             inf_inp, inf_pos, inf_mask, inf_gate = self.prepare_gpt_inputs_for_stages(
        #                 stage0_only, B, L_orig, H, device,
        #                 original_repeated_position_ids, original_attention_mask_2d,
        #                 embed_scale, self.config,
        #                 w_var=None,
        #                 force_eval_prune=False
        #             )
        #             inf_out = self.gpt_neox(
        #                 inputs_embeds=inf_inp,
        #                 attention_mask=inf_mask,
        #                 position_ids=inf_pos,
        #                 head_mask=head_mask,
        #                 past_key_values=None,
        #                 use_cache=False,
        #                 output_attentions=False,
        #                 output_hidden_states=user_requested_output_hidden_states,
        #                 return_dict=True,
        #                 cache_position=None,
        #                 ponder_gate=inf_gate,
        #             )
        #             hs_E0_inf = self.extract_hidden_for_computation(inf_out, slice(0, None, 1), self.config)  # [B,L,H]
        #             s_inf, w_inf, expected_inf, ent_inf, ent_mean_inf = self._compute_ponder_weights(hs_E0_inf)

        #             w_inf_mse_vs_current = torch.mean((w_inf - w_var) ** 2)
        #             w_inf_mse_vs_post    = torch.mean((w_inf - w_post) ** 2)

        #             # ---- 3) top1 agreement ----
        #             top1_cur  = torch.argmax(s_var, dim=-1)
        #             top1_post = torch.argmax(s_post, dim=-1)
        #             top1_inf  = torch.argmax(s_inf, dim=-1)
        #             top1_agree_inf_cur  = (top1_inf == top1_cur).float().mean()
        #             top1_agree_inf_post = (top1_inf == top1_post).float().mean()

        #             # ---- 4) mask agreement ----
        #             thr = float(getattr(self.config, "ponder_gate_eval_thr", 1e-4))
        #             m_cur  = (w_var  < thr)
        #             m_post = (w_post < thr)
        #             m_inf  = (w_inf  < thr)

        #             ham_inf_cur  = (m_inf ^ m_cur).float().mean()
        #             ham_inf_post = (m_inf ^ m_post).float().mean()

        #             def _jacc(a, b):
        #                 inter = (a & b).float().sum()
        #                 union = (a | b).float().sum().clamp_min(1.0)
        #                 return inter / union

        #             jac_inf_cur  = _jacc(m_inf, m_cur)
        #             jac_inf_post = _jacc(m_inf, m_post)

        #             # ---- optional: aggregate scalars across ranks for stable printing ----
        #             # (not strictly needed, but nice; avoids rank-to-rank variance)
        #             if torch.distributed.is_initialized():
        #                 for t in [
        #                     w_post_mse_vs_current,
        #                     w_inf_mse_vs_current,
        #                     w_inf_mse_vs_post,
        #                     top1_agree_inf_cur,
        #                     top1_agree_inf_post,
        #                     ham_inf_cur,
        #                     ham_inf_post,
        #                     jac_inf_cur,
        #                     jac_inf_post,
        #                 ]:
        #                     torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        #                     t /= torch.distributed.get_world_size()

        #         # ---- print only on rank0 ----
        #         if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
        #             print(f"[REFINE][POST-P2] w_post_mse_vs_current={w_post_mse_vs_current.item():.3e}")
        #             print(f"[REFINE][POST-P2][INF-APPROX] w_inf_mse_vs_current={w_inf_mse_vs_current.item():.3e} "
        #                 f"w_inf_mse_vs_post={w_inf_mse_vs_post.item():.3e}")
        #             print(f"[REFINE][POST-P2][TOP1] agree_inf_cur={top1_agree_inf_cur.item():.4f} "
        #                 f"agree_inf_post={top1_agree_inf_post.item():.4f}")
        #             print(f"[REFINE][POST-P2][MASK thr={thr}] ham_inf_cur={ham_inf_cur.item():.6f} ham_inf_post={ham_inf_post.item():.6f} "
        #                 f"jac_inf_cur={jac_inf_cur.item():.6f} jac_inf_post={jac_inf_post.item():.6f}")

        #     except Exception as e:
        #         if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
        #             print(f"[REFINE][POST-P2] debug failed: {e}")


        # ============================================================
        # [CHANGED] Final pass MUST ALWAYS run (outside do_refine)
        # ============================================================

        final_stages_for_pass = list(all_embeddings_stages)
        num_total_stages_final = len(final_stages_for_pass)
        S = num_total_stages_final
        K = S - 1
        ponder_aux_loss = torch.tensor(0.0, device=device, dtype=initial_embeds_raw.dtype)

        # [ADD] always define these (no UnboundLocalError)
        (final_inp_np, final_pos_np, final_mask_np, final_gate_np) = self.prepare_gpt_inputs_for_stages(
            final_stages_for_pass, B, L_orig, H, device,
            original_repeated_position_ids, original_attention_mask_2d,
            embed_scale, self.config, w_var=w_var, force_eval_prune=False
        )

        final_pass_gpt_neox_outputs = self.gpt_neox(
            inputs_embeds=final_inp_np,
            attention_mask=final_mask_np,
            position_ids=final_pos_np,
            head_mask=head_mask, past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=user_requested_output_hidden_states,
            return_dict=True, cache_position=cache_position,
            ponder_gate=final_gate_np,
        )

        # =========================================================
        # ==== [ADDED] 定义 all_hidden_states_from_final_pass，避免你之前 NameError
        # =========================================================
        all_hidden_states_from_final_pass = final_pass_gpt_neox_outputs.last_hidden_state  # [B, L*S, H]

        # =========================================================
        # [ADDED] 先算出 final_output_logits（oracle debug 里要用）
        # =========================================================
        hidden_states_final = torch.zeros_like(initial_embeds_raw)
        for i in range(0, S):
            hidden_states_final += all_hidden_states_from_final_pass[:, i::S, :] * s_var[..., i].unsqueeze(-1)
        final_output_logits = self.embed_out(hidden_states_final)

        # =========================================================
        # [MOVED][FIXED] Eval-only oracle debug：必须放到 final_output_logits 之后
        # + 修复：用 self._last_hs_E0_ref / self._last_w_var，而不是 locals()
        # + 注意：额外 forward 必须所有 rank 都跑（你这里没有 rank0-only 包裹 forward，OK）
        # =========================================================
        if (not self.training) and getattr(self.config, "debug_align_metrics", True):
            def _rank0_print(msg: str):
                if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
                    print(msg)

            def _dbg_w_metrics(w_a: torch.Tensor, w_b: torch.Tensor, thr: float):
                # w_mse
                w_mse = torch.mean((w_a - w_b) ** 2)

                # mask: True 表示被剪（<thr），False 表示保留
                m_a = (w_a < thr)
                m_b = (w_b < thr)

                # hamming: 不同位置比例
                mask_hamming = (m_a ^ m_b).float().mean()

                # jaccard: 交并比（两个 mask 都为 True 的交集 / 并集）
                inter = (m_a & m_b).float().sum()
                union = (m_a | m_b).float().sum().clamp_min(1.0)
                mask_jaccard = inter / union
                return w_mse, mask_hamming, mask_jaccard

            def _dbg_logits_metrics(logits_a: torch.Tensor, logits_b: torch.Tensor, attn_2d: torch.Tensor):
                # 只在有效 token 上算
                # logits: [B,L,V]，attn_2d: [B,L]
                mask = attn_2d.to(torch.bool)
                # logits mse：对 V 做均方，再对 token 平均
                diff = (logits_a - logits_b) ** 2
                mse_per_tok = diff.mean(dim=-1)  # [B,L]
                logits_mse = mse_per_tok[mask].mean()

                # top1 agree
                top1_a = torch.argmax(logits_a, dim=-1)
                top1_b = torch.argmax(logits_b, dim=-1)
                top1_agree = (top1_a == top1_b).float()[mask].mean()
                return logits_mse, top1_agree

            try:
                # ==== [CHANGED] 你原来用 locals() 判断，永远拿不到。这里改为读取 self._last_hs_E0_ref / self._last_w_var ====
                has_hs = hasattr(self, "_last_hs_E0_ref") and (self._last_hs_E0_ref is not None)
                has_w  = (w_var is not None)  # 当前 w_var 一定有（K>0时）

                if (not has_hs) or (not has_w):
                    _rank0_print("[ALIGN][eval] skipped: missing self._last_hs_E0_ref or w_var is None")
                else:
                    thr = float(getattr(self.config, "ponder_gate_eval_thr", 1e-4))

                    # 1) recompute w from last refinement stage0 hidden
                    s_recomp, w_recomp, expected_recomp, ent_recomp, entm_recomp = self._compute_ponder_weights(self._last_hs_E0_ref)

                    # 2) w-space mismatch
                    w_mse, mask_hamming, mask_jaccard = _dbg_w_metrics(w_var, w_recomp, thr=thr)

                    # 3) ONE extra refinement step updating embeddings using w_recomp (no further w update)
                    oracle_embeddings_stages = list(all_embeddings_stages)

                    oracle_ref_inp, oracle_ref_pos, oracle_ref_mask, oracle_ref_gate = self.prepare_gpt_inputs_for_stages(
                        oracle_embeddings_stages, B, L_orig, H, device,
                        original_repeated_position_ids, original_attention_mask_2d,
                        embed_scale, self.config,
                        w_var=w_recomp,
                        force_eval_prune=False,
                    )

                    oracle_ref_outputs = self.gpt_neox(
                        inputs_embeds=oracle_ref_inp,
                        attention_mask=oracle_ref_mask,
                        position_ids=oracle_ref_pos,
                        head_mask=head_mask, past_key_values=None, use_cache=False,
                        output_attentions=False, output_hidden_states=False,
                        return_dict=True, cache_position=None,
                        ponder_gate=oracle_ref_gate,
                    )

                    interp_use_topk = getattr(self.config, 'interpolation_use_topk', False)
                    S_oracle_ref = len(oracle_embeddings_stages)

                    for i in range(num_initial_interpolation_stages):
                        sel = slice(i, None, S_oracle_ref)
                        hs_for_stage = self.extract_hidden_for_computation(oracle_ref_outputs, sel, self.config)
                        if self.config.interpolation:
                            refined_embeds = self.compute_interpolated_embeds(
                                weight_for_interpolation, hs_for_stage, use_topk=interp_use_topk
                            )
                        else:
                            refined_embeds = hs_for_stage
                        oracle_embeddings_stages[i + 1] = refined_embeds

                    # 4) oracle final pass with (oracle_embeddings + w_recomp)
                    oracle_final_inp, oracle_final_pos, oracle_final_mask, oracle_final_gate = self.prepare_gpt_inputs_for_stages(
                        oracle_embeddings_stages, B, L_orig, H, device,
                        original_repeated_position_ids, original_attention_mask_2d,
                        embed_scale, self.config,
                        w_var=w_recomp,
                        force_eval_prune=False,
                    )

                    oracle_final_outputs = self.gpt_neox(
                        inputs_embeds=oracle_final_inp,
                        attention_mask=oracle_final_mask,
                        position_ids=oracle_final_pos,
                        head_mask=head_mask, past_key_values=None, use_cache=False,
                        output_attentions=False, output_hidden_states=False,
                        return_dict=True, cache_position=None,
                        ponder_gate=oracle_final_gate,
                    )

                    oracle_all_hs = oracle_final_outputs.last_hidden_state
                    oracle_hidden_final = torch.zeros_like(initial_embeds_raw)

                    S_oracle = len(oracle_embeddings_stages)
                    for i in range(0, S_oracle):
                        oracle_hidden_final += oracle_all_hs[:, i::S_oracle, :] * s_recomp[..., i].unsqueeze(-1)

                    oracle_logits = self.embed_out(oracle_hidden_final)

                    # 5) logits mismatch
                    logits_mse, top1_agree = _dbg_logits_metrics(final_output_logits, oracle_logits, original_attention_mask_2d)

                    _rank0_print(
                        f"[ALIGN][eval] w_mse={w_mse.item():.6e} mask_hamming={mask_hamming.item():.6e} "
                        f"mask_jaccard={mask_jaccard.item():.6e} logits_mse={logits_mse.item():.6e} "
                        f"top1_agree={top1_agree.item():.6e} thr={thr}"
                    )

            except Exception as e:
                _rank0_print(f"[ALIGN][eval] exception: {repr(e)}")

        # ====== 下面保持你原逻辑继续走（pruned loss 等） ======

        
        # 计算每个ponder步骤的预测准确率和softmax概率（用于计算penalty_ratio）
        # step_accuracies[i]存储步骤i的准确率，用于wandb记录
        # step_penalty_ratios[i]存储步骤i的惩罚比例（基于softmax概率），用于惩罚步骤i+1（方法1）
        # step_ce_losses[i]存储步骤i的CE loss，用于计算delta loss（方法2）
        # 例如：step_penalty_ratios[0]是w0的惩罚比例，用于惩罚w1（w0永远等于1，不惩罚）
        #      step_penalty_ratios[1]是w1的惩罚比例，用于惩罚w2
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
                        logits_step_i = self.embed_out(hs_step_i)  # [B, L, vocab_size]
                        
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
                        logits_step_i = self.embed_out(hs_step_i)  # [B, L, vocab_size]
                        
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
                            per_token_penalty_ratio = self.sigma_slope * torch.sigmoid(50.0 * (per_token_delta_loss))  # [B, L-1]
                            
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
                        logits_step_i = self.embed_out(hs_step_i)  # [B, L, vocab_size]

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
        
        # Eval模式：额外计算剪枝版本的hidden_states和loss（仅用于wandb记录）
        # ==== [FIX] ensure S is defined before first use ====
        if 'final_stages_for_pass' in locals():
            S = len(final_stages_for_pass)
        elif 'final_pass_gpt_neox_outputs' in locals():
            # fallback: infer S from last_hidden_state length
            # last_hidden_state: [B, L_orig*S, H]
            S = int(final_pass_gpt_neox_outputs.last_hidden_state.shape[1] // L_orig)
        else:
            # safest fallback (no interleaving)
            S = 1
        # ==== [FIX END] ====

        hidden_states_final_pruned = None
        pruned_loss = None
        if not self.training and w_var is not None:

            # ==== [FIX] make sure these are defined in THIS scope ====
            all_hidden_states_from_final_pass = final_pass_gpt_neox_outputs.last_hidden_state
            # （可选但很建议）避免 s_var=None 时这里又炸
            if s_var is None:
                s_var = torch.softmax(self._compute_ponder_weights(hs_E0_pre, return_logits=True)[2], dim=-1)  # 仅在你确实有 hs_E0_pre 时才行
            # ==== [FIX END] ====

            hidden_states_final_pruned = torch.zeros_like(initial_embeds_raw)
            thr = 1e-4
            for i in range(0, S):
                weight_pruned = torch.where(w_var[..., i] >= thr, s_var[..., i], torch.zeros_like(s_var[..., i]))
                hidden_states_final_pruned += all_hidden_states_from_final_pass[:, i::S, :] * weight_pruned.unsqueeze(-1)

            final_output_logits_pruned = self.embed_out(hidden_states_final_pruned)

            if labels is not None and final_output_logits_pruned.shape[1] == L_orig:
                shift_logits_pruned = final_output_logits_pruned[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                pruned_loss = CrossEntropyLoss()(
                    shift_logits_pruned.view(-1, self.config.vocab_size),
                    shift_labels.view(-1)
                )


        # 训练时计算ponder_aux_loss
        if self.training and ent_mean_var is not None:
            ponder_aux_loss, ent_mean, ent_loss, cost, cost_loss, diverse_mean, diverse_loss, weight_dist_mean, weight_dist_loss, min_weight_penalty_loss, current_lambda_entropy, current_lambda_ponder_cost, current_lambda_diverse, current_lambda_weight_dist, current_lambda_min_weight_penalty, penalty_ratios = self._accumulate_ponder_losses(
                entropy_var, expected_steps_var, ent_mean_var, w_var, global_step, step_accuracies, step_penalty_ratios, step_ce_losses
            )
        elif self.training:
            # 如果ent_mean_var为None（即more_iterations == 0），不计算ponder loss
            ponder_aux_loss = torch.tensor(0.0, device=device, dtype=initial_embeds_raw.dtype)
            ent_mean = torch.tensor(0.0, device=device)
            ent_loss = torch.tensor(0.0, device=device)
            cost = torch.tensor(0.0, device=device)
            cost_loss = torch.tensor(0.0, device=device)
            diverse_mean = torch.tensor(0.0, device=device)
            diverse_loss = torch.tensor(0.0, device=device)
            weight_dist_mean = torch.tensor(0.0, device=device)
            weight_dist_loss = torch.tensor(0.0, device=device)
            min_weight_penalty_loss = torch.tensor(0.0, device=device)
            current_lambda_entropy = 0.0
            current_lambda_ponder_cost = 0.0
            current_lambda_diverse = 0.0
            current_lambda_weight_dist = 0.0
            current_lambda_min_weight_penalty = 0.0
            penalty_ratios = None

        loss = None
        if labels is not None and final_output_logits.shape[1] == L_orig:
            # 使用标准的Lce损失计算方式
            shift_logits = final_output_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            base_loss = CrossEntropyLoss()(shift_logits.view(-1, self.config.vocab_size),
                                        shift_labels.view(-1))
            
            # 训练时加上ponder_aux_loss
            if self.training:
                loss = base_loss + ponder_aux_loss
                try:
                    log_dict = {
                        "train/base_loss": float(base_loss.detach().cpu()),
                        "train/ent_loss":  float(ent_loss),
                        "train/cost_loss": float(cost_loss),
                        "train/diverse_loss": float(diverse_loss),  # 正值，在ponder_aux中做减法（奖励多样性）
                        "train/diverse_reward": float(-diverse_loss),  # 负值，表示从loss中减去的奖励
                        "train/weight_dist_loss": float(weight_dist_loss),
                        "train/min_weight_penalty_loss": float(min_weight_penalty_loss),
                        "train/total_loss": float(loss.detach().cpu()),
                        "train/ent_mean":  float(ent_mean),
                        "train/diverse_mean": float(diverse_mean),
                        "train/weight_dist_mean": float(weight_dist_mean),
                        "train/lambda_entropy": float(current_lambda_entropy),
                        "train/lambda_ponder_cost": float(current_lambda_ponder_cost),
                        "train/lambda_diverse": float(current_lambda_diverse),
                        "train/lambda_weight_dist": float(current_lambda_weight_dist),
                        "train/lambda_min_weight_penalty": float(current_lambda_min_weight_penalty),
                        "train/mean_steps":float(cost),
                    }
                    # 添加每个步骤的惩罚比例
                    if penalty_ratios is not None:
                        for step_name, ratio in penalty_ratios.items():
                            log_dict[f"train/penalty_ratio_{step_name}"] = float(ratio)
                    # 添加每个步骤的准确率
                    if step_accuracies is not None:
                        for i, acc in enumerate(step_accuracies):
                            # i=0对应w0，i=1对应w1，以此类推
                            log_dict[f"train/step_accuracy_w{i}"] = float(acc.detach().cpu())
                    wandb.log(log_dict, commit=False)
                except Exception:
                    pass
            else:
                loss = base_loss
                # Eval模式：记录剪枝后的loss到wandb
                if pruned_loss is not None:
                    try:
                        wandb.log({
                            "eval/pruned_loss": float(pruned_loss.detach().cpu()),
                        }, commit=False)
                    except Exception:
                        pass

        final_hidden_states_to_return = final_pass_gpt_neox_outputs.hidden_states if user_requested_output_hidden_states else None
        final_attentions_to_return    = final_pass_gpt_neox_outputs.attentions if output_attentions else None

        if not return_dict:
            items = [final_output_logits]
            if final_pass_gpt_neox_outputs.past_key_values is not None:
                items.append(final_pass_gpt_neox_outputs.past_key_values)
            if final_hidden_states_to_return is not None:
                items.append(final_hidden_states_to_return)
            if final_attentions_to_return is not None:
                items.append(final_attentions_to_return)
            tup = tuple(items)
            return ((loss,) + tup) if loss is not None else tup

        return CausalLMOutputWithPast(
            loss=loss,
            logits=final_output_logits,
            past_key_values=final_pass_gpt_neox_outputs.past_key_values,
            hidden_states=final_hidden_states_to_return,
            attentions=final_attentions_to_return,
        )
    @staticmethod
    def _calculate_mse_embedding_change(current_embedding: torch.Tensor, previous_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the Mean Squared Error and Relative MSE between two embedding tensors."""
        if current_embedding.shape != previous_embedding.shape:
            logger.warning(
                f"Shape mismatch for MSE calculation: current {current_embedding.shape} vs previous {previous_embedding.shape}. Returning NaN."
            )
            nan_tensor = torch.tensor(float('nan'), device=current_embedding.device, dtype=current_embedding.dtype)
            return nan_tensor, nan_tensor
        
        if current_embedding.dtype != previous_embedding.dtype:
            logger.warning(
                f"Dtype mismatch for MSE calculation: current {current_embedding.dtype} vs previous {previous_embedding.dtype}. Casting previous to current."
            )
            previous_embedding = previous_embedding.to(current_embedding.dtype)
            
        mse = torch.mean((current_embedding - previous_embedding) ** 2)

        # Calculate relative MSE, avoiding division by zero
        mean_sq_prev = torch.mean(previous_embedding ** 2)
        if mean_sq_prev > 1e-9:
            relative_mse = mse / mean_sq_prev
        else:
            relative_mse = torch.tensor(float('nan'), device=current_embedding.device, dtype=current_embedding.dtype)


        return mse, relative_mse

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