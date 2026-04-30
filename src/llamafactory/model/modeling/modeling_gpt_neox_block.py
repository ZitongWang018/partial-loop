# coding=utf-8
# Copyright 2022 EleutherAI The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
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

# Import from original BlockRecurrentTransformer
from random import random
from functools import wraps, partial
from itertools import zip_longest
from collections import namedtuple, defaultdict
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
# from beartype import beartype # Used for type checking, can be optionally removed if not desired
# from beartype.door import is_bearable


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
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig # This config is kept as is
import math
import torch.distributed as dist
import time
# 在现有代码后添加以下模块
# ================ 新增模块开始 ================
import math
from random import random
from functools import wraps, partial
from itertools import zip_longest
from collections import namedtuple, defaultdict
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# 辅助函数
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)

# 旋转位置编码
class RotaryEmbedding(nn.Module): # Renamed to avoid conflict with HF's GPTNeoXRotaryEmbedding
    def __init__(self, dim, width, scale_base=512, theta=10000):
        super().__init__()
        self.width = width
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent=False)
        self.register_buffer('cached_freqs', None, persistent=False)
        self.register_buffer('cached_scales', None, persistent=False)
    
    @property
    def device(self):
        return next(self.buffers()).device
        
    def forward(self):
        device, seq_len = self.device, self.width
        if exists(self.cached_freqs):
            cached_seq_len = self.cached_freqs.shape[-2]
            if cached_seq_len >= seq_len:
                return self.cached_freqs[:seq_len], self.cached_scales[:seq_len]
        
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1) # Concatenate for cos and sin
        
        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim=-1)
        
        self.register_buffer('cached_freqs', freqs, persistent=False)
        self.register_buffer('cached_scales', scale, persistent=False)
        return freqs, scale

def rotate_half_custom(x): # Renamed to avoid conflict with HF's rotate_half
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_custom(t, freqs, scale=1.): # Renamed to avoid conflict with HF's apply_rotary_pos_emb
    scale = default(scale, 1.)
    seq_len = t.shape[-2]
    # Ensure freqs and scale are correctly sliced for the current sequence length
    freqs = freqs[-seq_len:]
    if isinstance(scale, torch.Tensor):
        scale = scale[-seq_len:]

    return (t * freqs.cos() * scale) + (rotate_half_custom(t) * freqs.sin() * scale)


# 记忆管理器
class MemoryManager(nn.Module):
    def __init__(self, dim, layers=1, mem_lengths=512, compress_factors=1):
        super().__init__()
        mem_lengths = cast_tuple(mem_lengths)
        compress_factors = cast_tuple(compress_factors)
        self.mem_lengths = mem_lengths
        self.compress_factors = compress_factors
        self.layers = nn.ModuleList([])
        
        for _ in range(layers):
            compress_fns = nn.ModuleList([])
            for compress_factor in compress_factors:
                compress_fn = nn.Identity()
                if compress_factor > 1:
                    compress_fn = nn.Sequential(
                        Rearrange('b n d -> b d n'),
                        nn.Conv1d(dim * 2, dim * 2, compress_factor, 
                                  stride=compress_factor, groups=2),
                        Rearrange('b d n -> b n d'),
                    )
                compress_fns.append(compress_fn)
            self.layers.append(compress_fns)
    
    def forward(self, past_memories, new_memories):
        next_memories = []
        # past_memories and new_memories are lists of (k, v) tuples.
        # k, v are of shape (B, H, N, D_head)
        # We need to flatten (B, H) into a single batch dimension for Conv1D.
        
        for layer_idx, (past_mem, new_mem, compress_fns) in enumerate(zip_longest(past_memories, new_memories, self.layers)):

            if not (exists(past_mem) or exists(new_mem)):
                next_memories.append(None)
                continue
                
            # Unpack k, v for current and new memories
            past_k, past_v = past_mem if exists(past_mem) else (None, None)
            new_k, new_v = new_mem if exists(new_mem) else (None, None) # new_mem might be None if a layer doesn't produce it

            # Combine k and v for processing (dim=-1 concatenates head_dim for k and v)
            past_kv = F.pad(safe_cat(past_k, past_v, dim=-1), (0,0,0,0,0,0,0,0), value=0.) if exists(past_k) else None
            new_kv = F.pad(safe_cat(new_k, new_v, dim=-1), (0,0,0,0,0,0,0,0), value=0.) if exists(new_k) else None

            current_layer_next_memory = None
            
            # Process through compression stages
            current_past_kv_for_stage = past_kv # This will be the remaining past_kv for the next compression stage

            for stage_idx, (mem_length, compress_factor, compress_fn) in enumerate(zip(self.mem_lengths, self.compress_factors, compress_fns)):
                
                current_mem_segment = None
                if exists(current_past_kv_for_stage):
                    # Take the latest `mem_length` segment from past_kv
                    if current_past_kv_for_stage.shape[-2] > mem_length:
                        current_past_kv_for_stage, current_mem_segment = current_past_kv_for_stage[..., :-mem_length, :], current_past_kv_for_stage[..., -mem_length:, :]
                    else:
                        current_mem_segment = current_past_kv_for_stage
                        current_past_kv_for_stage = None # All consumed
                        
                # Compress the new memories for this stage
                compressed_new_kv_for_stage = None
                if exists(new_kv) and new_kv.shape[-2] > 0 and compress_factor > 1:
                    # new_kv has shape (B, H, N, D_head*2)
                    original_batch, original_heads, new_mem_len_seq, original_dim = new_kv.shape
                    curtailed_length = (new_mem_len_seq // compress_factor) * compress_factor
                    curtailed_slice = slice(-curtailed_length, None) if curtailed_length > 0 else slice(0, 0)
                    
                    if curtailed_length > 0:
                        temp_new_kv = new_kv[..., curtailed_slice, :].reshape(original_batch * original_heads, curtailed_length, original_dim)
                        temp_new_kv = compress_fn(temp_new_kv) # (B*H, N_compressed, D_head*2)
                        compressed_new_kv_for_stage = temp_new_kv.reshape(original_batch, original_heads, -1, original_dim)
                    else:
                        compressed_new_kv_for_stage = new_kv.new_empty(original_batch, original_heads, 0, original_dim) # Empty with correct dims
                else: # No compression needed, or new_kv is empty
                    compressed_new_kv_for_stage = new_kv

                # FIFO memory queue: add the compressed new memory on the right
                combined_kv_segment = safe_cat(current_mem_segment, compressed_new_kv_for_stage, dim = -2)
                
                # Update new_kv for the next compression stage (what overflowed)
                if exists(combined_kv_segment) and combined_kv_segment.shape[-2] > mem_length:
                    new_kv_for_next_stage, current_mem_segment_final = combined_kv_segment[..., :-mem_length, :], combined_kv_segment[..., -mem_length:, :]
                else:
                    new_kv_for_next_stage = None
                    current_mem_segment_final = combined_kv_segment

                # Concat the current segment (potentially trimmed) to the left into current_layer_next_memory
                current_layer_next_memory = safe_cat(current_mem_segment_final, current_layer_next_memory, dim = -2)
                
                # The 'new_kv' for the next iteration of compression is what just overflowed
                new_kv = new_kv_for_next_stage


            # After all compression stages for this layer, split back into k and v
            if exists(current_layer_next_memory):
                k, v = current_layer_next_memory.chunk(2, dim=-1)
                next_memories.append((k, v))
            else:
                next_memories.append(None)

        return next_memories


# 状态容器
class StateContainer(nn.Module):
    def __init__(self, dim, num_state_vectors, dim_head=64, heads=8, 
                 qk_rmsnorm=False, qk_rmsnorm_scale=8, use_flash_attn=False,
                 config: GPTNeoXConfig = None # Added config for proper attention module initialization
                ):
        super().__init__()
        assert num_state_vectors > 0
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.state_norm = nn.LayerNorm(dim) # Changed to nn.LayerNorm for consistency with HF

        self.q_to_state = nn.Linear(dim, inner_dim, bias=False)
        self.q_from_state = nn.Linear(dim, inner_dim, bias=False)

        self.state_to_q = nn.Linear(dim, inner_dim, bias=False)
        self.state_to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.init_state = nn.Parameter(torch.randn(num_state_vectors, dim))
        torch.nn.init.normal_(self.init_state, 0, .1)
        self.state_pos_ids = nn.Parameter(torch.randn(num_state_vectors, dim))
        torch.nn.init.normal_(self.state_pos_ids, 0, 1)

        self.to_state_out = nn.Linear(inner_dim * 2, dim, bias=False)

        # Use GPT_NEOX_ATTENTION_CLASSES for compatibility with HF's attention implementations
        # Pass config and a dummy layer_idx as it's not used internally by StateContainer's attention
        attn_class = GPT_NEOX_ATTENTION_CLASSES[config._attn_implementation] if config else GPTNeoXAttention
        self.to_state_cross_attn = attn_class(config, layer_idx=None) 
        self.state_self_attn = attn_class(config, layer_idx=None)
        self.from_state_cross_attn = attn_class(config, layer_idx=None)

        # Gating related parameters
        self.state_out_to_gate = nn.Linear(dim, dim)
        self.learned_ema_beta = nn.Parameter(torch.randn(dim))
        torch.nn.init.normal_(self.learned_ema_beta, 0, .1)

        self.cache = None
        self.next_read_state = None

    def set_next_read_state(self, states):
        if not exists(states):
            # Ensure init_state has a batch dimension if `states` typically does
            # GPTNeoX's default attention expects (B, H, N, D_head) for Q, K, V
            # So states should typically be (B, N_states, D)
            states = self.init_state.unsqueeze(0) # (1, num_state_vectors, dim)
        self.next_read_state = (states,)

    def read(self, x_current_block): # x_current_block: (B, N_block, D)
        assert exists(self.next_read_state), 'States to be read must be set with .set_next_read_state'

        states, = self.next_read_state
        self.next_read_state = None

        normed_states = self.state_norm(states)
        normed_states = normed_states + self.state_pos_ids.unsqueeze(0) # Add positional ids, expanding for batch

        # Prepare Q for cross-attention from current block input
        q_to_state = self.q_to_state(x_current_block) # (B, N_block, inner_dim)
        q_to_state = rearrange(q_to_state, 'b n (h d) -> b h n d', h=self.heads) # (B, H, N_block, D_head)

        # Prepare K, V for state self-attention (these will be the K,V for query_to_state)
        state_k, state_v = self.state_to_kv(normed_states).chunk(2, dim=-1) # (B, N_states, D_head) each
        state_k = rearrange(state_k, 'b n d -> b h n d', h=self.heads) # (B, H, N_states, D_head)
        state_v = rearrange(state_v, 'b n d -> b h n d', h=self.heads) # (B, H, N_states, D_head)

        # Cross-attend: q_to_state queries the internal states
        # The GPTNeoXAttention forward expects (query, key, value, attention_mask, position_ids, ...)
        # Here, query is q_to_state, key is state_k, value is state_v
        attn_outputs_to_state = self.to_state_cross_attn(
            q_to_state,
            attention_mask=None, # No attention mask for state-to-input cross-attention usually
            position_ids=None, # RoPE for states is typically not used, or handled differently within StateContainer
            layer_past=(state_k, state_v), # Pass as layer_past to Attention to be used as K,V for cross-attention
            use_cache=False, # Not caching here
            output_attentions=False,
            cache_position=None,
            position_embeddings=None,
            rope_freqs_and_scales=None # Not applying global RoPE here
        )
        to_state_out = attn_outputs_to_state[0] # (B, H, N_block, D_head)
        to_state_out = rearrange(to_state_out, 'b h n d -> b n (h d)') # (B, N_block, H*D_head)

        self.cache = (states, normed_states, state_k, state_v)
        return to_state_out

    def write(self, *, memories: Tuple[torch.Tensor, torch.Tensor]): # memories: (k, v) from the main attention block, (B, H, N_block, D_head)
        assert exists(self.cache), "Cache must exist from a previous read operation."

        k_from_block, v_from_block = memories 
        batch_size = k_from_block.shape[0]

        states, normed_states, state_k_cached, state_v_cached = self.cache
        self.cache = None # Clear cache after use

        # Derive queries from states
        q_from_state = self.q_from_state(normed_states) # (B, N_states, inner_dim)
        q_from_state = rearrange(q_from_state, 'b n (h d) -> b h n d', h=self.heads) # (B, H, N_states, D_head)

        state_q = self.state_to_q(normed_states) # (B, N_states, inner_dim)
        state_q = rearrange(state_q, 'b n (h d) -> b h n d', h=self.heads) # (B, H, N_states, D_head)

        # States self-attention: states query cached states
        # K, V are the cached ones from the read step
        attn_outputs_state_self = self.state_self_attn(
            state_q,
            attention_mask=None,
            position_ids=None,
            layer_past=(state_k_cached, state_v_cached), # These are used as K,V for self-attention
            use_cache=False,
            output_attentions=False,
            cache_position=None,
            position_embeddings=None,
            rope_freqs_and_scales=None
        )
        state_out_self_attn = attn_outputs_state_self[0] # (B, H, N_states, D_head)

        # Cross-attention: states query current block's K, V
        attn_outputs_from_state_cross = self.from_state_cross_attn(
            q_from_state, # Query from states
            attention_mask=None,
            position_ids=None,
            layer_past=(k_from_block, v_from_block), # Current block's K,V are the source
            use_cache=False,
            output_attentions=False,
            cache_position=None,
            position_embeddings=None,
            rope_freqs_and_scales=None
        )
        from_state_out_cross_attn = attn_outputs_from_state_cross[0] # (B, H, N_states, D_head)

        state_out = torch.cat((state_out_self_attn, from_state_out_cross_attn), dim=-1)
        state_out = rearrange(state_out, 'b h n d -> b n (h d)') # (B, N_states, 2*H*D_head)

        state_out = self.to_state_out(state_out) # (B, N_states, D)

        # Learned EMA gating
        z = self.state_out_to_gate(state_out)
        learned_ema_decay = self.learned_ema_beta.sigmoid()

        # Update states
        # Ensure states from cache has batch dimension for consistency with z
        if states.ndim == 2: # If states is (N_states, D) (e.g., from init_state)
            states = states.unsqueeze(0).expand(batch_size, -1, -1) # (B, N_states, D)
        # states = states.expand_as(z) # This might cause issues if z has varying batch sizes, safer to expand explicitly

        return learned_ema_decay * z + (1 - learned_ema_decay) * states

    def forward(self, x):
        # This forward is not used directly, read/write are called by GPTNeoXLayer
        raise NotImplementedError

# ================ 新增模块结束 ================

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
                # Note: self.config.more_iterations is a custom config parameter for the recurrent block logic
                # Ensure it's handled gracefully if not present in default GPTNeoXConfig
                num_iterations = getattr(self.config, 'more_iterations', 0)
                std = std / math.sqrt(2.0 * self.config.num_hidden_layers * (num_iterations + 1))
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
        self.rotary_emb = GPTNeoXRotaryEmbedding(config=self.config) # Original HF RoPE

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
        # Added for BlockRecurrentTransformer, will be non-None when used by GPTNeoXModel
        # These are pre-computed RoPE freqs and scales from RotaryEmbedding (renamed to avoid conflict)
        # Note: 'rope_freqs_and_scales' is for BlockRecurrentTransformer's xPos RoPE.
        # It's different from the standard HF `position_embeddings` which are just (cos, sin).
        # We need to distinguish these two.
        rope_freqs_and_scales: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        read_from_state_containers: List['StateContainer'] = [], # New parameter for reading states
        current_block_hidden_states: Optional[torch.FloatTensor] = None, # New parameter for StateContainer.read(x)
    ):
        # Apply attention-specific projections and rope
        query, key, value, present = self._attn_projections_and_rope(
            hidden_states=hidden_states,
            position_ids=position_ids,
            layer_past=layer_past,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            rope_freqs_and_scales=rope_freqs_and_scales, # Pass through here
            cache_position=cache_position # Pass cache_position
        )

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        
        # Merge heads
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        
        # Read from state containers (new logic)
        # This will contain the aggregated output from state reading
        state_read_outputs = []
        if self.is_recurrent_layer and self.state_read_before_write and self.state_container is not None:
            # The current hidden_states for reading should be the *pre-normed* hidden_states
            # for the current block, before the attention and MLP
            if current_block_hidden_states is not None:
                state_read_outputs.append(self.state_container.read(current_block_hidden_states))
            else:
                logger.warning("`current_block_hidden_states` is None when `is_recurrent_layer` is True. State container read might be incorrect.")

        for read_state_container in read_from_state_containers:
            # Ensure these external state containers are also given the current block's hidden states
            if current_block_hidden_states is not None:
                state_read_outputs.append(read_state_container.read(current_block_hidden_states))
            else:
                logger.warning("`current_block_hidden_states` is None for external state container read. Check implementation.")

        # Concatenate state outputs with attention output
        if len(state_read_outputs) > 0:
            # Use safe_cat to handle potential None values if some state reads were skipped
            attn_output = safe_cat(attn_output, *state_read_outputs, dim=-1) # Concatenate along the feature dimension


        # Final dense projection
        attn_output = self.dense(attn_output)

        # Prepare outputs
        # Note: present is already updated within _attn_projections_and_rope for standard KV cache.
        # We need to also return the new_memories (for XL) and new_states (for recurrent)
        # from this attention block to be passed to the main GPTNeoXModel loop.
        
        # Original outputs
        outputs = (attn_output, present) 
        if output_attentions:
            outputs += (attn_weights,)
        
        # For recurrent block, also return new_memories (from this layer) and new_states (from this layer's state_container)
        # These will be filled in the GPTNeoXLayer or GPTNeoXModel.
        new_memories_from_layer = None
        new_states_from_layer = None

        # Pass current K, V from attention for potential memory management or state writing
        # This is (B, H, N_block, D_head)
        current_block_kv_for_memories = (key, value) # 'key' and 'value' here are the *processed* Q/K/V after RoPE, but before caching

        # Cache QKV values for BlockRecurrentTransformer's MemoryManager or StateContainer.write
        # These are what the MemoryManager expects as 'new_memories'
        # The 'key' and 'value' here are the post-RoPE, pre-concat (with layer_past) K/V.
        # The MemoryManager will handle concatenating with past memories internally.
        # For simplicity, we can pass (query_rot, key_rot) for xPos scaling, which is also returned by _attn_projections_and_rope
        # OR just use the key/value that comes out of _attn_projections_and_rope which already includes RoPE.
        # Given the previous context, 'key' and 'value' are already in the correct format (B, H, N, D_head) after RoPE.
        
        return (attn_output, present, attn_weights if output_attentions else None, current_block_kv_for_memories)

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
        rope_freqs_and_scales: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # New param
    ):
        # Compute QKV
        qkv = self.query_key_value(hidden_states)

        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # --- IMPORTANT CHANGE FOR BlockRecurrentTransformer RoPE ---
        # Prioritize rope_freqs_and_scales if provided (from the new RotaryEmbedding)
        if rope_freqs_and_scales is not None:
            freqs, scale = rope_freqs_and_scales
            query_rot = apply_rotary_pos_emb_custom(query_rot, freqs, scale)
            key_rot = apply_rotary_pos_emb_custom(key_rot, freqs, scale**-1) # xPos for keys
            # The apply_rotary_pos_emb_custom already returns the rotated part.
            # No need to call self.rotary_emb or apply_rotary_pos_emb here.
        elif position_embeddings is None:
            # Fallback to original GPTNeoX RoPE if no pre-computed freqs/scales
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            # HF's GPTNeoXRotaryEmbedding takes (value, position_ids)
            cos, sin = self.rotary_emb(value, position_ids) # This 'value' is actually the input hidden state to RoPE as per original HF GPTNeoX
            query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin) # This original apply_rotary_pos_emb is for the rotated part only
        else:
            # Use externally provided position_embeddings (cos, sin) as per HF standard
            cos, sin = position_embeddings
            query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin) # This original apply_rotary_pos_emb is for the rotated part only
        # --- END IMPORTANT CHANGE ---


        query = torch.cat((query_rot, query_pass), dim=-1) # query_rot is already the result of apply_rotary_pos_emb_adapted or apply_rotary_pos_emb
        key = torch.cat((key_rot, key_pass), dim=-1) # key_rot is already the result of apply_rotary_pos_emb_adapted or apply_rotary_pos_emb

        # Cache QKV values
        present = None
        if layer_past is not None:
            cache_kwargs = {
                "sin": position_embeddings[1] if position_embeddings is not None else None, # Use the sin from position_embeddings if available
                "cos": position_embeddings[0] if position_embeddings is not None else None, # Use the cos from position_embeddings if available
                "partial_rotation_size": self.rotary_ndims,
                "cache_position": cache_position,
            }
            key, value = layer_past.update(key, value, self.layer_idx, cache_kwargs)
            present = layer_past

        return query, key, value, present

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        # dynamically increase the causal mask with the key length, if needed.
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length, device=key.device)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        # SDPA requires q,k,v to be (B, H, N, D_head)
        # Here, query, key, value are already (B, H, N, D_head)

        attn_scores = torch.baddbmm(
            torch.empty(
                batch_size * num_attention_heads,
                query_length,
                key_length,
                dtype=query.dtype,
                device=key.device,
            ),
            query.reshape(-1, query_length, attn_head_size), # Reshape for baddbmm
            key.reshape(-1, key_length, attn_head_size).transpose(1, 2), # Reshape for baddbmm
            beta=1.0,
            alpha=self.norm_factor,
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:  # no matter the length, we just slice it
            # HF's attention_mask is typically (B, 1, 1, L) or (B, 1, L_q, L_kv) for broadcasting.
            # Ensure it aligns with current attn_scores shape (B, H, L_q, L_kv).
            # The causal_mask is (1, 1, L_q, L_kv)
            # Here we need to combine them.
            
            # This part of the logic needs to be careful:
            # original_attention_mask should be (B, 1, 1, max_seq_len) from GPTNeoXModel's _update_causal_mask
            # The current attention_mask here might already be a combined mask if coming from Model.forward,
            # or it's the raw input attention_mask.
            
            # Let's assume 'attention_mask' passed to _attn is the one from GPTNeoXModel's _update_causal_mask,
            # which is already a 4D mask (B, 1, L_q, L_kv) or (B, 1, L_q, L_kv) with padding
            # And it's already masked with large negative values for padding.
            
            # For simplicity, if attention_mask is provided and is a 4D mask (which it should be from _update_causal_mask)
            # then combine it with the causal mask.
            
            # Adjust attention_mask shape for current query_length and key_length if needed
            # The attention_mask from Model._update_causal_mask is already shaped for the full sequence.
            # Here, we only need to take the relevant part.
            # Assuming attention_mask is (B, 1, full_seq_len, full_seq_len_kv)
            # We need to slice it to (B, 1, query_length, key_length)
            
            # The `causal_mask` created above is a strict lower triangle.
            # The `attention_mask` from `_update_causal_mask` usually has `-inf` for masked tokens.
            # Adding them (`attn_scores + causal_mask`) effectively applies both.
            
            # If `attention_mask` is a boolean mask (1s and 0s)
            if attention_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~attention_mask[:, :, -query_length:, :key_length], mask_value)
            else: # If attention_mask already contains additive values like -inf
                attn_scores = attn_scores + attention_mask[:, :, -query_length:, :key_length]


        attn_weights = F.softmax(attn_scores, dim=-1)
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
        rope_freqs_and_scales: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # New param
        read_from_state_containers: List['StateContainer'] = [], # New param
        current_block_hidden_states: Optional[torch.FloatTensor] = None, # New param
    ):
        # Apply attention-specific projections and rope
        query, key, value, present = self._attn_projections_and_rope(
            hidden_states=hidden_states,
            position_ids=position_ids,
            layer_past=layer_past,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            rope_freqs_and_scales=rope_freqs_and_scales, # Pass through here
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
        attn_output = _flash_attention_forward(
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
        attn_output = attn_output.reshape(
            attn_output.shape[0], attn_output.shape[1], self.num_attention_heads * self.head_size
        )
        # Note: FlashAttention doesn't return raw attention weights. We return None if output_attentions is True.
        # This is a common pattern for FlashAttention integration in HF.

        # Read from state containers (new logic)
        state_read_outputs = []
        # is_recurrent_layer and state_read_before_write logic will be handled at GPTNeoXLayer level.
        # Here we just check for `read_from_state_containers`
        for read_state_container in read_from_state_containers:
            if current_block_hidden_states is not None:
                state_read_outputs.append(read_state_container.read(current_block_hidden_states))
            else:
                logger.warning("`current_block_hidden_states` is None for external state container read in FlashAttention2. Check implementation.")

        # Concatenate state outputs with attention output
        if len(state_read_outputs) > 0:
            attn_output = safe_cat(attn_output, *state_read_outputs, dim=-1)

        attn_output = self.dense(attn_output)

        # Pass current K, V from attention for potential memory management or state writing
        current_block_kv_for_memories = (key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)) # Convert back to (B, H, N, D_head)

        return (attn_output, present, None if output_attentions else None, current_block_kv_for_memories)
    
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
        rope_freqs_and_scales: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # New param
        read_from_state_containers: List['StateContainer'] = [], # New param
        current_block_hidden_states: Optional[torch.FloatTensor] = None, # New param
    ):
        if output_attentions or head_mask is not None:
            logger.warning_once(
                "`GPTNeoXSdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "`output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, but "
                "specifying the manual implementation will be required from Transformers version v5.0.0 onwards. "
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            # Fallback will now also pass the new parameters
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                rope_freqs_and_scales=rope_freqs_and_scales, # Pass through here
                read_from_state_containers=read_from_state_containers, # Pass through here
                current_block_hidden_states=current_block_hidden_states, # Pass through here
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
            rope_freqs_and_scales=rope_freqs_and_scales, # Pass through here
        )

        causal_mask = attention_mask
        if attention_mask is not None:
            # Slicing attention_mask to match key.shape[-2] should be handled in _update_causal_mask
            # Here, causal_mask should already be correct for SDPA
            pass 

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
        # `is_causal=True` is typically used when `attn_mask` is `None` and it's a standard causal attention.
        # If `attn_mask` is not None, `is_causal` should be `False` and the mask handles causality.
        is_causal = False
        if causal_mask is None and q_len > 1: # Only when no explicit mask and sequence length > 1 (for self-attention)
            is_causal = True # Rely on SDPA's internal causal mask

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=causal_mask, # SDPA uses attn_mask directly
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape outputs
        attn_output = attn_output.transpose(1, 2).contiguous() # (B, N_block, H, D_head) -> (B, N_block, H*D_head)
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        # Read from state containers (new logic)
        state_read_outputs = []
        for read_state_container in read_from_state_containers:
            if current_block_hidden_states is not None:
                state_read_outputs.append(read_state_container.read(current_block_hidden_states))
            else:
                logger.warning("`current_block_hidden_states` is None for external state container read in SDPA. Check implementation.")

        # Concatenate state outputs with attention output
        if len(state_read_outputs) > 0:
            attn_output = safe_cat(attn_output, *state_read_outputs, dim=-1)

        attn_output = self.dense(attn_output)

        # Pass current K, V from attention for potential memory management or state writing
        current_block_kv_for_memories = (key, value) # These are (B, H, N_block, D_head)

        # SDPA does not return attention weights. Return None for consistency.
        return attn_output, present, None, current_block_kv_for_memories
    
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


# 将 GPT_NEOX_ATTENTION_CLASSES 中的键 'eager'、'flash_attention_2'、'sdpa' 映射到新的 GPTNeoXAttention 类
# 这个字典在 GPTNeoXLayer 中使用，因此需要更新其引用。

# 新增的 GPTNeoXRecurrentAttention 类来作为基础注意力模块，它将包含 StateContainer 逻辑
class GPTNeoXRecurrentAttention(nn.Module):
    def __init__(self, config, layer_idx=None, num_state_vectors=0, 
                 num_external_state_reads=0, state_read_before_write=True):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.dense._is_attention_output = True # 标记为注意力输出层
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.layer_idx = layer_idx
        
        # 新增：状态容器
        self.is_recurrent_layer = num_state_vectors > 0
        self.state_read_before_write = state_read_before_write # Keep this for layer-level control
        self.num_external_state_reads = num_external_state_reads

        self.state_container = None
        if self.is_recurrent_layer:
            # Pass config to StateContainer for proper attention module initialization
            self.state_container = StateContainer(
                config.hidden_size,
                num_state_vectors=num_state_vectors,
                dim_head=self.head_size,
                heads=self.num_attention_heads,
                use_flash_attn=config._attn_implementation == "flash_attention_2",
                config=config # Pass config here
            )
        
        # 使用原始的HF Attention类，但需要确保它能够接收 `rope_freqs_and_scales` 和 `read_from_state_containers`
        # 我们可以直接使用 GPTNeoXAttention 作为基类，因为我们已经修改了它的 forward 方法。
        attn_class = GPT_NEOX_ATTENTION_CLASSES[config._attn_implementation]
        # 注意：此处不再是简单的 `attn = GPTNeoXAttention(...)`，
        # 而是为了实现 BlockRecurrentTransformer 的逻辑，需要一个能够管理状态和记忆的Attention模块。
        # 上面的 `GPTNeoXAttention` 的 `forward` 已经修改以支持 `rope_freqs_and_scales` 和 `read_from_state_containers`
        # 因此，这里直接实例化它即可。
        self.base_attention = attn_class(config, layer_idx=layer_idx)
        # 将一些属性从 base_attention 复制过来，或者在 forward 中直接调用 base_attention 的方法
        # 这样可以避免重复的QKV投影层，所有QKV投影都通过 base_attention 来完成。

    # forward method for GPTNeoXRecurrentAttention
    def forward(
        self, 
        hidden_states: torch.FloatTensor, # This is the hidden_states *before* normalization of this block
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Cache] = None, # This refers to the standard HF KV cache
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        # New params from GPTNeoXModel for BlockRecurrentTransformer
        rope_freqs_and_scales: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        xl_memories: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # (k,v) for XL memories
        read_from_state_containers: List['StateContainer'] = [], # List of StateContainer objects to read from
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns: (attn_output, current_block_memories_for_xl, new_states_from_this_layer)
        """
        # Save the current block's input hidden states for state container reads
        current_block_hidden_states_for_read = hidden_states 

        # Prepare layer_past for the base_attention if XL memories are provided
        # XL memories are (k,v) tuples for each head, (B, H, N_mem, D_head)
        # The base_attention's `layer_past` is the HF standard KV cache.
        # If `xl_memories` is provided, we should use it as the `layer_past` for the base attention.
        # Note: HF's `layer_past` for Attention modules is a `Cache` object, not a simple tuple.
        # We need to correctly wrap `xl_memories` into a Cache if that's the intended behavior.
        # Or, more simply, we will concatenate `xl_memories` with the current block's keys/values *before* passing to base_attention.

        # The `query`, `key`, `value` are computed inside `base_attention`.
        # `base_attention` will return the current block's processed (Q, K, V) which are then used by MemoryManager.
        
        # Read from state containers if this is a recurrent layer AND state_read_before_write
        # or if external state containers are explicitly passed to read from.
        actual_read_state_containers = []
        if self.is_recurrent_layer and self.state_container and self.state_read_before_write:
            actual_read_state_containers.append(self.state_container)
        actual_read_state_containers.extend(read_from_state_containers) # Add external ones


        # Call the base attention mechanism
        # It now returns (attn_output, present_kv_cache, attn_weights, current_block_kv_for_memories)
        attn_outputs_tuple = self.base_attention(
            hidden_states=hidden_states, # This is the input to QKV projection, not yet normalized
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            layer_past=layer_past, # Standard HF KV cache, if used for generation
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            rope_freqs_and_scales=rope_freqs_and_scales,
            read_from_state_containers=actual_read_state_containers, # Pass list of containers
            current_block_hidden_states=current_block_hidden_states_for_read, # Pass raw hidden states for state read
        )
        
        attn_output = attn_outputs_tuple[0] # (B, N, D_model) - includes potential state read concat
        present_kv_cache = attn_outputs_tuple[1] # Updated standard HF KV cache
        attn_weights = attn_outputs_tuple[2] # Attention weights, or None
        # `current_block_kv_for_memories` are the (K, V) from the current block, after RoPE but *before* concatenation with `layer_past`.
        # This is what MemoryManager and StateContainer.write needs.
        current_block_kv_for_memories = attn_outputs_tuple[3] # (K, V) tuple, (B, H, N_block, D_head)

        # State writing
        new_states_from_this_layer = None
        if self.is_recurrent_layer and self.state_container:
            new_states_from_this_layer = self.state_container.write(memories=current_block_kv_for_memories)

        # Return the relevant outputs for the BlockRecurrentTransformer logic
        # Output format: (processed_hidden_states, current_block_memories_for_xl, new_states_from_this_layer)
        # where current_block_memories_for_xl is a (K, V) tuple from this block/layer.
        return (attn_output, present_kv_cache, attn_weights, current_block_kv_for_memories, new_states_from_this_layer)

# 将 GPT_NEOX_ATTENTION_CLASSES 中的键 'eager'、'flash_attention_2'、'sdpa' 映射到新的 GPTNeoXAttention 类
# 这个字典在 GPTNeoXLayer 中使用，因此需要更新其引用。

# 新增的 GPTNeoXRecurrentAttention 类来作为基础注意力模块，它将包含 StateContainer 逻辑
class GPTNeoXRecurrentAttention(nn.Module):
    def __init__(self, config, layer_idx=None, num_state_vectors=0, 
                 num_external_state_reads=0, state_read_before_write=True):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.dense._is_attention_output = True # 标记为注意力输出层
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.layer_idx = layer_idx
        
        # 新增：状态容器
        self.is_recurrent_layer = num_state_vectors > 0
        self.state_read_before_write = state_read_before_write # Keep this for layer-level control
        self.num_external_state_reads = num_external_state_reads

        self.state_container = None
        if self.is_recurrent_layer:
            # Pass config to StateContainer for proper attention module initialization
            self.state_container = StateContainer(
                config.hidden_size,
                num_state_vectors=num_state_vectors,
                dim_head=self.head_size,
                heads=self.num_attention_heads,
                use_flash_attn=config._attn_implementation == "flash_attention_2",
                config=config # Pass config here
            )
        
        # 使用原始的HF Attention类，但需要确保它能够接收 `rope_freqs_and_scales` 和 `read_from_state_containers`
        # 我们可以直接使用 GPTNeoXAttention 作为基类，因为我们已经修改了它的 forward 方法。
        attn_class = GPT_NEOX_ATTENTION_CLASSES[config._attn_implementation]
        # 注意：此处不再是简单的 `attn = GPTNeoXAttention(...)`，
        # 而是为了实现 BlockRecurrentTransformer 的逻辑，需要一个能够管理状态和记忆的Attention模块。
        # 上面的 `GPTNeoXAttention` 的 `forward` 已经修改以支持 `rope_freqs_and_scales` 和 `read_from_state_containers`
        # 因此，这里直接实例化它即可。
        self.base_attention = attn_class(config, layer_idx=layer_idx)
        # 将一些属性从 base_attention 复制过来，或者在 forward 中直接调用 base_attention 的方法
        # 这样可以避免重复的QKV投影层，所有QKV投影都通过 base_attention 来完成。

    # forward method for GPTNeoXRecurrentAttention
    def forward(
        self, 
        hidden_states: torch.FloatTensor, # This is the hidden_states *before* normalization of this block
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Cache] = None, # This refers to the standard HF KV cache
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        # New params from GPTNeoXModel for BlockRecurrentTransformer
        rope_freqs_and_scales: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        xl_memories: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # (k,v) for XL memories
        read_from_state_containers: List['StateContainer'] = [], # List of StateContainer objects to read from
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns: (attn_output, current_block_memories_for_xl, new_states_from_this_layer)
        """
        # Save the current block's input hidden states for state container reads
        current_block_hidden_states_for_read = hidden_states 

        # Prepare layer_past for the base_attention if XL memories are provided
        # XL memories are (k,v) tuples for each head, (B, H, N_mem, D_head)
        # The base_attention's `layer_past` is the HF standard KV cache.
        # If `xl_memories` is provided, we should use it as the `layer_past` for the base attention.
        # Note: HF's `layer_past` for Attention modules is a `Cache` object, not a simple tuple.
        # We need to correctly wrap `xl_memories` into a Cache if that's the intended behavior.
        # Or, more simply, we will concatenate `xl_memories` with the current block's keys/values *before* passing to base_attention.

        # The `query`, `key`, `value` are computed inside `base_attention`.
        # `base_attention` will return the current block's processed (Q, K, V) which are then used by MemoryManager.
        
        # Read from state containers if this is a recurrent layer AND state_read_before_write
        # or if external state containers are explicitly passed to read from.
        actual_read_state_containers = []
        if self.is_recurrent_layer and self.state_container and self.state_read_before_write:
            actual_read_state_containers.append(self.state_container)
        actual_read_state_containers.extend(read_from_state_containers) # Add external ones


        # Call the base attention mechanism
        # It now returns (attn_output, present_kv_cache, attn_weights, current_block_kv_for_memories)
        attn_outputs_tuple = self.base_attention(
            hidden_states=hidden_states, # This is the input to QKV projection, not yet normalized
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            layer_past=layer_past, # Standard HF KV cache, if used for generation
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            rope_freqs_and_scales=rope_freqs_and_scales,
            read_from_state_containers=actual_read_state_containers, # Pass list of containers
            current_block_hidden_states=current_block_hidden_states_for_read, # Pass raw hidden states for state read
        )
        
        attn_output = attn_outputs_tuple[0] # (B, N, D_model) - includes potential state read concat
        present_kv_cache = attn_outputs_tuple[1] # Updated standard HF KV cache
        attn_weights = attn_outputs_tuple[2] # Attention weights, or None
        # `current_block_kv_for_memories` are the (K, V) from the current block, after RoPE but *before* concatenation with `layer_past`.
        # This is what MemoryManager and StateContainer.write needs.
        current_block_kv_for_memories = attn_outputs_tuple[3] # (K, V) tuple, (B, H, N_block, D_head)

        # State writing
        new_states_from_this_layer = None
        if self.is_recurrent_layer and self.state_container:
            new_states_from_this_layer = self.state_container.write(memories=current_block_kv_for_memories)

        # Return the relevant outputs for the BlockRecurrentTransformer logic
        # Output format: (processed_hidden_states, current_block_memories_for_xl, new_states_from_this_layer)
        # where current_block_memories_for_xl is a (K, V) tuple from this block/layer.
        return (attn_output, present_kv_cache, attn_weights, current_block_kv_for_memories, new_states_from_this_layer)

class GPTNeoXRecurrentLayer(nn.Module):
    def __init__(self, config, layer_idx, recurrent_layers: Tuple[int] = (),
                 read_recurrent_layers: Tuple[int] = (), num_state_vectors: int = 0):
        super().__init__()
        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Determine if this layer is a recurrent layer (writes states)
        self.is_recurrent_layer = layer_idx in recurrent_layers
        
        # Determine how many *external* state containers this layer needs to read from
        # Note: self.is_recurrent_layer handling its *own* state read is implicitly controlled by
        # `state_read_before_write` within GPTNeoXRecurrentAttention
        self.num_external_state_reads = sum(1 for read_layer_idx in read_recurrent_layers if read_layer_idx == layer_idx)

        # Initialize the attention block with recurrent capabilities
        # This will be an instance of the modified GPTNeoXAttention or its Flash/SDPA variants
        attn_class = GPT_NEOX_ATTENTION_CLASSES[config._attn_implementation]
        # We need to pass the config object to the attention module so it can correctly initialize its StateContainer.
        # The GPTNeoXRecurrentAttention class itself is now essentially the "AttentionBlock"
        # from the original BlockRecurrentTransformer, adapted to the HF format.
        self.attention = attn_class( # This will be one of GPTNeoXAttention, FlashAttention2, SDPA, customized for recurrent
            config,
            layer_idx=layer_idx,
        )
        # Manually set recurrent properties on the attention module if it's a recurrent layer
        # This is a bit of a hack, ideally these properties would be passed during __init__ of GPTNeoXAttention itself
        # but for minimal modification, we can set them post-init.
        self.attention.is_recurrent_layer = self.is_recurrent_layer
        self.attention.state_read_before_write = getattr(config, 'state_read_before_write', True) # From config or default
        self.attention.state_container = None
        if self.is_recurrent_layer:
            self.attention.state_container = StateContainer(
                config.hidden_size,
                num_state_vectors=num_state_vectors,
                dim_head=config.hidden_size // config.num_attention_heads,
                heads=config.num_attention_heads,
                qk_rmsnorm=False, # Original HF GPTNeoX doesn't use qk_rmsnorm by default
                use_flash_attn=config._attn_implementation == "flash_attention_2",
                config=config # Pass config here
            )
        
        self.mlp = GPTNeoXMLP(config)
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
    
    def forward(
        self, 
        hidden_states: torch.FloatTensor, # Input to this layer
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Cache] = None, # HF standard KV cache for this layer
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # RoPE from GPTNeoXModel
        # New params from GPTNeoXModel for BlockRecurrentTransformer
        rope_freqs_and_scales: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # RoPE from new RotaryEmbedding
        xl_memories: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # XL memories from previous block for this layer
        read_from_state_containers: List['StateContainer'] = [], # External state containers to read from
    ) -> Tuple[torch.Tensor, Optional[Cache], Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Returns: (output_hidden_states, present_kv_cache, attentions_weights, new_memories_for_xl, new_states_from_this_layer)
        """
        # Save original hidden_states for parallel residual connection and for state container read.
        residual = hidden_states
        
        # Pre-normalization for attention
        normed_hidden_states_for_attn = self.input_layernorm(hidden_states)

        # Attention layer (now GPTNeoXRecurrentAttention or its Flash/SDPA variants)
        # The attention module now needs the *current block's raw hidden states* for reading.
        # This `normed_hidden_states_for_attn` will serve as the input `x` to `StateContainer.read()`.
        
        # The `layer_past` here is for the standard HF KV cache,
        # `xl_memories` is for Transformer-XL, which might be concatenated to keys/values *before* attention calculation.
        # We need to pass `xl_memories` into the `self.attention` call if it handles the concatenation.
        # Let's assume `self.attention` (which is GPTNeoXAttention or its variants) will handle XL memories.
        # Modify `GPTNeoXAttention._attn_projections_and_rope` or `_attn` to merge `xl_memories` with `key` and `value`.
        # For now, we pass `xl_memories` as `layer_past` to mimic the original `layer_past` behavior.
        # This requires `GPTNeoXAttention` to know how to handle it.
        # OR, we explicitly concatenate `xl_memories` here before calling `self.attention`.
        
        # Simpler approach:
        # Pass xl_memories to the attention module. Inside the attention module's _attn_projections_and_rope,
        # we can modify the `key` and `value` by prepending `xl_memories` to them.
        
        # The `layer_past` parameter in HF attention modules is typically used for incremental decoding
        # and manages the KV cache across *decode steps*.
        # `xl_memories` in BlockRecurrentTransformer are *full blocks* of past KV pairs.
        # We should concatenate `xl_memories` with the *current block's* K,V, then pass the result to the attention.
        
        # Let's adjust `GPTNeoXAttention._attn` (or its variants) to incorporate `xl_memories`.
        # This will require `_attn` to accept `xl_memories` as input.
        # Or, even better, modify `GPTNeoXAttention.forward` to take `xl_memories` and handle it there.
        # For now, let's just assume `xl_memories` is used to extend the `layer_past` being sent.
        # This is crucial: the `layer_past` in HF models is a `Cache` object.
        # We need to convert `xl_memories` into a compatible format or integrate it differently.
        
        # Instead of modifying `layer_past` directly, let's pass `xl_memories` as a new parameter
        # to the attention module, and handle the concatenation of current K,V with XL memories *inside* the attention module.
        # This will require modifying `GPTNeoXAttention.forward` and `_attn`.

        # Okay, let's stick to the convention: `layer_past` is for current decoding, `xl_memories` are separate.
        # `GPTNeoXAttention.forward` will receive `xl_memories` as a new param and merge them.

        attn_outputs = self.attention(
            hidden_states=normed_hidden_states_for_attn, # Input after first LayerNorm
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            layer_past=layer_past, # Standard HF KV cache
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            rope_freqs_and_scales=rope_freqs_and_scales, # From RotaryEmbedding (our custom one)
            xl_memories=xl_memories, # Pass XL memories to attention module
            read_from_state_containers=read_from_state_containers, # Pass external state containers
            current_block_hidden_states=hidden_states # Raw hidden states for state reads (pre-normalization)
        )
        # attn_outputs is now a 4-tuple: (attn_output, present_kv_cache, attn_weights, current_block_kv_for_memories)
        attn_output = attn_outputs[0] # The main attention output (B, N_block, D_model)
        present_kv_cache = attn_outputs[1] # Updated HF KV cache for decoding
        attn_weights = attn_outputs[2] # Attention weights
        current_block_kv_for_memories = attn_outputs[3] # (K,V) from this block, (B, H, N_block, D_head)

        attn_output = self.post_attention_dropout(attn_output)
        
        # Apply residual connection after attention
        if self.use_parallel_residual:
            mlp_input = self.post_attention_layernorm(hidden_states) # original hidden_states for parallel residual
            mlp_output = self.mlp(mlp_input)
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output + residual # Add all three components
        else:
            attn_output_with_residual = attn_output + residual
            mlp_input = self.post_attention_layernorm(attn_output_with_residual)
            mlp_output = self.mlp(mlp_input)
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output_with_residual # Add mlp output to attn_output_with_residual

        # State writing (if this is a recurrent layer)
        new_states_from_this_layer = None
        if self.is_recurrent_layer and self.attention.state_container:
            new_states_from_this_layer = self.attention.state_container.write(memories=current_block_kv_for_memories)


        # Prepare outputs for GPTNeoXModel
        # This layer needs to return:
        # 1. The final hidden_states after attention and MLP
        # 2. The updated standard HF KV cache (`present_kv_cache`)
        # 3. Attention weights (if requested)
        # 4. The *new XL memories* from this layer's attention (raw K,V for MemoryManager)
        # 5. The *new recurrent states* from this layer's state container
        
        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_kv_cache,) # Standard HF KV cache
        if output_attentions:
            outputs += (attn_weights,)
            
        # Add XL memories and new recurrent states
        # These are specific to BlockRecurrentTransformer logic and will be processed by GPTNeoXModel
        outputs += (current_block_kv_for_memories,) # This is the new memory for the MemoryManager
        outputs += (new_states_from_this_layer,) # This is the new state for the next block's read


        return outputs

@add_start_docstrings(
    "The bare GPTNeoX Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_NEOX_START_DOCSTRING,
)
class GPTNeoXModel(GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # 新增：循环层和记忆管理相关配置
        # 从 config 中获取这些参数，如果没有则使用默认值
        self.recurrent_layers = default(getattr(config, 'recurrent_layers', None), ())
        self.read_recurrent_layers = default(getattr(config, 'read_recurrent_layers', None), ())
        self.num_state_vectors = default(getattr(config, 'num_state_vectors', None), 0)
        self.block_width = default(getattr(config, 'block_width', None), config.max_position_embeddings) # 默认 block_width 为 max_position_embeddings
        
        # 验证循环层配置
        assert all([0 < layer <= config.num_hidden_layers for layer in self.recurrent_layers]), \
            f'recurrent layers must range from 1 to the depth {config.num_hidden_layers}'
        # assert all_unique(self.recurrent_layers), 'recurrent layers must be all unique. no duplicate layers' # 移除，因为HF层可能不是从1开始的，且允许重复（虽然通常不这么做）

        assert all([read_layer <= write_layer for read_layer, write_layer in zip(self.read_recurrent_layers, self.recurrent_layers)]), \
            'the recurrent read layer must be always less than or equal to the write layer'
        assert all([0 < layer <= config.num_hidden_layers for layer in self.read_recurrent_layers])
        assert len(self.read_recurrent_layers) == len(self.recurrent_layers), \
            'Length of read_recurrent_layers must match recurrent_layers'

        self.write_to_read_map = {write_layer: read_layer for write_layer, read_layer in zip(self.recurrent_layers, self.read_recurrent_layers)}
        self.read_state_router = defaultdict(list) # 存储每个层需要读取的状态容器

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        
        # 修改：使用新的 GPTNeoXRecurrentLayer
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            layer_idx = i + 1 # 约定层索引从1开始
            is_recurrent_layer = layer_idx in self.recurrent_layers
            
            # num_external_state_reads 统计当前层作为 `read_recurrent_layers` 出现的次数
            num_external_state_reads = sum([int(layer_idx == read_layer) for read_layer in self.read_recurrent_layers])

            # qk_rmsnorm 可以在 config 中设置，或者根据是否为循环层决定
            qk_rmsnorm = getattr(config, 'all_layers_qk_rmsnorm', False) or is_recurrent_layer

            layer = GPTNeoXRecurrentLayer(
                config, 
                layer_idx=layer_idx,
                recurrent_layers=self.recurrent_layers,
                read_recurrent_layers=self.read_recurrent_layers,
                num_state_vectors=self.num_state_vectors if is_recurrent_layer else 0
            )
            self.layers.append(layer)
            
            if is_recurrent_layer:
                # 记录哪些层需要读取当前循环层的状态容器
                read_layer_for_this_write = self.write_to_read_map[layer_idx]
                if layer.attention.state_container is not None: # 确保 StateContainer 被创建
                    self.read_state_router[read_layer_for_this_write].append(layer.attention.state_container)

        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 旋转位置编码，使用我们自定义的 `RotaryEmbedding`
        # `width` 参数决定了 RoPE 缓存的最大序列长度，这里设置为 `block_width` 的某个倍数，以覆盖当前块和记忆。
        # 这里使用 (2 * block_width) + (compressed_mem_factor * block_width) 作为一个粗略的估计，
        # 实际需要根据 MemoryManager 的 `mem_lengths` 和 `compress_factors` 来精确计算最大有效宽度
        max_rope_width = self.block_width # 最小是当前块的长度
        if hasattr(config, 'use_compressed_mem') and config.use_compressed_mem:
            # 如果使用压缩记忆，RoPE 需要覆盖最长的记忆 + 当前块长度
            max_mem_length = max(cast_tuple(getattr(config, 'mem_lengths', self.block_width)))
            max_rope_width = max_mem_length + self.block_width
        
        self.rotary_emb_custom = RotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads, 
            width=max_rope_width, # 确保覆盖最大可能序列长度
            scale_base=getattr(config, 'xpos_scale_base', 512),
            theta=config.rotary_emb_base
        )
        
        # 新增：记忆管理器
        self.mem_manager = MemoryManager(
            dim=config.hidden_size // config.num_attention_heads, # dim 是 head_size
            layers=config.num_hidden_layers, # 每层都有自己的记忆链
            mem_lengths=cast_tuple(getattr(config, 'mem_lengths', self.block_width)),
            compress_factors=cast_tuple(getattr(config, 'compress_factors', 1))
        )
        
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
        # 新增：Block Recurrent Transformer 特有参数
        xl_memories: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, # 列表，每个元素是 (k, v) for a layer
        states: Optional[List[torch.Tensor]] = None, # 列表，每个元素是该层的状态向量
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

        # 梯度检查点与 use_cache 的兼容性处理
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        # 处理 HF 内部的 KV 缓存兼容性，转换为 `DynamicCache`
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

        batch_size, seq_length, hidden_dim = inputs_embeds.shape
        device = inputs_embeds.device

        # 处理 position_ids
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1) # (B, L_current_block)

        # `_update_causal_mask` 生成 attention_mask，此 mask 包含了因果性和 padding 信息
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # Prepare head mask
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        hidden_states = self.emb_dropout(inputs_embeds)

        # 创建用于旋转位置嵌入的频率和尺度（来自我们自定义的 RotaryEmbedding）
        # 这里的 `seq_length` 是当前输入块的长度。
        # 但 `RotaryEmbedding` 的 `width` 应该基于 `max_rope_width` 来预计算最大长度的 freqs 和 scales。
        # 然后，这里只取当前块所需的切片。
        # 这也意味着 `RotaryEmbedding` 需要知道实际的 `sequence_length` 来计算 freqs 和 scales。
        # 这里，我们让 `RotaryEmbedding` 基于 `max_rope_width` 预计算，然后在这里进行切片。
        
        # 获取用于当前块的 RoPE freqs 和 scales
        # 注意：这里的 `seq_length` 指的是当前处理的 **整个** QKV 序列长度，
        # 也就是当前 block 的长度 + XL memories 的长度。
        # 我们需要在 `self.rotary_emb_custom` 的 forward 中传递这个长度，以便正确缓存或生成 freqs。
        # 由于我们是分块处理，这里的 `seq_length` 是当前 `input_block` 的长度。
        # 然而，注意力机制会看到 `input_block_length + xl_memories_length` 的总长度。
        # 因此，我们需要确保 `rotary_emb_custom` 生成的 freqs 足够长。
        # `self.rotary_emb_custom.width` 已经设置为最大的可能长度，所以这里只需要传入当前注意力计算的有效长度。
        # 这个长度在 `GPTNeoXAttention` 内部计算，所以这里先提供基础的 `seq_length`。
        
        # 注意：这里的 `position_embeddings` 是 HF 原始的 RoPE 输出，而 `rope_freqs_and_scales` 是我们自定义的。
        # 我们在 `GPTNeoXAttention` 中优先使用 `rope_freqs_and_scales`。
        
        # 为每个块分配 XL memories 和 states
        # xl_memories 和 states 都是列表，每个元素对应一个 Transformer 层
        xl_memories = [None] * self.config.num_hidden_layers if xl_memories is None else xl_memories
        states = [None] * self.config.num_hidden_layers if states is None else states

        next_decoder_cache = None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_xl_memories = [None] * self.config.num_hidden_layers # 存储每一层输出的 XL 记忆
        all_next_states = [None] * self.config.num_hidden_layers # 存储每一层输出的新状态

        # 分块处理输入
        input_blocks = hidden_states.split(self.block_width, dim=1) # (B, block_width, D)

        final_hidden_states_list = [] # 收集所有块的最终隐状态

        for block_idx, input_block in enumerate(input_blocks):
            current_block_length = input_block.shape[1]

            # 计算当前块所需的 RoPE 频率和尺度
            # 注意：这里的 `seq_len` 是当前块的长度，但是 RoPE 需要覆盖当前块 + XL 记忆的长度。
            # 这里的 `self.rotary_emb_custom` 的 `width` 应该设置为最大可能序列长度（`max_seq_len_with_memories`）。
            # `current_block_total_len` 应该是 `current_block_length + sum(mem_lengths) for this layer`
            # 为了简化，这里我们让 `RotaryEmbedding` (自定义的) 基于 `block_width` 计算，
            # 并在 attention 内部根据实际的 query_length 和 key_length 进行切片。
            
            # 由于 `GPTNeoXAttention` 现在负责 RoPE 的应用，我们只需给它传入总的 sequence length
            # 以及从 `self.rotary_emb_custom` 获取的 freqs 和 scales。
            # `self.rotary_emb_custom` 的 `forward` 方法会根据其内部 `self.width`（预设的最大值）生成。
            
            # 这里是 GPTNeoXModel 层面，我们计算整个块的 position_ids
            # 但 `GPTNeoXAttention` 内部会根据 key_length 和 query_length 来切片 RoPE
            # current_block_pos_ids 应该基于当前块在整个序列中的绝对位置
            # current_block_cache_position 也是同理
            current_block_start_idx = block_idx * self.block_width
            current_block_position_ids = torch.arange(
                current_block_start_idx, 
                current_block_start_idx + current_block_length, 
                dtype=torch.long, 
                device=device
            ).unsqueeze(0).expand(batch_size, -1)

            # 获取全局 RoPE 频率和尺度 (一次性生成，然后切片使用)
            # `self.rotary_emb_custom` 的 width 应该足够大，以覆盖 `max_seq_len + max_mem_length`
            # 这样每次调用 `forward()` 才会更新缓存。
            # `self.rotary_emb_custom.forward()` 仅在需要时重新计算。
            # 它返回的是用于整个 `self.width` 的 freqs 和 scales。
            # 实际传递给 `Attention` 的是这些 freqs 和 scales 的切片，基于 `attn.forward` 内部的 `seq_len`。
            rope_freqs, xpos_scales = self.rotary_emb_custom.forward() # 假设这里返回的 freqs 和 scales 足够长
            
            # 为当前块设置状态容器的读取状态
            for layer_idx, layer_module in enumerate(self.layers):
                if layer_idx + 1 in self.recurrent_layers:
                    # 获取该层对应的循环状态
                    layer_states = states[layer_idx] if layer_idx < len(states) else None
                    if layer_module.attention.state_container is not None:
                        layer_module.attention.state_container.set_next_read_state(layer_states)

            # 遍历 Transformer 层
            current_hidden_states_in_block = input_block
            current_block_xl_memories = [None] * self.config.num_hidden_layers # 当前块新生成的 XL 记忆
            current_block_new_states = [None] * self.config.num_hidden_layers # 当前块新生成的循环状态

            layer_past_current_block = past_key_values # HF Standard KV cache, passed through layers

            for i, layer_module in enumerate(self.layers):
                layer_num = i + 1
                
                # 为当前层准备 external_read_state_containers
                read_state_containers_for_this_layer = self.read_state_router[layer_num]

                # 注意力层
                # outputs: (processed_hidden_states, present_kv_cache, attentions_weights, current_block_kv_for_memories, new_states_from_this_layer)
                # `current_hidden_states_in_block` 是该层的输入
                # `hidden_states` 是整个块的原始输入，用于 `StateContainer.read(x)`
                layer_outputs = layer_module(
                    hidden_states=current_hidden_states_in_block,
                    attention_mask=causal_mask, # 整个序列的掩码
                    position_ids=current_block_position_ids, # 当前块的 position_ids
                    head_mask=head_mask[i],
                    layer_past=layer_past_current_block, # Standard HF KV cache for incremental decoding
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cache_position=cache_position[block_idx*current_block_length : (block_idx+1)*current_block_length], # 仅传递当前块的缓存位置
                    position_embeddings=None, # 由 `rope_freqs_and_scales` 代替
                    rope_freqs_and_scales=(rope_freqs, xpos_scales), # 传递预计算的 RoPE freqs 和 scales
                    xl_memories=xl_memories[i], # 传递给当前层的 XL 记忆
                    read_from_state_containers=read_state_containers_for_this_layer, # 传递外部状态容器
                )

                current_hidden_states_in_block = layer_outputs[0] # 该层的输出隐状态
                layer_past_current_block = layer_outputs[1] # 更新的标准 HF KV cache
                attn_weights_from_layer = layer_outputs[2] # 注意力权重
                kv_for_memories_from_layer = layer_outputs[3] # 当前块的 K,V 用于 XL 记忆更新
                new_states_from_layer = layer_outputs[4] # 该层生成的新状态


                if output_hidden_states:
                    if len(all_hidden_states) <= i:
                        all_hidden_states += (current_hidden_states_in_block,)
                    else:
                        all_hidden_states = all_hidden_states[:i] + (torch.cat((all_hidden_states[i], current_hidden_states_in_block), dim=1),) + all_hidden_states[i+1:]

                if output_attentions:
                    if attn_weights_from_layer is not None: # FlashAttention may return None
                        if len(all_attentions) <= i:
                            all_attentions += (attn_weights_from_layer,)
                        else:
                            all_attentions = all_attentions[:i] + (torch.cat((all_attentions[i], attn_weights_from_layer), dim=1),) + all_attentions[i+1:]
                    else: # If FlashAttention returns None, append None for consistency
                        if len(all_attentions) <= i:
                            all_attentions += (None,)

                # 收集当前块生成的 XL 记忆和状态
                current_block_xl_memories[i] = kv_for_memories_from_layer
                if new_states_from_layer is not None:
                    current_block_new_states[i] = new_states_from_layer

            final_hidden_states_list.append(current_hidden_states_in_block)

            # 更新下一迭代的 XL memories 和 states
            # 仅当当前块的长度等于 block_width 时才更新（或者在最后一个块，如果它小于 block_width）
            if current_block_length == self.block_width or (block_idx == len(input_blocks) - 1):
                # 调用记忆管理器更新 XL 记忆
                # current_block_xl_memories 是一个列表，每个元素是 (k, v) for a layer
                xl_memories = self.mem_manager(xl_memories, current_block_xl_memories)
                
                # 更新状态
                states = current_block_new_states
            
            # 如果是最后一个块，且请求了缓存，则 next_decoder_cache 应该是最终的 past_key_values
            if block_idx == len(input_blocks) - 1 and use_cache:
                next_decoder_cache = layer_past_current_block # 最终的标准 HF KV cache


        # 合并所有块的最终隐状态
        last_hidden_state = torch.cat(final_hidden_states_list, dim=1)
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # Add last hidden state to all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (last_hidden_state,)

        # 确保返回的记忆和状态是 None 或者 detach 后的张量列表
        returned_xl_memories = list(map(lambda x: (x[0].detach(), x[1].detach()) if x is not None else None, xl_memories))
        returned_states = list(map(torch.detach, states)) if all(s is not None for s in states) else None


        if return_legacy_cache:
            next_decoder_cache = next_decoder_cache.to_legacy_cache() if next_decoder_cache else None

        if not return_dict:
            # 调整返回顺序以符合 BaseModelOutputWithPast 的结构
            # (last_hidden_state, past_key_values, hidden_states, attentions)
            output_tuple = (last_hidden_state, next_decoder_cache, all_hidden_states, all_attentions)
            # 添加返回的 XL memories 和 states
            # 这些是 BlockRecurrentTransformer 特有的输出，需要额外处理
            # 暂时将它们作为额外的元素附加到元组中，但需要确保消费者能够正确解析
            return tuple(v for v in output_tuple if v is not None) + (returned_xl_memories, returned_states)


        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            # 将 BlockRecurrentTransformer 特有的输出添加到 ModelOutput 中
            # 可能需要扩展 BaseModelOutputWithPast 或者定义新的输出类
            # 临时方案：在 BaseModelOutputWithPast 中添加自定义属性
            # 如果不修改 BaseModelOutputWithPast，则只能通过 return_dict=False 获取
            # 为了兼容性，这里暂时不直接添加到 BaseModelOutputWithPast
        ), returned_xl_memories, returned_states # 返回额外的记忆和状态

    # `_update_causal_mask` 和 `_prepare_4d_causal_attention_mask_with_cache_position` 保持不变
    # ... (这两个方法与前面给出的一致，此处省略以节省篇幅) ...
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

        self.gpt_neox = GPTNeoXModel(config) # GPTNeoXModel 已经包含了 Block Recurrent Transformer 逻辑
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()        

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings
        
    # 移除 prepare_gpt_inputs_for_stages 和 extract_hidden_for_computation
    # 它们不再是 GPTNeoXForCausalLM 的直接职责
    # def prepare_gpt_inputs_for_stages(...):
    #     ...
    # def extract_hidden_for_computation(...):
    #     ...

    # compute_interpolated_embeds 也不再是 GPTNeoXForCausalLM 的直接职责
    # 如果 interpolation 逻辑被完全移到 GPTNeoXModel 内部，则这个方法可以被删除
    # 或者如果它只是一个辅助函数，可以移到顶层作为 helper。
    # 为了保持简洁，我们假设它作为 GPTNeoXModel 内部的 helper 函数。
    # 如果你需要它，应该把它放在 GPTNeoXModel 内部，或者作为公共辅助函数。
    # def compute_interpolated_embeds(...):
    #     ...

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        # high_entropy_mask: Optional[torch.LongTensor] = None, # 移除，不再是官方 CLM 签名的一部分
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
        # global_step: Optional[int] = None, # 移除，不再是官方 CLM 签名的一部分
        # xl_memories 和 states 作为模型外部输入，应该在 RecurrentTrainerWrapper 或 GenerationMixin 中处理
        xl_memories: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, 
        states: Optional[List[torch.Tensor]] = None, 
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 调用基础的 GPTNeoXModel
        # GPTNeoXModel 现在内部处理分块、XL 记忆和循环状态
        outputs_from_base_model = self.gpt_neox(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True, # 始终返回字典以方便解析
            cache_position=cache_position,
            # 将 Block Recurrent Transformer 特有的输入传递给 gpt_neox 模型
            xl_memories=xl_memories, 
            states=states,
        )

        # gpt_neox 的 forward 方法现在返回 (BaseModelOutputWithPast, returned_xl_memories, returned_states)
        base_model_output = outputs_from_base_model[0] # BaseModelOutputWithPast
        returned_xl_memories = outputs_from_base_model[1] # List[Tuple[Tensor, Tensor]]
        returned_states = outputs_from_base_model[2] # List[Tensor]


        # 应用语言模型头
        logits = self.embed_out(base_model_output.last_hidden_state)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + base_model_output[1:]
            # 根据需求，可能需要将 returned_xl_memories 和 returned_states 附加到输出元组的末尾
            # 但官方 CLM 输出不包含这些，所以通常会在训练/生成 wrapper 中处理
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=base_model_output.past_key_values,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions,
            # 为了能够返回这些额外信息，如果 `return_dict=True`，
            # 它们需要被添加到 `CausalLMOutputWithPast` 中，或者通过其他方式访问。
            # 如果不修改 `CausalLMOutputWithPast`，这些信息将只能在 `return_dict=False` 时作为额外元组元素返回，
            # 或者通过 `RecurrentTrainerWrapper` 封装调用。
            # 为了保持官方写法，我们在这里不直接添加它们到 CausalLMOutputWithPast。
            # 它们会通过 `GPTNeoXModel` 的额外返回值被上层（例如 `RecurrentTrainerWrapper`）捕获。
        )
    
    # 移除 _calculate_mse_embedding_change，它已在 GPTNeoXModel 中处理，或移到顶层作为辅助函数。
    # def _calculate_mse_embedding_change(...):
    #     ...

    # _reorder_cache 保持不变，它是 GenerationMixin 的一部分
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
