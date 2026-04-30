# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/training_args.py
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

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class ModelArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from Hugging Face Hub."},
    )
    adapter_name_or_path: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "Path to adapter(s) from Hugging Face Hub. If you use multiple adapters, separate them with commas."
            )
        },
    )
    adapter_folder: Optional[str] = field(
        default=None,
        metadata={"help": "The folder containing the adapter weights to load."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from Hugging Face Hub."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by tokenizers library)."},
    )
    resize_vocab: bool = field(
        default=False,
        metadata={"help": "Whether or not to resize the tokenizer vocab and embedding layer."},
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether or not to split special tokens during tokenization."},
    )
    new_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "Special tokens to add to the tokenizer, separated by commas."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={"help": "Whether or not to use limited CPU memory by loading the model in 8bit steps."},
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model using bitsandbytes."},
    )
    quantization_type: Literal["fp4", "nf4"] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in int4 training."},
    )
    double_quantization: bool = field(
        default=True,
        metadata={"help": "Whether or not to use double quantization in int4 training."},
    )
    quantization_device_map: Optional[Literal["auto", "balanced", "sequential"]] = field(
        default=None,
        metadata={"help": "Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0."},
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Which scaling strategy to use for the RoPE embeddings."},
    )
    rope_theta: Optional[float] = field(
        default=None,
        metadata={"help": "The theta value to use for the RoPE embeddings."},
    )
    rope_factor: float = field(
        default=1.0,
        metadata={"help": "The scaling factor to use for the RoPE embeddings."},
    )
    mix_docter_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "The ratio of mix docter."},
    )
    mix_docter_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "The alpha of mix docter."},
    )
    mix_docter_layer_num: Optional[int] = field(
        default=None,
        metadata={"help": "The number of layers to apply mix docter."},
    )
    mix_docter_start_layer: Optional[int] = field(
        default=None,
        metadata={"help": "The start layer to apply mix docter."},
    )
    mix_docter_use_residual: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use residual in mix docter."},
    )
    mix_docter_use_softmax: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use softmax in mix docter."},
    )
    mix_docter_use_balance: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use balance in mix docter."},
    )
    mix_docter_balance_coef: Optional[float] = field(
        default=None,
        metadata={"help": "The balance coefficient in mix docter."},
    )
    mix_docter_balance_tolerance: Optional[float] = field(
        default=None,
        metadata={"help": "The balance tolerance in mix docter."},
    )
    use_ponder: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use ponder."},
    )
    ponder_strategy: Optional[str] = field(
        default=None,
        metadata={"help": "The strategy of ponder."},
    )
    ponder_temperature: Optional[float] = field(
        default=None,
        metadata={"help": "The temperature of ponder."},
    )
    ponder_temperature_end: Optional[float] = field(
        default=None,
        metadata={"help": "The end temperature of ponder."},
    )
    ponder_temperature_warmup_steps: Optional[int] = field(
        default=None,
        metadata={"help": "The warmup steps of ponder temperature."},
    )
    ponder_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "The alpha of ponder."},
    )
    ponder_beta: Optional[float] = field(
        default=None,
        metadata={"help": "The beta of ponder."},
    )
    ponder_gamma: Optional[float] = field(
        default=None,
        metadata={"help": "The gamma of ponder."},
    )
    ponder_top_p: Optional[float] = field(
        default=None,
        metadata={"help": "The top_p of ponder."},
    )
    ponder_min_n_layer: Optional[int] = field(
        default=None,
        metadata={"help": "The minimum number of layers to ponder."},
    )
    ponder_max_n_layer: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum number of layers to ponder."},
    )
    ponder_n_layer: Optional[int] = field(
        default=None,
        metadata={"help": "The number of layers to ponder."},
    )
    ponder_share_weights: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to share weights in ponder."},
    )
    ponder_use_residual: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use residual in ponder."},
    )
    ponder_use_softmax: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use softmax in ponder."},
    )
    ponder_use_balance: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use balance in ponder."},
    )
    ponder_balance_coef: Optional[float] = field(
        default=None,
        metadata={"help": "The balance coefficient in ponder."},
    )
    ponder_balance_tolerance: Optional[float] = field(
        default=None,
        metadata={"help": "The balance tolerance in ponder."},
    )
    ponder_use_aux_ce_loss: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use auxiliary CE loss in ponder."},
    )
    ponder_aux_ce_loss_coef: Optional[float] = field(
        default=None,
        metadata={"help": "The auxiliary CE loss coefficient in ponder."},
    )
    ponder_use_aux_ctc_loss: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use auxiliary CTC loss in ponder."},
    )
    ponder_aux_ctc_loss_coef: Optional[float] = field(
        default=None,
        metadata={"help": "The auxiliary CTC loss coefficient in ponder."},
    )
    ponder_use_aux_entropy_loss: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use auxiliary entropy loss in ponder."},
    )
    ponder_aux_entropy_loss_coef: Optional[float] = field(
        default=None,
        metadata={"help": "The auxiliary entropy loss coefficient in ponder."},
    )
    ponder_use_aux_regularization: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use auxiliary regularization in ponder."},
    )
    ponder_aux_regularization_coef: Optional[float] = field(
        default=None,
        metadata={"help": "The auxiliary regularization coefficient in ponder."},
    )
    ponder_use_aux_ce_loss_ponder: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use auxiliary CE loss in ponder for ponderer."},
    )
    ponder_aux_ce_loss_ponder_coef: Optional[float] = field(
        default=None,
        metadata={"help": "The auxiliary CE loss coefficient in ponder for ponderer."},
    )
    ponder_use_aux_ctc_loss_ponder: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use auxiliary CTC loss in ponder for ponderer."},
    )
    ponder_aux_ctc_loss_ponder_coef: Optional[float] = field(
        default=None,
        metadata={"help": "The auxiliary CTC loss coefficient in ponder for ponderer."},
    )
    ponder_use_aux_entropy_loss_ponder: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use auxiliary entropy loss in ponder for ponderer."},
    )
    ponder_aux_entropy_loss_ponder_coef: Optional[float] = field(
        default=None,
        metadata={"help": "The auxiliary entropy loss coefficient in ponder for ponderer."},
    )
    ponder_use_aux_regularization_ponder: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use auxiliary regularization in ponder for ponderer."},
    )
    ponder_aux_regularization_ponder_coef: Optional[float] = field(
        default=None,
        metadata={"help": "The auxiliary regularization coefficient in ponder for ponderer."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether or not to allow for custom models defined on the Hub in their own modeling files."},
    )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "The dtype to use for inference."},
    )
    flash_attn: Literal["auto", "disabled", "sdpa", "fa2"] = field(
        default="auto",
        metadata={"help": "Enable FlashAttention for faster training and inference."},
    )
    shift_attn: bool = field(
        default=False,
        metadata={"help": "Enable shift attention for the llama model (S^2-Attn)."},
    )
    disable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    upcast_layernorm: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the layernorm weights to fp32."},
    )
    upcast_lmhead_output: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the output of lm_head to fp32."},
    )
    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "Whether or not to randomly initialize the model weights."},
    )
    infer_backend: Literal["huggingface", "vllm"] = field(
        default="huggingface",
        metadata={"help": "Inference engine backend to use."},
    )
    vllm_maxlen: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for vLLM."},
    )
    vllm_gpu_util: float = field(
        default=0.9,
        metadata={"help": "The fraction of GPU memory to use for vLLM."},
    )
    vllm_enforce_eager: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable CUDA graph in vLLM."},
    )
    vllm_max_lora_rank: int = field(
        default=32,
        metadata={"help": "Maximum rank for LoRA adapters."},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face access token for private model."},
    )
    ms_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "ModelScope access token for private model."},
    )
    om_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "OpenMind access token for private model."},
    )
    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."},
    )
    export_size: int = field(
        default=1,
        metadata={"help": "The file shard size (in GB) for the exported model."},
    )
    export_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the exported model."},
    )
    export_quantization_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the dataset to use for quantizing the exported model."},
    )
    export_legacy_format: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the `.bin` files instead of `.safetensors`."},
    )
    export_hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the repository on Hugging Face Hub to push the exported model."},
    )
    print_param_status: bool = field(
        default=False,
        metadata={"help": "For debugging purposes, print the status of the parameters."},
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
