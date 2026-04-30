# Copyright 2024 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from .adapter import init_adapter
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model
from .llama_patch import patch_llama, addhidden_patch_gpt_neox, weightshare_patch_gpt_neox, pausetoken_patch_gpt_neox, orin_patch_gpt_neox, mlp_patch_gpt_neox, orinrandom_patch_gpt_neox, patch_gpt2, patch_mamba2, patch_qwen2, patch_mixtral, bptt_patch_gpt_neox,patch_gpt_neox,patch_gpt_neox_base,patch_gpt_neox_baseline,patch_llama_pause, patch_llama_loop, patch_llama_orin, patch_llama_ours

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info_rank0("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New tokens have been added, changed `resize_vocab` to True.")

    patch_tokenizer(tokenizer)
    try:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        patch_processor(processor, config, tokenizer, model_args)
    except Exception as e:
        logger.debug(f"Processor was not found: {e}.")
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model.
    """
    if model_args.latent is True:
        # patch_llama()
        if model_args.patch_method == "addhidden":
            addhidden_patch_gpt_neox()
            patch_llama()
        elif model_args.patch_method == "weightshare":
            weightshare_patch_gpt_neox()
            patch_llama_loop()
        elif model_args.patch_method == "pausetoken":
            pausetoken_patch_gpt_neox()
            patch_llama_pause()  
        elif model_args.patch_method == "orin":
            orin_patch_gpt_neox()
            patch_llama_orin()
        elif model_args.patch_method == "mlp":
            mlp_patch_gpt_neox()
        elif model_args.patch_method == "orinrandom":
            orinrandom_patch_gpt_neox()
        elif model_args.patch_method == "bptt":
            bptt_patch_gpt_neox()
        elif model_args.patch_method == "baseline":
            patch_gpt_neox_baseline()
        elif model_args.patch_method == "base":
            patch_gpt_neox_base()
        elif model_args.patch_method == "ours":
            patch_llama_ours()
        else:
            patch_gpt_neox()
            # patch_gpt_neox_base()
            # patch_gpt2()
            # patch_mamba2()
            # patch_qwen2()
            # patch_mixtral()
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    # config.output_hidden_states = True
    config.output_hidden_states = model_args.output_hidden_states
    config.ponder_size = model_args.ponder_size
    config.is_normalize_hidden_states = model_args.is_normalize_hidden_states
    config.normalize_topk_sample = model_args.normalize_topk_sample
    config.hidden_layer_num = model_args.hidden_layer_num
    config.more_iterations = model_args.more_iterations
    config.more_eval_iterations = model_args.more_eval_iterations
    config.vary_position = model_args.vary_position
    config.add_loss_for_ponderer = model_args.add_loss_for_ponderer
    config.replace_embeddings = model_args.replace_embeddings
    config.softmax_temperature = model_args.softmax_temperature
    config.add_ponderer_token = model_args.add_ponderer_token
    config.recurrent_model = model_args.recurrent_model
    config.add_adapter = model_args.add_adapter
    config.scale_embeds = model_args.scale_embeds
    config.residual_interpolated_embeds = model_args.residual_interpolated_embeds
    config.recurrent_interval = model_args.recurrent_interval
    config.recurrent_layer = model_args.recurrent_layer
    config.high_memory_mode = model_args.high_memory_mode
    config.add_gate = model_args.add_gate
    config.top_k_num = model_args.top_k_num
    config.back_iterations = model_args.back_iterations
    # config.attention_dropout = 0.1
    # config.hidden_dropout = 0.1
    config.classifier_dropout = 0
    config.mutiply_iterations = model_args.mutiply_iterations
    config.checkpoint_num_layers = model_args.checkpoint_num_layers
    config.uniform_real_time = model_args.uniform_real_time
    config.max_position_embeddings = model_args.max_position_embeddings
    config.use_all_logits = model_args.use_all_logits
    config.interpolation_use_topk = model_args.interpolation_use_topk
    config.stage_router_update_w = model_args.stage_router_update_w
    # config.tie_word_embeddings = False
    config.training_refinement_steps = model_args.training_refinement_steps
    config.eval_refinement_steps = model_args.eval_refinement_steps
    config.interpolation = model_args.interpolation
    config.vary_refine_steps = model_args.vary_refine_steps
    config.use_anderson = model_args.use_anderson
    config.anderson_depth = model_args.anderson_depth
    config.anderson_beta = model_args.anderson_beta
    config.anderson_regularization = model_args.anderson_regularization
    config.anderson_convex_only = model_args.anderson_convex_only
    config.anderson_residual_increase_thr = model_args.anderson_residual_increase_thr
    config.anderson_reset_interval = model_args.anderson_reset_interval
    config.consistency_weight = model_args.consistency_weight
    config.ponder_ent_lambda_start = model_args.ponder_ent_lambda_start
    config.ponder_ent_lambda_max = model_args.ponder_ent_lambda_max
    config.ponder_ent_warmup_steps = model_args.ponder_ent_warmup_steps
    config.ponder_ent_peak_steps = model_args.ponder_ent_peak_steps
    config.ponder_cost_lambda_start = model_args.ponder_cost_lambda_start
    config.ponder_cost_lambda_max = model_args.ponder_cost_lambda_max
    config.ponder_cost_warmup_steps = model_args.ponder_cost_warmup_steps
    config.ponder_cost_peak_steps = model_args.ponder_cost_peak_steps
    config.diverse_lambda_start = model_args.diverse_lambda_start
    config.diverse_lambda_max = model_args.diverse_lambda_max
    config.diverse_warmup_steps = model_args.diverse_warmup_steps
    config.diverse_peak_steps = model_args.diverse_peak_steps
    config.weight_dist_lambda_start = model_args.weight_dist_lambda_start
    config.weight_dist_lambda_max = model_args.weight_dist_lambda_max
    config.weight_dist_warmup_steps = model_args.weight_dist_warmup_steps
    config.weight_dist_peak_steps = model_args.weight_dist_peak_steps
    config.min_weight_penalty_lambda_start = model_args.min_weight_penalty_lambda_start
    config.min_weight_penalty_lambda_max = model_args.min_weight_penalty_lambda_max
    config.min_weight_penalty_warmup_steps = model_args.min_weight_penalty_warmup_steps
    config.min_weight_penalty_peak_steps = model_args.min_weight_penalty_peak_steps
    config.min_weight_penalty_method = model_args.min_weight_penalty_method
    config.delta_method = model_args.delta_method
    config.sigma_slope = model_args.sigma_slope
    config.last_n_steps_update_w = model_args.last_n_steps_update_w
    config.damping_alpha = model_args.damping_alpha
    config.anderson_ridge = model_args.anderson_ridge
    config.sigma_2 = model_args.sigma_2
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            if type(config) in AutoModelForVision2Seq._model_mapping.keys():  # assume built-in models
                load_class = AutoModelForVision2Seq
            else:
                load_class = AutoModelForCausalLM

            if model_args.train_from_scratch:
                model = load_class.from_config(config, trust_remote_code=True)
            else:
                model = load_class.from_pretrained(**init_kwargs)

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model
