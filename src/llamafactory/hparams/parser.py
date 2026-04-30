# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

from ..extras import logging
from ..extras.misc import check_dependencies, skip_check_imports
from .data_args import DataArguments
from .finetuning_args import FinetuningArguments
from .generating_args import GeneratingArguments
from .model_args import ModelArguments


logger = logging.get_logger(__name__)


def _parse_dict(
    args: Optional[Dict[str, Any]],
    args_class,
    **kwargs,
):
    if args is not None:
        return args_class(**args, **kwargs)
    else:
        return args_class(**kwargs)


def get_train_args(
    args: Optional[Dict[str, Any]] = None,
) -> Tuple["ModelArguments", "DataArguments", "Seq2SeqTrainingArguments", "FinetuningArguments", "GeneratingArguments"]:
    r"""
    Parses command-line arguments and returns the parsed args.
    """
    skip_check_imports()
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_combined_args(args)

    # Setup logging
    if training_args.should_log:
        log_level = training_args.get_process_log_level()
        logging.set_logger(log_level)

    # Check dependencies
    check_dependencies()

    if finetuning_args.stage != "pt" and data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    if finetuning_args.stage != "sft":
        if training_args.predict_with_generate:
            raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

    if finetuning_args.stage == "ppo" and training_args.gradient_checkpointing:
        raise ValueError("Gradient checkpointing is not supported in PPO training.")

    if finetuning_args.stage == "ppo" and training_args.optim != "adamw_torch":
        raise ValueError("PPO training requires `adamw_torch` optimizer.")

    if finetuning_args.stage == "ppo" and model_args.use_ponder:
        raise ValueError("PPO training is not supported with ponder.")

    if finetuning_args.stage == "rm" and training_args.gradient_checkpointing:
        raise ValueError("Gradient checkpointing is not supported in reward modeling.")

    if finetuning_args.stage == "rm" and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

    if finetuning_args.stage == "dpo" and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

    if finetuning_args.stage == "kto" and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

    if finetuning_args.stage == "pt" and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

    if finetuning_args.stage == "pt" and data_args.template is not None:
        logger.warning_rank0("`template` is not used in pre-training.")

    if finetuning_args.stage == "sft" and training_args.do_predict:
        if data_args.eval_dataset is None:
            raise ValueError("Cannot predict without an evaluation dataset.")

    if finetuning_args.stage == "pt" and data_args.streaming:
        if training_args.gradient_checkpointing:
            raise ValueError("Gradient checkpointing is not supported in streaming mode.")

    if finetuning_args.stage == "pt" and data_args.packing is None:
        data_args.packing = True  # enable packing in pre-training by default

    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    if training_args.do_train and data_args.dataset is None:
        raise ValueError("Please specify `dataset` for training.")

    if model_args.train_from_scratch and model_args.model_name_or_path is not None:
        logger.warning_rank0("`model_name_or_path` is ignored when `train_from_scratch` is True.")
        model_args.model_name_or_path = None

    if model_args.adapter_name_or_path is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Adapter is only valid for the LoRA method.")

    if model_args.adapter_name_or_path is not None and finetuning_args.create_new_adapter:
        raise ValueError("Cannot create new adapter with an existing adapter.")

    if model_args.adapter_name_or_path is not None and finetuning_args.resume_from_checkpoint is not None:
        raise ValueError("Cannot resume from checkpoint with an existing adapter.")

    if model_args.use_ponder and model_args.ponder_strategy is None:
        raise ValueError("Please specify `ponder_strategy` when using ponder.")

    return model_args, data_args, training_args, finetuning_args, generating_args


def get_infer_args(
    args: Optional[Dict[str, Any]] = None,
) -> Tuple["ModelArguments", "DataArguments", "FinetuningArguments", "GeneratingArguments"]:
    r"""
    Parses command-line arguments and returns the parsed args for inference.
    """
    skip_check_imports()
    model_args, data_args, training_args, finetuning_args, generating_args = _parse_combined_args(args)

    if data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    return model_args, data_args, finetuning_args, generating_args


def _parse_combined_args(
    args: Optional[Dict[str, Any]] = None,
) -> Tuple["ModelArguments", "DataArguments", "Seq2SeqTrainingArguments", "FinetuningArguments", "GeneratingArguments"]:
    r"""
    Parses command-line arguments and returns the parsed args.
    """
    if args is not None:
        model_args = _parse_dict(args, ModelArguments)
        data_args = _parse_dict(args, DataArguments)
        training_args = _parse_dict(args, Seq2SeqTrainingArguments)
        finetuning_args = _parse_dict(args, FinetuningArguments)
        generating_args = _parse_dict(args, GeneratingArguments)
    else:
        parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments))
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            model_args, data_args, training_args, finetuning_args, generating_args = parser.parse_yaml_file(
                yaml_file=os.path.abspath(sys.argv[1])
            )
        elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            model_args, data_args, training_args, finetuning_args, generating_args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1])
            )
        else:
            model_args, data_args, training_args, finetuning_args, generating_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args, finetuning_args, generating_args
