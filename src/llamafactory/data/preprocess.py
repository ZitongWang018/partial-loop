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

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from ..extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ..hparams import DataArguments
    from .template import Template


logger = logging.get_logger(__name__)


def get_preprocess_and_print_func(
    data_args: "DataArguments",
    stage: str,
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    do_generate: bool = False,
) -> Tuple[Callable, Callable]:
    r"""
    Gets the preprocessing function and the print function.
    """
    if stage == "pt":
        preprocess_func = _preprocess_pretraining_dataset
        print_function = _print_pretraining_dataset
    else:
        preprocess_func = _preprocess_supervised_dataset
        print_function = _print_supervised_dataset

    return preprocess_func, print_function


def _preprocess_pretraining_dataset(examples: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Preprocesses the pretraining dataset.
    """
    # For pre-training with npy data, input_ids are already tokenized
    if "input_ids" in examples:
        return {
            "input_ids": examples["input_ids"],
            "labels": examples["input_ids"],
            "attention_mask": examples.get("attention_mask", None),
        }

    return examples


def _print_pretraining_dataset(examples: Dict[str, Any]) -> None:
    r"""
    Prints a sample from the pretraining dataset.
    """
    if "input_ids" in examples:
        print(f"input_ids: {examples['input_ids'][:50]}...")
        print(f"labels: {examples['labels'][:50]}...")


def _preprocess_supervised_dataset(examples: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Preprocesses the supervised dataset.
    """
    return examples


def _print_supervised_dataset(examples: Dict[str, Any]) -> None:
    r"""
    Prints a sample from the supervised dataset.
    """
    print(examples)
