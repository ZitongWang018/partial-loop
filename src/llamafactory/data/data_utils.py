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

from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

from datasets import DatasetDict, concatenate_datasets, interleave_datasets

from ..extras import logging


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


DatasetModule = Dict[str, "Dataset"]


def merge_dataset(
    datasets: Sequence[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    seed: int,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Merges multiple datasets into one.
    """
    if len(datasets) == 0:
        return None
    elif len(datasets) == 1:
        return datasets[0]
    else:
        if data_args.mix_strategy == "concat":
            return concatenate_datasets(datasets)
        elif data_args.mix_strategy == "interleave_under" or data_args.mix_strategy == "interleave_over":
            if data_args.interleave_probs is not None:
                probabilities = data_args.interleave_probs
            else:
                probabilities = [1.0 / len(datasets)] * len(datasets)

            return interleave_datasets(
                datasets=datasets,
                probabilities=probabilities,
                seed=seed,
                stopping_strategy="all_exhausted" if data_args.mix_strategy == "interleave_over" else "first_exhausted",
            )
        else:
            raise ValueError("Unknown mixing strategy.")


def split_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    data_args: "DataArguments",
    seed: int,
) -> "DatasetDict":
    r"""
    Splits the dataset and returns a dataset dict containing train and evaluation splits.
    """
    if data_args.streaming:
        dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)
        val_set = dataset.take(int(data_args.val_size))
        train_set = dataset.skip(int(data_args.val_size))
        return DatasetDict({"train": train_set, "validation": val_set})
    else:
        val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
        dataset = dataset.train_test_split(test_size=val_size, seed=seed)
        return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
