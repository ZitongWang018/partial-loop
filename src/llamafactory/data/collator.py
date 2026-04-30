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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from transformers import DataCollatorForSeq2Seq


@dataclass
class SFTDataCollatorWith4DAttentionMask(DataCollatorForSeq2Seq):
    r"""
    Data collator for 4d attention mask.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(features)
        if "attention_mask" not in batch:
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])

        return batch


@dataclass
class KTODataCollatorWithPadding:
    r"""
    Data collator for KTO.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class MultiModalDataCollatorForSeq2Seq:
    r"""
    Data collator for multimodal models.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class PairwiseDataCollatorWithPadding:
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError
