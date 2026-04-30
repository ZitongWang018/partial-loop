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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

from ..extras import logging
from ..extras.constants import DEFAULT_TEMPLATE


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


@dataclass
class Template:
    r"""
    Template class for formatting prompts.
    """
    prefix: Optional[List[Union[str, Dict[str, str]]]]
    prompt: Optional[List[Union[str, Dict[str, str]]]]
    system: Optional[str]
    sep: List[Union[str, Dict[str, str]]]
    stop_words: List[str]
    use_history: bool
    efficient_eos: bool

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 2048,
        reserved_label_len: int = 1,
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a tuple of input_ids and labels.
        """
        # Simplified version for pre-training
        return [], []


TEMPLATES: Dict[str, Template] = {}


def _register_template(
    name: str,
    prefix: Optional[List[Union[str, Dict[str, str]]]] = None,
    prompt: Optional[List[Union[str, Dict[str, str]]]] = None,
    system: Optional[str] = None,
    sep: Optional[List[Union[str, Dict[str, str]]]] = None,
    stop_words: Optional[List[str]] = None,
    use_history: bool = True,
    efficient_eos: bool = False,
) -> None:
    template = Template(
        prefix=prefix or [],
        prompt=prompt or [],
        system=system or "",
        sep=sep or [],
        stop_words=stop_words or [],
        use_history=use_history,
        efficient_eos=efficient_eos,
    )
    TEMPLATES[name] = template


def get_template_and_fix_tokenizer(
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
) -> "Template":
    r"""
    Gets the template for the given dataset and fixes the tokenizer.
    """
    if data_args.template is not None and data_args.template in TEMPLATES:
        template = TEMPLATES[data_args.template]
    elif data_args.template is None:
        template = TEMPLATES["empty"]
    else:
        raise ValueError(f"Template {data_args.template} is not defined.")

    return template


# Register empty template for pre-training
_register_template(
    name="empty",
    prefix=[],
    prompt=[],
    system="",
    sep=[],
    stop_words=[],
    use_history=False,
    efficient_eos=False,
)

# Register default template
_register_template(
    name="default",
    prefix=[
        {
            "token": "<|im_start|>system\n{{system}}<|im_end|>\n"
        }
    ],
    prompt=[
        {
            "token": "<|im_start|>user\n{{query}}<|im_end|>\n<|im_start|>assistant\n"
        }
    ],
    system="You are a helpful assistant.",
    sep=[
        {
            "token": "<|im_end|>\n"
        }
    ],
    stop_words=["<|im_end|>", "<|im_start|>"],
    use_history=True,
)
