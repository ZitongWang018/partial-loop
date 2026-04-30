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
class FinetuningArguments:
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    finetuning_type: Literal["lora", "freeze", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    use_galore: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the GaLore optimizer."},
    )
    use_badam: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the BAdam optimizer."},
    )
    use_adam_mini: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Adam-mini optimizer."},
    )
    use_swanlab: bool = field(
        default=False,
        metadata={"help": "Whether or not to use SwanLab logger."},
    )
    pissa_convert: bool = field(
        default=False,
        metadata={"help": "Whether or not to convert the PiSSA adapter to a normal LoRA adapter."},
    )
    # lora arguments
    lora_rank: int = field(
        default=8,
        metadata={"help": "The rank of LoRA matrices."},
    )
    lora_alpha: float = field(
        default=None,
        metadata={"help": "The alpha parameter of LoRA matrices."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout rate of LoRA matrices."},
    )
    loraplus_lr_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "The LR ratio for the LoRA layers."},
    )
    loraplus_lr_embedding: Optional[float] = field(
        default=None,
        metadata={"help": "The LR for the LoRA embedding layers."},
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with random weights."},
    )
    additional_target: Optional[str] = field(
        default=None,
        metadata={"help": "Name(s) of module(s) to be added as trainable parameters."},
    )
    lora_target: Optional[str] = field(
        default=None,
        metadata={"help": "Name(s) of module(s) to apply LoRA to."},
    )
    lora_target_linear: bool = field(
        default=True,
        metadata={"help": "Whether or not to apply LoRA to all linear layers."},
    )
    lora_modules_to_save: Optional[str] = field(
        default=None,
        metadata={"help": "Name(s) of module(s) to save in addition to LoRA layers."},
    )
    lora_dtype: Optional[Literal["auto", "float16", "bfloat16", "float32"]] = field(
        default=None,
        metadata={"help": "The dtype for the LoRA weights."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Rank Stabilized LoRA (rank-stabilized lora)."},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Weight-Decomposed Low-Rank Adaptation (DoRA)."},
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the LoRA initialization."},
    )
    # freeze arguments
    freeze_trainable_layers: int = field(
        default=2,
        metadata={"help": "The number of trainable layers for freeze method."},
    )
    freeze_trainable_modules: str = field(
        default="all",
        metadata={"help": "The trainable modules for freeze method."},
    )
    freeze_extra_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Name(s) of module(s) to be added as trainable parameters."},
    )
    # dpo arguments
    pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )
    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter for the DPO loss."},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in DPO method."},
    )
    pref_loss_weight: float = field(
        default=1.0,
        metadata={"help": "The weight of the preference loss."},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The gamma parameter for the SimPO loss."},
    )
    # ppo arguments
    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the reward model for PPO training."},
    )
    # kto arguments
    kl_target: float = field(
        default=0.1,
        metadata={"help": "The target KL divergence for KTO training."},
    )
    # rm arguments
    rm_beta: float = field(
        default=0.01,
        metadata={"help": "The beta parameter for the Reward Modeling loss."},
    )
    rm_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout rate for the Reward Model."},
    )
    # packing arguments
    packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Enable sequences packing in training."},
    )
    # neftune arguments
    neftune_noise_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "The noise alpha parameter for NEFTune."},
    )
    # train arguments
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    mask_history: bool = field(
        default=False,
        metadata={"help": "Whether or not to mask the history and train on the last turn only."},
    )
    # deepspeed arguments
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the DeepSpeed config file."},
    )
    # compute arguments
    compute_metrics: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute metrics during training."},
    )
    # logging arguments
    logging_steps: float = field(
        default=100,
        metadata={"help": "Log every X updates steps."},
    )
    save_steps: float = field(
        default=100,
        metadata={"help": "Save checkpoint every X updates steps."},
    )
    save_total_limit: int = field(
        default=None,
        metadata={"help": "Limit the total amount of checkpoints."},
    )
    # report arguments
    report_to: str = field(
        default="wandb",
        metadata={"help": "The list of integrations to report the results and logs to."},
    )
    # resume arguments
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    # ignore arguments
    ignore_data_skip: bool = field(
        default=False,
        metadata={"help": "When resuming training, whether or not to skip the epochs and batches to get the data loading."},
    )
    # plot loss
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to plot the loss after training."},
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
