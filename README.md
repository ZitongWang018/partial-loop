# Ponder-Train

基于 LLaMA-Factory 修改的 70M Orin Iter3 训练代码仓库。

## 背景

本仓库是从 `/data/home/fcsong/LLaMA-Factory-hli` 中提取出 `train_70m_orin_iter3.sh` 脚本所依赖的所有代码文件，整理成一个独立的可 push 到 GitHub 的仓库。

原始脚本路径：`/data/home/ztwang/nips/Train_algo/train_70m_orin_iter3.sh`

## 仓库结构

```
ponder-train/
├── train_70m_orin_iter3.sh          # 训练脚本（入口）
├── README.md                        # 本文件
├── data/
│   └── dataset_info.json            # 数据集配置（smallpile, testpile_streaming 等）
├── examples/deepspeed/              # DeepSpeed 配置文件（ds_z0, ds_z2, ds_z3 等）
├── Llama_config/70m/                # 70M 模型配置（config.json, tokenizer）
└── src/llamafactory/                # 核心代码（基于 LLaMA-Factory 修改）
    ├── __init__.py
    ├── data/                        # 数据处理模块
    │   ├── aligner.py               # 数据对齐
    │   ├── collator.py              # 数据 collator
    │   ├── data_utils.py            # 数据工具
    │   ├── loader.py                # 数据加载
    │   ├── parser.py                # 数据解析
    │   ├── preprocess.py            # 预处理
    │   └── template.py              # 模板
    ├── extras/                      # 扩展工具
    │   ├── constants.py             # 常量
    │   ├── env.py                   # 环境变量
    │   ├── logging.py               # 日志
    │   ├── misc.py                  # 杂项工具
    │   ├── optim.py                 # 优化器
    │   ├── packages.py              # 包管理
    │   └── ploting.py               # 绘图
    ├── hparams/                     # 超参数配置
    │   ├── data_args.py             # 数据参数
    │   ├── finetuning_args.py       # 微调参数
    │   ├── generating_args.py       # 生成参数
    │   ├── model_args.py            # 模型参数
    │   └── parser.py                # 参数解析
    ├── model/                       # 模型模块
    │   ├── adapter.py               # 适配器
    │   ├── loader.py                # 模型加载
    │   ├── llama_patch.py           # Llama patch
    │   ├── patcher.py               # Patcher
    │   ├── modeling/                # 所有模型实现
    │   │   ├── modeling_llama.py            # Llama 模型
    │   │   ├── modeling_llama_orin.py       # Orin 变体
    │   │   ├── modeling_llama_loop.py       # Loop 变体
    │   │   ├── modeling_llama_new.py        # New 变体
    │   │   ├── modeling_llama_pause.py      # Pause token 变体
    │   │   ├── modeling_gpt_neox.py         # GPT-NeoX
    │   │   ├── modeling_gpt_neox_orin.py    # GPT-NeoX Orin
    │   │   ├── modeling_mamba2.py           # Mamba2
    │   │   ├── modeling_mixtral.py          # Mixtral
    │   │   ├── modeling_qwen2.py            # Qwen2
    │   │   ├── flash_attention_vecbias.py   # Flash attention with vec bias
    │   │   └── ...                          # 其他变体
    │   └── model_utils/             # 模型工具
    │       ├── attention.py         # 注意力实现
    │       ├── checkpointing.py     # 梯度检查点
    │       ├── embedding.py         # 嵌入层
    │       ├── moe.py               # MoE
    │       ├── packing.py           # Packing
    │       ├── quantization.py      # 量化
    │       ├── rope.py              # RoPE
    │       └── ...                  # 其他工具
    └── train/                       # 训练模块
        ├── callbacks.py             # 回调
        ├── trainer_utils.py         # 训练工具
        ├── tuner.py                 # Tuner
        └── pt/                      # PT（Pre-Training）流程
            ├── trainer.py           # PT Trainer
            └── workflow.py          # PT 工作流
```

## 使用方法

### 1. 克隆仓库

```bash
git clone git@github.com:ZitongWang018/partial-loop.git
cd partial-loop
```

### 2. 运行训练

**必要参数（必须传入）：**
- `--model_name_or_path`：模型权重路径（config.json 所在目录）
- `--tokenized_path`：tokenized 数据集路径

**示例：**
```bash
bash train_70m_orin_iter3.sh \
    --model_name_or_path /path/to/model \
    --tokenized_path /path/to/tokenized_data
```

**可选参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_gpus` | GPU 数量 | 自动检测所有可用 GPU |
| `--output_dir` | 输出目录 | `./outputs/70m-orin-iter3` |
| `--deepspeed_config` | DeepSpeed 配置 | `./examples/deepspeed/ds_z0_config_llama.json` |
| `--wandb_run_name` | W&B 运行名 | `70m-orin-iter3` |
| `--wandb_project` | W&B 项目 | `train-algo` |
| `--wandb_entity` | W&B entity | `wangzitong344-sun-yat-sen-university` |
| `--max_steps` | 最大步数 | 7500 |
| `--per_device_batch_size` | 每设备 batch size | 2 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 128 |
| `--learning_rate` | 学习率 | 1e-3 |
| `--save_steps` | 保存间隔 | 1000 |
| `--eval_steps` | 评估间隔 | 1000 |

### 3. 训练参数说明

脚本使用的核心训练参数：
- **模型架构**：70M 参数，Orin patch method，recurrent model
- **训练方式**：Full fine-tuning，Pre-Training (pt) stage
- **优化器**：AdamW (beta1=0.9, beta2=0.95, weight_decay=0.1)
- **学习率调度**：cosine_with_min_lr (min_lr_rate=0.1)
- **精度**：bf16
- **Attention**：Flash Attention 2 (fa2)
- **序列长度**：max_position_embeddings=4096, cutoff_len=2048
- **特殊配置**：latent=true, more_iterations=3, scale_embeds=true, residual_interpolated_embeds=true

## 与原始脚本的区别

原始脚本（`/data/home/ztwang/nips/Train_algo/train_70m_orin_iter3.sh`）有以下特点：
1. 使用 **Slurm sbatch** 提交任务
2. 代码路径指向 `/data/home/fcsong/LLaMA-Factory-hli`
3. 包含集群特定的 NCCL 和代理配置

本仓库的脚本做了以下适配：
1. **移除 sbatch**，改为纯 bash 脚本，通过 `nvidia-smi` 自动检测 GPU 数量
2. **代码路径自包含**，`PYTHONPATH` 指向本仓库的 `src/` 目录
3. **模型路径可配置**，通过 `--model_name_or_path` 参数传入
4. **支持单卡/多卡**：单卡直接运行，多卡自动使用 `torchrun` 分布式训练

## 如何 Push 到 GitHub

```bash
cd /path/to/ponder-train

# 配置 git（首次）
git config user.email "your_email@example.com"
git config user.name "Your Name"

# 初始化并提交
git init
git add -A
git commit -m "Initial commit"

# 关联远程仓库并推送
git branch -m main
git remote add origin git@github.com:YourUsername/your-repo.git
git push -u origin main
```

> **注意**：push 前需要确保 SSH key 已注册到 GitHub（Settings → SSH and GPG keys → New SSH key）。
