#!/bin/bash
# =============================================================================
# 70M Orin Iter3 训练脚本
# 本脚本与 ponder-train 代码仓库配套使用。
#
# 使用方法：
#   1. git clone <本仓库> ponder-train
#   2. cd ponder-train
#   3. 调整下方必要参数
#   4. bash train_70m_orin_iter3.sh
#
# 必要参数（必须设置）：
#   --model_name_or_path  模型权重路径（config.json 所在目录）
#   --tokenized_path      tokenized 数据集路径
#
# 可选参数：
#   --num_gpus            GPU 数量（默认：所有可用 GPU）
#   --output_dir          输出目录（默认：./outputs/70m-orin-iter3）
#   --deepspeed_config    DeepSpeed 配置文件（默认：./examples/deepspeed/ds_z0_config_llama.json）
#   --wandb_run_name      W&B 运行名称（默认：70m-orin-iter3）
#   --wandb_project       W&B 项目名（默认：train-algo）
#   --wandb_entity        W&B entity（默认：wangzitong344-sun-yat-sen-university）
#   --wandb_api_key       W&B API key
#   --wandb_endpoint      W&B 端点（默认：https://api.bandw.top）
#   --max_steps           最大步数（默认：7500）
#   --per_device_batch_size  每设备 batch size（默认：2）
#   --gradient_accumulation_steps  梯度累积步数（默认：128）
#   --learning_rate       学习率（默认：1e-3）
#   --save_steps          保存间隔（默认：1000）
#   --eval_steps          评估间隔（默认：1000）
# =============================================================================

set -euo pipefail

# ================= 解析命令行参数 =================
# 必要参数（无默认值）
MODEL_NAME_OR_PATH=""
TOKENIZED_PATH=""

# 可选参数（有默认值）
NUM_GPUS=""
OUTPUT_DIR=""
DEEPSPEED_CONFIG=""
WANDB_RUN_NAME="70m-orin-iter3"
WANDB_PROJECT="train-algo"
WANDB_ENTITY="wangzitong344-sun-yat-sen-university"
WANDB_API_KEY="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"
WANDB_ENDPOINT="https://api.bandw.top"
MAX_STEPS=7500
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=128
LEARNING_RATE=1e-3
SAVE_STEPS=1000
EVAL_STEPS=1000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path) MODEL_NAME_OR_PATH="$2"; shift 2 ;;
    --tokenized_path)     TOKENIZED_PATH="$2"; shift 2 ;;
    --num_gpus)           NUM_GPUS="$2"; shift 2 ;;
    --output_dir)         OUTPUT_DIR="$2"; shift 2 ;;
    --deepspeed_config)   DEEPSPEED_CONFIG="$2"; shift 2 ;;
    --wandb_run_name)     WANDB_RUN_NAME="$2"; shift 2 ;;
    --wandb_project)      WANDB_PROJECT="$2"; shift 2 ;;
    --wandb_entity)       WANDB_ENTITY="$2"; shift 2 ;;
    --wandb_api_key)      WANDB_API_KEY="$2"; shift 2 ;;
    --wandb_endpoint)     WANDB_ENDPOINT="$2"; shift 2 ;;
    --max_steps)          MAX_STEPS="$2"; shift 2 ;;
    --per_device_batch_size) PER_DEVICE_BATCH_SIZE="$2"; shift 2 ;;
    --gradient_accumulation_steps) GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --learning_rate)      LEARNING_RATE="$2"; shift 2 ;;
    --save_steps)         SAVE_STEPS="$2"; shift 2 ;;
    --eval_steps)         EVAL_STEPS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ================= 检查必要参数 =================
if [ -z "$MODEL_NAME_OR_PATH" ]; then
  echo "ERROR: --model_name_or_path is required"
  echo "Usage: bash train_70m_orin_iter3.sh --model_name_or_path <path> --tokenized_path <path>"
  exit 1
fi
if [ -z "$TOKENIZED_PATH" ]; then
  echo "ERROR: --tokenized_path is required"
  echo "Usage: bash train_70m_orin_iter3.sh --model_name_or_path <path> --tokenized_path <path>"
  exit 1
fi

# ================= 自动计算路径 =================
PONDER_TRAIN_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PONDER_TRAIN_ROOT/outputs/70m-orin-iter3}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$PONDER_TRAIN_ROOT/examples/deepspeed/ds_z0_config_llama.json}"

# ================= 自动检测 GPU 数量 =================
if [ -z "$NUM_GPUS" ]; then
  NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
  if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected via nvidia-smi"
    exit 1
  fi
fi
echo "Using $NUM_GPUS GPU(s)"

# ================= 环境变量 =================
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ================= W&B =================
export WANDB_API_KEY="$WANDB_API_KEY"
export WANDB_ENDPOINT="$WANDB_ENDPOINT"
export WANDB_BASE_URL="$WANDB_ENDPOINT"
export WANDB_ENTITY="$WANDB_ENTITY"
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_RUN_NAME="$WANDB_RUN_NAME"
export WANDB_NAME="$WANDB_RUN_NAME"
export WANDB_DIR="$PONDER_TRAIN_ROOT/wandb"

echo "============================================="
echo "Date:   $(date)"
echo "GPUs:   $NUM_GPUS"
echo "WandB:  entity=$WANDB_ENTITY project=$WANDB_PROJECT run=$WANDB_RUN_NAME"
echo "Out:    $OUTPUT_DIR"
echo "Data:   TOKENIZED_PATH=$TOKENIZED_PATH"
echo "Model:  MODEL_NAME_OR_PATH=$MODEL_NAME_OR_PATH"
echo "============================================="

# ================= Conda 环境 =================
source /data/home/ztwang/miniconda3/bin/activate llamafactory
PY_EXEC="/data/home/ztwang/miniconda3/envs/llamafactory/bin/python"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PONDER_TRAIN_ROOT}/src:${PYTHONPATH:-}"
echo "Using Python: $PY_EXEC"
"$PY_EXEC" --version

# 验证 llamafactory 加载自本仓库
"$PY_EXEC" - <<'PY'
import llamafactory
print("Using llamafactory from:", llamafactory.__file__)
PY

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR"

LLAMAFACTORY_CLI="$(dirname "$PY_EXEC")/llamafactory-cli"
cd "$PONDER_TRAIN_ROOT"

# ================= 启动训练 =================
set +e
if [ "$NUM_GPUS" -gt 1 ]; then
  echo "Starting distributed training with $NUM_GPUS GPUs via torchrun..."
  FORCE_TORCHRUN=1 "$LLAMAFACTORY_CLI" train \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --stage pt \
    --do_train \
    --finetuning_type full \
    --max_steps "$MAX_STEPS" \
    --dataset smallpile \
    --template default \
    --cutoff_len 2048 \
    --interpolation false \
    --output_dir "$OUTPUT_DIR" \
    --logging_steps 1 \
    --num_train_epochs 1.0 \
    --save_steps "$SAVE_STEPS" \
    --plot_loss \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_type cosine_with_min_lr \
    --warmup_ratio 0.01 \
    --report_to wandb \
    --run_name "$WANDB_RUN_NAME" \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --weight_decay 0.1 \
    --ddp_timeout 180000000 \
    --eval_dataset testpile_streaming \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy steps \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --eval_steps "$EVAL_STEPS" \
    --save_total_limit 10 \
    --max_position_embeddings 4096 \
    --latent true \
    --more_iterations 3 \
    --bf16 \
    --patch_method orin \
    --scale_embeds true \
    --recurrent_model true \
    --residual_interpolated_embeds true \
    --disable_gradient_checkpointing true \
    --flash_attn fa2 \
    --train_from_scratch true \
    --dataloader_num_workers 4 \
    --tokenized_path "$TOKENIZED_PATH" \
    --lr_scheduler_kwargs '{"min_lr_rate":0.1}' \
    --do_eval
else
  echo "Starting single-GPU training..."
  "$LLAMAFACTORY_CLI" train \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --stage pt \
    --do_train \
    --finetuning_type full \
    --max_steps "$MAX_STEPS" \
    --dataset smallpile \
    --template default \
    --cutoff_len 2048 \
    --interpolation false \
    --output_dir "$OUTPUT_DIR" \
    --logging_steps 1 \
    --num_train_epochs 1.0 \
    --save_steps "$SAVE_STEPS" \
    --plot_loss \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_type cosine_with_min_lr \
    --warmup_ratio 0.01 \
    --report_to wandb \
    --run_name "$WANDB_RUN_NAME" \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --weight_decay 0.1 \
    --ddp_timeout 180000000 \
    --eval_dataset testpile_streaming \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy steps \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --eval_steps "$EVAL_STEPS" \
    --save_total_limit 10 \
    --max_position_embeddings 4096 \
    --latent true \
    --more_iterations 3 \
    --bf16 \
    --patch_method orin \
    --scale_embeds true \
    --recurrent_model true \
    --residual_interpolated_embeds true \
    --disable_gradient_checkpointing true \
    --flash_attn fa2 \
    --train_from_scratch true \
    --dataloader_num_workers 4 \
    --tokenized_path "$TOKENIZED_PATH" \
    --lr_scheduler_kwargs '{"min_lr_rate":0.1}' \
    --do_eval
fi

EXIT_CODE=$?
set -e
if [ "$EXIT_CODE" -ne 0 ]; then
  echo "Training FAILED with exit code $EXIT_CODE" >&2
else
  echo "Training completed successfully."
fi
echo "============================================="
echo "Finished at: $(date)"
echo "============================================="
exit "$EXIT_CODE"
