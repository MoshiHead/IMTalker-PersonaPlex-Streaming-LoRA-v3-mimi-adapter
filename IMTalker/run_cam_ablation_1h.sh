#!/usr/bin/env bash
set -euo pipefail

MODE="$1"
EXP_NAME="$2"
VISIBLE_GPUS="$3"
MASTER_PORT_VALUE="${4:-12910}"

ROOT=/workspace/imtalker_static/IMTalker
LOG_DIR=/workspace/imtalker_static/logs
mkdir -p "$LOG_DIR" /workspace/exps

cd "$ROOT"
source /workspace/preprocess_5090/bin/activate
export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
export MASTER_PORT="$MASTER_PORT_VALUE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u generator/train.py \
  --dataset_path /workspace/datasets/hdtf_dataset_original \
  --list_path /workspace/datasets/hdtf_dataset_original/hdtf_valid_stems.txt \
  --skip_dataset_filter \
  --wav2vec_model_path /workspace/IMTalker/checkpoints/wav2vec2-base-960h \
  --resume_ckpt /workspace/IMTalker/checkpoints/generator.ckpt \
  --batch_size 128 \
  --iter 100000 \
  --save_freq 5000 \
  --display_freq 5000 \
  --cam_condition_mode "$MODE" \
  --max_minutes 60 \
  --exp_path /workspace/exps \
  --exp_name "$EXP_NAME" \
  --rank cuda
