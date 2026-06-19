#!/usr/bin/env bash
set -euo pipefail

MODE="$1"
EXP_NAME="$2"
GPU_ID="$3"
PID_FILE="$4"

ROOT=/workspace/imtalker_static/IMTalker
EXP_DIR=/workspace/exps/${EXP_NAME}
SEEN_FILE=${EXP_DIR}/previews/.seen_checkpoints

cd "$ROOT"
source /workspace/preprocess_5090/bin/activate
export PYTHONPATH="$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${EXP_DIR}/previews"
touch "$SEEN_FILE"

preview_ckpt() {
  local ckpt="$1"
  local base step name
  base=$(basename "$ckpt" .ckpt)
  step=${base#step=}
  name="${MODE}_${step}"
  echo "[preview-monitor:${MODE}] rendering ${ckpt}" >&2
  python -u generator/preview_full_checkpoint.py \
    --ref_path /workspace/imtalker_static/IMTalker/assets/source_5.png \
    --aud_path /workspace/imtalker_static/IMTalker/assets/audio_3.wav \
    --renderer_path /workspace/IMTalker/checkpoints/renderer.ckpt \
    --baseline_generator_path /workspace/IMTalker/checkpoints/generator.ckpt \
    --checkpoint "$ckpt" \
    --out_dir "${EXP_DIR}/previews" \
    --name "$name" \
    --wav2vec_model_path /workspace/IMTalker/checkpoints/wav2vec2-base-960h \
    --cam_condition_mode "$MODE" \
    --a_cfg_scale 1.0 \
    --nfe 5 \
    --seed 25 \
    --crop
}

while true; do
  for ckpt in "${EXP_DIR}"/checkpoints/step=*.ckpt; do
    [[ -e "$ckpt" ]] || continue
    if ! grep -Fxq "$ckpt" "$SEEN_FILE"; then
      preview_ckpt "$ckpt" || echo "[preview-monitor:${MODE}] preview failed for ${ckpt}" >&2
      echo "$ckpt" >> "$SEEN_FILE"
    fi
  done

  if [[ -f "$PID_FILE" ]]; then
    pid=$(cat "$PID_FILE")
    if ! kill -0 "$pid" 2>/dev/null; then
      ckpt="${EXP_DIR}/checkpoints/last.ckpt"
      if [[ -e "$ckpt" ]] && ! grep -Fxq "$ckpt" "$SEEN_FILE"; then
        preview_ckpt "$ckpt" || echo "[preview-monitor:${MODE}] final preview failed for ${ckpt}" >&2
        echo "$ckpt" >> "$SEEN_FILE"
      fi
      echo "[preview-monitor:${MODE}] training pid ended; monitor done" >&2
      exit 0
    fi
  fi
  sleep 60
done
