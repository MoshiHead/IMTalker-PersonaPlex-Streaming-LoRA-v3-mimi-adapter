#!/usr/bin/env bash
set -euo pipefail
IMTALKER_ROOT=/workspace/IMTalker-PersonaPlex-Streaming-v1/IMTalker
PERSONAPLEX_ROOT=/workspace/IMTalker-PersonaPlex-Streaming-v1/personaplex
cd "$IMTALKER_ROOT"
source /workspace/preprocess_5090/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$IMTALKER_ROOT:$PERSONAPLEX_ROOT/checkpoints/moshi"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${HF_TOKEN:?Set HF_TOKEN in the environment before running this script}"
# Place your trained LoRA checkpoint at checkpoints/lora.ckpt, or remove the
# --lora_path/--lora_rank/--lora_alpha lines below to run without LoRA.
#
# To use the acoustic-only Mimi-decoder frontend tap (fixes random muted
# lip-sync caused by Helium's hidden state carrying text/dialogue context;
# train the checkpoint first with generator/train_frontend_adapter.py), add:
#   --frontend_source mimi_decoder \
#   --mimi_adapter_path /path/to/mimi_decoder_best.pt \
python -u "$IMTALKER_ROOT/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary.py" \
  --generator_path "$IMTALKER_ROOT/checkpoints/generator.ckpt" \
  --renderer_path "$IMTALKER_ROOT/checkpoints/renderer.ckpt" \
  --lora_path "$IMTALKER_ROOT/checkpoints/lora.ckpt" \
  --lora_rank 64 \
  --lora_alpha 128 \
  --adapter_path /workspace/IMTalker-PersonaPlex-Streaming-v1/checkpoints/personaplex_helium_w2v_frontend_adapter/checkpoints/phase2_best_wav2vec_final_loss.pt \
  --adapter_num_layers 6 \
  --adapter_dropout 0.1 \
  --wav2vec_model_path "$IMTALKER_ROOT/checkpoints/wav2vec2-base-960h" \
  --ref_path "$IMTALKER_ROOT/assets/2_vid_robert.png" \
  --host 0.0.0.0 \
  --port 8998 \
  --device cuda \
  --enable_moshi_reply \
  --direct_reply_hidden \
  --moshi_root "$PERSONAPLEX_ROOT" \
  --mimi_hf_repo nvidia/personaplex-7b-v1 \
  --moshi_weight "$PERSONAPLEX_ROOT/checkpoints/model_bnb_4bit.pt" \
  --quantize_4bit \
  --num_codebooks 8 \
  --moshi_reply_device cuda \
  --moshi_cfg_coef 1.0 \
  --voice_prompt NATM0.pt \
  --text_prompt "You work for North South University which is a university and your name is Nabeel Mohammed. Information: you are answering Computer science related questions explicitly about models and telling about how moshi and personaplex are trained to ordinary people. So in lighter terms." \
  --a_cfg_scale 1.34 \
  --nfe 5 \
  --wav2vec_sec 0.96 \
  --audio_chunk_sec 0.96 \
  --fm_chunk_frames 24 \
  --reply_hidden_steps_per_chunk 0 \
  --prebuffer_chunks 0 \
  --frame_q_backpressure 160 \
  --render_sub_batch 8 \
  --jpeg_quality 58 \
  --dump_motion \
  --dump_dir "$IMTALKER_ROOT/live_try_dumps_personaplex_frontend_source5_cfg134" \
  --shared_noise \
  --noise_seed 42 \
  --noise_max_frames 5000 \
  --fp32 \
  --tf32
