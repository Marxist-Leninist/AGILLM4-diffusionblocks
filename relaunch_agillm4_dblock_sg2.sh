#!/usr/bin/env bash
# Relaunch AGILLM-4 dblock with SG2's tuned config + improved sublinear attention v2.
set -Eeuo pipefail
cd /workspace/agillm-4
export TOKENIZERS_PARALLELISM=false
export TOKENIZER_ID="${TOKENIZER_ID:-deepseek-ai/DeepSeek-V4-Pro}"
unset PYTORCH_CUDA_ALLOC_CONF  # B6/L1024 ar70: avoid near-full allocator reservation slowdown
export AGILLM_ATTN_BACKEND=sublinear
[ -f /root/.cache/huggingface/token ] && { export HF_TOKEN="$(tr -d '\r\n' </root/.cache/huggingface/token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; }
SAVE_DIR=/workspace/agillm4_4090_ckpts
TOKEN_PARAM_RATIO="${TOKEN_PARAM_RATIO:-55}"
CKPT="$(ls -1t "$SAVE_DIR"/pretrain_step*.pt 2>/dev/null | head -1)"
exec >> /workspace/agillm4_floor_train.log 2>&1
echo "RELAUNCH_AGILLM4_DBLOCK_SG2 $(date -u +%Y-%m-%dT%H:%M:%SZ) resume=$CKPT (quality ratio55 + B6/L1024 + ar70 backward-math + sublinear v2)"
exec python -u nB300_agillm4.py train --preset agillm4_floor --resume "$CKPT" \
  --dblock --dblock_blocks 4 --dblock_schedule loss_balanced --dblock_warmup_steps 16 \
  --dblock_sigma_curriculum_steps 2000 --dblock_log_every 25 --dblock_objective_mode stochastic \
  --dblock_ar_prob 0.70 --dblock_sat_prob 0.15 --dblock_nat_prob 0.15 \
  --dblock_ar_loss_tokens 512 --dblock_sat_loss_tokens 0 --dblock_nat_loss_tokens 512 \
  --tie_weights --batch_size 6 --block 1024 --amp --attn_backend sublinear \
  --sublinear_window 128 --sublinear_stride 128 --sublinear_max_anchors 128 --sublinear_chunk 128 \
  --sublinear_sinks 4 --sublinear_recent_anchors 64 --no-sublinear_pooled_landmarks \
  --grad_checkpoint --dblock_checkpoint_stride 1 --optimizer paged_adamw8bit --sat_every 4 --nat_every 4 --nat_max_tokens 768 --nat_mask_ratio 0.5 \
  --token_param_ratio "$TOKEN_PARAM_RATIO" --save_dir "$SAVE_DIR" --save_every_sec 86400 --heartbeat_every_sec 300 \
  --empty_cache_every_steps 0 --delta_every_steps 25000 --delta_max_keep 1 --max_ckpts 1
