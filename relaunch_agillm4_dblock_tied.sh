#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace/agillm-4
export TOKENIZERS_PARALLELISM=false
export TOKENIZER_ID="${TOKENIZER_ID:-deepseek-ai/DeepSeek-V4-Pro}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export AGILLM_ATTN_BACKEND=sublinear
[ -f /root/.cache/huggingface/token ] && { export HF_TOKEN="$(tr -d '\r\n' </root/.cache/huggingface/token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; }
SAVE_DIR=/workspace/agillm4_4090_ckpts
CKPT="$(ls -1t "$SAVE_DIR"/pretrain_step*.pt 2>/dev/null | head -1)"
exec >> /workspace/agillm4_floor_train.log 2>&1
echo "RELAUNCH_AGILLM4_DBLOCK_TIED $(date -u +%Y-%m-%dT%H:%M:%SZ) resume=$CKPT --dblock --tie_weights --attn_backend sublinear (fused_ce fixed)"
exec python -u nB300_agillm4.py train --preset agillm4_floor --resume "$CKPT" \
  --dblock --dblock_blocks "${AGILLM4_DBLOCKS:-4}" --dblock_schedule "${AGILLM4_DBLOCK_SCHEDULE:-loss_balanced}" \
  --dblock_warmup_steps "${AGILLM4_DBLOCK_WARMUP:-16}" --dblock_sigma_curriculum_steps "${AGILLM4_DBLOCK_SIGMA_CURRICULUM:-2000}" \
  --dblock_log_every "${AGILLM4_DBLOCK_LOG_EVERY:-25}" --tie_weights \
  --batch_size 1 --block 1280 --amp --attn_backend sublinear --grad_checkpoint \
  --optimizer paged_adamw8bit --sat_every 1 --nat_every 1 --nat_max_tokens 768 --nat_mask_ratio 0.5 \
  --token_param_ratio 100 --save_dir "$SAVE_DIR" \
  --save_every_sec 86400 --heartbeat_every_sec "${AGILLM4_HEARTBEAT_EVERY_SEC:-300}" \
  --empty_cache_every_steps "${AGILLM4_EMPTY_CACHE_EVERY_STEPS:-1}" \
  --delta_every_steps 25000 --delta_max_keep 1 --max_ckpts 1
