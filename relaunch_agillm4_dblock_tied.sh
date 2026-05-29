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
BATCH_SIZE="${AGILLM4_BATCH_SIZE:-2}"
SAT_EVERY="${AGILLM4_SAT_EVERY:-4}"
NAT_EVERY="${AGILLM4_NAT_EVERY:-4}"
EMPTY_CACHE_EVERY="${AGILLM4_EMPTY_CACHE_EVERY_STEPS:-0}"
GRAD_CHECKPOINT="${AGILLM4_GRAD_CHECKPOINT:-1}"
DBLOCK_OBJECTIVE_MODE="${AGILLM4_DBLOCK_OBJECTIVE_MODE:-stochastic}"
DBLOCK_AR_PROB="${AGILLM4_DBLOCK_AR_PROB:-0.85}"
DBLOCK_SAT_PROB="${AGILLM4_DBLOCK_SAT_PROB:-0.075}"
DBLOCK_NAT_PROB="${AGILLM4_DBLOCK_NAT_PROB:-0.075}"
DBLOCK_AR_LOSS_TOKENS="${AGILLM4_DBLOCK_AR_LOSS_TOKENS:-512}"
DBLOCK_SAT_LOSS_TOKENS="${AGILLM4_DBLOCK_SAT_LOSS_TOKENS:-0}"
DBLOCK_NAT_LOSS_TOKENS="${AGILLM4_DBLOCK_NAT_LOSS_TOKENS:-512}"
SUBLINEAR_WINDOW="${AGILLM4_SUBLINEAR_WINDOW:-128}"
SUBLINEAR_STRIDE="${AGILLM4_SUBLINEAR_STRIDE:-128}"
SUBLINEAR_MAX_ANCHORS="${AGILLM4_SUBLINEAR_MAX_ANCHORS:-128}"
SUBLINEAR_CHUNK="${AGILLM4_SUBLINEAR_CHUNK:-128}"
GC_FLAG=()
if [ "$GRAD_CHECKPOINT" = "1" ] || [ "$GRAD_CHECKPOINT" = "true" ] || [ "$GRAD_CHECKPOINT" = "yes" ]; then GC_FLAG=(--grad_checkpoint); fi
exec >> /workspace/agillm4_floor_train.log 2>&1
echo "RELAUNCH_AGILLM4_DBLOCK_TIED_SPEED $(date -u +%Y-%m-%dT%H:%M:%SZ) resume=$CKPT --dblock --tie_weights --attn_backend sublinear batch=$BATCH_SIZE sat_every=$SAT_EVERY nat_every=$NAT_EVERY empty_cache_every=$EMPTY_CACHE_EVERY grad_checkpoint=$GRAD_CHECKPOINT objective=$DBLOCK_OBJECTIVE_MODE ar_prob=$DBLOCK_AR_PROB sat_prob=$DBLOCK_SAT_PROB nat_prob=$DBLOCK_NAT_PROB ar_loss_tokens=$DBLOCK_AR_LOSS_TOKENS nat_loss_tokens=$DBLOCK_NAT_LOSS_TOKENS sublinear_window=$SUBLINEAR_WINDOW sublinear_stride=$SUBLINEAR_STRIDE sublinear_max_anchors=$SUBLINEAR_MAX_ANCHORS"
exec python -u nB300_agillm4.py train --preset agillm4_floor --resume "$CKPT" \
  --dblock --dblock_blocks "${AGILLM4_DBLOCKS:-4}" --dblock_schedule "${AGILLM4_DBLOCK_SCHEDULE:-loss_balanced}" \
  --dblock_warmup_steps "${AGILLM4_DBLOCK_WARMUP:-16}" --dblock_sigma_curriculum_steps "${AGILLM4_DBLOCK_SIGMA_CURRICULUM:-2000}" \
  --dblock_log_every "${AGILLM4_DBLOCK_LOG_EVERY:-25}" --dblock_objective_mode "$DBLOCK_OBJECTIVE_MODE" \
  --dblock_ar_prob "$DBLOCK_AR_PROB" --dblock_sat_prob "$DBLOCK_SAT_PROB" --dblock_nat_prob "$DBLOCK_NAT_PROB" \
  --dblock_ar_loss_tokens "$DBLOCK_AR_LOSS_TOKENS" --dblock_sat_loss_tokens "$DBLOCK_SAT_LOSS_TOKENS" --dblock_nat_loss_tokens "$DBLOCK_NAT_LOSS_TOKENS" \
  --tie_weights \
  --batch_size "$BATCH_SIZE" --block 1280 --amp --attn_backend sublinear --sublinear_window "$SUBLINEAR_WINDOW" --sublinear_stride "$SUBLINEAR_STRIDE" --sublinear_max_anchors "$SUBLINEAR_MAX_ANCHORS" --sublinear_chunk "$SUBLINEAR_CHUNK" "${GC_FLAG[@]}" \
  --optimizer paged_adamw8bit --sat_every "$SAT_EVERY" --nat_every "$NAT_EVERY" --nat_max_tokens 768 --nat_mask_ratio 0.5 \
  --token_param_ratio 100 --save_dir "$SAVE_DIR" \
  --save_every_sec 86400 --heartbeat_every_sec "${AGILLM4_HEARTBEAT_EVERY_SEC:-300}" \
  --empty_cache_every_steps "$EMPTY_CACHE_EVERY" \
  --delta_every_steps 25000 --delta_max_keep 1 --max_ckpts 1
