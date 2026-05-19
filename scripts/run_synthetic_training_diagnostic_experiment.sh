#!/usr/bin/env bash
set -euo pipefail
set -x

CONFIG="${CONFIG:-configs/smoke_original_dnachunker.yaml}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
SEQ_LENS="${SEQ_LENS:-64,128}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-128}"
SEEDS="${SEEDS:-1,2}"
SYNTHETIC_MODES="${SYNTHETIC_MODES:-${MODES:-random,regions,repeat_heavy,motif_heavy,conserved_heavy}}"
STEPS="${STEPS:-50}"
LR="${LR:-1e-4}"
MASK_PROB="${MASK_PROB:-0.15}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUT:-outputs/synthetic_training_diagnostics}}"

python scripts/benchmark_original_tokenizer_diagnostics.py \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-lens "${SEQ_LENS}" \
  --seeds "${SEEDS}" \
  --synthetic-modes "${SYNTHETIC_MODES}" \
  --output-dir "${OUTPUT_DIR}/before" \
  --no-backward

python scripts/train_original_synthetic_mlm.py \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}/train" \
  --steps "${STEPS}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --seq-len "${TRAIN_SEQ_LEN}" \
  --synthetic-mode regions \
  --seed 1 \
  --lr "${LR}" \
  --mask-prob "${MASK_PROB}" \
  --log-every 10 \
  --no-wandb

python scripts/benchmark_original_tokenizer_diagnostics.py \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-lens "${SEQ_LENS}" \
  --seeds "${SEEDS}" \
  --synthetic-modes "${SYNTHETIC_MODES}" \
  --output-dir "${OUTPUT_DIR}/after" \
  --checkpoint "${OUTPUT_DIR}/train/final.pt" \
  --no-backward

python scripts/compare_tokenizer_before_after_training.py \
  --before "${OUTPUT_DIR}/before/summary.csv" \
  --after "${OUTPUT_DIR}/after/summary.csv" \
  --output-dir "${OUTPUT_DIR}/compare"
