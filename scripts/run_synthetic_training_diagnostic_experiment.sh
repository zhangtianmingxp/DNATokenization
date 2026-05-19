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
MODES="${MODES:-random,regions,repeat_heavy,motif_heavy,conserved_heavy}"
STEPS="${STEPS:-50}"
LR="${LR:-1e-4}"
MASK_PROB="${MASK_PROB:-0.15}"
OUT="${OUT:-outputs/synthetic_training_diagnostics}"

python scripts/benchmark_original_tokenizer_diagnostics.py \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-lens "${SEQ_LENS}" \
  --seeds "${SEEDS}" \
  --synthetic-modes "${MODES}" \
  --output-dir "${OUT}/before" \
  --no-backward

python scripts/train_original_synthetic_mlm.py \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --output-dir "${OUT}/train" \
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
  --synthetic-modes "${MODES}" \
  --output-dir "${OUT}/after" \
  --checkpoint "${OUT}/train/final.pt" \
  --no-backward

python scripts/compare_tokenizer_before_after_training.py \
  --before "${OUT}/before/summary.csv" \
  --after "${OUT}/after/summary.csv" \
  --output-dir "${OUT}/compare"
