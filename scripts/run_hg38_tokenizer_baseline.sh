#!/usr/bin/env bash
set -euo pipefail
set -x

FASTA="${FASTA:-data/hg38.fa}"
CONFIG="${CONFIG:-configs/smoke_original_dnachunker.yaml}"
DEVICE="${DEVICE:-cuda}"
CHROMS="${CHROMS:-}"
SEQ_LENS="${SEQ_LENS:-128,512,1024}"
NUM_WINDOWS="${NUM_WINDOWS:-32}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hg38_tokenizer_baseline}"
CHECKPOINT="${CHECKPOINT:-}"

if [[ -z "${CHROMS}" ]]; then
  CHROMS="$(python scripts/check_hg38_fasta.py --fasta "${FASTA}" --suggest-chroms-only)"
fi

IFS=',' read -r -a LENS <<< "${SEQ_LENS}"
for SEQ_LEN in "${LENS[@]}"; do
  OUT_DIR="${OUTPUT_DIR}/seq${SEQ_LEN}"
  CHECKPOINT_ARGS=()
  if [[ -n "${CHECKPOINT}" ]]; then
    CHECKPOINT_ARGS=(--checkpoint "${CHECKPOINT}")
  fi
  python scripts/diagnose_real_genome_tokenizer.py \
    --config "${CONFIG}" \
    --device "${DEVICE}" \
    --fasta "${FASTA}" \
    --chroms "${CHROMS}" \
    --seq-len "${SEQ_LEN}" \
    --num-windows "${NUM_WINDOWS}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --output-dir "${OUT_DIR}" \
    "${CHECKPOINT_ARGS[@]}"

  python scripts/analyze_real_genome_tokenizer.py \
    --summary "${OUT_DIR}/summary.csv"
done

python - <<PY
import json
from pathlib import Path

out = Path("${OUTPUT_DIR}")
out.mkdir(parents=True, exist_ok=True)
(out / "run_config.json").write_text(json.dumps({
    "fasta": "${FASTA}",
    "config": "${CONFIG}",
    "device": "${DEVICE}",
    "chroms": "${CHROMS}",
    "seq_lens": "${SEQ_LENS}",
    "num_windows": "${NUM_WINDOWS}",
    "batch_size": "${BATCH_SIZE}",
    "seed": "${SEED}",
    "checkpoint": "${CHECKPOINT}" or None,
}, indent=2), encoding="utf-8")
PY
