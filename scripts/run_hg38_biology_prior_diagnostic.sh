#!/usr/bin/env bash
set -euo pipefail
set -x

FASTA="${FASTA:-data/hg38.fa}"
CONFIG="${CONFIG:-configs/smoke_original_dnachunker.yaml}"
DEVICE="${DEVICE:-cuda}"
SEQ_LENS="${SEQ_LENS:-128,512,1024}"
NUM_WINDOWS="${NUM_WINDOWS:-32}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEED="${SEED:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hg38_biology_prior_diagnostic}"
CHECKPOINT="${CHECKPOINT:-}"
CHROMS="${CHROMS:-}"

if [[ -z "${CHROMS}" ]]; then
  CHROMS="$(python scripts/check_hg38_fasta.py --fasta "${FASTA}" --suggest-chroms-only)"
fi

mkdir -p "${OUTPUT_DIR}"

IFS=',' read -ra LENGTHS <<< "${SEQ_LENS}"
for seq_len in "${LENGTHS[@]}"; do
  seq_len="$(echo "${seq_len}" | xargs)"
  out_dir="${OUTPUT_DIR}/seq${seq_len}"
  checkpoint_args=()
  if [[ -n "${CHECKPOINT}" ]]; then
    checkpoint_args=(--checkpoint "${CHECKPOINT}")
  fi
  python scripts/diagnose_real_genome_biology_prior.py \
    --config "${CONFIG}" \
    --device "${DEVICE}" \
    --fasta "${FASTA}" \
    --chroms "${CHROMS}" \
    --seq-len "${seq_len}" \
    --num-windows "${NUM_WINDOWS}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --output-dir "${out_dir}" \
    "${checkpoint_args[@]}"
done

python scripts/report_biology_prior_diagnostics.py --input-dir "${OUTPUT_DIR}"
