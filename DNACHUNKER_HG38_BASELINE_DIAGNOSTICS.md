# DNACHUNKER HG38 Baseline Diagnostics

## Purpose

This workflow runs the original author `HNetTransformerForMaskedLM` tokenizer/chunker diagnostics on a local hg38 FASTA file. It does not modify model logic, train on hg38, download data, or add biological priors.

## Expected Local FASTA

Default path:

```text
data/hg38.fa
```

Check chromosome names first:

```bash
conda run -n tokenize python scripts/check_hg38_fasta.py --fasta data/hg38.fa --max-chroms 30
```

The checker prints whether names look like `chr1,chr2` or `1,2`, plus a suggested `--chroms` value.

## Run Baseline

```bash
conda run -n tokenize bash scripts/run_hg38_tokenizer_baseline.sh
```

Useful overrides:

```bash
FASTA=data/hg38.fa \
CHROMS=chr1,chr2 \
SEQ_LENS=128,512 \
NUM_WINDOWS=32 \
BATCH_SIZE=2 \
OUTPUT_DIR=outputs/hg38_tokenizer_baseline \
conda run -n tokenize bash scripts/run_hg38_tokenizer_baseline.sh
```

## Optional Checkpoint

Later, after training a tiny checkpoint:

```bash
CHECKPOINT=outputs/synthetic_training_diagnostics/train/final.pt \
conda run -n tokenize bash scripts/run_hg38_tokenizer_baseline.sh
```

## Aggregate And Report

```bash
conda run -n tokenize python scripts/aggregate_hg38_tokenizer_baseline.py \
  --input-dir outputs/hg38_tokenizer_baseline

conda run -n tokenize python scripts/report_hg38_tokenizer_baseline.py \
  --input-dir outputs/hg38_tokenizer_baseline
```

## Output Files

- `seq*/summary.csv`
- `seq*/window_diagnostics.jsonl`
- `seq*/summary_by_chrom.csv`
- `seq*/summary_by_gc_bin.csv`
- `seq*/summary_by_n_bin.csv`
- `seq*/summary_by_region_label.csv`
- `hg38_summary_all.csv`
- `hg38_summary_by_seq_len.csv`
- `hg38_summary_by_gc_bin.csv`
- `hg38_summary_by_region_label.csv`
- `HG38_BASELINE_REPORT.md`

## Metrics

- stage-1 boundary density
- stage-1 compression ratio
- ratio loss
- GC fraction
- N fraction
- motif-like break rate
- heuristic region-label boundary densities

Heuristic labels include `neutral`, `repeat_like`, `motif_like`, `GC_rich`, `AT_rich`, and `N_rich`.

## Limitations

- The default model is tiny/random unless a checkpoint is provided.
- Heuristic labels are not biological ground truth.
- No biological prior is active.
- No hg38 training occurs.
- Do not interpret these diagnostics as biological results.
