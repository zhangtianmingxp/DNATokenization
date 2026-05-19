# DNAChunker Real Genome Tokenizer Diagnostics

## Purpose

This workflow runs the original author `HNetTransformerForMaskedLM` tokenizer/chunker diagnostics on small windows sampled from a local FASTA file. It is designed for local hg38-style files without downloading any large dataset automatically.

The model code in `hnet_twostage/modeling_hnet.py` is not modified.

## Required Local Inputs

- FASTA file, for example `hg38.ml.fa` or another indexed genome FASTA.
- Optional BED file with at least `chrom`, `start`, and `end` columns.

`pyfaidx` is used to read FASTA windows without loading the full genome into memory.

## Random Window Example

```bash
conda run -n tokenize python scripts/diagnose_real_genome_tokenizer.py \
  --config configs/smoke_original_dnachunker.yaml \
  --device cuda \
  --fasta /path/to/hg38.ml.fa \
  --chroms chr1,chr2 \
  --seq-len 1024 \
  --num-windows 32 \
  --batch-size 2 \
  --seed 1 \
  --output-dir outputs/real_genome_tokenizer_hg38_small
```

## BED Window Example

```bash
conda run -n tokenize python scripts/diagnose_real_genome_tokenizer.py \
  --config configs/smoke_original_dnachunker.yaml \
  --device cuda \
  --fasta /path/to/hg38.ml.fa \
  --bed /path/to/regions.bed \
  --sampling bed \
  --chroms chr1,chr2 \
  --seq-len 1024 \
  --num-windows 32 \
  --batch-size 2 \
  --output-dir outputs/real_genome_tokenizer_hg38_bed
```

## Output Files

- `window_diagnostics.jsonl`: one JSON record per sampled window.
- `summary.csv`: one compact CSV row per sampled window.
- `summary_by_chrom.csv`: grouped analysis by chromosome.
- `summary_by_gc_bin.csv`: grouped analysis by GC fraction bin.
- `summary_by_n_bin.csv`: grouped analysis by N fraction bin.
- `summary_by_region_label.csv`: grouped boundary density by diagnostic label.

## Metrics

Per window:

- chromosome, start, end, strand
- sequence length
- N fraction
- GC fraction
- loss and ratio_loss, if labels are provided
- stage-1 boundary density
- stage-1 compression ratio
- stage-1 number of chunks
- stage-1 average chunk length
- stage-2 number of chunks

Heuristic region labels:

- `neutral`
- `N_rich`
- `GC_rich`
- `AT_rich`
- `repeat_like`
- `motif_like`

These labels are simple diagnostics, not biological ground truth. Default motif-like examples are `TATA`, `CCAAT`, `CGCG`, and `GATA`.

## Limitations

- No biological prior is used yet.
- The tiny/random or short-trained checkpoints do not imply biological behavior.
- Simple motif/repeat labels are heuristic and only meant to support diagnostics.
- Stage-2 chunk boundaries are not mapped back to original genomic coordinates.
- The scripts do not download hg38 automatically.

## Next Step

Run this on a small local hg38 FASTA sample with the tiny model, confirm outputs are stable, then decide where to add biological-prior experiments.
