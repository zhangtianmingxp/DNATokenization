# DNAChunker Biology Prior Diagnostics

This stage adds an offline diagnostic for biology-guided boundary routing. It does not change the original author model, `RoutingModule.forward`, or any HNet architecture code.

## Purpose

The prior answers one narrow engineering question:

If captured stage1 boundary probabilities were adjusted by simple biological heuristics, which boundary metrics would change?

The current heuristic prior can:

- reduce boundaries inside motif-like spans,
- reduce boundaries in repeat-like and N-rich spans,
- optionally increase boundaries near local GC-fraction shifts,
- compare original and prior-adjusted boundaries on the same sampled FASTA windows.

These labels and signals are heuristic diagnostics only. They are not biological ground truth.

## Main scripts

Run one FASTA diagnostic:

```bash
conda run -n tokenize python scripts/diagnose_real_genome_biology_prior.py \
  --config configs/smoke_original_dnachunker.yaml \
  --device cuda \
  --fasta tests/fixtures/tiny_genome.fa \
  --chroms chrTiny1,chrTiny2 \
  --seq-len 64 \
  --num-windows 4 \
  --batch-size 2 \
  --seed 1 \
  --output-dir outputs/biology_prior_tiny
```

Create a report:

```bash
conda run -n tokenize python scripts/report_biology_prior_diagnostics.py \
  --input-dir outputs/biology_prior_tiny
```

Run local hg38:

```bash
conda run -n tokenize bash scripts/run_hg38_biology_prior_diagnostic.sh
```

Useful overrides:

```bash
SEQ_LENS=128 NUM_WINDOWS=8 OUTPUT_DIR=outputs/hg38_biology_prior_debug \
conda run -n tokenize bash scripts/run_hg38_biology_prior_diagnostic.sh
```

Optional checkpoint:

```bash
CHECKPOINT=outputs/synthetic_training_diagnostics/train/final.pt \
conda run -n tokenize bash scripts/run_hg38_biology_prior_diagnostic.sh
```

## Outputs

Each run writes:

- `biology_prior_summary.csv`
- `biology_prior_window_diagnostics.jsonl`
- `biology_prior_config.json`

The report script writes:

- `BIOLOGY_PRIOR_DIAGNOSTIC_REPORT.md`
- `biology_prior_summary_all.csv`
- `biology_prior_summary_by_seq_len.csv`
- `biology_prior_summary_by_region_label.csv`

## Key metrics

- `base_boundary_density`: original stage1 post-merge boundary density.
- `prior_boundary_density`: boundary density after offline prior adjustment.
- `boundary_density_delta`: prior minus base.
- `base_compression_ratio` and `prior_compression_ratio`: sequence length divided by number of boundaries.
- `boundary_flip_rate`: fraction of positions whose boundary decision changed.
- `boundary_added_density` and `boundary_removed_density`: directional changes.
- `base_motif_like_break_rate` and `prior_motif_like_break_rate`: fraction of motif spans with internal boundaries.

## Limitations

- The prior is not part of the original model forward pass.
- It is not trained and does not update model weights.
- It uses heuristic region labels from local sequence content.
- It does not use curated annotations or validated regulatory labels.
- It should not be interpreted as a biological result.
