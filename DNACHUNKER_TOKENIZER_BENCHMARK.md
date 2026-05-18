# DNAChunker Tokenizer Diagnostic Benchmark

## Purpose

This benchmark repeatedly runs the original author `HNetTransformerForMaskedLM` on controlled synthetic DNA token sequences and records tokenizer/chunker diagnostics collected with forward hooks. It is meant to establish a baseline view of boundary and chunk behavior before adding any biological-prior logic.

The benchmark uses the original model path:

- `hnet_twostage.modeling_hnet.HNetTransformerForMaskedLM`
- `RoutingModule`
- `Downsampler`

It does not use `MinimalDNAChunker`.

## Synthetic, Not Scientific Evidence

The inputs are synthetic and the tiny smoke config is randomly initialized. The resulting boundary densities and compression ratios are useful for debugging, instrumentation, and comparing later controlled experiments, but they are not biological conclusions.

## Synthetic Modes

- `random`: random DNA/N tokens with neutral labels.
- `regions`: balanced neutral, repeat, and conserved/motif-rich thirds.
- `repeat_heavy`: a large low-complexity repeat block.
- `motif_heavy`: many short motif spans embedded in neutral sequence.
- `conserved_heavy`: longer conserved-like motif-rich blocks.

## Run Benchmark

```bash
conda run -n tokenize python scripts/benchmark_original_tokenizer_diagnostics.py \
  --config configs/smoke_original_dnachunker.yaml \
  --device cuda \
  --batch-size 2 \
  --seq-lens 64,128 \
  --seeds 1,2 \
  --synthetic-modes random,regions,repeat_heavy,motif_heavy,conserved_heavy \
  --output-dir outputs/tokenizer_benchmark \
  --no-backward
```

## Analyze Benchmark

```bash
conda run -n tokenize python scripts/analyze_tokenizer_benchmark.py \
  --input outputs/tokenizer_benchmark/summary.csv
```

## Output Files

Each run writes one scalar-only JSON file:

```text
outputs/tokenizer_benchmark/run_0000_random_L64_seed1.json
```

The benchmark also writes:

```text
outputs/tokenizer_benchmark/summary.csv
outputs/tokenizer_benchmark/summary_by_mode.csv
outputs/tokenizer_benchmark/summary_by_length.csv
```

## Important Metrics

- `stage1_boundary_prob_mean/std/min/max`
- `stage1_pre_merge_boundary_density`
- `stage1_post_merge_boundary_density`
- `stage1_num_chunks_mean/std/min/max`
- `stage1_avg_chunk_length_mean/std`
- `stage1_compression_ratio_mean/std`
- `stage2_*` equivalents when available
- `neutral_boundary_density`
- `repeat_boundary_density`
- `conserved_boundary_density`
- `motif_break_rate`
- `motif_internal_boundary_count_mean`

## How This Supports Later Prior Experiments

The benchmark provides a stable, reproducible set of synthetic cases. After biological priors are implemented later, the same benchmark can compare pre-prior and post-prior behavior without changing data, seeds, or model size. That makes it easier to catch accidental changes to chunking behavior before running real hg38 experiments.
