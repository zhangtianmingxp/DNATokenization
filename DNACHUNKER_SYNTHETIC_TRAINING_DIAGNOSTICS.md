# DNAChunker Synthetic Training Diagnostics

## Purpose

This workflow tests whether the original author DNAChunker dynamic tokenizer changes its measured chunking behavior after a very small controlled MLM training run. It uses the original `hnet_twostage.modeling_hnet.HNetTransformerForMaskedLM` and the hook-based diagnostic benchmark.

No biological priors are implemented here, and `RoutingModule.forward` is not modified.

## Why This Is Not Scientific Evidence

The model is tiny, randomly initialized, and trained for only a few synthetic MLM steps. The synthetic modes are useful for instrumentation and regression testing, but they are not real genomic evidence and should not be interpreted as learned biology.

## Run A Small Debug Training

```bash
conda run -n tokenize python scripts/train_original_synthetic_mlm.py \
  --config configs/smoke_original_dnachunker.yaml \
  --device cuda \
  --output-dir outputs/synthetic_training_debug/train \
  --steps 5 \
  --batch-size 2 \
  --seq-len 128 \
  --synthetic-mode regions \
  --seed 1 \
  --lr 1e-4 \
  --mask-prob 0.15 \
  --log-every 1 \
  --no-wandb
```

## Run The Full Tiny Before/After Diagnostic Experiment

```bash
conda run -n tokenize bash scripts/run_synthetic_training_diagnostic_experiment.sh
```

Environment variables such as `STEPS`, `SEQ_LENS`, `BATCH_SIZE`, and `OUT` can override the shell script defaults.

## Output Files

Training:

- `step_000000.pt`
- `step_000050.pt`, or the configured final step
- `final.pt`
- `train_log.csv`

Diagnostics:

- `before/summary.csv`
- `after/summary.csv`
- per-run JSON files in `before/` and `after/`

Checkpoint loading:

- `scripts/diagnose_original_tokenizer.py --checkpoint PATH`
- `scripts/benchmark_original_tokenizer_diagnostics.py --checkpoint PATH`
- Checkpoints are loaded into `HNetTransformerForMaskedLM` with `strict=True`.
- Supported checkpoint formats are this workflow's `model_state_dict`, Lightning-style `state_dict`, or a raw model state dict.

Comparison:

- `compare/comparison_by_mode.csv`

## Metrics To Inspect

- training `loss`
- training `ratio_loss`
- stage-1 compression ratio delta
- stage-1 post-merge boundary density delta
- motif break-rate delta
- repeat boundary-density delta
- conserved/motif-rich boundary-density delta
- ratio-loss delta

## How This Informs Later Biological Prior Work

This gives a stable before/after measurement loop. Once biological priors are added later, the same benchmark can check whether prior injection changes boundary behavior in the intended synthetic regimes before moving to real hg38 diagnostics.
