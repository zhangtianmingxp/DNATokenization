# Minimal DNACHUNKER-style Prototype Plan

## Stage 0 audit summary

- The repository already contains a Hydra + PyTorch Lightning training stack under `train.py`, `configs/`, and `src/`.
- Existing model-related files include Hyena, Caduceus, Mamba registry entries, DNA embedding utilities, and an `hnet_twostage/` Hugging Face-style model.
- Existing data files include genomic benchmark, hg38, nucleotide-transformer datasets, tokenizer helpers, and MLM utilities under `src/dataloaders/`.
- Existing configs cover model, dataset, trainer, optimizer, scheduler, task, callbacks, and hg38 experiments.
- Existing dependencies are listed in `environment.yml`; PyTorch is already used.
- Heavy or compatibility-sensitive dependencies are present: `mamba-ssm`, `flash-attn`, `triton`, `causal-conv1d`, CUDA packages, PyTorch Lightning, Hydra, WandB, Enformer, and genomic dataset packages.
- Potential compatibility issues:
  - `environment.yml` pins Python 3.8 and PyTorch 2.2.0 while local environments may use newer PyTorch/Python.
  - `flash-attn`, `mamba-ssm`, `triton`, and CUDA-specific packages can be difficult to install on CPU-only machines.
  - The full training stack expects external genomic datasets and optional logging infrastructure.
- Missing for a minimal first DNACHUNKER-style experiment:
  - A tiny synthetic DNA dataset that needs no external files.
  - A small standalone dynamic boundary router/chunker.
  - CPU-friendly smoke training.
  - A lightweight test that validates shapes and metrics without Mamba/flash-attn.

## Stage 1 file-level plan

- `src/dnachunker_minimal/__init__.py`
  - Export minimal prototype components.
- `src/dnachunker_minimal/data.py`
  - DNA token constants, synthetic dataset, and BERT-style nucleotide masking collate function.
- `src/dnachunker_minimal/model.py`
  - Embedding, local encoder, straight-through boundary router, chunk pooling, tiny chunk Transformer, dechunking, and MLM head.
- `src/dnachunker_minimal/train.py`
  - Seed control, device selection, training loop, metrics printing, config loading, and CLI.
- `configs/minimal_dnachunker.yaml`
  - CPU/GPU-friendly default hyperparameters.
- `scripts/run_minimal_dnachunker.py`
  - Thin runnable entry point from the repository root.
- `tests/test_minimal_dnachunker.py`
  - Unit/smoke tests for forward shapes and one training batch.

## Scope limits

- Do not add Mamba, DeepSeek sparse attention, biological priors, long-context tasks, Enformer data, or real genomic datasets.
- Do not claim scientific results from the synthetic smoke test.
- Keep implementation readable and small enough for CPU execution.
