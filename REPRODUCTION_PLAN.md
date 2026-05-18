# DNAChunker Reproduction Plan

## Current Reproduction Status

The original two-stage HNet/DNAChunker model is now importable, instantiable, and runnable on CUDA with synthetic DNA token ids. The next reproduction target is the full training stack:

```bash
python -m train experiment=hg38/hnet_twostage
```

This requires the original model, tokenizer, hg38-style BED/FASTA files, the Lightning training wrapper, optimizer/scheduler configs, and CUDA Mamba/Triton runtime.

## What Is Now Available

- Original model smoke: `scripts/smoke_original_dnachunker.py`
- Dependency check: `scripts/check_original_dependencies.py`
- Original import check: `scripts/check_original_import.py`
- Tiny local hg38-like data generator: `scripts/prepare_tiny_hg38_data.py`
- Tiny training config: `configs/experiment/hg38/hnet_twostage_tiny_repro.yaml`
- Local `caduceus.tokenization_caduceus.CaduceusTokenizer` compatibility module for the repo's character-token training path

## Reproduction Ladder

1. Model smoke, no dataloader:

```bash
conda run -n tokenize python scripts/smoke_original_dnachunker.py --config configs/smoke_original_dnachunker.yaml --device cuda
```

2. Training-stack smoke with tiny local FASTA/BED:

```bash
conda run -n tokenize python scripts/prepare_tiny_hg38_data.py
conda run -n tokenize python -m train experiment=hg38/hnet_twostage_tiny_repro
```

3. Small real-data run:

Use real `human-sequences.bed` and `hg38.ml.fa`, but keep `d_model`, layers, `max_length`, `batch_size`, and `max_steps` small.

4. Original-scale run:

Use `scripts/run_pretrain_hnet_twostage.sh` after updating data paths, GPU count, sequence length, batch size, and logging/checkpoint settings for the local machine.

## Required Real Data For Full Reproduction

The original hg38 pipeline expects:

- BED file with columns: `chr_name`, `start`, `end`, `split`
- Splits named exactly: `train`, `valid`, `test`
- FASTA file containing chromosomes referenced by the BED file

Default config paths point to:

- `data/hg38/human-sequences.bed`
- `data/hg38/hg38.ml.fa`

The checked-in large run script uses an absolute repeat BED path:

- `/workspace/caduceus_proj/data/hg38/hg38_repeats.bed`

That path must be replaced or repeat penalization disabled.

## Current Known Constraints

- CPU training is not supported in this installed Mamba/causal-conv1d path; CUDA is required.
- Triton needs a C compiler. The `tokenize` conda env has `x86_64-conda-linux-gnu-cc`, and smoke scripts set `CC` automatically.
- The original large script assumes 8 GPUs and very large token budgets. Do not start from it unchanged on a smaller machine.
- No scientific result is reproduced until real hg38 data, comparable token budget, and comparable hyperparameters are used.
