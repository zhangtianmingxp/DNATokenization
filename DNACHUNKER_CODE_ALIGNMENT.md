# DNACHUNKER Code Alignment

## Executive summary

The current command:

```bash
python scripts/run_minimal_dnachunker.py --steps 5 --device cpu
```

does **not** instantiate the original author DNAChunker/H-Net model. It runs only the simplified prototype added in the previous Codex run under `src/dnachunker_minimal/`.

The original dynamic chunking code that is present in this checkout is concentrated in `hnet_twostage/modeling_hnet.py`, with training orchestration through `train.py`, Hydra configs under `configs/`, and model registration in `src/utils/registry.py`.

An original-model smoke test is not currently feasible in the lightweight `dnachunker-minimal` environment because the original model import path requires missing heavy dependencies:

- `transformers`
- `mamba_ssm`
- likely CUDA/Triton-linked Mamba components when full import proceeds

No original author model logic was modified.

## Repository separation audit

### A. Original author code

| Path | Purpose | Used by current minimal smoke test? |
|---|---|---|
| `train.py` | Hydra/PyTorch Lightning training entry point. Builds datasets, model, tasks, optimizer, trainer. | No |
| `environment.yml` | Original full environment including CUDA, Mamba, flash-attn, Triton, Lightning, Hydra, genomic libraries. | No |
| `hnet_twostage/configuration_hnet.py` | Hugging Face-style `HNetConfig`. | No |
| `hnet_twostage/modeling_hnet.py` | Two-stage H-Net/DNAChunker-style model, routing, downsampling, upsampling, MLM head, ratio loss. | No |
| `hnet_twostage/rope.py` | Rotary embedding implementation with flash-attn/Triton dependency. | No |
| `configs/model/hnet_twostage.yaml` | Hydra model config for `hnet_twostage.modeling_hnet.HNetTransformerForMaskedLM`. | No |
| `configs/model/hnet.yaml` | Hydra model config referencing `hnet.configuration_hnet.HNetConfig`; the corresponding `hnet/` package is not present in this checkout. | No |
| `configs/experiment/hg38/hnet_twostage.yaml` | hg38 MLM experiment config for the two-stage HNet model. | No |
| `configs/experiment/hg38/hnet.yaml` | hg38 MLM experiment config for `hnet`. The `hnet/` package is not present in this checkout. | No |
| `src/utils/registry.py` | Registry maps `hnet_twostage` to `hnet_twostage.modeling_hnet.HNetTransformerForMaskedLM`. | No |
| `src/dataloaders/utils/mlm.py` | Original BERT-style MLM masking helper. | No |
| `src/dataloaders/datasets/hg38_dataset.py` | hg38 dataset; calls `mlm_getitem` when `mlm=True`. Requires external genomic files. | No |
| `src/dataloaders/genomics.py` | Dataset setup/tokenizer wrapper for hg38 and benchmark tasks. | No |
| `src/models/sequence/dna_embedding.py` | Downstream embedding wrappers, including `DNAEmbeddingModelHNetTwostage`. Imports heavy dependencies at module import time. | No |
| `scripts/run_pretrain_hnet_twostage.sh` | Original pretraining script for two-stage HNet. | No |
| `scripts/run_pretrain_hnet.sh` | Original pretraining script for `hnet`; the referenced `hnet/` package is absent here. | No |

### B. Files added by the previous Codex run

| Path | Purpose | Used by current minimal smoke test? |
|---|---|---|
| `IMPLEMENTATION_PLAN.md` | Stage 0 audit summary and Stage 1 file-level plan. | No |
| `configs/minimal_dnachunker.yaml` | YAML config for lightweight prototype smoke training. | Yes |
| `scripts/run_minimal_dnachunker.py` | Thin entry point for the prototype training CLI. | Yes |
| `src/dnachunker_minimal/__init__.py` | Exports prototype dataset/model helpers. | Yes |
| `src/dnachunker_minimal/data.py` | Synthetic DNA dataset and prototype MLM masking. | Yes |
| `src/dnachunker_minimal/model.py` | Simplified dynamic chunker model. | Yes |
| `src/dnachunker_minimal/train.py` | Prototype seed/device/config/training loop. | Yes |
| `tests/test_minimal_dnachunker.py` | Prototype forward and one-batch training tests. | Used by tests, not by smoke command |

## Original implementation discovery

| Component | Original path | Class/function | Explanation | Matches paper idea? | Used by prototype? |
|---|---|---|---|---|---|
| DNAChunker main model | `hnet_twostage/modeling_hnet.py` | `HNetMixerModel`, `HNetTransformer`, `HNetTransformerForMaskedLM` | Two-stage H-Net-style model with encoder, routing, downsampling, chunk Transformer, cross-attention upsampling, decoder, MLM head. | Partially. It implements a two-stage dynamic H-Net-like chunker, but this audit did not verify against the paper line-by-line. | No |
| Nucleotide embedding | `hnet_twostage/modeling_hnet.py` | `HNetEmbeddings` | Token embedding plus optional tokenizer variants. | Yes at a high level. | No |
| Local encoder | `hnet_twostage/modeling_hnet.py` | `encoder1_layers`, `encoder2_layers`, `create_block`, `BiMambaWrapper` | Mamba-based encoder blocks before each chunking stage. | Different from the prototype; aligned with author code's Mamba-backed H-Net variant. | No |
| Boundary router | `hnet_twostage/modeling_hnet.py` | `RoutingModule` | Projects adjacent states to q/k and computes cosine-similarity-derived boundary probability. | Yes at a high level for learned dynamic boundaries. | No |
| Chunk boundary prediction | `hnet_twostage/modeling_hnet.py` | `RoutingModule.forward` | Computes `p_t = 0.5 * (1 - cosine(q_t, k_{t-1}))`; first token forced boundary. | Yes at a high level. | No |
| Boundary discretization | `hnet_twostage/modeling_hnet.py` | `StraightThroughEstimator` | Rounds probabilities in forward pass and passes gradients through unchanged. | Yes, straight-through discretization. | No |
| Chunk assignment | `hnet_twostage/modeling_hnet.py` | `Downsampler.forward` | Does not assign every token to a chunk; selects boundary-position vectors as compressed chunks. | Different from mean-pooling chunk assignment. | No |
| Chunk pooling/compression | `hnet_twostage/modeling_hnet.py` | `Downsampler` | Compresses by selecting hidden states at boundary locations and padding per batch. | Partially; this is selection/downsampling, not average pooling. | No |
| Main chunk-level network | `hnet_twostage/modeling_hnet.py` | `TransformerBlock`, `HNetMixerModel.main_model` | Transformer over finest chunks after two-stage downsampling. | Yes at a high level. | No |
| Dechunking/upsampling | `hnet_twostage/modeling_hnet.py` | `CrossAttentionUpsampler`, `upsampler_stage1`, `upsampler_stage2` | Cross-attention from higher-resolution queries to lower-resolution chunk states. | Yes at a high level. | No |
| MLM head | `hnet_twostage/modeling_hnet.py` | `HNetTransformerForMaskedLM.lm_head` | Linear head over full-resolution hidden states. | Yes. | No |
| Masking function | `src/dataloaders/utils/mlm.py` | `mlm_getitem` | Hugging Face-style 80/10/10 MLM replacement; unmasked targets become pad token id. | Yes for MLM training. | No |
| Ratio/compression loss | `hnet_twostage/modeling_hnet.py` | `HNetMixerModel.forward` ratio loss block | Combines stage 1 and stage 2 ratio losses using `target_ratio_stage1/2`. Added to MLM loss in `HNetTransformerForMaskedLM.forward`. | Yes at a high level. | No |
| Training loop | `train.py` | `SequenceLightningModule`, `train`, `main` | Hydra + PyTorch Lightning training, dataset/task/model registry wiring. | Not paper-specific; training infrastructure. | No |
| Config system | `configs/`, `train.py`, `hnet_twostage/configuration_hnet.py` | Hydra configs + `HNetConfig` | Hydra composes experiment/model/dataset/trainer settings, then instantiates HF-style config/model. | Not paper-specific; repo system. | No |

## Prototype vs original comparison

| Component | Minimal prototype implementation | Original author implementation | Status | Notes |
|---|---|---|---|---|
| Entry point | `scripts/run_minimal_dnachunker.py` -> `src.dnachunker_minimal.train.main` | `train.py` via Hydra scripts such as `scripts/run_pretrain_hnet_twostage.sh` | Different | Current smoke bypasses Hydra/Lightning entirely. |
| Main model | `MinimalDNAChunker` | `HNetTransformerForMaskedLM` / `HNetMixerModel` | Different | Prototype is intentionally small and standalone. |
| Embedding | `nn.Embedding` over A/C/G/T/N/MASK/PAD ids | `HNetEmbeddings` | Similar | Same broad role, not same class. |
| Local encoder | Two Conv1d + GELU layers | Mamba `create_block`/`BiMambaWrapper` encoder stages | Different | Prototype avoids Mamba/CUDA dependencies. |
| Boundary router logic | Linear head + sigmoid per position | Adjacent q/k cosine similarity: `0.5 * (1 - similarity)` | Different | Future alignment should start at `RoutingModule`. |
| Boundary discretization | Thresholded hard boundary with straight-through expression | `StraightThroughEstimator.apply(p)` using `round()` | Similar but different | Both use straight-through behavior; thresholds differ. |
| Special token boundaries | None | `calculate_special_token_boundaries` ORs special-token boundaries into stage 1 | Missing | Prototype has no special-token treatment. |
| Chunk assignment | `cumsum` chunk ids for every position | No per-token chunk ids; select boundary states | Different | This is one of the biggest semantic differences. |
| Chunk pooling/compression | Weighted average by chunk ids | `Downsampler` selects boundary vectors and pads | Different | Original does not mean-pool all tokens. |
| Chunk-level network | Tiny `nn.TransformerEncoder` | Custom `TransformerBlock` stack with RoPE attention | Different | Same broad role. |
| Dechunking | Gather chunk states back to positions by chunk id | Cross-attention upsampling in two stages | Different | Original learns upsampling through attention. |
| Decoder after upsampling | None beyond norm + MLM head | Mamba decoder layers + final norm | Missing | Prototype omits decoder stack. |
| MLM head | Linear vocab projection | Linear `lm_head` in `HNetTransformerForMaskedLM` | Similar | Same broad role. |
| Masking | `mask_dna_batch`, labels use `-100` ignore index | `mlm_getitem`, labels use tokenizer pad id as ignore index | Different | Original loss ignores `config.pad_token_id`; prototype ignores `-100`. |
| Ratio/compression loss | Metrics only, no loss term | Two-stage ratio loss added to MLM loss | Missing | Important for later alignment. |
| Data | Tiny synthetic dataset | hg38/genomic benchmark datasets and tokenizers | Different | Prototype intentionally avoids external data. |

## Current smoke command import path

Command:

```bash
python scripts/run_minimal_dnachunker.py --steps 5 --device cpu
```

Actual imports:

```text
scripts/run_minimal_dnachunker.py
  -> src.dnachunker_minimal.train.main
     -> src.dnachunker_minimal.data.SyntheticDNADataset
     -> src.dnachunker_minimal.data.mask_dna_batch
     -> src.dnachunker_minimal.model.MinimalDNAChunker
     -> src.dnachunker_minimal.model.mlm_loss
```

Answer:

- It does **not** instantiate the original DNAChunker model.
- It instantiates the simplified prototype model `src.dnachunker_minimal.model.MinimalDNAChunker`.
- It bypasses `hnet_twostage/modeling_hnet.py`, `train.py`, Hydra configs except `configs/minimal_dnachunker.yaml`, original dataloaders, original `mlm_getitem`, Mamba blocks, ratio loss, and cross-attention dechunking.

## Original-model smoke test feasibility

Attempted in `dnachunker-minimal`:

```bash
conda run -n dnachunker-minimal python -c "import hnet_twostage.modeling_hnet as m; print(m.HNetTransformerForMaskedLM)"
```

Result:

```text
ModuleNotFoundError: No module named 'transformers'
```

Dependency probe:

```text
transformers None
mamba_ssm None
triton None
flash_attn None
```

The base environment also lacks `transformers`, `mamba_ssm`, and `flash_attn`.

Because `hnet_twostage/modeling_hnet.py` imports `mamba_ssm` at module import time, adding only `transformers` would not be enough. The smallest later fix is one of:

1. Create a full original-code environment from `environment.yml` on a compatible CUDA machine.
2. Add a CPU-only original-model smoke path by making Mamba imports lazy or optional and allowing tiny Transformer-only substitute blocks. This would modify original model logic and is therefore out of scope for this audit.
3. Add a narrow import-only smoke test after installing `transformers` and `mamba-ssm`, if compatible wheels are available.

No `scripts/smoke_original_dnachunker.py` or `tests/test_original_dnachunker_forward.py` was added because the original model cannot currently be imported in the available environments without heavy missing dependencies.

## Recommended next modification point

For biology-prior boundary routing later, the most direct original-code insertion point is:

- `hnet_twostage/modeling_hnet.py`
  - `RoutingModule.forward`: add biology-aware prior logits/probabilities before discretization.
  - `HNetMixerModel.forward`: pass input-derived biological prior signals into `routing_module_stage1` and/or `routing_module_stage2`.
  - Ratio-loss block in `HNetMixerModel.forward`: keep compression regularization intact when adding routing priors.

The prototype-side exploratory insertion point remains:

- `src/dnachunker_minimal/model.py`
  - `MinimalDNAChunker.forward`, before thresholding boundary probabilities.

However, changes intended to align with the author implementation should be prototyped against `RoutingModule` and `Downsampler` semantics, not the prototype's mean-pooling chunk assignment.

## Validation commands

Run from repository root:

```bash
cd /home/ztm/mycode/idea2/DNAChunker_final
conda run -n dnachunker-minimal python scripts/run_minimal_dnachunker.py --steps 5 --device cpu
conda run -n dnachunker-minimal python -m pytest -q tests/test_minimal_dnachunker.py
conda run -n dnachunker-minimal python -c "import hnet_twostage.modeling_hnet as m; print(m.HNetTransformerForMaskedLM)"
```

Expected status:

- Minimal prototype smoke should run and print `step=... loss=... logits_shape=...` lines.
- Minimal prototype tests should pass.
- Original model import should fail until missing original dependencies are installed.
