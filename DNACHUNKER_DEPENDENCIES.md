# DNACHUNKER Dependencies

## Files inspected

- `environment.yml`
- `setup_env.sh`
- `scripts/README_hnet_ntv2.md`
- `hnet_twostage/modeling_hnet.py`
- `hnet_twostage/configuration_hnet.py`
- `train.py`
- `scripts/*.py`
- `scripts/*.sh`
- `src/`

No `requirements.txt`, `pyproject.toml`, `setup.py`, or top-level README installation guide is present in this checkout.

## Original environment

The original `environment.yml` defines a full research/training environment named `tokenize` with Python 3.8, PyTorch 2.2.0, CUDA packages, and many training/data dependencies. The most relevant entries are:

- Core model/runtime: `pytorch`, `transformers`, `einops`
- Config/training: `hydra-core`, `omegaconf`, `pytorch-lightning`, `wandb`, `torchmetrics`, `timm`
- Genomics/data: `biopython`, `datasets`, `genomic-benchmarks`, `h5py`, `pyfaidx`, `pysam`, `pandas`, `scikit-learn`
- Heavy sequence kernels: `mamba-ssm`, `causal-conv1d`, `flash-attn`, `triton`, CUDA packages

## Minimum dependencies by use case

| Use case | Needed dependencies | Notes |
|---|---|---|
| Minimal prototype smoke | `torch`, `pyyaml`, `pytest`, `numpy` | Already works in `dnachunker-minimal`; does not use original author code. |
| Original `hnet_twostage.configuration_hnet` | `transformers` | The package `hnet_twostage/__init__.py` imports `modeling_hnet`, so normal package import also trips Mamba unless imported by a file-level workaround. |
| Original `hnet_twostage.modeling_hnet` import | `torch`, `transformers`, `mamba_ssm` | `mamba_ssm` is imported at module import time. No config can disable this before import. |
| Original HNet encoder/decoder blocks | `mamba_ssm`, likely `causal-conv1d`, compatible CUDA/PyTorch toolchain | `create_block` and `BiMambaWrapper` use Mamba-backed blocks. |
| Flash attention path | `flash-attn`, `triton` | `hnet_twostage/modeling_hnet.py` gracefully falls back for RMSNorm helpers if flash/Triton layer norm imports fail, but other repo modules import `flash_attn` directly. |
| Full original training | `hydra-core`, `omegaconf`, `pytorch-lightning`, `wandb`, `fsspec`, `torchmetrics`, dataset/tokenizer dependencies | Required by `train.py` and original dataloaders. |
| Original genomic datasets | `datasets`, `genomic-benchmarks`, `pyfaidx`, `pysam`, `pandas`, `numpy`, external FASTA/BED data | Not needed for tiny synthetic model smoke. |

## Installed during this run

Installed into `dnachunker-minimal`:

```bash
conda install -n dnachunker-minimal transformers einops -c conda-forge -y
```

Observed versions:

```text
transformers 5.8.1
einops 0.8.2
```

After this install, original model import advanced past `transformers` and failed at:

```text
ModuleNotFoundError: No module named 'mamba_ssm'
```

## Import-time blockers

`hnet_twostage/modeling_hnet.py` contains:

```python
from mamba_ssm.modules.mamba_simple import Mamba
```

This is a hard module-import dependency. The code does not provide a config option that can disable Mamba before `modeling_hnet.py` is imported.

`flash_attn` is less immediately blocking inside `hnet_twostage/modeling_hnet.py` because the RMSNorm/layer-norm import is wrapped in `try/except`. However, other original repo modules, such as `src/models/sequence/dna_embedding.py` and `src/models/sequence/long_conv_lm.py`, import `flash_attn` directly.

## CPU-only fallback limitations

The original two-stage HNet author model is not currently CPU-smoke-testable in this lightweight environment without either:

1. Installing `mamba_ssm` and its compiled/kernel dependencies, or
2. Modifying original model logic to make Mamba imports lazy/optional or to provide a non-Mamba block.

Option 2 is out of scope for this run because the instruction is not to change original author model logic.

Given the current CPU-only `dnachunker-minimal` environment (`torch.cuda.is_available() == False`) and the original `environment.yml` pinning CUDA/Mamba/flash-attn packages, installing `mamba_ssm` here is likely to require a compatible compiled wheel or a CUDA build toolchain. It was not attempted in this run.

## Smallest next fix

For a true original-model smoke test, use a compatible full environment close to `environment.yml` on a CUDA-capable machine:

```bash
conda env create -f environment.yml
conda activate tokenize
python scripts/check_original_import.py
```

If a CPU-only original smoke test is required, the smallest code change later would be to make the Mamba imports in `hnet_twostage/modeling_hnet.py` lazy/optional and add a test-only non-Mamba block. That would be a model-logic change and should be done in a separate, explicit task.
