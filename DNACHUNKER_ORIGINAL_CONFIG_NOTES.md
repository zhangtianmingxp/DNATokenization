# Original HNet/DNAChunker Config Notes

## Config class

Original two-stage model config:

- File: `hnet_twostage/configuration_hnet.py`
- Class: `HNetConfig`
- Base class: `transformers.PretrainedConfig`
- Model type: `hnet`

Hydra config:

- File: `configs/model/hnet_twostage.yaml`
- Model registry name: `hnet_twostage`
- Registry path: `src/utils/registry.py` maps `hnet_twostage` to `hnet_twostage.modeling_hnet.HNetTransformerForMaskedLM`

## Key config fields

`HNetConfig` default values include:

| Field | Default | Purpose |
|---|---:|---|
| `d_model` | `2560` | Hidden size. |
| `vocab_size` | `50277` | Token vocabulary size; padded to `pad_vocab_size_multiple` inside `HNetTransformer`. |
| `ssm_cfg` | `None` | Mamba SSM block configuration. |
| `rms_norm` | `True` | Use RMSNorm if available. |
| `fused_add_norm` | `True` | Use fused add+norm path. |
| `residual_in_fp32` | `True` | Residual precision behavior. |
| `pad_vocab_size_multiple` | `8` | Vocab padding multiple. |
| `norm_epsilon` | `1e-5` | Norm epsilon. |
| `bidirectional` | `True` | Bidirectional Mamba wrapper. |
| `bidirectional_strategy` | `"add"` | How forward/reverse streams are combined. |
| `bidirectional_weight_tie` | `True` | Tie selected forward/reverse Mamba weights. |
| `target_ratio` | `0.3` | Legacy ratio parameter. |
| `target_ratio_stage1` | `0.5` | Stage 1 compression target. |
| `target_ratio_stage2` | `0.3` | Stage 2 compression target. |
| `motif_ratio` | `0.0` | Present but not used in the audited forward path. |
| `n_enc_layer` | `1` | Number of encoder layers split into two stages. |
| `n_main_layer` | `2` | Number of chunk-level Transformer blocks. |
| `n_dec_layer` | `1` | Number of decoder layers. |
| `transformer_n_head` | `8` | Chunk Transformer and upsampler attention heads. |
| `transformer_mlp_mult` | `4` | Chunk Transformer MLP multiplier. |
| `tokenizer_type` | `"default"` | Uses `HNetEmbeddings`; `"stft"` uses `HNetEmbeddingsSTFT`. |

The config also sets:

```python
self.n_layer = self.n_enc_layer + self.n_main_layer + self.n_dec_layer
```

## Instantiation path

The original masked-language-model class is instantiated directly with `HNetConfig`:

```python
from hnet_twostage.configuration_hnet import HNetConfig
from hnet_twostage.modeling_hnet import HNetTransformerForMaskedLM

config = HNetConfig(...)
model = HNetTransformerForMaskedLM(config)
```

`HNetTransformerForMaskedLM.__init__` wraps `HNetTransformer`, which mutates `config.vocab_size` upward when it is not divisible by `pad_vocab_size_multiple`, then creates `HNetMixerModel` and an LM head.

## Required fields for a runnable tiny config

No constructor argument is strictly required because `HNetConfig` provides defaults, but a smoke-sized model should set these fields explicitly:

| Field | Smoke value | Why it is needed |
|---|---:|---|
| `d_model` | `16` | Hidden size. Must be divisible by `transformer_n_head`. |
| `vocab_size` | `12` | DNA character-token experiments in this repo use 12; padded to 16 by the model. |
| `n_enc_layer` | `1` | Keeps one encoder Mamba layer total. |
| `n_main_layer` | `1` | Keeps one chunk-level Transformer block. |
| `n_dec_layer` | `1` | Keeps one decoder Mamba layer. |
| `transformer_n_head` | `4` | Divides `d_model=16`; used by chunk Transformer and upsamplers. |
| `transformer_mlp_mult` | `2` | Small MLP size in the chunk Transformer. |
| `tokenizer_type` | `"default"` | Avoids STFT path and external tokenizer files. |
| `ssm_cfg.d_state` | `4` | Small Mamba state size. |
| `ssm_cfg.d_conv` | `2` | Small Mamba convolution width. |
| `ssm_cfg.expand` | `1` | Small Mamba expansion. |
| `ssm_cfg.dt_rank` | `"auto"` | Accepted by installed `mamba_ssm`. |
| `ssm_cfg.use_fast_path` | `false` | Smaller/debuggier path, but the installed `causal_conv1d` still requires CUDA tensors. |
| `rms_norm` | `false` | Uses PyTorch LayerNorm for Mamba blocks where configurable. |
| `fused_add_norm` | `false` | Avoids fused final norm path where configurable. |
| `residual_in_fp32` | `false` | Matches lightweight smoke behavior. |
| `pad_vocab_size_multiple` | `8` | Original model pads vocab size to this multiple. |
| `target_ratio_stage1` | `0.5` | Required for stage-1 ratio loss. |
| `target_ratio_stage2` | `0.5` | Required for stage-2 ratio loss. |
| `pad_token_id` | `-100` | Lets labels use the standard PyTorch ignore index. |

## Smallest validated tiny config

A validated config is stored in `configs/smoke_original_dnachunker.yaml`. Its core model settings are:

```python
HNetConfig(
    d_model=16,
    vocab_size=16,
    n_enc_layer=1,
    n_main_layer=1,
    n_dec_layer=1,
    transformer_n_head=4,
    tokenizer_type="default",
    fused_add_norm=False,
    rms_norm=False,
    residual_in_fp32=False,
    ssm_cfg={
        "d_state": 4,
        "d_conv": 2,
        "expand": 1,
        "use_fast_path": False,
    },
    target_ratio_stage1=0.5,
    target_ratio_stage2=0.5,
    pad_token_id=-100,
)
```

This config instantiates, runs a CUDA forward pass, and runs backward in the `tokenize` conda environment.

## Token/vocab assumptions

The repo configs use `vocab_size: 12` for character-token DNA experiments. The model itself only requires integer `input_ids` within `[0, vocab_size)`. Special-token handling in `HNetMixerModel.calculate_special_token_boundaries` treats `input_ids <= 6` as special tokens for stage-1 boundaries.

For a synthetic smoke test, a safe input uses token ids sampled from `7..11` when `vocab_size=12`, with labels in the same range and some labels set to `-100`.

## Sequence length constraints

No explicit fixed sequence-length constraint was found in `HNetConfig` for `tokenizer_type="default"`. The forward pass dynamically uses `input_ids.shape[1]`, predicted boundary masks, and padded chunk lengths.

`tokenizer_type="stft"` has implicit constraints from `torch.stft` parameters (`n_fft=256`, `hop_length=64`) and is not recommended for tiny smoke tests.

## Is Mamba mandatory?

Yes for the current code as written.

Reasons:

- `hnet_twostage/modeling_hnet.py` imports `Mamba` and `Block` from `mamba_ssm` at module import time.
- `HNetMixerModel` constructs encoder and decoder layers via `create_block`.
- `create_block` builds `Block` instances with `BiMambaWrapper`.
- There is no config flag that replaces these with a pure PyTorch fallback.

## Is flash attention mandatory?

Not for the immediate `hnet_twostage/modeling_hnet.py` import path observed after installing `transformers`; the current blocker is `mamba_ssm`.

Inside `modeling_hnet.py`, RMSNorm/layer norm helpers from `mamba_ssm.ops.triton` are attempted and then fall back to `None`. `CrossAttentionUpsampler` falls back to `nn.LayerNorm` if RMSNorm is unavailable.

However, full repo imports and other model wrappers may require `flash_attn` directly.

## Output fields

`HNetMixerModel.forward` returns:

- `hidden_states`
- `all_hidden_states`
- `ratio_loss`

`HNetTransformer.forward(return_dict=True)` wraps these in `HNetTransformerModelOutput`, including named `ratio_loss`.

`HNetTransformerForMaskedLM.forward(return_dict=False)` returns:

- `(loss, logits, hidden_states, ratio_loss)` when labels and hidden states are requested

`HNetTransformerForMaskedLM.forward(return_dict=True)` returns a standard `MaskedLMOutput` with:

- `loss`
- `logits`
- `hidden_states`

The final `MaskedLMOutput` does not expose `ratio_loss` as a named field. Boundary/router tensors are computed internally but not exposed by the public masked-LM return object.

## Is CUDA mandatory?

The code does not explicitly call `.cuda()` in `hnet_twostage/modeling_hnet.py`, but the required `mamba_ssm`, `causal-conv1d`, `flash-attn`, and Triton ecosystem is commonly CUDA/toolchain-sensitive. The original `environment.yml` includes CUDA packages and pinned CUDA-oriented dependencies.

In the installed `tokenize` environment, CPU forward fails with:

```text
RuntimeError: Expected x.is_cuda() to be true, but got false.
```

CUDA forward/backward succeeds after installing a conda C/C++ compiler and setting `CC` to `x86_64-conda-linux-gnu-cc` for Triton kernel compilation.
