# DNAChunker Tokenizer/Chunker Diagnostics Notes

These notes describe the currently available non-invasive diagnostics for the original author two-stage HNet/DNAChunker model. No biological prior logic is implemented here, and the original model outputs are unchanged by default.

## Extraction Method

Diagnostics use PyTorch forward hooks in `scripts/diagnose_original_tokenizer.py`.

Hooked modules:

- `model.caduceus.backbone.routing_module_stage1`
- `model.caduceus.backbone.routing_module_stage2`
- `model.caduceus.backbone.downsampler`

The original model code in `hnet_twostage/modeling_hnet.py` is not modified. Hooks read tensors that are already produced during the forward pass.

## Internal Variables

| File | Class/function | Variable | Shape | Differentiable | Returned by public model output |
|---|---|---|---|---|---|
| `hnet_twostage/modeling_hnet.py` | `RoutingModule.forward` | `q_aligned`, `k_aligned` | `(batch, seq_len - 1, d_model)` | yes | no |
| `hnet_twostage/modeling_hnet.py` | `RoutingModule.forward` | `similarity` | `(batch, seq_len - 1)` | yes | no |
| `hnet_twostage/modeling_hnet.py` | `RoutingModule.forward` | `p_values` | `(batch, seq_len - 1)` | yes | no |
| `hnet_twostage/modeling_hnet.py` | `RoutingModule.forward` | `p` | `(batch, seq_len)` | yes | no, but captured by hook |
| `hnet_twostage/modeling_hnet.py` | `StraightThroughEstimator.forward` | returned rounded tensor | `(batch, seq_len)` | straight-through gradient in backward | no, but captured by hook as `b` |
| `hnet_twostage/modeling_hnet.py` | `HNetMixerModel.calculate_special_token_boundaries` | `special_token_boundaries` | `(batch, seq_len)` | no | no |
| `hnet_twostage/modeling_hnet.py` | `HNetMixerModel.forward` | `b_stage1` after merge | `(batch, seq_len)` | yes through STE branch before bool/float conversion is limited by bool conversion | no, but captured as `Downsampler` input |
| `hnet_twostage/modeling_hnet.py` | `Downsampler.forward` | `mask` | `(batch, seq_len)` | no | no |
| `hnet_twostage/modeling_hnet.py` | `Downsampler.forward` | `padded_chunks` | `(batch, max_chunks, d_model)` | yes with respect to selected hidden states | no, but shape is captured by hook |
| `hnet_twostage/modeling_hnet.py` | `Downsampler.forward` | `chunk_lengths` | `(batch,)` | no | hidden-state list contains stage-1 lengths; hook captures both downsampler calls |
| `hnet_twostage/modeling_hnet.py` | `HNetMixerModel.forward` | `ratio_loss_stage1`, `ratio_loss_stage2`, `ratio_loss` | scalar after mean | yes through `p_stage*`/`b_stage*` path where applicable | `ratio_loss` is returned by `HNetTransformer`; masked-LM exposes it only in tuple mode |
| `hnet_twostage/modeling_hnet.py` | `HNetMixerModel.forward` | `all_hidden_states.append((z_dechunked_s1, chunk_lengths_s1))` | tensor `(batch, max_chunks_stage1, d_model)`, lengths `(batch,)` | tensor yes, lengths no | yes when `output_hidden_states=True` |

## Available Diagnostics

`scripts/diagnose_original_tokenizer.py` now reports scalar summaries for each chunking stage:

- boundary probability shape, mean, std, min, max
- whether boundary probabilities require gradients
- pre-merge discretized boundary density from `RoutingModule`
- post-merge boundary density from `Downsampler` input
- number of chunks per sample
- average chunk length, computed as source sequence length divided by average number of chunks
- compression ratio, same scalar interpretation for this boundary-selected downsampler
- min/max of the original `chunk_lengths` tensor, which is the number of chunks per sample
- chunked hidden-state shape
- ratio loss from the original model output

For `--synthetic-mode regions`, stage-1 token-level boundary tensors also support:

- boundary density in neutral regions
- boundary density in repeat regions
- boundary density in conserved/motif-rich regions
- motif break rate, defined as the fraction of synthetic motif spans that contain an internal boundary

## Currently Unavailable Without Model-Code Changes

- Boundary logits are not available because the original `RoutingModule` computes probabilities directly from cosine similarity; no separate logits tensor exists.
- Public masked-LM `return_dict=True` output does not include `ratio_loss` as a named field because it returns a standard `MaskedLMOutput`.
- Public masked-LM outputs do not include raw `p_stage1`, `b_stage1`, `p_stage2`, or `b_stage2`.
- Stage-2 region-wise token statistics are not directly meaningful because stage 2 operates on stage-1 chunks rather than original token positions.

## Commands

Random synthetic input:

```bash
conda run -n tokenize python scripts/diagnose_original_tokenizer.py --config configs/smoke_original_dnachunker.yaml --device cuda --batch-size 2 --seq-len 64 --synthetic-mode random
```

Region-labeled synthetic input with JSON output:

```bash
conda run -n tokenize python scripts/diagnose_original_tokenizer.py --config configs/smoke_original_dnachunker.yaml --device cuda --batch-size 2 --seq-len 128 --synthetic-mode regions --save-json outputs/tokenizer_diag_regions.json
```

Test:

```bash
conda run -n tokenize python -m pytest -q tests/test_original_tokenizer_diagnostics.py
```
