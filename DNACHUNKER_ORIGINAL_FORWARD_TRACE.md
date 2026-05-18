# Original HNet/DNAChunker Forward Trace

This trace follows `hnet_twostage/modeling_hnet.py` without changing model behavior. The smoke config in `configs/smoke_original_dnachunker.yaml` uses `batch_size=2`, `seq_len=64`, `d_model=16`, `n_enc_layer=1`, `n_main_layer=1`, `n_dec_layer=1`, and DNA-like token ids `7..11`.

## 1. `input_ids`

- File: `scripts/smoke_original_dnachunker.py`
- Shape in smoke config: `(2, 64)`
- Values: synthetic integer ids from `[7, 8, 9, 10, 11]`
- Active in smoke config: yes

The original code treats `input_ids <= 6` as special tokens in `HNetMixerModel.calculate_special_token_boundaries`, so the smoke config uses ids above 6 to exercise learned routing rather than only forced special-token boundaries.

## 2. Embedding

- File: `hnet_twostage/modeling_hnet.py`
- Class/function: `HNetEmbeddings.forward`
- Input shape: `(batch, seq_len)`
- Output shape: `(batch, seq_len, d_model)`, smoke `(2, 64, 16)`
- Active in smoke config: yes

`tokenizer_type: default` selects `HNetEmbeddings`. `tokenizer_type: stft` is not active in the smoke config.

## 3. Encoder / Mamba Mixer

- File: `hnet_twostage/modeling_hnet.py`
- Class/function: `HNetMixerModel.forward`, `create_block`, `BiMambaWrapper.forward`
- Input shape: `(batch, seq_len, d_model)`
- Output shape: `(batch, seq_len, d_model)`
- Active in smoke config: yes

The smoke config has `n_enc_layer=1`. The code splits encoder layers into `encoder1_layers` and `encoder2_layers`; with one encoder layer, stage 1 has zero Mamba blocks and stage 2 has one Mamba block operating after first downsampling.

## 4. RoutingModule Stage 1

- File: `hnet_twostage/modeling_hnet.py`
- Class/function: `RoutingModule.forward`
- Input shape: `(batch, seq_len, d_model)`
- Output shapes: probabilities `(batch, seq_len)`, hard boundaries `(batch, seq_len)`
- Active in smoke config: yes

The module computes adjacent-vector cosine similarities after learned query/key projections, converts them to boundary probabilities, then applies `StraightThroughEstimator`.

## 5. StraightThroughEstimator

- File: `hnet_twostage/modeling_hnet.py`
- Class/function: `StraightThroughEstimator.forward/backward`
- Input shape: `(batch, seq_len)`
- Output shape: `(batch, seq_len)`
- Active in smoke config: yes

Forward rounds probabilities to 0/1. Backward passes gradients through unchanged.

## 6. Downsampler Stage 1

- File: `hnet_twostage/modeling_hnet.py`
- Class/function: `Downsampler.forward`
- Input shapes: hidden states `(batch, seq_len, d_model)`, boundaries `(batch, seq_len)`
- Output shapes: padded chunks `(batch, max_chunks_stage1, d_model)`, lengths `(batch,)`
- Active in smoke config: yes

The downsampler selects hidden states at boundary positions and pads each batch item to the same chunk count.

## 7. RoutingModule + Downsampler Stage 2

- File: `hnet_twostage/modeling_hnet.py`
- Class/function: `RoutingModule.forward`, `Downsampler.forward`
- Input shape: stage-1 chunks `(batch, max_chunks_stage1, d_model)`
- Output shapes: stage-2 chunks `(batch, max_chunks_stage2, d_model)`, lengths `(batch,)`
- Active in smoke config: yes

Stage 2 repeats routing and downsampling on the coarse chunk sequence.

## 8. Chunk-Level Model

- File: `hnet_twostage/modeling_hnet.py`
- Class/function: `TransformerBlock.forward`, `RotarySelfAttention.forward`
- Input shape: `(batch, max_chunks_stage2, d_model)`
- Output shape: `(batch, max_chunks_stage2, d_model)`
- Active in smoke config: yes

The smoke config uses one chunk-level Transformer block with 4 heads and MLP multiplier 2.

## 9. CrossAttentionUpsampler Stage 2

- File: `hnet_twostage/modeling_hnet.py`
- Class/function: `CrossAttentionUpsampler.forward`
- Query shape: stage-1 encoded chunks `(batch, max_chunks_stage1, d_model)`
- Key/value shape: stage-2 chunk states `(batch, max_chunks_stage2, d_model)`
- Output shape: `(batch, max_chunks_stage1, d_model)`
- Active in smoke config: yes

This maps fine chunk states back to the coarser stage-1 chunk positions.

## 10. CrossAttentionUpsampler Stage 1

- File: `hnet_twostage/modeling_hnet.py`
- Class/function: `CrossAttentionUpsampler.forward`
- Query shape: full sequence encoder states `(batch, seq_len, d_model)`
- Key/value shape: stage-1 dechunked states `(batch, max_chunks_stage1, d_model)`
- Output shape: `(batch, seq_len, d_model)`
- Active in smoke config: yes

This maps chunk representations back to the full token sequence.

## 11. Decoder / Mamba Mixer

- File: `hnet_twostage/modeling_hnet.py`
- Class/function: `HNetMixerModel.forward`, `BiMambaWrapper.forward`
- Input shape: `(batch, seq_len, d_model)`
- Output shape: `(batch, seq_len, d_model)`
- Active in smoke config: yes

The smoke config uses one decoder Mamba block.

## 12. MLM Head / Logits

- File: `hnet_twostage/modeling_hnet.py`
- Class/function: `HNetTransformerForMaskedLM.forward`
- Input shape: `(batch, seq_len, d_model)`
- Output shape: `(batch, seq_len, vocab_size)`
- Active in smoke config: yes

`vocab_size: 12` is padded by `HNetTransformer.__init__` to 16 because `pad_vocab_size_multiple: 8`.

## Exposed Outputs

With `return_dict=False`, `HNetTransformerForMaskedLM.forward` returns:

- `loss`, if labels are provided
- `logits`
- `hidden_states`, when `output_hidden_states=True`
- `ratio_loss`

With `return_dict=True`, the final `MaskedLMOutput` exposes `loss`, `logits`, and `hidden_states`; it does not expose `ratio_loss` as a named field. Boundary tensors are computed internally but not returned by the original public forward API.

## Tokenizer Diagnostic Extraction Path

Tokenizer/chunker diagnostics are extracted in `scripts/diagnose_original_tokenizer.py` with PyTorch forward hooks. This keeps the original model behavior and output signatures unchanged.

Hook path:

- `HNetTransformerForMaskedLM`
- `model.caduceus`
- `model.caduceus.backbone`
- `routing_module_stage1`: captures `(p_stage1, b_stage1)` before special-token boundary merging
- `downsampler` first call: captures stage-1 downsampler input boundaries after special-token merging, plus `(x_s1, chunk_lengths_s1)`
- `routing_module_stage2`: captures `(p_stage2, b_stage2)`
- `downsampler` second call: captures `(x_s2, chunk_lengths_s2)`

The hook summaries expose boundary probabilities, discretized boundaries, post-merge boundary density, chunk counts, approximate average chunk length, compression ratio, ratio loss, and chunked hidden-state shapes.

For `--synthetic-mode regions`, the diagnostic script also creates token-level synthetic region labels. Stage-1 boundaries can then be summarized by neutral, repeat, and conserved/motif-rich regions. Stage-2 boundaries are not mapped back to original token regions because stage 2 operates on stage-1 chunks.
