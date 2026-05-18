"""A small DNACHUNKER-style dynamic chunker for masked nucleotide prediction."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .data import DNA_PAD_ID, DNA_VOCAB_SIZE


class MinimalDNAChunker(nn.Module):
    """Embed nucleotides, predict boundaries, pool chunks, process chunks, and dechunk."""

    def __init__(
        self,
        vocab_size: int = DNA_VOCAB_SIZE,
        d_model: int = 32,
        local_kernel_size: int = 5,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        dropout: float = 0.10,
        boundary_threshold: float = 0.50,
    ):
        super().__init__()
        if d_model % transformer_heads != 0:
            raise ValueError("d_model must be divisible by transformer_heads")

        self.boundary_threshold = boundary_threshold
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=DNA_PAD_ID)
        padding = local_kernel_size // 2
        self.local_encoder = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=local_kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=local_kernel_size, padding=padding),
            nn.GELU(),
        )
        self.boundary_router = nn.Linear(d_model, 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.chunk_model = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        embedded = self.embedding(input_ids)
        encoded = self.local_encoder(embedded.transpose(1, 2)).transpose(1, 2)

        boundary_prob = torch.sigmoid(self.boundary_router(encoded).squeeze(-1))
        hard_boundaries = (boundary_prob >= self.boundary_threshold).float()
        hard_boundaries[:, 0] = 1.0
        boundaries = hard_boundaries + boundary_prob - boundary_prob.detach()

        chunk_ids = torch.cumsum(hard_boundaries.long(), dim=1) - 1
        pooled_chunks, chunk_padding_mask = self._pool_chunks(encoded, chunk_ids, boundaries)
        chunk_states = self.chunk_model(pooled_chunks, src_key_padding_mask=chunk_padding_mask)
        nucleotide_states = self._dechunk(chunk_states, chunk_ids)
        logits = self.mlm_head(self.output_norm(nucleotide_states))

        chunk_counts = hard_boundaries.sum(dim=1).clamp_min(1.0)
        seq_len = input_ids.shape[1]
        metrics = {
            "avg_chunk_length": (seq_len / chunk_counts).mean().detach(),
            "compression_ratio": (seq_len / chunk_counts).mean().detach(),
            "boundary_density": ((chunk_counts - 1.0) / max(seq_len - 1, 1)).mean().detach(),
            "num_chunks": chunk_counts.mean().detach(),
            "boundary_prob_mean": boundary_prob.mean().detach(),
        }
        return logits, metrics

    @staticmethod
    def _pool_chunks(
        encoded: torch.Tensor, chunk_ids: torch.Tensor, boundaries: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, d_model = encoded.shape
        chunk_counts = chunk_ids.max(dim=1).values + 1
        max_chunks = int(chunk_counts.max().item())

        pooled = encoded.new_zeros(batch_size, max_chunks, d_model)
        counts = encoded.new_zeros(batch_size, max_chunks, 1)
        weights = boundaries.unsqueeze(-1).clamp_min(1e-3)

        for batch_index in range(batch_size):
            ids = chunk_ids[batch_index]
            pooled[batch_index].index_add_(0, ids, encoded[batch_index] * weights[batch_index])
            counts[batch_index].index_add_(0, ids, weights[batch_index])

        pooled = pooled / counts.clamp_min(1e-6)
        valid_chunks = torch.arange(max_chunks, device=encoded.device).unsqueeze(0) < chunk_counts.unsqueeze(1)
        return pooled, ~valid_chunks

    @staticmethod
    def _dechunk(chunk_states: torch.Tensor, chunk_ids: torch.Tensor) -> torch.Tensor:
        gathered = []
        for batch_index in range(chunk_ids.shape[0]):
            gathered.append(chunk_states[batch_index].index_select(0, chunk_ids[batch_index]))
        return torch.stack(gathered, dim=0)


def mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
