"""Synthetic DNA inputs with simple region labels for tokenizer diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


DNA_IDS = {
    "A": 7,
    "C": 8,
    "G": 9,
    "T": 10,
    "N": 11,
}

REGION_NAMES = {
    0: "neutral",
    1: "repeat",
    2: "conserved_motif_rich",
}


@dataclass
class SyntheticRegions:
    input_ids: torch.LongTensor
    region_ids: torch.LongTensor | None
    motif_mask: torch.BoolTensor | None
    motif_spans: list[list[tuple[int, int]]]


def make_random_input_ids(batch_size: int, seq_len: int, device: torch.device) -> SyntheticRegions:
    token_ids = torch.tensor([DNA_IDS[k] for k in ["A", "C", "G", "T", "N"]], dtype=torch.long, device=device)
    indices = torch.randint(0, token_ids.numel(), (batch_size, seq_len), device=device)
    return SyntheticRegions(
        input_ids=token_ids[indices],
        region_ids=None,
        motif_mask=None,
        motif_spans=[[] for _ in range(batch_size)],
    )


def make_region_input_ids(batch_size: int, seq_len: int, device: torch.device) -> SyntheticRegions:
    input_ids = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    region_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    motif_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    motif_spans: list[list[tuple[int, int]]] = []

    neutral_end = max(seq_len // 3, 1)
    repeat_end = max((2 * seq_len) // 3, neutral_end + 1)
    motif = [DNA_IDS[ch] for ch in "GATTACA"]
    repeat = [DNA_IDS["A"], DNA_IDS["C"]]
    bases = torch.tensor([DNA_IDS[k] for k in ["A", "C", "G", "T"]], dtype=torch.long, device=device)

    for batch_idx in range(batch_size):
        neutral = bases[torch.randint(0, bases.numel(), (neutral_end,), device=device)]
        input_ids[batch_idx, :neutral_end] = neutral

        for pos in range(neutral_end, min(repeat_end, seq_len)):
            input_ids[batch_idx, pos] = repeat[(pos - neutral_end) % len(repeat)]
            region_ids[batch_idx, pos] = 1

        spans: list[tuple[int, int]] = []
        pos = repeat_end
        motif_offset = batch_idx % max(len(motif), 1)
        while pos < seq_len:
            for motif_pos in range(len(motif)):
                if pos >= seq_len:
                    break
                token = motif[(motif_pos + motif_offset) % len(motif)]
                input_ids[batch_idx, pos] = token
                region_ids[batch_idx, pos] = 2
                motif_mask[batch_idx, pos] = True
                pos += 1
            spans.append((max(repeat_end, pos - len(motif)), pos))
            for token in [DNA_IDS["C"], DNA_IDS["G"], DNA_IDS["T"]]:
                if pos >= seq_len:
                    break
                input_ids[batch_idx, pos] = token
                region_ids[batch_idx, pos] = 2
                pos += 1
        motif_spans.append(spans)

    return SyntheticRegions(
        input_ids=input_ids,
        region_ids=region_ids,
        motif_mask=motif_mask,
        motif_spans=motif_spans,
    )
