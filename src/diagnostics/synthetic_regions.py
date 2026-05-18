"""Synthetic DNA inputs with simple region labels for tokenizer diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

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

SYNTHETIC_MODES = [
    "random",
    "regions",
    "repeat_heavy",
    "motif_heavy",
    "conserved_heavy",
]


@dataclass
class SyntheticRegions:
    input_ids: torch.LongTensor
    region_ids: torch.LongTensor | None
    motif_mask: torch.BoolTensor | None
    motif_spans: list[list[tuple[int, int]]]
    metadata: dict[str, object] = field(default_factory=dict)


def _bases(device: torch.device, include_n: bool = False) -> torch.LongTensor:
    keys = ["A", "C", "G", "T", "N"] if include_n else ["A", "C", "G", "T"]
    return torch.tensor([DNA_IDS[k] for k in keys], dtype=torch.long, device=device)


def _random_bases(length: int, device: torch.device, include_n: bool = False) -> torch.LongTensor:
    bases = _bases(device, include_n=include_n)
    indices = torch.randint(0, bases.numel(), (length,), device=device)
    return bases[indices]


def _write_motif_block(
    input_ids: torch.LongTensor,
    region_ids: torch.LongTensor,
    motif_mask: torch.BoolTensor,
    batch_idx: int,
    start: int,
    end: int,
    motif: list[int],
    offset: int = 0,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        motif_start = pos
        for motif_pos in range(len(motif)):
            if pos >= end:
                break
            input_ids[batch_idx, pos] = motif[(motif_pos + offset) % len(motif)]
            region_ids[batch_idx, pos] = 2
            motif_mask[batch_idx, pos] = True
            pos += 1
        if pos > motif_start:
            spans.append((motif_start, pos))
        spacer = [DNA_IDS["C"], DNA_IDS["G"], DNA_IDS["T"]]
        for token in spacer:
            if pos >= end:
                break
            input_ids[batch_idx, pos] = token
            region_ids[batch_idx, pos] = 2
            pos += 1
    return spans


def make_random_input_ids(batch_size: int, seq_len: int, device: torch.device) -> SyntheticRegions:
    input_ids = torch.stack(
        [_random_bases(seq_len, device=device, include_n=True) for _ in range(batch_size)]
    )
    region_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    return SyntheticRegions(
        input_ids=input_ids,
        region_ids=region_ids,
        motif_mask=torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device),
        motif_spans=[[] for _ in range(batch_size)],
        metadata={"mode": "random", "description": "random DNA/N tokens, all neutral labels"},
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

    for batch_idx in range(batch_size):
        input_ids[batch_idx, :neutral_end] = _random_bases(neutral_end, device=device)

        for pos in range(neutral_end, min(repeat_end, seq_len)):
            input_ids[batch_idx, pos] = repeat[(pos - neutral_end) % len(repeat)]
            region_ids[batch_idx, pos] = 1

        spans = _write_motif_block(
            input_ids,
            region_ids,
            motif_mask,
            batch_idx=batch_idx,
            start=repeat_end,
            end=seq_len,
            motif=motif,
            offset=batch_idx % len(motif),
        )
        motif_spans.append(spans)

    return SyntheticRegions(
        input_ids=input_ids,
        region_ids=region_ids,
        motif_mask=motif_mask,
        motif_spans=motif_spans,
        metadata={"mode": "regions", "description": "balanced neutral/repeat/conserved motif-rich thirds"},
    )


def make_repeat_heavy_input_ids(batch_size: int, seq_len: int, device: torch.device) -> SyntheticRegions:
    input_ids = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    region_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    motif_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    repeat_patterns = [
        [DNA_IDS["A"]],
        [DNA_IDS["A"], DNA_IDS["C"]],
        [DNA_IDS["G"], DNA_IDS["T"], DNA_IDS["G"]],
    ]

    for batch_idx in range(batch_size):
        input_ids[batch_idx] = _random_bases(seq_len, device=device)
        block_start = seq_len // 8
        block_end = min(seq_len, block_start + (3 * seq_len) // 4)
        pattern = repeat_patterns[batch_idx % len(repeat_patterns)]
        for pos in range(block_start, block_end):
            input_ids[batch_idx, pos] = pattern[(pos - block_start) % len(pattern)]
            region_ids[batch_idx, pos] = 1

    return SyntheticRegions(
        input_ids=input_ids,
        region_ids=region_ids,
        motif_mask=motif_mask,
        motif_spans=[[] for _ in range(batch_size)],
        metadata={"mode": "repeat_heavy", "description": "large low-complexity repeat block"},
    )


def make_motif_heavy_input_ids(batch_size: int, seq_len: int, device: torch.device) -> SyntheticRegions:
    input_ids = torch.stack([_random_bases(seq_len, device=device) for _ in range(batch_size)])
    region_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    motif_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    motif = [DNA_IDS[ch] for ch in "GATTACA"]
    motif_spans: list[list[tuple[int, int]]] = []
    stride = max(len(motif) + 4, seq_len // 10)

    for batch_idx in range(batch_size):
        spans: list[tuple[int, int]] = []
        start = (batch_idx * 3) % max(stride, 1)
        while start + len(motif) <= seq_len:
            for motif_pos, token in enumerate(motif):
                pos = start + motif_pos
                input_ids[batch_idx, pos] = token
                region_ids[batch_idx, pos] = 2
                motif_mask[batch_idx, pos] = True
            spans.append((start, start + len(motif)))
            start += stride
        motif_spans.append(spans)

    return SyntheticRegions(
        input_ids=input_ids,
        region_ids=region_ids,
        motif_mask=motif_mask,
        motif_spans=motif_spans,
        metadata={"mode": "motif_heavy", "description": "many short motif spans embedded in neutral sequence"},
    )


def make_conserved_heavy_input_ids(batch_size: int, seq_len: int, device: torch.device) -> SyntheticRegions:
    input_ids = torch.stack([_random_bases(seq_len, device=device) for _ in range(batch_size)])
    region_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    motif_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    motif = [DNA_IDS[ch] for ch in "GATTACAGGCGT"]
    motif_spans: list[list[tuple[int, int]]] = []

    for batch_idx in range(batch_size):
        spans: list[tuple[int, int]] = []
        first_start = seq_len // 6
        first_end = min(seq_len, first_start + seq_len // 3)
        second_start = min(seq_len, (2 * seq_len) // 3)
        second_end = min(seq_len, second_start + seq_len // 5)
        for start, end in [(first_start, first_end), (second_start, second_end)]:
            if start >= end:
                continue
            spans.extend(
                _write_motif_block(
                    input_ids,
                    region_ids,
                    motif_mask,
                    batch_idx=batch_idx,
                    start=start,
                    end=end,
                    motif=motif,
                    offset=batch_idx % len(motif),
                )
            )
        motif_spans.append(spans)

    return SyntheticRegions(
        input_ids=input_ids,
        region_ids=region_ids,
        motif_mask=motif_mask,
        motif_spans=motif_spans,
        metadata={"mode": "conserved_heavy", "description": "longer conserved-like motif-rich blocks"},
    )


def make_synthetic_input_ids(
    mode: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> SyntheticRegions:
    if mode == "random":
        return make_random_input_ids(batch_size, seq_len, device)
    if mode == "regions":
        return make_region_input_ids(batch_size, seq_len, device)
    if mode == "repeat_heavy":
        return make_repeat_heavy_input_ids(batch_size, seq_len, device)
    if mode == "motif_heavy":
        return make_motif_heavy_input_ids(batch_size, seq_len, device)
    if mode == "conserved_heavy":
        return make_conserved_heavy_input_ids(batch_size, seq_len, device)
    raise ValueError(f"Unknown synthetic mode: {mode}")
