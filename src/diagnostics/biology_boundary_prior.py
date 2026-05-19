"""Heuristic biology-guided boundary prior for tokenizer diagnostics.

This module does not modify the original HNet model. It only applies an
offline logit adjustment to captured boundary probabilities so diagnostics can
measure what a simple biological prior would change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from src.diagnostics.real_genome_windows import DNA_IDS, REAL_REGION_LABELS


ID_TO_DNA = {value: key for key, value in DNA_IDS.items()}
DEFAULT_LABEL_LOGIT_BIAS = {
    "neutral": 0.0,
    "N_rich": -1.0,
    "GC_rich": 0.25,
    "AT_rich": 0.0,
    "repeat_like": -0.5,
    "motif_like": -0.75,
}


@dataclass(frozen=True)
class BiologyPriorConfig:
    """Parameters for offline boundary probability adjustment."""

    label_logit_bias: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LABEL_LOGIT_BIAS))
    motif_interior_logit_penalty: float = 2.0
    motif_edge_logit_bonus: float = 0.5
    gc_shift_logit_bonus: float = 1.0
    gc_shift_threshold: float = 0.25
    gc_window_radius: int = 8
    threshold: float = 0.5
    probability_eps: float = 1.0e-4


@dataclass
class PriorAdjustment:
    adjusted_probabilities: torch.Tensor
    adjusted_boundaries: torch.Tensor
    logit_adjustment: torch.Tensor
    region_logit_bias: torch.Tensor
    gc_shift_score: torch.Tensor
    motif_interior_mask: torch.Tensor
    motif_edge_mask: torch.Tensor


def ids_to_sequence(input_ids: torch.Tensor) -> str:
    """Convert DNA token IDs used by the original project back to bases."""

    values = input_ids.detach().cpu().tolist()
    return "".join(ID_TO_DNA.get(int(value), "N") for value in values)


def gc_fraction(seq: str) -> float:
    clean = [base for base in seq.upper() if base in {"A", "C", "G", "T"}]
    if not clean:
        return 0.0
    return sum(base in {"G", "C"} for base in clean) / len(clean)


def gc_shift_scores(
    seq: str,
    radius: int = 8,
    threshold: float = 0.25,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Score candidate boundaries by local GC fraction contrast."""

    seq = seq.upper()
    scores = torch.zeros(len(seq), dtype=torch.float32, device=device)
    if len(seq) <= 1:
        return scores
    radius = max(1, int(radius))
    for pos in range(1, len(seq)):
        left = seq[max(0, pos - radius) : pos]
        right = seq[pos : min(len(seq), pos + radius)]
        if not left or not right:
            continue
        delta = abs(gc_fraction(left) - gc_fraction(right))
        if delta >= threshold:
            scores[pos] = float(delta)
    return scores


def motif_masks(
    length: int,
    motif_spans: Iterable[tuple[int, int]],
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masks for motif-internal boundary positions and motif starts."""

    interior = torch.zeros(length, dtype=torch.bool, device=device)
    edge = torch.zeros(length, dtype=torch.bool, device=device)
    for start, end in motif_spans:
        start = max(0, min(int(start), length))
        end = max(start, min(int(end), length))
        if start < length:
            edge[start] = True
        if end - start > 1:
            interior[start + 1 : end] = True
    return interior, edge


def region_logit_biases(
    region_ids: torch.Tensor,
    config: BiologyPriorConfig,
    device: torch.device | None = None,
) -> torch.Tensor:
    values = region_ids.detach().cpu().tolist()
    biases = torch.zeros(len(values), dtype=torch.float32, device=device)
    for idx, region_id in enumerate(values):
        name = REAL_REGION_LABELS.get(int(region_id), "neutral")
        biases[idx] = float(config.label_logit_bias.get(name, 0.0))
    return biases


def safe_logit(probabilities: torch.Tensor, eps: float) -> torch.Tensor:
    probs = probabilities.detach().float().clamp(min=eps, max=1.0 - eps)
    return torch.logit(probs)


def apply_biology_boundary_prior(
    probabilities: torch.Tensor,
    input_ids: torch.Tensor,
    region_ids: torch.Tensor,
    motif_spans: Iterable[tuple[int, int]],
    config: BiologyPriorConfig | None = None,
) -> PriorAdjustment:
    """Apply the heuristic prior to one sequence of boundary probabilities."""

    config = config or BiologyPriorConfig()
    device = probabilities.device
    seq = ids_to_sequence(input_ids)
    logits = safe_logit(probabilities, config.probability_eps)
    region_bias = region_logit_biases(region_ids, config, device=device)
    gc_scores = gc_shift_scores(
        seq,
        radius=config.gc_window_radius,
        threshold=config.gc_shift_threshold,
        device=device,
    )
    motif_interior, motif_edge = motif_masks(len(seq), motif_spans, device=device)

    adjustment = region_bias + config.gc_shift_logit_bonus * gc_scores
    adjustment = adjustment.clone()
    adjustment[motif_interior] -= float(config.motif_interior_logit_penalty)
    adjustment[motif_edge] += float(config.motif_edge_logit_bonus)

    adjusted_probabilities = torch.sigmoid(logits + adjustment)
    adjusted_boundaries = (adjusted_probabilities >= float(config.threshold)).float()
    if adjusted_boundaries.numel() > 0:
        adjusted_boundaries[0] = 1.0
        adjusted_probabilities[0] = torch.maximum(
            adjusted_probabilities[0],
            torch.tensor(float(config.threshold), device=device, dtype=adjusted_probabilities.dtype),
        )

    return PriorAdjustment(
        adjusted_probabilities=adjusted_probabilities,
        adjusted_boundaries=adjusted_boundaries,
        logit_adjustment=adjustment,
        region_logit_bias=region_bias,
        gc_shift_score=gc_scores,
        motif_interior_mask=motif_interior,
        motif_edge_mask=motif_edge,
    )


def motif_break_rate(boundaries: torch.Tensor, motif_spans: Iterable[tuple[int, int]]) -> float | None:
    values = boundaries.detach().bool().cpu()
    usable = 0
    broken = 0
    for start, end in motif_spans:
        if end - start <= 1:
            continue
        usable += 1
        if values[start + 1 : end].any().item():
            broken += 1
    if usable == 0:
        return None
    return broken / usable


def boundary_density_by_label(boundaries: torch.Tensor, region_ids: torch.Tensor) -> dict[str, float]:
    values = boundaries.detach().float().cpu()
    labels = region_ids.detach().cpu()
    densities: dict[str, float] = {}
    for region_id, name in REAL_REGION_LABELS.items():
        mask = labels == int(region_id)
        if mask.any():
            densities[name] = float(values[mask].mean().item())
    return densities


def compression_ratio(boundaries: torch.Tensor) -> float:
    length = int(boundaries.numel())
    chunks = int(boundaries.detach().float().sum().cpu().item())
    return length / max(chunks, 1)


def boundary_flip_metrics(base_boundaries: torch.Tensor, adjusted_boundaries: torch.Tensor) -> dict[str, float]:
    base = base_boundaries.detach().bool().cpu()
    adjusted = adjusted_boundaries.detach().bool().cpu()
    changed = base != adjusted
    added = adjusted & ~base
    removed = base & ~adjusted
    denom = max(int(base.numel()), 1)
    return {
        "boundary_flip_rate": float(changed.float().mean().item()),
        "boundary_added_density": float(added.float().sum().item() / denom),
        "boundary_removed_density": float(removed.float().sum().item() / denom),
    }


def config_to_dict(config: BiologyPriorConfig) -> dict[str, object]:
    return {
        "label_logit_bias": dict(config.label_logit_bias),
        "motif_interior_logit_penalty": config.motif_interior_logit_penalty,
        "motif_edge_logit_bonus": config.motif_edge_logit_bonus,
        "gc_shift_logit_bonus": config.gc_shift_logit_bonus,
        "gc_shift_threshold": config.gc_shift_threshold,
        "gc_window_radius": config.gc_window_radius,
        "threshold": config.threshold,
        "probability_eps": config.probability_eps,
    }
