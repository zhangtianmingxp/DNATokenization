"""Small FASTA/BED window utilities for real-genome tokenizer diagnostics."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch


DNA_IDS = {
    "A": 7,
    "C": 8,
    "G": 9,
    "T": 10,
    "N": 11,
}

REAL_REGION_LABELS = {
    0: "neutral",
    1: "N_rich",
    2: "GC_rich",
    3: "AT_rich",
    4: "repeat_like",
    5: "motif_like",
}

LABEL_TO_ID = {value: key for key, value in REAL_REGION_LABELS.items()}
DEFAULT_MOTIFS = ["TATA", "CCAAT", "CGCG", "GATA"]


@dataclass
class GenomeWindow:
    input_ids: torch.LongTensor
    region_ids: torch.LongTensor
    motif_spans: list[tuple[int, int]]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BedInterval:
    chrom: str
    start: int
    end: int
    strand: str = "+"


def require_pyfaidx():
    try:
        from pyfaidx import Fasta
    except ImportError as exc:
        raise ImportError("pyfaidx is required for real FASTA diagnostics. Install pyfaidx first.") from exc
    return Fasta


def open_fasta(fasta_path: str | Path):
    Fasta = require_pyfaidx()
    path = Path(fasta_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file does not exist: {path}")
    return Fasta(str(path), as_raw=True, sequence_always_upper=True)


def fasta_chrom_lengths(fasta) -> dict[str, int]:
    return {chrom: len(fasta[chrom]) for chrom in fasta.keys()}


def parse_chroms(chroms: str | None, fasta) -> list[str]:
    available = list(fasta.keys())
    if not chroms:
        return available
    selected = [chrom.strip() for chrom in chroms.split(",") if chrom.strip()]
    missing = [chrom for chrom in selected if chrom not in fasta]
    if missing:
        raise ValueError(f"Chromosomes not found in FASTA: {missing}")
    return selected


def read_bed(path: str | Path) -> list[BedInterval]:
    intervals = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            strand = fields[5] if len(fields) >= 6 and fields[5] in {"+", "-"} else "+"
            intervals.append(BedInterval(fields[0], int(fields[1]), int(fields[2]), strand))
    return intervals


def clean_sequence(seq: str) -> str:
    seq = seq.upper()
    return "".join(base if base in DNA_IDS else "N" for base in seq)


def reverse_complement(seq: str) -> str:
    complement = str.maketrans("ACGTN", "TGCAN")
    return seq.translate(complement)[::-1]


def sequence_to_ids(seq: str) -> torch.LongTensor:
    return torch.tensor([DNA_IDS.get(base, DNA_IDS["N"]) for base in clean_sequence(seq)], dtype=torch.long)


def sequence_fractions(seq: str) -> tuple[float, float]:
    seq = clean_sequence(seq)
    if not seq:
        return 0.0, 0.0
    n_frac = seq.count("N") / len(seq)
    gc_frac = (seq.count("G") + seq.count("C")) / len(seq)
    return n_frac, gc_frac


def local_fraction(seq: str, pos: int, chars: set[str], radius: int = 8) -> float:
    start = max(0, pos - radius)
    end = min(len(seq), pos + radius + 1)
    if start >= end:
        return 0.0
    window = seq[start:end]
    return sum(base in chars for base in window) / len(window)


def mark_repeat_like(seq: str, labels: list[int]) -> None:
    n = len(seq)
    for i in range(n):
        if i + 5 <= n and len(set(seq[i : i + 5])) == 1 and seq[i] != "N":
            for pos in range(i, i + 5):
                labels[pos] = LABEL_TO_ID["repeat_like"]
        if i + 6 <= n:
            pattern = seq[i : i + 2]
            if "N" not in pattern and pattern * 3 == seq[i : i + 6]:
                for pos in range(i, i + 6):
                    labels[pos] = LABEL_TO_ID["repeat_like"]


def mark_motifs(seq: str, labels: list[int], motifs: Iterable[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for motif in motifs:
        motif = clean_sequence(motif)
        if not motif:
            continue
        start = 0
        while True:
            pos = seq.find(motif, start)
            if pos < 0:
                break
            end = pos + len(motif)
            spans.append((pos, end))
            for idx in range(pos, end):
                labels[idx] = LABEL_TO_ID["motif_like"]
            start = pos + 1
    return spans


def label_sequence_regions(
    seq: str,
    motifs: Iterable[str] = DEFAULT_MOTIFS,
    n_rich_threshold: float = 0.30,
    gc_rich_threshold: float = 0.65,
    at_rich_threshold: float = 0.75,
) -> tuple[torch.LongTensor, list[tuple[int, int]]]:
    seq = clean_sequence(seq)
    labels = [LABEL_TO_ID["neutral"]] * len(seq)
    for pos, base in enumerate(seq):
        if base == "N" or local_fraction(seq, pos, {"N"}) >= n_rich_threshold:
            labels[pos] = LABEL_TO_ID["N_rich"]
        elif local_fraction(seq, pos, {"G", "C"}) >= gc_rich_threshold:
            labels[pos] = LABEL_TO_ID["GC_rich"]
        elif local_fraction(seq, pos, {"A", "T"}) >= at_rich_threshold:
            labels[pos] = LABEL_TO_ID["AT_rich"]

    mark_repeat_like(seq, labels)
    motif_spans = mark_motifs(seq, labels, motifs)
    return torch.tensor(labels, dtype=torch.long), motif_spans


def extract_window(
    fasta,
    chrom: str,
    start: int,
    end: int,
    strand: str = "+",
    motifs: Iterable[str] = DEFAULT_MOTIFS,
) -> GenomeWindow:
    chrom_len = len(fasta[chrom])
    start = max(0, min(start, chrom_len))
    end = max(start, min(end, chrom_len))
    seq = clean_sequence(str(fasta[chrom][start:end]))
    if strand == "-":
        seq = reverse_complement(seq)
    input_ids = sequence_to_ids(seq)
    region_ids, motif_spans = label_sequence_regions(seq, motifs=motifs)
    n_frac, gc_frac = sequence_fractions(seq)
    metadata = {
        "chrom": chrom,
        "start": start,
        "end": end,
        "strand": strand,
        "seq_len": len(seq),
        "n_frac": n_frac,
        "gc_frac": gc_frac,
    }
    return GenomeWindow(input_ids=input_ids, region_ids=region_ids, motif_spans=motif_spans, metadata=metadata)


def sample_random_windows(
    fasta,
    chroms: list[str],
    seq_len: int,
    num_windows: int,
    rng: random.Random,
    motifs: Iterable[str] = DEFAULT_MOTIFS,
    min_n_frac: float | None = None,
    max_n_frac: float | None = None,
    max_attempts_per_window: int = 100,
) -> list[GenomeWindow]:
    windows: list[GenomeWindow] = []
    lengths = fasta_chrom_lengths(fasta)
    valid_chroms = [chrom for chrom in chroms if lengths[chrom] >= seq_len]
    if not valid_chroms:
        raise ValueError(f"No selected chromosome is at least seq_len={seq_len}.")

    for _ in range(num_windows):
        last_window = None
        for _attempt in range(max_attempts_per_window):
            chrom = rng.choice(valid_chroms)
            start = rng.randint(0, lengths[chrom] - seq_len)
            window = extract_window(fasta, chrom, start, start + seq_len, motifs=motifs)
            last_window = window
            n_frac = float(window.metadata["n_frac"])
            if min_n_frac is not None and n_frac < min_n_frac:
                continue
            if max_n_frac is not None and n_frac > max_n_frac:
                continue
            windows.append(window)
            break
        else:
            if last_window is not None:
                windows.append(last_window)
    return windows


def sample_bed_windows(
    fasta,
    bed_path: str | Path,
    seq_len: int,
    num_windows: int,
    rng: random.Random,
    chroms: list[str] | None = None,
    motifs: Iterable[str] = DEFAULT_MOTIFS,
) -> list[GenomeWindow]:
    intervals = read_bed(bed_path)
    if chroms:
        chrom_set = set(chroms)
        intervals = [interval for interval in intervals if interval.chrom in chrom_set]
    fasta_keys = set(fasta.keys())
    intervals = [interval for interval in intervals if interval.chrom in fasta_keys and interval.end > interval.start]
    if not intervals:
        raise ValueError("No usable BED intervals were found for the selected chromosomes.")

    windows = []
    for _ in range(num_windows):
        interval = rng.choice(intervals)
        span = interval.end - interval.start
        if span >= seq_len:
            start = rng.randint(interval.start, interval.end - seq_len)
        else:
            center = (interval.start + interval.end) // 2
            start = center - seq_len // 2
        chrom_len = len(fasta[interval.chrom])
        if chrom_len >= seq_len:
            start = max(0, min(start, chrom_len - seq_len))
        windows.append(
            extract_window(
                fasta,
                interval.chrom,
                start,
                start + seq_len,
                strand=interval.strand,
                motifs=motifs,
            )
        )
    return windows
