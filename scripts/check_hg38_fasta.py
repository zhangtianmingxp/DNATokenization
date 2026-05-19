"""Inspect a local hg38-style FASTA file for tokenizer diagnostics."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.diagnostics.real_genome_windows import open_fasta, sequence_fractions  # noqa: E402


def suggest_chroms(names: list[str]) -> list[str]:
    if "chr1" in names and "chr2" in names:
        return ["chr1", "chr2"]
    if "1" in names and "2" in names:
        return ["1", "2"]
    autosomes_chr = [f"chr{i}" for i in range(1, 23) if f"chr{i}" in names]
    if len(autosomes_chr) >= 2:
        return autosomes_chr[:2]
    autosomes_plain = [str(i) for i in range(1, 23) if str(i) in names]
    if len(autosomes_plain) >= 2:
        return autosomes_plain[:2]
    return names[:2]


def estimate_n_fraction(fasta, chroms: list[str], sample_windows: int = 20, window_size: int = 1000) -> float:
    rng = random.Random(1)
    values = []
    valid = [chrom for chrom in chroms if len(fasta[chrom]) >= window_size]
    for _ in range(min(sample_windows, max(len(valid), 1) * 2)):
        if not valid:
            break
        chrom = rng.choice(valid)
        start = rng.randint(0, len(fasta[chrom]) - window_size)
        seq = str(fasta[chrom][start : start + window_size])
        n_frac, _ = sequence_fractions(seq)
        values.append(n_frac)
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", default="data/hg38.fa")
    parser.add_argument("--max-chroms", type=int, default=30)
    parser.add_argument("--suggest-chroms-only", action="store_true")
    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    fasta = open_fasta(fasta_path)
    names = list(fasta.keys())
    suggested = suggest_chroms(names)
    if args.suggest_chroms_only:
        print(",".join(suggested))
        return 0

    print(f"fasta_path: {fasta_path}")
    print(f"exists: {fasta_path.exists()}")
    print(f"num_sequences: {len(names)}")
    print(f"first_20_chrom_names: {names[:20]}")
    print(f"name_style: {'chr-prefixed' if any(name.startswith('chr') for name in names[:30]) else 'plain-or-other'}")
    print(f"chr1_available: {'chr1' in names}")
    print(f"chr2_available: {'chr2' in names}")
    print(f"1_available: {'1' in names}")
    print(f"2_available: {'2' in names}")
    print("chrom_lengths:")
    for name in names[: args.max_chroms]:
        print(f"  {name}: {len(fasta[name])}")
    major = [name for name in ["chr1", "chr2", "chr3", "1", "2", "3"] if name in names]
    sample_chroms = major or names[: min(len(names), args.max_chroms)]
    print(f"estimated_sample_n_fraction: {estimate_n_fraction(fasta, sample_chroms):.6f}")
    print(f"suggested_chroms: {','.join(suggested)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
