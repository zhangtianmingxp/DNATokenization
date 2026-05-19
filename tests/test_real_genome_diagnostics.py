from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import torch


def test_real_genome_window_utilities_cpu() -> None:
    from src.diagnostics.real_genome_windows import open_fasta, parse_chroms, sample_random_windows

    fasta = open_fasta("tests/fixtures/tiny_genome.fa")
    chroms = parse_chroms("chrTiny1,chrTiny2", fasta)
    import random

    windows = sample_random_windows(
        fasta,
        chroms=chroms,
        seq_len=64,
        num_windows=2,
        rng=random.Random(1),
    )
    assert len(windows) == 2
    assert all(window.input_ids.shape[0] == 64 for window in windows)
    assert all("gc_frac" in window.metadata for window in windows)
    assert all(window.region_ids.shape[0] == 64 for window in windows)


def test_real_genome_tokenizer_diagnostics_cuda(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Original HNet/Mamba diagnostics require CUDA.")

    from scripts.diagnose_real_genome_tokenizer import main

    output_dir = tmp_path / "real_genome_tokenizer"
    previous_argv = sys.argv
    sys.argv = [
        "diagnose_real_genome_tokenizer.py",
        "--config",
        "configs/smoke_original_dnachunker.yaml",
        "--device",
        "cuda",
        "--fasta",
        "tests/fixtures/tiny_genome.fa",
        "--chroms",
        "chrTiny1,chrTiny2",
        "--seq-len",
        "64",
        "--num-windows",
        "2",
        "--batch-size",
        "1",
        "--seed",
        "1",
        "--output-dir",
        str(output_dir),
    ]
    try:
        assert main() == 0
    finally:
        sys.argv = previous_argv

    summary_path = output_dir / "summary.csv"
    jsonl_path = output_dir / "window_diagnostics.jsonl"
    assert summary_path.exists()
    assert jsonl_path.exists()
    rows = list(csv.DictReader(summary_path.open("r", newline="", encoding="utf-8")))
    assert len(rows) == 2
    required = {
        "chrom",
        "start",
        "end",
        "n_frac",
        "gc_frac",
        "stage1_boundary_density",
        "stage1_compression_ratio",
        "stage1_num_chunks",
    }
    assert required.issubset(rows[0].keys())
