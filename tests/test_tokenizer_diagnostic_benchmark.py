from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch


pytest.importorskip("mamba_ssm")
pytest.importorskip("causal_conv1d")


def test_tokenizer_diagnostic_benchmark_smoke(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Original Mamba/causal_conv1d path requires CUDA for diagnostics.")

    from scripts.benchmark_original_tokenizer_diagnostics import main

    output_dir = tmp_path / "tokenizer_benchmark"
    old_argv = [
        "benchmark_original_tokenizer_diagnostics.py",
        "--config",
        "configs/smoke_original_dnachunker.yaml",
        "--device",
        "cuda",
        "--batch-size",
        "1",
        "--seq-lens",
        "64",
        "--seeds",
        "1",
        "--synthetic-modes",
        "random,regions",
        "--output-dir",
        str(output_dir),
        "--max-runs",
        "2",
        "--no-backward",
    ]

    import sys

    previous_argv = sys.argv
    sys.argv = old_argv
    try:
        assert main() == 0
    finally:
        sys.argv = previous_argv

    summary_path = output_dir / "summary.csv"
    assert summary_path.exists()
    rows = list(csv.DictReader(summary_path.open("r", newline="", encoding="utf-8")))
    assert len(rows) == 2
    assert all(row["status"] == "ok" for row in rows)
    required_columns = {
        "synthetic_mode",
        "seq_len",
        "seed",
        "stage1_compression_ratio_mean",
        "stage1_post_merge_boundary_density",
        "stage1_boundary_prob_mean",
        "json_path",
    }
    assert required_columns.issubset(rows[0].keys())
