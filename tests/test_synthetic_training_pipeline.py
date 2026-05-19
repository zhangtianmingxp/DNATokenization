from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import torch


pytest.importorskip("mamba_ssm")
pytest.importorskip("causal_conv1d")


def run_main(main_func, argv: list[str]) -> None:
    previous_argv = sys.argv
    sys.argv = argv
    try:
        assert main_func() == 0
    finally:
        sys.argv = previous_argv


def test_synthetic_training_pipeline(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Original Mamba/causal_conv1d path requires CUDA for synthetic training.")

    from scripts.benchmark_original_tokenizer_diagnostics import main as benchmark_main
    from scripts.compare_tokenizer_before_after_training import main as compare_main
    from scripts.train_original_synthetic_mlm import main as train_main

    train_dir = tmp_path / "train"
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    compare_dir = tmp_path / "compare"

    run_main(
        benchmark_main,
        [
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
            "regions",
            "--output-dir",
            str(before_dir),
            "--max-runs",
            "1",
            "--no-backward",
        ],
    )

    run_main(
        train_main,
        [
            "train_original_synthetic_mlm.py",
            "--config",
            "configs/smoke_original_dnachunker.yaml",
            "--device",
            "cuda",
            "--output-dir",
            str(train_dir),
            "--steps",
            "2",
            "--batch-size",
            "1",
            "--seq-len",
            "64",
            "--synthetic-mode",
            "regions",
            "--seed",
            "1",
            "--lr",
            "1e-4",
            "--mask-prob",
            "0.15",
            "--log-every",
            "1",
            "--no-wandb",
        ],
    )

    assert (train_dir / "step_000000.pt").exists()
    assert (train_dir / "final.pt").exists()
    assert (train_dir / "train_log.csv").exists()

    run_main(
        benchmark_main,
        [
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
            "regions",
            "--output-dir",
            str(after_dir),
            "--checkpoint",
            str(train_dir / "final.pt"),
            "--max-runs",
            "1",
            "--no-backward",
        ],
    )

    rows = list(csv.DictReader((after_dir / "summary.csv").open("r", newline="", encoding="utf-8")))
    assert rows and rows[0]["status"] == "ok"

    run_main(
        compare_main,
        [
            "compare_tokenizer_before_after_training.py",
            "--before",
            str(before_dir / "summary.csv"),
            "--after",
            str(after_dir / "summary.csv"),
            "--output-dir",
            str(compare_dir),
        ],
    )

    assert (compare_dir / "comparison_by_mode.csv").exists()
