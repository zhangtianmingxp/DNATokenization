from __future__ import annotations

import csv
import sys
from pathlib import Path


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_synthetic_training_report_generates_markdown(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    summary_row = {
        "status": "ok",
        "synthetic_mode": "regions",
        "seq_len": "64",
        "stage1_compression_ratio_mean": "2.0",
    }
    write_csv(experiment_dir / "before" / "summary.csv", [summary_row])
    write_csv(experiment_dir / "after" / "summary.csv", [summary_row])
    write_csv(
        experiment_dir / "train" / "train_log.csv",
        [
            {"step": "1", "loss": "4.0", "ratio_loss": "2.0", "lr": "0.0001"},
            {"step": "2", "loss": "3.5", "ratio_loss": "1.8", "lr": "0.0001"},
        ],
    )
    comparison_row = {
        "synthetic_mode": "regions",
        "seq_len": "64",
        "before_n": "1",
        "after_n": "1",
        "stage1_compression_ratio_mean_before": "2.0",
        "stage1_compression_ratio_mean_after": "2.1",
        "stage1_compression_ratio_mean_delta": "0.1",
        "stage1_post_merge_boundary_density_delta": "0.01",
        "motif_break_rate_delta": "0.0",
        "repeat_boundary_density_delta": "0.0",
        "conserved_boundary_density_delta": "0.0",
        "ratio_loss_delta": "-0.2",
    }
    write_csv(experiment_dir / "compare" / "comparison_by_mode.csv", [comparison_row])
    write_csv(experiment_dir / "compare" / "comparison_by_length.csv", [comparison_row])
    write_csv(experiment_dir / "compare" / "comparison_by_mode_length.csv", [comparison_row])

    from scripts.report_synthetic_training_diagnostics import main

    previous_argv = sys.argv
    sys.argv = [
        "report_synthetic_training_diagnostics.py",
        "--experiment-dir",
        str(experiment_dir),
    ]
    try:
        assert main() == 0
    finally:
        sys.argv = previous_argv

    report_path = experiment_dir / "BASELINE_REPORT.md"
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "Experiment summary" in text
    assert "Training summary" in text
    assert "Tokenizer before/after comparison" in text
    assert "Limitations" in text
