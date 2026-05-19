"""Generate a Markdown baseline report for synthetic training diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


REPORT_SECTIONS = [
    "Experiment summary",
    "Training summary",
    "Tokenizer before/after comparison",
    "Limitations",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(rows: list[dict[str, str]], columns: list[str], max_rows: int | None = None) -> str:
    if not rows:
        return "_No rows available._"
    rows = rows[:max_rows] if max_rows is not None else rows
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def count_ok(rows: Iterable[dict[str, str]]) -> int:
    return sum(row.get("status") == "ok" for row in rows)


def training_summary(train_rows: list[dict[str, str]]) -> dict[str, str]:
    if not train_rows:
        return {
            "steps": "0",
            "initial_loss": "",
            "final_loss": "",
            "initial_ratio_loss": "",
            "final_ratio_loss": "",
        }
    first = train_rows[0]
    last = train_rows[-1]
    return {
        "steps": last.get("step", str(len(train_rows))),
        "initial_loss": first.get("loss", ""),
        "final_loss": last.get("loss", ""),
        "initial_ratio_loss": first.get("ratio_loss", ""),
        "final_ratio_loss": last.get("ratio_loss", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", default="outputs/synthetic_training_diagnostics")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    before_rows = read_csv(experiment_dir / "before" / "summary.csv")
    after_rows = read_csv(experiment_dir / "after" / "summary.csv")
    train_rows = read_csv(experiment_dir / "train" / "train_log.csv")
    compare_by_mode = read_csv(experiment_dir / "compare" / "comparison_by_mode.csv")
    compare_by_length = read_csv(experiment_dir / "compare" / "comparison_by_length.csv")
    compare_by_mode_length = read_csv(experiment_dir / "compare" / "comparison_by_mode_length.csv")
    train = training_summary(train_rows)

    report_path = experiment_dir / "BASELINE_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    mode_columns = [
        "synthetic_mode",
        "before_n",
        "after_n",
        "stage1_compression_ratio_mean_before",
        "stage1_compression_ratio_mean_after",
        "stage1_compression_ratio_mean_delta",
        "stage1_post_merge_boundary_density_delta",
        "motif_break_rate_delta",
        "repeat_boundary_density_delta",
        "conserved_boundary_density_delta",
        "ratio_loss_delta",
    ]
    length_columns = [
        "seq_len",
        "before_n",
        "after_n",
        "stage1_compression_ratio_mean_before",
        "stage1_compression_ratio_mean_after",
        "stage1_compression_ratio_mean_delta",
        "stage1_post_merge_boundary_density_delta",
        "ratio_loss_delta",
    ]

    report = f"""# Synthetic Training Diagnostics Baseline Report

## Experiment summary

- Experiment directory: `{experiment_dir}`
- Before benchmark runs completed: `{count_ok(before_rows)}`
- After benchmark runs completed: `{count_ok(after_rows)}`
- After checkpoint: `{experiment_dir / "train" / "final.pt"}`
- Report type: engineering baseline for a tiny synthetic experiment

## Training summary

- Training steps: `{train["steps"]}`
- Initial loss: `{fmt(train["initial_loss"])}`
- Final loss: `{fmt(train["final_loss"])}`
- Initial ratio_loss: `{fmt(train["initial_ratio_loss"])}`
- Final ratio_loss: `{fmt(train["final_ratio_loss"])}`

## Tokenizer before/after comparison

### By synthetic mode

{markdown_table(compare_by_mode, mode_columns)}

### By sequence length

{markdown_table(compare_by_length, length_columns)}

### By synthetic mode and sequence length

{markdown_table(compare_by_mode_length, ["synthetic_mode", "seq_len", "before_n", "after_n", "stage1_compression_ratio_mean_delta", "stage1_post_merge_boundary_density_delta", "motif_break_rate_delta", "ratio_loss_delta"], max_rows=20)}

## Limitations

- This uses a tiny randomly initialized model and synthetic sequences.
- The training run is intentionally short and is not intended to reproduce biological behavior.
- Boundary changes are engineering diagnostics only, not scientific evidence.
- Stage-2 boundaries operate on stage-1 chunks and are not mapped back to original token regions here.

## Next recommended experiment

Run the same before/after diagnostic loop on a small real hg38 sample, still with the tiny model and short training, to check that the instrumentation behaves sensibly before adding any biological prior.
"""
    report_path.write_text(report, encoding="utf-8")
    print(f"baseline_report: {report_path}")
    print(f"before_runs: {count_ok(before_rows)}")
    print(f"after_runs: {count_ok(after_rows)}")
    print(f"training_steps: {train['steps']}")
    print(f"initial_loss: {fmt(train['initial_loss'])}")
    print(f"final_loss: {fmt(train['final_loss'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
