"""Generate a Markdown report for hg38 tokenizer baseline diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
    except (ValueError, TypeError):
        return str(value)


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/hg38_tokenizer_baseline")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    config_path = input_dir / "run_config.json"
    run_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    by_seq = read_csv(input_dir / "hg38_summary_by_seq_len.csv")
    by_region = read_csv(input_dir / "hg38_summary_by_region_label.csv")
    all_rows = read_csv(input_dir / "hg38_summary_all.csv")
    seq_lens = ",".join(sorted({row.get("seq_len_group", "") for row in all_rows if row.get("seq_len_group")}))
    windows = len(all_rows)

    report = f"""# HG38 Tokenizer Baseline Diagnostics

## Experiment summary

- FASTA path: `{run_config.get("fasta", "unknown")}`
- Chromosomes used: `{run_config.get("chroms", "unknown")}`
- Sequence lengths: `{seq_lens or run_config.get("seq_lens", "unknown")}`
- Sampled windows: `{windows}`
- Windows per sequence length target: `{run_config.get("num_windows", "unknown")}`
- Model config: `{run_config.get("config", "unknown")}`
- Checkpoint: `{run_config.get("checkpoint") or "none"}`

## Metrics by sequence length

{markdown_table(by_seq, ["seq_len_group", "n", "stage1_boundary_density", "stage1_compression_ratio", "ratio_loss", "gc_frac", "n_frac", "motif_like_break_rate"])}

## Metrics by heuristic region label

{markdown_table(by_region, ["region_label", "n", "boundary_density"])}

## Limitations

- This is a tiny/random model baseline unless a checkpoint was explicitly provided.
- Region labels are heuristic diagnostics only, not biological ground truth.
- No biological prior is used.
- No training on hg38 is performed.
- These metrics are engineering diagnostics and should not be interpreted as biological conclusions.

## Next safest step

Run the same baseline with a short synthetic-trained checkpoint, then compare random-tiny versus short-trained behavior on the same sampled hg38 windows before adding any biological prior.
"""
    report_path = input_dir / "HG38_BASELINE_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"hg38_baseline_report: {report_path}")
    print(f"windows: {windows}")
    print(f"seq_lens: {seq_lens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
