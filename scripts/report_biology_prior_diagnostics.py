"""Summarize offline biology-prior tokenizer diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


LABELS = ["neutral", "N_rich", "GC_rich", "AT_rich", "repeat_like", "motif_like"]


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def load_rows(input_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(input_dir.glob("seq*/biology_prior_summary.csv"))
    if not paths and (input_dir / "biology_prior_summary.csv").exists():
        paths = [input_dir / "biology_prior_summary.csv"]
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                if "seq_len_group" not in row:
                    row["seq_len_group"] = row.get("seq_len", "")
                row["source_summary"] = str(path)
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No biology_prior_summary.csv files found under {input_dir}")
    return rows


def load_configs(input_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(input_dir.glob("seq*/biology_prior_config.json"))
    if not paths and (input_dir / "biology_prior_config.json").exists():
        paths = [input_dir / "biology_prior_config.json"]
    configs = []
    for path in paths:
        configs.append(json.loads(path.read_text(encoding="utf-8")))
    return configs


def mean_column(rows: list[dict[str, Any]], column: str) -> float | None:
    values = [parse_float(row.get(column)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return mean(values)


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)
    return groups


def summarize_rows(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    return {
        "group": label,
        "n": len(rows),
        "base_boundary_density": mean_column(rows, "base_boundary_density"),
        "prior_boundary_density": mean_column(rows, "prior_boundary_density"),
        "boundary_density_delta": mean_column(rows, "boundary_density_delta"),
        "base_compression_ratio": mean_column(rows, "base_compression_ratio"),
        "prior_compression_ratio": mean_column(rows, "prior_compression_ratio"),
        "compression_ratio_delta": mean_column(rows, "compression_ratio_delta"),
        "boundary_flip_rate": mean_column(rows, "boundary_flip_rate"),
        "boundary_added_density": mean_column(rows, "boundary_added_density"),
        "boundary_removed_density": mean_column(rows, "boundary_removed_density"),
        "base_motif_like_break_rate": mean_column(rows, "base_motif_like_break_rate"),
        "prior_motif_like_break_rate": mean_column(rows, "prior_motif_like_break_rate"),
        "motif_break_rate_delta": mean_column(rows, "motif_break_rate_delta"),
    }


def summarize_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for label in LABELS:
        base_col = f"base_{label}_boundary_density"
        prior_col = f"prior_{label}_boundary_density"
        delta_col = f"{label}_boundary_density_delta"
        usable = [row for row in rows if parse_float(row.get(base_col)) is not None]
        if not usable:
            continue
        summary.append(
            {
                "region_label": label,
                "n": len(usable),
                "base_boundary_density": mean_column(usable, base_col),
                "prior_boundary_density": mean_column(usable, prior_col),
                "boundary_density_delta": mean_column(usable, delta_col),
            }
        )
    return summary


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")
    return lines


def write_report(input_dir: Path, rows: list[dict[str, Any]], configs: list[dict[str, Any]]) -> Path:
    by_seq = [summarize_rows(group, seq_len) for seq_len, group in sorted(group_by(rows, "seq_len_group").items())]
    by_label = summarize_labels(rows)
    overall = summarize_rows(rows, "all")

    seq_columns = [
        "group",
        "n",
        "base_boundary_density",
        "prior_boundary_density",
        "boundary_density_delta",
        "base_compression_ratio",
        "prior_compression_ratio",
        "compression_ratio_delta",
        "boundary_flip_rate",
        "base_motif_like_break_rate",
        "prior_motif_like_break_rate",
    ]
    label_columns = [
        "region_label",
        "n",
        "base_boundary_density",
        "prior_boundary_density",
        "boundary_density_delta",
    ]
    output_all = input_dir / "biology_prior_summary_all.csv"
    output_seq = input_dir / "biology_prior_summary_by_seq_len.csv"
    output_label = input_dir / "biology_prior_summary_by_region_label.csv"
    write_csv(output_all, [overall], seq_columns)
    write_csv(output_seq, by_seq, seq_columns)
    write_csv(output_label, by_label, label_columns)

    config = configs[0] if configs else {}
    lines = [
        "# Biology Prior Tokenizer Diagnostic Report",
        "",
        "## Experiment summary",
        "",
        f"- FASTA path: `{config.get('fasta', '')}`",
        f"- Chromosomes used: `{','.join(config.get('chroms', [])) if config.get('chroms') else ''}`",
        f"- Sampled windows: `{len(rows)}`",
        f"- Model config: `{config.get('config', '')}`",
        f"- Checkpoint: `{config.get('checkpoint') or 'none'}`",
        "- Prior mode: offline diagnostic logit adjustment only",
        "",
        "## Overall metrics",
        "",
        *markdown_table([overall], seq_columns),
        "",
        "## Metrics by sequence length",
        "",
        *markdown_table(by_seq, seq_columns),
        "",
        "## Metrics by heuristic region label",
        "",
        *markdown_table(by_label, label_columns),
        "",
        "## Prior configuration",
        "",
        "```json",
        json.dumps(config.get("prior", {}), indent=2),
        "```",
        "",
        "## Limitations",
        "",
        "- The prior is not part of the original model forward pass.",
        "- The original author architecture and RoutingModule are unchanged.",
        "- Region labels are heuristic diagnostics, not biological ground truth.",
        "- No hg38 training is performed.",
        "- These metrics describe an engineering intervention and are not biological conclusions.",
    ]
    report_path = input_dir / "BIOLOGY_PRIOR_DIAGNOSTIC_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/hg38_biology_prior_diagnostic")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows = load_rows(input_dir)
    configs = load_configs(input_dir)
    report = write_report(input_dir, rows, configs)
    print(f"report: {report}")
    print(f"windows: {len(rows)}")
    print(f"mean_base_boundary_density: {mean_column(rows, 'base_boundary_density'):.6f}")
    print(f"mean_prior_boundary_density: {mean_column(rows, 'prior_boundary_density'):.6f}")
    print(f"mean_base_compression_ratio: {mean_column(rows, 'base_compression_ratio'):.6f}")
    print(f"mean_prior_compression_ratio: {mean_column(rows, 'prior_compression_ratio'):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
