"""Analyze real-genome tokenizer diagnostic CSV summaries."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


METRICS = [
    "stage1_boundary_density",
    "stage1_compression_ratio",
    "stage1_num_chunks",
    "stage2_num_chunks",
    "ratio_loss",
    "neutral_boundary_density",
    "N_rich_boundary_density",
    "GC_rich_boundary_density",
    "AT_rich_boundary_density",
    "repeat_like_boundary_density",
    "motif_like_boundary_density",
    "motif_like_break_rate",
]


def as_float(value: str):
    if value in ("", "None", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bin_gc(value: float) -> str:
    if value < 0.35:
        return "low_gc"
    if value > 0.60:
        return "high_gc"
    return "mid_gc"


def bin_n(value: float) -> str:
    if value == 0:
        return "no_N"
    if value < 0.10:
        return "low_N"
    return "high_N"


def summarize(rows: list[dict[str, str]], group_key: str) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    summaries = []
    for key, group_rows in sorted(groups.items()):
        summary: dict[str, object] = {group_key: key, "n": len(group_rows)}
        for metric in METRICS:
            values = [as_float(row.get(metric, "")) for row in group_rows]
            values = [value for value in values if value is not None]
            summary[metric] = mean(values) if values else ""
        summaries.append(summary)
    return summaries


def summarize_region_labels(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    label_columns = [
        "neutral_boundary_density",
        "N_rich_boundary_density",
        "GC_rich_boundary_density",
        "AT_rich_boundary_density",
        "repeat_like_boundary_density",
        "motif_like_boundary_density",
    ]
    summaries = []
    for column in label_columns:
        values = [as_float(row.get(column, "")) for row in rows]
        values = [value for value in values if value is not None]
        summaries.append(
            {
                "region_label": column.removesuffix("_boundary_density"),
                "n": len(values),
                "boundary_density": mean(values) if values else "",
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(title: str, rows: list[dict[str, object]]) -> None:
    print(title)
    if not rows:
        print("  no rows")
        return
    columns = list(rows[0].keys())
    print("  " + "\t".join(columns))
    for row in rows:
        rendered = []
        for column in columns:
            value = row[column]
            rendered.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        print("  " + "\t".join(rendered))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="outputs/real_genome_tokenizer/summary.csv")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    rows = read_rows(summary_path)
    for row in rows:
        row["gc_bin"] = bin_gc(float(row["gc_frac"]))
        row["n_bin"] = bin_n(float(row["n_frac"]))

    by_chrom = summarize(rows, "chrom")
    by_gc = summarize(rows, "gc_bin")
    by_n = summarize(rows, "n_bin")
    by_region = summarize_region_labels(rows)

    output_dir = summary_path.parent
    write_csv(output_dir / "summary_by_chrom.csv", by_chrom)
    write_csv(output_dir / "summary_by_gc_bin.csv", by_gc)
    write_csv(output_dir / "summary_by_n_bin.csv", by_n)
    write_csv(output_dir / "summary_by_region_label.csv", by_region)

    print(f"input_summary: {summary_path}")
    print(f"windows: {len(rows)}")
    print_table("\nBy chromosome:", by_chrom)
    print_table("\nBy GC bin:", by_gc)
    print_table("\nBy N bin:", by_n)
    print_table("\nBy region label:", by_region)
    print(f"\nsaved_csv: {output_dir / 'summary_by_chrom.csv'}")
    print(f"saved_csv: {output_dir / 'summary_by_gc_bin.csv'}")
    print(f"saved_csv: {output_dir / 'summary_by_n_bin.csv'}")
    print(f"saved_csv: {output_dir / 'summary_by_region_label.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
