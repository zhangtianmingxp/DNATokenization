"""Aggregate hg38 tokenizer baseline summaries across sequence lengths."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


METRICS = [
    "stage1_boundary_density",
    "stage1_compression_ratio",
    "ratio_loss",
    "gc_frac",
    "n_frac",
    "motif_like_break_rate",
    "repeat_like_boundary_density",
    "GC_rich_boundary_density",
    "AT_rich_boundary_density",
    "N_rich_boundary_density",
    "neutral_boundary_density",
    "motif_like_boundary_density",
]


def as_float(value: str):
    if value in ("", "None", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def bin_gc(value: float) -> str:
    if value < 0.35:
        return "low_gc"
    if value > 0.60:
        return "high_gc"
    return "mid_gc"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]], group_key: str) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    out = []
    for key, group in sorted(groups.items()):
        row: dict[str, object] = {group_key: key, "n": len(group)}
        for metric in METRICS:
            values = [as_float(item.get(metric, "")) for item in group]
            values = [value for value in values if value is not None]
            row[metric] = mean(values) if values else ""
        out.append(row)
    return out


def summarize_region_labels(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    columns = {
        "neutral": "neutral_boundary_density",
        "N_rich": "N_rich_boundary_density",
        "GC_rich": "GC_rich_boundary_density",
        "AT_rich": "AT_rich_boundary_density",
        "repeat_like": "repeat_like_boundary_density",
        "motif_like": "motif_like_boundary_density",
    }
    out = []
    for label, column in columns.items():
        values = [as_float(row.get(column, "")) for row in rows]
        values = [value for value in values if value is not None]
        out.append({"region_label": label, "n": len(values), "boundary_density": mean(values) if values else ""})
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0].keys()))
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
        print("  " + "\t".join(f"{row[col]:.6f}" if isinstance(row[col], float) else str(row[col]) for col in columns))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/hg38_tokenizer_baseline")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows: list[dict[str, str]] = []
    for summary in sorted(input_dir.glob("seq*/summary.csv")):
        seq_label = summary.parent.name.removeprefix("seq")
        for row in read_csv(summary):
            row["seq_len_group"] = seq_label
            row["gc_bin"] = bin_gc(float(row["gc_frac"]))
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No seq*/summary.csv files found under {input_dir}")

    by_seq = summarize(rows, "seq_len_group")
    by_gc = summarize(rows, "gc_bin")
    by_region = summarize_region_labels(rows)

    all_path = input_dir / "hg38_summary_all.csv"
    write_csv(all_path, rows, fieldnames=list(rows[0].keys()))
    write_csv(input_dir / "hg38_summary_by_seq_len.csv", by_seq)
    write_csv(input_dir / "hg38_summary_by_gc_bin.csv", by_gc)
    write_csv(input_dir / "hg38_summary_by_region_label.csv", by_region)

    print(f"input_dir: {input_dir}")
    print(f"windows: {len(rows)}")
    print_table("\nBy seq_len:", by_seq)
    print_table("\nBy GC bin:", by_gc)
    print_table("\nBy heuristic region label:", by_region)
    print(f"\nsaved_csv: {all_path}")
    print(f"saved_csv: {input_dir / 'hg38_summary_by_seq_len.csv'}")
    print(f"saved_csv: {input_dir / 'hg38_summary_by_gc_bin.csv'}")
    print(f"saved_csv: {input_dir / 'hg38_summary_by_region_label.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
