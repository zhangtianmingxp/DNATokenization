"""Compare tokenizer diagnostic benchmark summaries before and after training."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


METRIC_MAP = {
    "compression_ratio": "stage1_compression_ratio_mean",
    "boundary_density": "stage1_post_merge_boundary_density",
    "motif_break_rate": "motif_break_rate",
    "repeat_boundary_density": "repeat_boundary_density",
    "conserved_boundary_density": "conserved_boundary_density",
    "ratio_loss": "ratio_loss",
}


def as_float(value: str):
    if value in ("", "None", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == "ok"]


def grouped_means(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, float | str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["synthetic_mode"], row["seq_len"])].append(row)

    result: dict[tuple[str, str], dict[str, float | str]] = {}
    for key, group_rows in groups.items():
        summary: dict[str, float | str] = {
            "synthetic_mode": key[0],
            "seq_len": key[1],
        }
        for public_name, column in METRIC_MAP.items():
            values = [as_float(row.get(column, "")) for row in group_rows]
            values = [value for value in values if value is not None]
            summary[public_name] = mean(values) if values else ""
        result[key] = summary
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output-dir", default="outputs/synthetic_training_compare")
    args = parser.parse_args()

    before = grouped_means(read_rows(Path(args.before)))
    after = grouped_means(read_rows(Path(args.after)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    keys = sorted(set(before) & set(after))
    for key in keys:
        row = {
            "synthetic_mode": key[0],
            "seq_len": key[1],
        }
        for metric in METRIC_MAP:
            before_value = before[key].get(metric, "")
            after_value = after[key].get(metric, "")
            row[f"{metric}_before"] = before_value
            row[f"{metric}_after"] = after_value
            row[f"{metric}_delta"] = (
                after_value - before_value
                if isinstance(before_value, float) and isinstance(after_value, float)
                else ""
            )
        rows.append(row)

    columns = ["synthetic_mode", "seq_len"]
    for metric in METRIC_MAP:
        columns.extend([f"{metric}_before", f"{metric}_after", f"{metric}_delta"])

    output_path = output_dir / "comparison_by_mode.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("Tokenizer diagnostics before/after comparison:")
    print("\t".join(columns))
    for row in rows:
        rendered = []
        for column in columns:
            value = row[column]
            rendered.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        print("\t".join(rendered))
    print(f"comparison_csv: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
