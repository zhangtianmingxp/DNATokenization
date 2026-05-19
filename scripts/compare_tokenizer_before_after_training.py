"""Compare tokenizer diagnostic benchmark summaries before and after training."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


METRIC_COLUMNS = [
    "stage1_compression_ratio_mean",
    "stage1_post_merge_boundary_density",
    "motif_break_rate",
    "repeat_boundary_density",
    "conserved_boundary_density",
    "neutral_boundary_density",
    "ratio_loss",
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
        return [row for row in csv.DictReader(handle) if row.get("status") == "ok"]


def group_mean_rows(rows: Iterable[dict[str, str]], group_keys: list[str]) -> dict[tuple[str, ...], dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in group_keys)].append(row)

    result: dict[tuple[str, ...], dict[str, object]] = {}
    for group_id, group_rows in groups.items():
        summary: dict[str, object] = {key: value for key, value in zip(group_keys, group_id)}
        summary["n"] = len(group_rows)
        for metric in METRIC_COLUMNS:
            values = [as_float(row.get(metric, "")) for row in group_rows]
            values = [value for value in values if value is not None]
            summary[metric] = mean(values) if values else ""
        result[group_id] = summary
    return result


def comparison_rows(
    before_rows: list[dict[str, str]],
    after_rows: list[dict[str, str]],
    group_keys: list[str],
) -> list[dict[str, object]]:
    before = group_mean_rows(before_rows, group_keys)
    after = group_mean_rows(after_rows, group_keys)
    rows = []
    for group_id in sorted(set(before) & set(after)):
        row: dict[str, object] = {key: value for key, value in zip(group_keys, group_id)}
        row["before_n"] = before[group_id].get("n", 0)
        row["after_n"] = after[group_id].get("n", 0)
        for metric in METRIC_COLUMNS:
            before_value = before[group_id].get(metric, "")
            after_value = after[group_id].get(metric, "")
            row[f"{metric}_before"] = before_value
            row[f"{metric}_after"] = after_value
            row[f"{metric}_delta"] = (
                after_value - before_value
                if isinstance(before_value, float) and isinstance(after_value, float)
                else ""
            )
        rows.append(row)
    return rows


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
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output-dir", default="outputs/synthetic_training_compare")
    args = parser.parse_args()

    before_rows = read_rows(Path(args.before))
    after_rows = read_rows(Path(args.after))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_mode = comparison_rows(before_rows, after_rows, ["synthetic_mode"])
    by_length = comparison_rows(before_rows, after_rows, ["seq_len"])
    by_mode_length = comparison_rows(before_rows, after_rows, ["synthetic_mode", "seq_len"])

    write_csv(output_dir / "comparison_by_mode.csv", by_mode)
    write_csv(output_dir / "comparison_by_length.csv", by_length)
    write_csv(output_dir / "comparison_by_mode_length.csv", by_mode_length)

    print_table("Tokenizer diagnostics before/after by synthetic_mode:", by_mode)
    print_table("\nTokenizer diagnostics before/after by seq_len:", by_length)
    print_table("\nTokenizer diagnostics before/after by synthetic_mode + seq_len:", by_mode_length)
    print(f"\ncomparison_by_mode_csv: {output_dir / 'comparison_by_mode.csv'}")
    print(f"comparison_by_length_csv: {output_dir / 'comparison_by_length.csv'}")
    print(f"comparison_by_mode_length_csv: {output_dir / 'comparison_by_mode_length.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
