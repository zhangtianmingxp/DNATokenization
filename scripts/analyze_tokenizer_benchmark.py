"""Compact text/CSV summaries for tokenizer diagnostic benchmark output."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


METRICS = [
    "stage1_compression_ratio_mean",
    "stage1_post_merge_boundary_density",
    "motif_break_rate",
    "repeat_boundary_density",
    "conserved_boundary_density",
    "neutral_boundary_density",
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


def summarize(rows: Iterable[dict[str, str]], keys: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        groups[tuple(row.get(key, "") for key in keys)].append(row)

    summaries = []
    for group_key, group_rows in sorted(groups.items()):
        summary: dict[str, object] = {key: value for key, value in zip(keys, group_key)}
        summary["n"] = len(group_rows)
        for metric in METRICS:
            values = [as_float(row.get(metric, "")) for row in group_rows]
            values = [value for value in values if value is not None]
            summary[metric] = mean(values) if values else ""
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
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
            if isinstance(value, float):
                rendered.append(f"{value:.6f}")
            else:
                rendered.append(str(value))
        print("  " + "\t".join(rendered))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/tokenizer_benchmark/summary.csv")
    parser.add_argument("--save-csv", action="store_true", default=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = read_rows(input_path)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]
    print(f"input: {input_path}")
    print(f"ok_runs: {len(ok_rows)}")
    print(f"failed_runs: {len(failed_rows)}")

    by_mode = summarize(rows, ["synthetic_mode"])
    by_length = summarize(rows, ["seq_len"])
    by_mode_length = summarize(rows, ["synthetic_mode", "seq_len"])
    by_seed = summarize(rows, ["seed"])

    print_table("\nMean metrics by synthetic_mode:", by_mode)
    print_table("\nMean metrics by seq_len:", by_length)
    print_table("\nMean metrics by synthetic_mode and seq_len:", by_mode_length)
    print_table("\nMean metrics by seed:", by_seed)

    if args.save_csv:
        output_dir = input_path.parent
        write_csv(output_dir / "summary_by_mode.csv", by_mode)
        write_csv(output_dir / "summary_by_length.csv", by_length)
        print(f"\nsaved_csv: {output_dir / 'summary_by_mode.csv'}")
        print(f"saved_csv: {output_dir / 'summary_by_length.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
