"""Run a small synthetic tokenizer/chunker diagnostic benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_original_tokenizer import run_diagnostics  # noqa: E402
from src.diagnostics.synthetic_regions import SYNTHETIC_MODES  # noqa: E402


SUMMARY_COLUMNS = [
    "run_id",
    "status",
    "synthetic_mode",
    "seq_len",
    "seed",
    "batch_size",
    "device",
    "json_path",
    "loss",
    "ratio_loss",
    "input_shape",
    "logits_shape",
    "stage1_boundary_prob_mean",
    "stage1_boundary_prob_std",
    "stage1_boundary_prob_min",
    "stage1_boundary_prob_max",
    "stage1_pre_merge_boundary_density",
    "stage1_post_merge_boundary_density",
    "stage1_num_chunks_mean",
    "stage1_num_chunks_std",
    "stage1_num_chunks_min",
    "stage1_num_chunks_max",
    "stage1_avg_chunk_length_mean",
    "stage1_avg_chunk_length_std",
    "stage1_compression_ratio_mean",
    "stage1_compression_ratio_std",
    "stage2_boundary_prob_mean",
    "stage2_boundary_prob_std",
    "stage2_boundary_prob_min",
    "stage2_boundary_prob_max",
    "stage2_pre_merge_boundary_density",
    "stage2_post_merge_boundary_density",
    "stage2_num_chunks_mean",
    "stage2_num_chunks_std",
    "stage2_num_chunks_min",
    "stage2_num_chunks_max",
    "stage2_avg_chunk_length_mean",
    "stage2_avg_chunk_length_std",
    "stage2_compression_ratio_mean",
    "stage2_compression_ratio_std",
    "neutral_boundary_density",
    "repeat_boundary_density",
    "conserved_boundary_density",
    "neutral_boundary_prob_mean",
    "repeat_boundary_prob_mean",
    "conserved_boundary_prob_mean",
    "motif_break_rate",
    "motif_internal_boundary_count_mean",
    "error",
]


def parse_csv_arg(value: str, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def stage_value(diagnostics: dict[str, Any], stage: str, key: str, default=None):
    return diagnostics.get("stages", {}).get(stage, {}).get(key, default)


def probability_value(diagnostics: dict[str, Any], stage: str, key: str):
    probs = stage_value(diagnostics, stage, "boundary_probabilities", {}) or {}
    return probs.get(key)


def region_value(stage: dict[str, Any], key: str, region_name: str):
    values = stage.get(key, {}) or {}
    return values.get(region_name)


def flatten_diagnostics(
    run_id: int,
    diagnostics: dict[str, Any],
    json_path: Path,
    seed: int,
    seq_len: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    stage1 = diagnostics.get("stages", {}).get("stage1", {}) or {}
    stage2 = diagnostics.get("stages", {}).get("stage2", {}) or {}
    row = {
        "run_id": run_id,
        "status": "ok",
        "synthetic_mode": diagnostics.get("synthetic_mode"),
        "seq_len": seq_len,
        "seed": seed,
        "batch_size": batch_size,
        "device": device,
        "json_path": str(json_path),
        "loss": diagnostics.get("loss"),
        "ratio_loss": diagnostics.get("ratio_loss"),
        "input_shape": json.dumps(diagnostics.get("input_shape")),
        "logits_shape": json.dumps(diagnostics.get("logits_shape")),
        "stage1_boundary_prob_mean": probability_value(diagnostics, "stage1", "mean"),
        "stage1_boundary_prob_std": probability_value(diagnostics, "stage1", "std"),
        "stage1_boundary_prob_min": probability_value(diagnostics, "stage1", "min"),
        "stage1_boundary_prob_max": probability_value(diagnostics, "stage1", "max"),
        "stage1_pre_merge_boundary_density": stage1.get("pre_merge_discrete_boundary_density"),
        "stage1_post_merge_boundary_density": stage1.get("post_merge_boundary_density"),
        "stage1_num_chunks_mean": stage1.get("num_chunks_mean"),
        "stage1_num_chunks_std": stage1.get("num_chunks_std"),
        "stage1_num_chunks_min": stage1.get("num_chunks_min"),
        "stage1_num_chunks_max": stage1.get("num_chunks_max"),
        "stage1_avg_chunk_length_mean": stage1.get("average_chunk_length_mean"),
        "stage1_avg_chunk_length_std": stage1.get("average_chunk_length_std"),
        "stage1_compression_ratio_mean": stage1.get("compression_ratio_mean"),
        "stage1_compression_ratio_std": stage1.get("compression_ratio_std"),
        "stage2_boundary_prob_mean": probability_value(diagnostics, "stage2", "mean"),
        "stage2_boundary_prob_std": probability_value(diagnostics, "stage2", "std"),
        "stage2_boundary_prob_min": probability_value(diagnostics, "stage2", "min"),
        "stage2_boundary_prob_max": probability_value(diagnostics, "stage2", "max"),
        "stage2_pre_merge_boundary_density": stage2.get("pre_merge_discrete_boundary_density"),
        "stage2_post_merge_boundary_density": stage2.get("post_merge_boundary_density"),
        "stage2_num_chunks_mean": stage2.get("num_chunks_mean"),
        "stage2_num_chunks_std": stage2.get("num_chunks_std"),
        "stage2_num_chunks_min": stage2.get("num_chunks_min"),
        "stage2_num_chunks_max": stage2.get("num_chunks_max"),
        "stage2_avg_chunk_length_mean": stage2.get("average_chunk_length_mean"),
        "stage2_avg_chunk_length_std": stage2.get("average_chunk_length_std"),
        "stage2_compression_ratio_mean": stage2.get("compression_ratio_mean"),
        "stage2_compression_ratio_std": stage2.get("compression_ratio_std"),
        "neutral_boundary_density": region_value(stage1, "region_boundary_density", "neutral"),
        "repeat_boundary_density": region_value(stage1, "region_boundary_density", "repeat"),
        "conserved_boundary_density": region_value(stage1, "region_boundary_density", "conserved_motif_rich"),
        "neutral_boundary_prob_mean": region_value(stage1, "region_boundary_probability_mean", "neutral"),
        "repeat_boundary_prob_mean": region_value(stage1, "region_boundary_probability_mean", "repeat"),
        "conserved_boundary_prob_mean": region_value(stage1, "region_boundary_probability_mean", "conserved_motif_rich"),
        "motif_break_rate": stage1.get("motif_break_rate"),
        "motif_internal_boundary_count_mean": stage1.get("motif_internal_boundary_count_mean"),
        "error": "",
    }
    return row


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_original_dnachunker.yaml")
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-lens", default="64,128,256")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--synthetic-modes", default=",".join(SYNTHETIC_MODES))
    parser.add_argument("--output-dir", default="outputs/tokenizer_benchmark")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--no-backward", action="store_true", default=True)
    parser.add_argument("--backward", dest="no_backward", action="store_false")
    args = parser.parse_args()

    seq_lens = parse_csv_arg(args.seq_lens, int)
    seeds = parse_csv_arg(args.seeds, int)
    modes = parse_csv_arg(args.synthetic_modes, str)
    unknown_modes = sorted(set(modes) - set(SYNTHETIC_MODES))
    if unknown_modes:
        raise ValueError(f"Unknown synthetic modes: {unknown_modes}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    run_id = 0
    for mode in modes:
        for seq_len in seq_lens:
            for seed in seeds:
                if args.max_runs is not None and run_id >= args.max_runs:
                    break
                json_path = output_dir / f"run_{run_id:04d}_{mode}_L{seq_len}_seed{seed}.json"
                print(f"run {run_id}: mode={mode} seq_len={seq_len} seed={seed}")
                try:
                    diagnostics = run_diagnostics(
                        config=args.config,
                        device=args.device,
                        batch_size=args.batch_size,
                        seq_len=seq_len,
                        seed=seed,
                        synthetic_mode=mode,
                        run_backward=not args.no_backward,
                    )
                    json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
                    rows.append(
                        flatten_diagnostics(
                            run_id=run_id,
                            diagnostics=diagnostics,
                            json_path=json_path,
                            seed=seed,
                            seq_len=seq_len,
                            batch_size=args.batch_size,
                            device=args.device,
                        )
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "run_id": run_id,
                            "status": "failed",
                            "synthetic_mode": mode,
                            "seq_len": seq_len,
                            "seed": seed,
                            "batch_size": args.batch_size,
                            "device": args.device,
                            "json_path": str(json_path),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    write_summary(output_dir / "summary.csv", rows)
                    raise
                run_id += 1
            if args.max_runs is not None and run_id >= args.max_runs:
                break
        if args.max_runs is not None and run_id >= args.max_runs:
            break

    summary_path = output_dir / "summary.csv"
    write_summary(summary_path, rows)
    print(f"completed_runs: {sum(row.get('status') == 'ok' for row in rows)}")
    print(f"summary_csv: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
