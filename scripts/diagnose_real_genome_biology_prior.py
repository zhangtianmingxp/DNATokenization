"""Compare original HNet boundaries with an offline biology-guided prior."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hnet_twostage.configuration_hnet import HNetConfig  # noqa: E402
from hnet_twostage.modeling_hnet import HNetTransformerForMaskedLM  # noqa: E402
from scripts.diagnose_original_tokenizer import (  # noqa: E402
    TokenizerDiagnosticHooks,
    ensure_triton_compiler,
    load_checkpoint,
    load_config,
)
from scripts.diagnose_real_genome_tokenizer import batch_windows, choose_device, parse_csv_arg  # noqa: E402
from src.diagnostics.biology_boundary_prior import (  # noqa: E402
    BiologyPriorConfig,
    apply_biology_boundary_prior,
    boundary_density_by_label,
    boundary_flip_metrics,
    compression_ratio,
    config_to_dict,
    motif_break_rate,
)
from src.diagnostics.real_genome_windows import (  # noqa: E402
    DEFAULT_MOTIFS,
    REAL_REGION_LABELS,
    GenomeWindow,
    open_fasta,
    parse_chroms,
    sample_bed_windows,
    sample_random_windows,
)


SUMMARY_COLUMNS = [
    "window_id",
    "chrom",
    "start",
    "end",
    "strand",
    "seq_len",
    "n_frac",
    "gc_frac",
    "loss",
    "ratio_loss",
    "base_boundary_density",
    "prior_boundary_density",
    "boundary_density_delta",
    "base_compression_ratio",
    "prior_compression_ratio",
    "compression_ratio_delta",
    "base_num_chunks",
    "prior_num_chunks",
    "boundary_flip_rate",
    "boundary_added_density",
    "boundary_removed_density",
    "base_motif_like_break_rate",
    "prior_motif_like_break_rate",
    "motif_break_rate_delta",
    "mean_prior_logit_adjustment",
    "mean_gc_shift_score",
    "motif_interior_fraction",
]

for label_name in REAL_REGION_LABELS.values():
    SUMMARY_COLUMNS.extend(
        [
            f"base_{label_name}_boundary_density",
            f"prior_{label_name}_boundary_density",
            f"{label_name}_boundary_density_delta",
        ]
    )


def make_prior_config(args: argparse.Namespace) -> BiologyPriorConfig:
    return BiologyPriorConfig(
        label_logit_bias={
            "neutral": args.neutral_logit_bias,
            "N_rich": args.n_rich_logit_bias,
            "GC_rich": args.gc_rich_logit_bias,
            "AT_rich": args.at_rich_logit_bias,
            "repeat_like": args.repeat_logit_bias,
            "motif_like": args.motif_logit_bias,
        },
        motif_interior_logit_penalty=args.motif_interior_logit_penalty,
        motif_edge_logit_bonus=args.motif_edge_logit_bonus,
        gc_shift_logit_bonus=args.gc_shift_logit_bonus,
        gc_shift_threshold=args.gc_shift_threshold,
        gc_window_radius=args.gc_window_radius,
        threshold=args.boundary_threshold,
    )


def nullable_delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return float(after) - float(before)


def run_batch(
    model: HNetTransformerForMaskedLM,
    windows: list[GenomeWindow],
    device: torch.device,
    prior_config: BiologyPriorConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_ids = torch.stack([window.input_ids for window in windows]).to(device)
    labels = input_ids.clone()
    labels[:, ::3] = int(model.config.pad_token_id)

    with TokenizerDiagnosticHooks(model) as hooks:
        outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=True, return_dict=False)

    loss = outputs[0]
    ratio_loss = outputs[-1]
    probabilities = hooks.routing["stage1"]["probabilities"].detach()
    stage1 = hooks.downsampler[0]
    base_boundaries = stage1["boundaries"].detach()

    json_rows = []
    csv_rows = []
    for idx, window in enumerate(windows):
        seq_len = int(window.metadata["seq_len"])
        base = base_boundaries[idx]
        adjustment = apply_biology_boundary_prior(
            probabilities[idx],
            input_ids[idx],
            window.region_ids.to(device),
            window.motif_spans,
            prior_config,
        )
        prior = adjustment.adjusted_boundaries

        base_density = float(base.float().mean().cpu().item())
        prior_density = float(prior.float().mean().cpu().item())
        base_compression = compression_ratio(base)
        prior_compression = compression_ratio(prior)
        base_chunks = int(base.float().sum().cpu().item())
        prior_chunks = int(prior.float().sum().cpu().item())
        base_motif_break = motif_break_rate(base, window.motif_spans)
        prior_motif_break = motif_break_rate(prior, window.motif_spans)
        flip = boundary_flip_metrics(base, prior)
        base_region_density = boundary_density_by_label(base, window.region_ids)
        prior_region_density = boundary_density_by_label(prior, window.region_ids)

        row: dict[str, Any] = {
            "chrom": window.metadata["chrom"],
            "start": window.metadata["start"],
            "end": window.metadata["end"],
            "strand": window.metadata["strand"],
            "seq_len": seq_len,
            "n_frac": window.metadata["n_frac"],
            "gc_frac": window.metadata["gc_frac"],
            "loss": float(loss.detach().cpu().item()) if loss is not None else None,
            "ratio_loss": float(ratio_loss.detach().cpu().item()) if ratio_loss is not None else None,
            "base_boundary_density": base_density,
            "prior_boundary_density": prior_density,
            "boundary_density_delta": prior_density - base_density,
            "base_compression_ratio": base_compression,
            "prior_compression_ratio": prior_compression,
            "compression_ratio_delta": prior_compression - base_compression,
            "base_num_chunks": base_chunks,
            "prior_num_chunks": prior_chunks,
            "base_motif_like_break_rate": base_motif_break,
            "prior_motif_like_break_rate": prior_motif_break,
            "motif_break_rate_delta": nullable_delta(prior_motif_break, base_motif_break),
            "mean_prior_logit_adjustment": float(adjustment.logit_adjustment.mean().cpu().item()),
            "mean_gc_shift_score": float(adjustment.gc_shift_score.mean().cpu().item()),
            "motif_interior_fraction": float(adjustment.motif_interior_mask.float().mean().cpu().item()),
            "base_region_boundary_density": base_region_density,
            "prior_region_boundary_density": prior_region_density,
            "motif_spans": window.motif_spans,
        }
        row.update(flip)
        for label_name in REAL_REGION_LABELS.values():
            base_value = base_region_density.get(label_name)
            prior_value = prior_region_density.get(label_name)
            row[f"base_{label_name}_boundary_density"] = base_value
            row[f"prior_{label_name}_boundary_density"] = prior_value
            row[f"{label_name}_boundary_density_delta"] = nullable_delta(prior_value, base_value)

        json_rows.append(row)
        csv_rows.append({column: row.get(column, "") for column in SUMMARY_COLUMNS if column != "window_id"})
    return json_rows, csv_rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for idx, row in enumerate(rows):
            output = {"window_id": idx, **row}
            writer.writerow({column: output.get(column, "") for column in SUMMARY_COLUMNS})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_original_dnachunker.yaml")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--bed", default=None)
    parser.add_argument("--chroms", default=None)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--num-windows", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/real_genome_biology_prior")
    parser.add_argument("--motifs", default=",".join(DEFAULT_MOTIFS))
    parser.add_argument("--min-n-frac", type=float, default=None)
    parser.add_argument("--max-n-frac", type=float, default=None)
    parser.add_argument("--sampling", choices=["random", "bed"], default="random")
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--neutral-logit-bias", type=float, default=0.0)
    parser.add_argument("--n-rich-logit-bias", type=float, default=-1.0)
    parser.add_argument("--gc-rich-logit-bias", type=float, default=0.25)
    parser.add_argument("--at-rich-logit-bias", type=float, default=0.0)
    parser.add_argument("--repeat-logit-bias", type=float, default=-0.5)
    parser.add_argument("--motif-logit-bias", type=float, default=-0.75)
    parser.add_argument("--motif-interior-logit-penalty", type=float, default=2.0)
    parser.add_argument("--motif-edge-logit-bonus", type=float, default=0.5)
    parser.add_argument("--gc-shift-logit-bonus", type=float, default=1.0)
    parser.add_argument("--gc-shift-threshold", type=float, default=0.25)
    parser.add_argument("--gc-window-radius", type=int, default=8)
    args = parser.parse_args()

    ensure_triton_compiler()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    motifs = parse_csv_arg(args.motifs) or DEFAULT_MOTIFS
    prior_config = make_prior_config(args)

    fasta = open_fasta(args.fasta)
    chroms = parse_chroms(args.chroms, fasta)
    if args.sampling == "bed":
        if not args.bed:
            raise ValueError("--bed is required when --sampling bed")
        windows = sample_bed_windows(
            fasta,
            args.bed,
            seq_len=args.seq_len,
            num_windows=args.num_windows,
            rng=rng,
            chroms=chroms,
            motifs=motifs,
        )
    else:
        windows = sample_random_windows(
            fasta,
            chroms=chroms,
            seq_len=args.seq_len,
            num_windows=args.num_windows,
            rng=rng,
            motifs=motifs,
            min_n_frac=args.min_n_frac,
            max_n_frac=args.max_n_frac,
        )

    cfg = load_config(REPO_ROOT / args.config)
    model_config = HNetConfig(**cfg["model"])
    model_config.pad_token_id = -100
    model = HNetTransformerForMaskedLM(model_config).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "biology_prior_window_diagnostics.jsonl"
    summary_path = output_dir / "biology_prior_summary.csv"
    config_path = output_dir / "biology_prior_config.json"

    all_json_rows: list[dict[str, Any]] = []
    all_csv_rows: list[dict[str, Any]] = []
    for batch_start, batch in batch_windows(windows, args.batch_size):
        json_rows, csv_rows = run_batch(model, batch, device, prior_config)
        for offset, row in enumerate(json_rows):
            row["window_id"] = batch_start + offset
        all_json_rows.extend(json_rows)
        all_csv_rows.extend(csv_rows)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in all_json_rows:
            handle.write(json.dumps(row) + "\n")
    write_summary(summary_path, all_csv_rows)
    config_payload = {
        "fasta": args.fasta,
        "chroms": chroms,
        "seq_len": args.seq_len,
        "num_windows": args.num_windows,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "prior": config_to_dict(prior_config),
    }
    config_path.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

    mean_base_density = sum(float(row["base_boundary_density"]) for row in all_csv_rows) / len(all_csv_rows)
    mean_prior_density = sum(float(row["prior_boundary_density"]) for row in all_csv_rows) / len(all_csv_rows)
    mean_base_compression = sum(float(row["base_compression_ratio"]) for row in all_csv_rows) / len(all_csv_rows)
    mean_prior_compression = sum(float(row["prior_compression_ratio"]) for row in all_csv_rows) / len(all_csv_rows)
    print("model_class: HNetTransformerForMaskedLM")
    print(f"windows: {len(all_csv_rows)}")
    print(f"summary_csv: {summary_path}")
    print(f"window_jsonl: {jsonl_path}")
    print(f"prior_config: {config_path}")
    print(f"mean_base_boundary_density: {mean_base_density:.6f}")
    print(f"mean_prior_boundary_density: {mean_prior_density:.6f}")
    print(f"mean_base_compression_ratio: {mean_base_compression:.6f}")
    print(f"mean_prior_compression_ratio: {mean_prior_compression:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
