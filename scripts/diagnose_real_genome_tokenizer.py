"""Run original HNet tokenizer diagnostics on local FASTA/BED genome windows."""

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
    "stage1_boundary_density",
    "stage1_compression_ratio",
    "stage1_num_chunks",
    "stage1_avg_chunk_length",
    "stage2_num_chunks",
    "neutral_boundary_density",
    "N_rich_boundary_density",
    "GC_rich_boundary_density",
    "AT_rich_boundary_density",
    "repeat_like_boundary_density",
    "motif_like_boundary_density",
    "motif_like_break_rate",
]


def choose_device(requested: str) -> torch.device:
    if requested != "cuda":
        raise RuntimeError("The installed original HNet/Mamba diagnostics path requires CUDA.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device("cuda")


def parse_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def batch_windows(windows: list[GenomeWindow], batch_size: int):
    for start in range(0, len(windows), batch_size):
        yield start, windows[start : start + batch_size]


def label_density(values: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    densities = {}
    values_cpu = values.detach().float().cpu()
    labels_cpu = labels.detach().cpu()
    for label_id, label_name in REAL_REGION_LABELS.items():
        mask = labels_cpu == label_id
        if mask.any():
            densities[label_name] = float(values_cpu[mask].mean().item())
    return densities


def motif_break_rate(boundaries: torch.Tensor, motif_spans: list[tuple[int, int]]) -> float | None:
    if not motif_spans:
        return None
    boundary_values = boundaries.detach().bool().cpu()
    usable = 0
    broken = 0
    for start, end in motif_spans:
        if end - start <= 1:
            continue
        usable += 1
        if boundary_values[start + 1 : end].any().item():
            broken += 1
    if usable == 0:
        return None
    return broken / usable


def run_batch(
    model: HNetTransformerForMaskedLM,
    windows: list[GenomeWindow],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_ids = torch.stack([window.input_ids for window in windows]).to(device)
    labels = input_ids.clone()
    labels[:, ::3] = int(model.config.pad_token_id)

    with TokenizerDiagnosticHooks(model) as hooks:
        outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=True, return_dict=False)

    loss = outputs[0]
    ratio_loss = outputs[-1]
    stage1 = hooks.downsampler[0]
    stage2 = hooks.downsampler[1] if len(hooks.downsampler) > 1 else None
    stage1_boundaries = stage1["boundaries"].detach()
    stage1_lengths = stage1["chunk_lengths"].detach()
    stage2_lengths = stage2["chunk_lengths"].detach() if stage2 is not None else None

    json_rows = []
    csv_rows = []
    for idx, window in enumerate(windows):
        seq_len = int(window.metadata["seq_len"])
        boundaries = stage1_boundaries[idx]
        num_chunks = int(stage1_lengths[idx].cpu().item())
        compression_ratio = seq_len / max(num_chunks, 1)
        label_values = window.region_ids.to(boundaries.device)
        densities = label_density(boundaries, label_values)
        motif_rate = motif_break_rate(boundaries, window.motif_spans)

        row = {
            "chrom": window.metadata["chrom"],
            "start": window.metadata["start"],
            "end": window.metadata["end"],
            "strand": window.metadata["strand"],
            "seq_len": seq_len,
            "n_frac": window.metadata["n_frac"],
            "gc_frac": window.metadata["gc_frac"],
            "loss": float(loss.detach().cpu().item()) if loss is not None else None,
            "ratio_loss": float(ratio_loss.detach().cpu().item()) if ratio_loss is not None else None,
            "stage1_boundary_density": float(boundaries.float().mean().cpu().item()),
            "stage1_compression_ratio": compression_ratio,
            "stage1_num_chunks": num_chunks,
            "stage1_avg_chunk_length": compression_ratio,
            "stage2_num_chunks": int(stage2_lengths[idx].cpu().item()) if stage2_lengths is not None else None,
            "region_boundary_density": densities,
            "motif_like_break_rate": motif_rate,
            "motif_spans": window.motif_spans,
        }
        json_rows.append(row)

        csv_row = {
            "chrom": row["chrom"],
            "start": row["start"],
            "end": row["end"],
            "strand": row["strand"],
            "seq_len": row["seq_len"],
            "n_frac": row["n_frac"],
            "gc_frac": row["gc_frac"],
            "loss": row["loss"],
            "ratio_loss": row["ratio_loss"],
            "stage1_boundary_density": row["stage1_boundary_density"],
            "stage1_compression_ratio": row["stage1_compression_ratio"],
            "stage1_num_chunks": row["stage1_num_chunks"],
            "stage1_avg_chunk_length": row["stage1_avg_chunk_length"],
            "stage2_num_chunks": row["stage2_num_chunks"],
            "neutral_boundary_density": densities.get("neutral"),
            "N_rich_boundary_density": densities.get("N_rich"),
            "GC_rich_boundary_density": densities.get("GC_rich"),
            "AT_rich_boundary_density": densities.get("AT_rich"),
            "repeat_like_boundary_density": densities.get("repeat_like"),
            "motif_like_boundary_density": densities.get("motif_like"),
            "motif_like_break_rate": motif_rate,
        }
        csv_rows.append(csv_row)
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
    parser.add_argument("--output-dir", default="outputs/real_genome_tokenizer")
    parser.add_argument("--motifs", default=",".join(DEFAULT_MOTIFS))
    parser.add_argument("--min-n-frac", type=float, default=None)
    parser.add_argument("--max-n-frac", type=float, default=None)
    parser.add_argument("--sampling", choices=["random", "bed"], default="random")
    parser.add_argument("--no-backward", action="store_true", default=True)
    args = parser.parse_args()

    ensure_triton_compiler()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    motifs = parse_csv_arg(args.motifs) or DEFAULT_MOTIFS

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
    jsonl_path = output_dir / "window_diagnostics.jsonl"
    summary_path = output_dir / "summary.csv"

    all_json_rows: list[dict[str, Any]] = []
    all_csv_rows: list[dict[str, Any]] = []
    for batch_start, batch in batch_windows(windows, args.batch_size):
        json_rows, csv_rows = run_batch(model, batch, device)
        for offset, row in enumerate(json_rows):
            row["window_id"] = batch_start + offset
        all_json_rows.extend(json_rows)
        all_csv_rows.extend(csv_rows)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in all_json_rows:
            handle.write(json.dumps(row) + "\n")
    write_summary(summary_path, all_csv_rows)

    mean_density = sum(float(row["stage1_boundary_density"]) for row in all_csv_rows) / len(all_csv_rows)
    mean_compression = sum(float(row["stage1_compression_ratio"]) for row in all_csv_rows) / len(all_csv_rows)
    print(f"model_class: HNetTransformerForMaskedLM")
    print(f"windows: {len(all_csv_rows)}")
    print(f"summary_csv: {summary_path}")
    print(f"window_jsonl: {jsonl_path}")
    print(f"mean_stage1_boundary_density: {mean_density:.6f}")
    print(f"mean_stage1_compression_ratio: {mean_compression:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
