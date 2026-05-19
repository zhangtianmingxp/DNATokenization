"""Hook-based tokenizer/chunker diagnostics for the original HNet model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hnet_twostage.configuration_hnet import HNetConfig  # noqa: E402
from hnet_twostage.modeling_hnet import HNetTransformerForMaskedLM  # noqa: E402
from src.diagnostics.synthetic_regions import (  # noqa: E402
    REGION_NAMES,
    SYNTHETIC_MODES,
    SyntheticRegions,
    make_synthetic_input_ids,
)


def ensure_triton_compiler() -> None:
    if os.environ.get("CC"):
        return
    cc = shutil.which("cc") or shutil.which("x86_64-conda-linux-gnu-cc")
    if cc:
        os.environ["CC"] = cc


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


def extract_model_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            if any(key.startswith("model.") for key in state_dict):
                return {key.removeprefix("model."): value for key, value in state_dict.items()}
            return state_dict
    return checkpoint


def load_checkpoint(
    model: HNetTransformerForMaskedLM,
    checkpoint_path: str | Path | None,
    device: torch.device,
) -> str | None:
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_model_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return str(checkpoint_path)


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float().cpu()
    return {
        "shape": list(values.shape),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "requires_grad": bool(tensor.requires_grad),
    }


def list_ints(tensor: torch.Tensor) -> list[int]:
    return [int(v) for v in tensor.detach().cpu().tolist()]


def region_boundary_density(boundaries: torch.Tensor, region_ids: torch.Tensor) -> dict[str, float]:
    return region_scalar_mean(boundaries, region_ids)


def region_scalar_mean(values: torch.Tensor, region_ids: torch.Tensor) -> dict[str, float]:
    result = {}
    scalar_values = values.detach().float().cpu()
    if scalar_values.shape != region_ids.shape:
        scalar_values = scalar_values[..., : region_ids.shape[-1]]
    region_values = region_ids.detach().cpu()
    for region_id, region_name in REGION_NAMES.items():
        mask = region_values == region_id
        if mask.any():
            result[region_name] = float(scalar_values[mask].mean().item())
    return result


def vector_stats(values: torch.Tensor) -> dict[str, float | int]:
    tensor = values.detach().float().cpu()
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": int(tensor.min().item()),
        "max": int(tensor.max().item()),
    }


def motif_break_rate(boundaries: torch.Tensor, motif_spans: list[list[tuple[int, int]]]) -> float | None:
    boundary_values = boundaries.detach().bool().cpu()
    total = 0
    broken = 0
    for batch_idx, spans in enumerate(motif_spans):
        for start, end in spans:
            if end - start <= 1:
                continue
            total += 1
            if boundary_values[batch_idx, start + 1 : end].any().item():
                broken += 1
    if total == 0:
        return None
    return broken / total


def motif_internal_boundary_counts(boundaries: torch.Tensor, motif_spans: list[list[tuple[int, int]]]) -> list[int]:
    boundary_values = boundaries.detach().bool().cpu()
    counts = []
    for batch_idx, spans in enumerate(motif_spans):
        per_sample = 0
        for start, end in spans:
            if end - start <= 1:
                continue
            per_sample += int(boundary_values[batch_idx, start + 1 : end].sum().item())
        counts.append(per_sample)
    return counts


class TokenizerDiagnosticHooks:
    def __init__(self, model: HNetTransformerForMaskedLM):
        self.model = model
        self.routing: dict[str, dict[str, torch.Tensor]] = {}
        self.downsampler: list[dict[str, torch.Tensor]] = []
        self.handles = []

    def __enter__(self):
        backbone = self.model.caduceus.backbone
        self.handles.append(
            backbone.routing_module_stage1.register_forward_hook(self._routing_hook("stage1"))
        )
        self.handles.append(
            backbone.routing_module_stage2.register_forward_hook(self._routing_hook("stage2"))
        )
        self.handles.append(backbone.downsampler.register_forward_hook(self._downsampler_hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()

    def _routing_hook(self, stage: str):
        def hook(module, inputs, output):
            probabilities, discrete = output
            self.routing[stage] = {
                "probabilities": probabilities.detach(),
                "discrete": discrete.detach(),
                "probabilities_requires_grad": probabilities.requires_grad,
                "discrete_requires_grad": discrete.requires_grad,
            }

        return hook

    def _downsampler_hook(self, module, inputs, output):
        hidden_states, boundaries = inputs
        chunks, chunk_lengths = output
        self.downsampler.append(
            {
                "source_hidden_shape": torch.tensor(hidden_states.shape, device=chunk_lengths.device),
                "boundaries": boundaries.detach(),
                "chunks_shape": torch.tensor(chunks.shape, device=chunk_lengths.device),
                "chunk_lengths": chunk_lengths.detach(),
                "chunks_requires_grad": torch.tensor(int(chunks.requires_grad), device=chunk_lengths.device),
            }
        )


def make_inputs(
    mode: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> SyntheticRegions:
    return make_synthetic_input_ids(mode, batch_size, seq_len, device)


def summarize_stage(
    stage: str,
    routing_record: dict[str, torch.Tensor] | None,
    downsample_record: dict[str, torch.Tensor] | None,
    seq_len: int,
    synthetic: SyntheticRegions,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if routing_record is not None:
        probabilities = routing_record["probabilities"]
        discrete = routing_record["discrete"]
        probability_stats = tensor_stats(probabilities)
        probability_stats["requires_grad"] = bool(routing_record["probabilities_requires_grad"])
        summary["boundary_probabilities"] = probability_stats
        summary["pre_merge_discrete_boundary_density"] = float(discrete.detach().float().mean().cpu().item())
        summary["pre_merge_discrete_requires_grad"] = bool(routing_record["discrete_requires_grad"])

    if downsample_record is not None:
        boundaries = downsample_record["boundaries"]
        chunk_lengths = downsample_record["chunk_lengths"]
        source_len = int(boundaries.shape[1])
        chunk_lengths_float = chunk_lengths.detach().float()
        avg_chunks = float(chunk_lengths_float.mean().cpu().item())
        per_sample_avg_chunk_length = source_len / chunk_lengths_float.clamp(min=1)
        per_sample_compression = per_sample_avg_chunk_length
        chunk_stats = vector_stats(chunk_lengths)
        avg_chunk_stats = vector_stats(per_sample_avg_chunk_length)
        compression_stats = vector_stats(per_sample_compression)
        summary["post_merge_boundary_density"] = float(boundaries.detach().float().mean().cpu().item())
        summary["num_chunks_per_sample"] = list_ints(chunk_lengths)
        summary["num_chunks_mean"] = avg_chunks
        summary["num_chunks_std"] = chunk_stats["std"]
        summary["num_chunks_min"] = chunk_stats["min"]
        summary["num_chunks_max"] = chunk_stats["max"]
        summary["chunk_length_min"] = chunk_stats["min"]
        summary["chunk_length_max"] = chunk_stats["max"]
        summary["average_chunk_length"] = float(source_len / avg_chunks) if avg_chunks > 0 else 0.0
        summary["average_chunk_length_mean"] = avg_chunk_stats["mean"]
        summary["average_chunk_length_std"] = avg_chunk_stats["std"]
        summary["compression_ratio"] = float(source_len / avg_chunks) if avg_chunks > 0 else 0.0
        summary["compression_ratio_mean"] = compression_stats["mean"]
        summary["compression_ratio_std"] = compression_stats["std"]
        summary["chunked_hidden_shape"] = [int(v) for v in downsample_record["chunks_shape"].detach().cpu().tolist()]
        summary["source_hidden_shape"] = [int(v) for v in downsample_record["source_hidden_shape"].detach().cpu().tolist()]
        summary["chunked_hidden_requires_grad"] = bool(int(downsample_record["chunks_requires_grad"].detach().cpu().item()))

        if stage == "stage1" and synthetic.region_ids is not None:
            summary["region_boundary_density"] = region_boundary_density(boundaries, synthetic.region_ids)
            if routing_record is not None:
                probabilities = routing_record["probabilities"]
                summary["region_boundary_probability_mean"] = region_scalar_mean(probabilities, synthetic.region_ids)
            motif_rate = motif_break_rate(boundaries, synthetic.motif_spans)
            summary["motif_break_rate"] = motif_rate
            motif_counts = motif_internal_boundary_counts(boundaries, synthetic.motif_spans)
            summary["motif_internal_boundary_count_per_sample"] = motif_counts
            summary["motif_internal_boundary_count_mean"] = (
                float(sum(motif_counts) / len(motif_counts)) if motif_counts else None
            )

    return summary


def run_diagnostics(
    config: str | Path = "configs/smoke_original_dnachunker.yaml",
    device: str = "auto",
    batch_size: int | None = None,
    seq_len: int | None = None,
    seed: int | None = None,
    synthetic_mode: str = "random",
    run_backward: bool = False,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    ensure_triton_compiler()
    cfg = load_config(REPO_ROOT / config)
    smoke_cfg = cfg["smoke"]
    batch_size = int(batch_size if batch_size is not None else smoke_cfg["batch_size"])
    seq_len = int(seq_len if seq_len is not None else smoke_cfg["seq_len"])
    seed = int(seed if seed is not None else smoke_cfg.get("seed", 13))
    torch.manual_seed(seed)
    torch_device = choose_device(device)

    model_config = HNetConfig(**cfg["model"])
    model = HNetTransformerForMaskedLM(model_config).to(torch_device)
    loaded_checkpoint = load_checkpoint(model, checkpoint, torch_device)
    model.train()

    synthetic = make_inputs(synthetic_mode, batch_size, seq_len, torch_device)
    input_ids = synthetic.input_ids
    labels = input_ids.clone()
    labels[:, ::3] = int(model_config.pad_token_id)

    with TokenizerDiagnosticHooks(model) as hooks:
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            return_dict=False,
        )

    loss = outputs[0]
    logits = outputs[1]
    hidden_states = outputs[2] if len(outputs) > 3 else None
    ratio_loss = outputs[-1]

    downsampler_stage1 = hooks.downsampler[0] if len(hooks.downsampler) > 0 else None
    downsampler_stage2 = hooks.downsampler[1] if len(hooks.downsampler) > 1 else None

    diagnostics: dict[str, Any] = {
        "model_class": model.__class__.__name__,
        "device": str(torch_device),
        "checkpoint": loaded_checkpoint,
        "checkpoint_strict": True if loaded_checkpoint is not None else None,
        "synthetic_mode": synthetic_mode,
        "synthetic_metadata": synthetic.metadata,
        "input_shape": list(input_ids.shape),
        "logits_shape": list(logits.shape),
        "has_loss": loss is not None,
        "loss": float(loss.detach().cpu().item()) if loss is not None else None,
        "has_ratio_loss": ratio_loss is not None,
        "ratio_loss": float(ratio_loss.detach().cpu().item()) if ratio_loss is not None else None,
        "backward_run": False,
        "hidden_states_type": type(hidden_states).__name__,
        "hidden_states_entries": len(hidden_states) if isinstance(hidden_states, list) else None,
        "stages": {
            "stage1": summarize_stage(
                "stage1",
                hooks.routing.get("stage1"),
                downsampler_stage1,
                seq_len,
                synthetic,
            ),
            "stage2": summarize_stage(
                "stage2",
                hooks.routing.get("stage2"),
                downsampler_stage2,
                seq_len,
                synthetic,
            ),
        },
    }

    stage1 = diagnostics["stages"]["stage1"]
    diagnostics["compression_ratio"] = stage1.get("compression_ratio")
    diagnostics["chunked_hidden_shape"] = stage1.get("chunked_hidden_shape")
    if run_backward and loss is not None:
        loss.backward()
        diagnostics["backward_run"] = True
    return diagnostics


def print_summary(diagnostics: dict[str, Any]) -> None:
    print(f"model_class: {diagnostics['model_class']}")
    print(f"device: {diagnostics['device']}")
    print(f"synthetic_mode: {diagnostics['synthetic_mode']}")
    print(f"input_shape: {tuple(diagnostics['input_shape'])}")
    print(f"logits_shape: {tuple(diagnostics['logits_shape'])}")
    print(f"loss: {diagnostics['loss']:.6f}" if diagnostics["loss"] is not None else "loss: None")
    print(
        f"ratio_loss: {diagnostics['ratio_loss']:.6f}"
        if diagnostics["ratio_loss"] is not None
        else "ratio_loss: None"
    )

    for stage_name, stage in diagnostics["stages"].items():
        print(f"{stage_name}:")
        probs = stage.get("boundary_probabilities")
        if probs is not None:
            print(
                "  boundary_probability_mean_std_min_max: "
                f"{probs['mean']:.6f} {probs['std']:.6f} {probs['min']:.6f} {probs['max']:.6f}"
            )
        if "pre_merge_discrete_boundary_density" in stage:
            print(f"  pre_merge_discrete_boundary_density: {stage['pre_merge_discrete_boundary_density']:.6f}")
        if "post_merge_boundary_density" in stage:
            print(f"  post_merge_boundary_density: {stage['post_merge_boundary_density']:.6f}")
        if "num_chunks_per_sample" in stage:
            print(f"  num_chunks_per_sample: {stage['num_chunks_per_sample']}")
            print(f"  average_chunk_length: {stage['average_chunk_length']:.6f}")
            print(f"  chunk_length_min_max: {stage['chunk_length_min']} {stage['chunk_length_max']}")
            print(f"  compression_ratio: {stage['compression_ratio']:.6f}")
            print(f"  chunked_hidden_shape: {tuple(stage['chunked_hidden_shape'])}")
        if "region_boundary_density" in stage:
            print(f"  region_boundary_density: {stage['region_boundary_density']}")
        if "region_boundary_probability_mean" in stage:
            print(f"  region_boundary_probability_mean: {stage['region_boundary_probability_mean']}")
        if "motif_break_rate" in stage:
            print(f"  motif_break_rate: {stage['motif_break_rate']}")
        if "motif_internal_boundary_count_mean" in stage:
            print(f"  motif_internal_boundary_count_mean: {stage['motif_internal_boundary_count_mean']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_original_dnachunker.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--synthetic-mode", choices=SYNTHETIC_MODES, default="random")
    parser.add_argument("--save-json", default=None)
    parser.add_argument("--backward", action="store_true", help="Run loss.backward() after collecting diagnostics.")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    diagnostics = run_diagnostics(
        config=args.config,
        device=args.device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        synthetic_mode=args.synthetic_mode,
        run_backward=args.backward,
        checkpoint=args.checkpoint,
    )
    print_summary(diagnostics)

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        print(f"saved_json: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
