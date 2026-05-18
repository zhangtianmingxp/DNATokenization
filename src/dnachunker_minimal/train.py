"""CLI and training loop for the minimal DNACHUNKER-style prototype."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from .data import MaskingConfig, SyntheticDNADataset, mask_dna_batch
from .model import MinimalDNAChunker, mlm_loss


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 13,
    "device": "auto",
    "data": {"num_sequences": 128, "seq_len": 96, "batch_size": 8, "mask_probability": 0.15},
    "model": {
        "d_model": 32,
        "local_kernel_size": 5,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "dropout": 0.1,
        "boundary_threshold": 0.5,
    },
    "optim": {"lr": 0.001, "weight_decay": 0.01},
    "train": {"steps": 10, "log_every": 1},
}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = {key: value.copy() if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | None) -> Dict[str, Any]:
    if path is None:
        return DEFAULT_CONFIG

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when using --config. Install pyyaml or use argparse defaults.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return deep_update(DEFAULT_CONFIG, loaded)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_smoke(config: Dict[str, Any]) -> None:
    set_seed(int(config["seed"]))
    device = resolve_device(str(config["device"]))

    data_cfg = config["data"]
    dataset = SyntheticDNADataset(
        num_sequences=int(data_cfg["num_sequences"]),
        seq_len=int(data_cfg["seq_len"]),
        seed=int(config["seed"]),
    )
    masking = MaskingConfig(mask_probability=float(data_cfg["mask_probability"]))
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        collate_fn=lambda batch: mask_dna_batch(batch, masking),
    )

    model = MinimalDNAChunker(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optim"]["lr"]),
        weight_decay=float(config["optim"]["weight_decay"]),
    )

    steps = int(config["train"]["steps"])
    log_every = int(config["train"]["log_every"])
    model.train()
    iterator = iter(loader)

    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, metrics = model(input_ids)
        loss = mlm_loss(logits, labels)
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps:
            metric_text = " ".join(f"{name}={value.item():.4f}" for name, value in metrics.items())
            print(
                f"step={step:03d} loss={loss.item():.4f} "
                f"logits_shape={tuple(logits.shape)} {metric_text}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/minimal_dnachunker.yaml")
    parser.add_argument("--steps", type=int, default=None, help="Override train.steps")
    parser.add_argument("--device", type=str, default=None, help="Override device: auto, cpu, or cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.steps is not None:
        config["train"]["steps"] = args.steps
    if args.device is not None:
        config["device"] = args.device
    train_smoke(config)


if __name__ == "__main__":
    main()
