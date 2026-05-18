"""Tiny synthetic smoke test for the original HNetTransformerForMaskedLM."""

from __future__ import annotations

import argparse
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


def ensure_triton_compiler() -> None:
    """Point Triton at the conda C compiler when the generic `cc` is absent."""
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


def make_inputs(smoke_cfg: dict[str, Any], device: torch.device) -> torch.LongTensor:
    token_ids = torch.tensor(smoke_cfg["dna_token_ids"], dtype=torch.long, device=device)
    batch_size = int(smoke_cfg["batch_size"])
    seq_len = int(smoke_cfg["seq_len"])
    indices = torch.randint(0, token_ids.numel(), (batch_size, seq_len), device=device)
    return token_ids[indices]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_original_dnachunker.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    ensure_triton_compiler()
    cfg = load_config(REPO_ROOT / args.config)
    torch.manual_seed(int(cfg["smoke"].get("seed", 13)))
    device = choose_device(args.device)

    model_config = HNetConfig(**cfg["model"])
    print(f"requested_device: {args.device}")
    print(f"device: {device}")
    print(f"triton_cc: {os.environ.get('CC', 'unset')}")
    print(f"model_config_vocab_size_before_padding: {model_config.vocab_size}")

    model = HNetTransformerForMaskedLM(model_config).to(device)
    model.train()

    input_ids = make_inputs(cfg["smoke"], device)
    labels = input_ids.clone()
    labels[:, ::3] = int(model_config.pad_token_id)

    try:
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            return_dict=False,
        )
    except RuntimeError as exc:
        if device.type == "cpu" and "Expected x.is_cuda() to be true" in str(exc):
            print("cpu_forward_supported: False")
            print("reason: causal_conv1d requires CUDA tensors in this installed path.")
            return 2
        raise

    loss = outputs[0]
    logits = outputs[1]
    hidden_states = outputs[2] if len(outputs) > 3 else None
    ratio_loss = outputs[-1]

    print(f"model_class: {model.__class__.__name__}")
    print(f"model_config_vocab_size_after_padding: {model_config.vocab_size}")
    print(f"input_ids_shape: {tuple(input_ids.shape)}")
    print(f"logits_shape: {tuple(logits.shape)}")
    print(f"loss: {float(loss.detach().cpu()):.6f}")
    print(f"ratio_loss: {float(ratio_loss.detach().cpu()):.6f}")
    print(f"hidden_states_type: {type(hidden_states).__name__}")
    if isinstance(hidden_states, list):
        print(f"hidden_states_entries: {len(hidden_states)}")
        chunk_entries = [item for item in hidden_states if isinstance(item, tuple)]
        print(f"chunking_hidden_state_entries: {len(chunk_entries)}")
        for idx, item in enumerate(chunk_entries):
            tensor, lengths = item
            print(f"chunk_entry_{idx}_tensor_shape: {tuple(tensor.shape)}")
            print(f"chunk_entry_{idx}_lengths: {lengths.detach().cpu().tolist()}")
    print("boundary_outputs_exposed: False")

    if bool(cfg["smoke"].get("run_backward", True)):
        loss.backward()
        print("backward: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
