"""Tiny synthetic MLM training for the original author HNet DNAChunker model."""

from __future__ import annotations

import argparse
import csv
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
from src.diagnostics.synthetic_regions import SYNTHETIC_MODES, make_synthetic_input_ids  # noqa: E402


DNA_TOKEN_IDS = torch.tensor([7, 8, 9, 10, 11], dtype=torch.long)
MASK_TOKEN_ID = 3
IGNORE_INDEX = -100


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
    if requested != "cuda":
        raise RuntimeError("Synthetic training for the installed original model requires CUDA.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device("cuda")


def apply_mlm_mask(
    input_ids: torch.LongTensor,
    mask_prob: float,
    device: torch.device,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    masked_input = input_ids.clone()
    labels = input_ids.clone()
    mask = torch.rand(input_ids.shape, device=device) < mask_prob
    if not mask.any():
        mask[0, 0] = True
    labels[~mask] = IGNORE_INDEX

    replace_with_mask = (torch.rand(input_ids.shape, device=device) < 0.8) & mask
    random_replace = (torch.rand(input_ids.shape, device=device) < 0.5) & mask & ~replace_with_mask
    masked_input[replace_with_mask] = MASK_TOKEN_ID

    dna_ids = DNA_TOKEN_IDS.to(device)
    random_ids = dna_ids[torch.randint(0, dna_ids.numel(), input_ids.shape, device=device)]
    masked_input[random_replace] = random_ids[random_replace]
    return masked_input, labels


def save_checkpoint(
    path: Path,
    model: HNetTransformerForMaskedLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": args.config,
            "synthetic_mode": args.synthetic_mode,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_original_dnachunker.yaml")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--output-dir", default="outputs/synthetic_training_debug/train")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--synthetic-mode", choices=SYNTHETIC_MODES, default="regions")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    ensure_triton_compiler()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(REPO_ROOT / args.config)
    model_config = HNetConfig(**cfg["model"])
    model_config.pad_token_id = IGNORE_INDEX
    model = HNetTransformerForMaskedLM(model_config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_checkpoint(output_dir / "step_000000.pt", model, optimizer, 0, args)

    log_path = output_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "ratio_loss", "lr", "synthetic_mode"])
        writer.writeheader()

        for step in range(1, args.steps + 1):
            synthetic = make_synthetic_input_ids(args.synthetic_mode, args.batch_size, args.seq_len, device)
            masked_input, labels = apply_mlm_mask(synthetic.input_ids, args.mask_prob, device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=masked_input, labels=labels, output_hidden_states=True, return_dict=False)
            loss = outputs[0]
            ratio_loss = outputs[-1]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            row = {
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "ratio_loss": float(ratio_loss.detach().cpu().item()) if ratio_loss is not None else "",
                "lr": optimizer.param_groups[0]["lr"],
                "synthetic_mode": args.synthetic_mode,
            }
            writer.writerow(row)
            handle.flush()

            if step % args.log_every == 0 or step == 1 or step == args.steps:
                print(
                    f"step={step} loss={row['loss']:.6f} "
                    f"ratio_loss={row['ratio_loss']:.6f} lr={row['lr']}"
                )

            if step % args.save_every == 0:
                save_checkpoint(output_dir / f"step_{step:06d}.pt", model, optimizer, step, args)

    final_step_path = output_dir / f"step_{args.steps:06d}.pt"
    save_checkpoint(final_step_path, model, optimizer, args.steps, args)
    save_checkpoint(output_dir / "final.pt", model, optimizer, args.steps, args)
    print(f"initial_checkpoint: {output_dir / 'step_000000.pt'}")
    print(f"final_step_checkpoint: {final_step_path}")
    print(f"final_checkpoint: {output_dir / 'final.pt'}")
    print(f"train_log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
