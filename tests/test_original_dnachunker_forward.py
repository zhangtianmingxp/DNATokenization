from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import torch
import yaml


pytest.importorskip("mamba_ssm")
pytest.importorskip("causal_conv1d")

from hnet_twostage.configuration_hnet import HNetConfig
from hnet_twostage.modeling_hnet import HNetTransformerForMaskedLM


def ensure_triton_compiler() -> None:
    if os.environ.get("CC"):
        return
    cc = shutil.which("cc") or shutil.which("x86_64-conda-linux-gnu-cc")
    if cc:
        os.environ["CC"] = cc


def test_original_dnachunker_tiny_cuda_forward() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Installed causal_conv1d/Mamba path requires CUDA for this original model smoke test.")

    ensure_triton_compiler()
    config_path = Path(__file__).resolve().parents[1] / "configs" / "smoke_original_dnachunker.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    model_config = HNetConfig(**cfg["model"])
    model = HNetTransformerForMaskedLM(model_config).cuda()
    model.train()

    torch.manual_seed(int(cfg["smoke"]["seed"]))
    token_ids = torch.tensor(cfg["smoke"]["dna_token_ids"], dtype=torch.long, device="cuda")
    indices = torch.randint(
        0,
        token_ids.numel(),
        (int(cfg["smoke"]["batch_size"]), int(cfg["smoke"]["seq_len"])),
        device="cuda",
    )
    input_ids = token_ids[indices]
    labels = input_ids.clone()
    labels[:, ::3] = int(model_config.pad_token_id)

    outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=True, return_dict=False)
    loss, logits = outputs[0], outputs[1]

    assert logits.shape == (cfg["smoke"]["batch_size"], cfg["smoke"]["seq_len"], model_config.vocab_size)
    assert torch.isfinite(loss)
    loss.backward()
