from __future__ import annotations

import pytest
import torch


pytest.importorskip("mamba_ssm")
pytest.importorskip("causal_conv1d")


def test_original_tokenizer_diagnostics_contains_basic_keys() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Original Mamba/causal_conv1d path requires CUDA for diagnostics.")

    from scripts.diagnose_original_tokenizer import run_diagnostics

    diagnostics = run_diagnostics(
        config="configs/smoke_original_dnachunker.yaml",
        device="cuda",
        batch_size=2,
        seq_len=64,
        seed=13,
        synthetic_mode="regions",
    )

    assert diagnostics["input_shape"] == [2, 64]
    assert diagnostics["logits_shape"][0:2] == [2, 64]
    assert diagnostics["has_loss"]
    assert diagnostics["has_ratio_loss"]
    assert "compression_ratio" in diagnostics
    assert "chunked_hidden_shape" in diagnostics
    assert "stage1" in diagnostics["stages"]
    assert "region_boundary_density" in diagnostics["stages"]["stage1"]
