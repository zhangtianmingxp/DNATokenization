from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import torch


def test_hg38_baseline_scripts_with_tiny_fixture(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Original HNet/Mamba diagnostics require CUDA.")

    output_dir = tmp_path / "hg38_baseline"
    env = os.environ.copy()
    env.update(
        {
            "FASTA": "tests/fixtures/tiny_genome.fa",
            "CONFIG": "configs/smoke_original_dnachunker.yaml",
            "DEVICE": "cuda",
            "CHROMS": "chrTiny1,chrTiny2",
            "SEQ_LENS": "64",
            "NUM_WINDOWS": "2",
            "BATCH_SIZE": "1",
            "SEED": "1",
            "OUTPUT_DIR": str(output_dir),
        }
    )
    subprocess.run(["bash", "scripts/run_hg38_tokenizer_baseline.sh"], check=True, env=env)
    subprocess.run(["python", "scripts/aggregate_hg38_tokenizer_baseline.py", "--input-dir", str(output_dir)], check=True)
    subprocess.run(["python", "scripts/report_hg38_tokenizer_baseline.py", "--input-dir", str(output_dir)], check=True)

    assert (output_dir / "seq64" / "summary.csv").exists()
    assert (output_dir / "hg38_summary_all.csv").exists()
    report = output_dir / "HG38_BASELINE_REPORT.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Experiment summary" in text
    assert "Metrics by sequence length" in text
    assert "Metrics by heuristic region label" in text
    assert "Limitations" in text
