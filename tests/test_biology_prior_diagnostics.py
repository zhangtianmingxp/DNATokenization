from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import torch


def test_biology_prior_suppresses_motif_internal_boundaries() -> None:
    from src.diagnostics.biology_boundary_prior import BiologyPriorConfig, apply_biology_boundary_prior
    from src.diagnostics.real_genome_windows import label_sequence_regions, sequence_to_ids

    seq = "TATAACGTAAAAAA"
    input_ids = sequence_to_ids(seq)
    region_ids, motif_spans = label_sequence_regions(seq, motifs=["TATA"])
    probabilities = torch.full((len(seq),), 0.8)
    probabilities[0] = 1.0

    adjustment = apply_biology_boundary_prior(
        probabilities,
        input_ids,
        region_ids,
        motif_spans,
        BiologyPriorConfig(motif_interior_logit_penalty=3.0, motif_edge_logit_bonus=0.0),
    )

    assert motif_spans == [(0, 4)]
    assert adjustment.adjusted_boundaries[0].item() == 1.0
    assert adjustment.adjusted_boundaries[1:4].sum().item() == 0.0


def test_real_genome_biology_prior_diagnostics_cuda(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Original HNet/Mamba diagnostics require CUDA.")

    from scripts.diagnose_real_genome_biology_prior import main as diagnose_main
    from scripts.report_biology_prior_diagnostics import main as report_main

    output_dir = tmp_path / "biology_prior"
    previous_argv = sys.argv
    sys.argv = [
        "diagnose_real_genome_biology_prior.py",
        "--config",
        "configs/smoke_original_dnachunker.yaml",
        "--device",
        "cuda",
        "--fasta",
        "tests/fixtures/tiny_genome.fa",
        "--chroms",
        "chrTiny1,chrTiny2",
        "--seq-len",
        "64",
        "--num-windows",
        "2",
        "--batch-size",
        "1",
        "--seed",
        "1",
        "--output-dir",
        str(output_dir),
    ]
    try:
        assert diagnose_main() == 0
        sys.argv = [
            "report_biology_prior_diagnostics.py",
            "--input-dir",
            str(output_dir),
        ]
        assert report_main() == 0
    finally:
        sys.argv = previous_argv

    summary_path = output_dir / "biology_prior_summary.csv"
    report_path = output_dir / "BIOLOGY_PRIOR_DIAGNOSTIC_REPORT.md"
    assert summary_path.exists()
    assert report_path.exists()
    rows = list(csv.DictReader(summary_path.open("r", newline="", encoding="utf-8")))
    assert len(rows) == 2
    assert "base_boundary_density" in rows[0]
    assert "prior_boundary_density" in rows[0]
    assert "boundary_flip_rate" in rows[0]
    assert "Prior configuration" in report_path.read_text(encoding="utf-8")
