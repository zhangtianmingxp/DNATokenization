"""Create a tiny local hg38-like FASTA/BED dataset for training smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path


SEQ = (
    "ACGTACGTNNACGTGCAATTCGGAACGTACGT"
    "TTGCAACGTACGTNNNNACGTGATTACAACGT"
)


def wrap_fasta(seq: str, width: int = 80) -> str:
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/hg38_tiny")
    parser.add_argument("--length", type=int, default=4096)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq = (SEQ * ((args.length // len(SEQ)) + 1))[: args.length]
    fasta_path = out_dir / "hg38_tiny.fa"
    bed_path = out_dir / "human-sequences-tiny.bed"

    fasta_path.write_text(f">chrTiny\n{wrap_fasta(seq)}\n", encoding="utf-8")
    bed_path.write_text(
        "\n".join(
            [
                "chrTiny\t0\t1048576\ttrain",
                "chrTiny\t0\t1048576\tvalid",
                "chrTiny\t0\t1048576\ttest",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"fasta_file: {fasta_path.resolve()}")
    print(f"bed_file: {bed_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
