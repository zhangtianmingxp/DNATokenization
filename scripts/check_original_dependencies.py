"""Check runtime dependencies required by the original HNet DNAChunker code."""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODULES = [
    "torch",
    "transformers",
    "datasets",
    "pytorch_lightning",
    "mamba_ssm",
    "flash_attn",
    "causal_conv1d",
    "hnet_twostage.modeling_hnet",
]


def version_of(module: object) -> str:
    return str(getattr(module, "__version__", "version unavailable"))


def main() -> int:
    ok = True
    print("Original DNAChunker dependency check:")
    for name in MODULES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            ok = False
            print(f"  {name}: FAILED ({type(exc).__name__}: {exc})")
            traceback.print_exc()
            continue
        print(f"  {name}: OK ({version_of(module)})")

    if ok:
        print("\nSUCCESS: all original DNAChunker dependencies are importable.")
        return 0

    print("\nFAILED: one or more dependencies could not be imported.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
