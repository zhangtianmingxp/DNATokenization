"""Check whether the original hnet_twostage DNAChunker model is importable."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CLASS_NAMES = [
    "RoutingModule",
    "StraightThroughEstimator",
    "Downsampler",
    "CrossAttentionUpsampler",
    "HNetMixerModel",
    "HNetTransformerForMaskedLM",
]

DEPENDENCIES = [
    "torch",
    "transformers",
    "einops",
    "mamba_ssm",
    "flash_attn",
    "triton",
]


def main() -> int:
    print("Dependency availability:")
    for name in DEPENDENCIES:
        print(f"  {name}: {importlib.util.find_spec(name) is not None}")

    print("\nOriginal module import:")
    try:
        module = importlib.import_module("hnet_twostage.modeling_hnet")
    except Exception as exc:
        print("  hnet_twostage.modeling_hnet: FAILED")
        print(f"  error_type: {type(exc).__name__}")
        print(f"  error: {exc}")
        print("\nRequested classes:")
        for class_name in CLASS_NAMES:
            print(f"  {class_name}: False")
        print("\nTraceback:")
        traceback.print_exc()
        return 1

    print("  hnet_twostage.modeling_hnet: OK")
    print("\nRequested classes:")
    all_present = True
    for class_name in CLASS_NAMES:
        present = hasattr(module, class_name)
        all_present = all_present and present
        print(f"  {class_name}: {present}")

    if all_present:
        print("\nSUCCESS: original hnet_twostage model symbols are importable.")
        return 0

    print("\nFAILED: module imported, but one or more expected symbols are missing.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
