"""Run the minimal DNACHUNKER-style smoke training loop."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dnachunker_minimal.train import main


if __name__ == "__main__":
    main()
