from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from scripts.leaderboard_adapter import main as adapter_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Pegasus leaderboard inference entrypoint")
    parser.add_argument("--stage", required=True, choices=["inference"])
    _known, rest = parser.parse_known_args()

    sys.argv = [sys.argv[0], *rest]
    return adapter_main()


if __name__ == "__main__":
    raise SystemExit(main())
