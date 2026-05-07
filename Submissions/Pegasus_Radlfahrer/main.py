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

    ckpt_path = ROOT / "runs" / "pegasus_best_checkpoint.pt"
    part1 = ROOT / "runs" / "pegasus_best_checkpoint.pt.part1"
    part2 = ROOT / "runs" / "pegasus_best_checkpoint.pt.part2"
    if not ckpt_path.exists() and part1.exists() and part2.exists():
        with open(ckpt_path, "wb") as f_out:
            for part in [part1, part2]:
                with open(part, "rb") as f_in:
                    f_out.write(f_in.read())

    sys.argv = [sys.argv[0], *rest]
    return adapter_main()


if __name__ == "__main__":
    raise SystemExit(main())
