from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resonance_field.hardware_frequency import sample_hardware_frequency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample CPU/GPU frequency telemetry into JSONL."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Optional fixed sample count. Zero means run until interrupted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    taken = 0
    with args.output.open("a", encoding="utf-8") as handle:
        while True:
            payload = sample_hardware_frequency()
            handle.write(json.dumps(payload) + "\n")
            handle.flush()
            taken += 1
            if args.count > 0 and taken >= args.count:
                break
            time.sleep(max(0.05, args.interval))


if __name__ == "__main__":
    main()
