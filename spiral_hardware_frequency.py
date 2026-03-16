from __future__ import annotations

import argparse
import json
from pathlib import Path

from resonance_field.hardware_frequency import sample_hardware_frequency, write_hardware_frequency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read current CPU and GPU hardware clock frequencies for SPIRAL."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is not None:
        payload = write_hardware_frequency(args.output)
    else:
        payload = sample_hardware_frequency()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
