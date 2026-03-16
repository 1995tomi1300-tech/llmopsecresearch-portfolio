from __future__ import annotations

import argparse
import json
from pathlib import Path

from resonance_field.spiral_bridge import write_bridge_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export fixed-schema SPIRAL bridge records for the Mojo AI engine."
    )
    parser.add_argument(
        "reports",
        nargs="+",
        help="One or more upstream report JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the bridge JSON payload.",
    )
    args = parser.parse_args()

    payload = write_bridge_payload(args.reports, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
