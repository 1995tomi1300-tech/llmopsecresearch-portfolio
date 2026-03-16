from __future__ import annotations

import argparse
import json
from pathlib import Path

from resonance_field.spiral_validation import SpiralValidationEngine, evaluate_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the minimal SPIRAL validation layer over structured upstream reports."
    )
    parser.add_argument(
        "reports",
        nargs="+",
        help="One or more upstream JSON report files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    engine = SpiralValidationEngine()
    payload: dict[str, list[dict[str, object]]] = {}
    for report_path in args.reports:
        results = [result.as_dict() for result in evaluate_report(report_path, engine)]
        payload[str(Path(report_path))] = results

    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
