from __future__ import annotations

import argparse
from pathlib import Path

from resonance_field.spiral_population_baseline import write_population_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SPIRAL population baseline without teaching/memory updates."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Bridge payloads or upstream report files.",
    )
    parser.add_argument(
        "--logical-total",
        type=float,
        default=None,
        help="Optional logical node total to distribute across cohorts, e.g. 1e100.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/d/spiral_validation_outputs/population_baseline.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_population_baseline(
        inputs=args.inputs,
        output_path=args.output,
        logical_total=args.logical_total,
    )
    print(f"Saved baseline to: {args.output}")
    print(f"cohort_count={payload['cohort_count']}")
    print(f"logical_node_total={payload['logical_node_total']}")
    print(f"stable_mass_fraction={payload['stable_mass_fraction']}")
    print(f"questionable_mass_fraction={payload['questionable_mass_fraction']}")
    print(f"discard_mass_fraction={payload['discard_mass_fraction']}")
    print(f"weighted_mean_trust_score={payload['weighted_mean_trust_score']}")


if __name__ == "__main__":
    main()
