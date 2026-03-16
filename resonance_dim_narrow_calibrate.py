from __future__ import annotations

import argparse
from pathlib import Path

from resonance_field.dim_narrow_calibration import calibrate_dim_narrow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate dimension-narrowed outputs against large-sample reference."
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("/mnt/d/resonance_phase16_boundary_outputs_v1"),
    )
    parser.add_argument(
        "--dim-root",
        type=Path,
        default=Path("/mnt/d/resonance_phase16_dim_narrow_synced"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/d/resonance_phase16_dim_narrow_calibrated"),
    )
    parser.add_argument("--reference-weight", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = calibrate_dim_narrow(
        reference_root=args.reference_root,
        dim_root=args.dim_root,
        output_dir=args.output_dir,
        reference_weight=args.reference_weight,
    )
    print(f"Saved calibration to: {Path(result['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
