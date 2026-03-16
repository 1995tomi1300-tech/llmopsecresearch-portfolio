from __future__ import annotations

import argparse
import json
from pathlib import Path

from resonance_field.spiral_dual_gpu_batch import run_dual_gpu_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SPIRAL Mojo validation layer across two GPUs with generated wrapper batches."
    )
    parser.add_argument(
        "reports",
        nargs="+",
        help="One or more upstream report JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/d/spiral_validation_outputs/dual_gpu_batch"),
        help="Directory for generated wrapper files and parsed results.",
    )
    args = parser.parse_args()

    result = run_dual_gpu_batch(args.reports, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
