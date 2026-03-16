from __future__ import annotations

import argparse
from pathlib import Path

from resonance_field.phase16_config import PhaseXVIConfig
from resonance_field.phase16_scaled_sweep import run_phase16_scaled_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase XVI scaled 1000-node manipulation sweep."
    )
    parser.add_argument("--runs", type=int, default=1, help="Runs per scenario and variant.")
    parser.add_argument("--nodes", type=int, default=1000, help="Node count.")
    parser.add_argument("--steps", type=int, default=600, help="Simulation steps.")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda"],
        help="Numeric backend selection.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="CUDA device index when GPU backend is active.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resonance_phase16_scaled_outputs"),
        help="Directory for scaled manipulation sweep outputs.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=20260314,
        help="Base seed used to derive per-run seeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PhaseXVIConfig(
        nodes=args.nodes,
        steps=args.steps,
        compute_backend=args.backend,
        backend_gpu_index=args.gpu_index,
    )
    result = run_phase16_scaled_sweep(
        base_config=config,
        output_dir=args.output_dir,
        runs=args.runs,
        seed_base=args.seed_base,
    )
    print(f"\nSaved outputs to: {Path(result['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
