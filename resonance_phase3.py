from __future__ import annotations

import argparse
from pathlib import Path

from resonance_field.phase3_config import PhaseIIIConfig
from resonance_field.phase3_experiment import run_phase3_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase III Resonant Manifold Intelligence experiments."
    )
    parser.add_argument("--runs", type=int, default=12, help="Runs per scenario.")
    parser.add_argument("--nodes", type=int, default=PhaseIIIConfig.nodes, help="Node count.")
    parser.add_argument("--steps", type=int, default=PhaseIIIConfig.steps, help="Simulation steps.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resonance_phase3_outputs"),
        help="Directory for metrics and visualizations.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=20260311,
        help="Base seed used to derive per-run seeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PhaseIIIConfig(nodes=args.nodes, steps=args.steps)
    result = run_phase3_experiments(
        config=config,
        output_dir=args.output_dir,
        runs=args.runs,
        seed_base=args.seed_base,
    )
    print(f"\nSaved outputs to: {Path(result['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
