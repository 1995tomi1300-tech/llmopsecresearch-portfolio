from __future__ import annotations

import argparse
from pathlib import Path

from resonance_field.phase11_config import PhaseXIConfig
from resonance_field.phase11_experiment import run_phase11_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase XI adaptive policy-learner resonance experiments."
    )
    parser.add_argument("--runs", type=int, default=12, help="Runs per scenario.")
    parser.add_argument("--nodes", type=int, default=PhaseXIConfig.nodes, help="Node count.")
    parser.add_argument("--steps", type=int, default=PhaseXIConfig.steps, help="Simulation steps.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resonance_phase11_outputs"),
        help="Directory for policy-learner metrics and visualizations.",
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
    config = PhaseXIConfig(nodes=args.nodes, steps=args.steps)
    result = run_phase11_experiments(
        config=config,
        output_dir=args.output_dir,
        runs=args.runs,
        seed_base=args.seed_base,
    )
    print(f"\nSaved outputs to: {Path(result['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
