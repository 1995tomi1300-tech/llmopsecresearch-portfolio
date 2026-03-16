from __future__ import annotations

import argparse
from pathlib import Path

from resonance_field.phase5_config import PhaseVConfig
from resonance_field.phase5_experiment import run_phase5_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase V shock and reordering recovery experiments."
    )
    parser.add_argument("--runs", type=int, default=12, help="Runs per scenario.")
    parser.add_argument("--nodes", type=int, default=PhaseVConfig.nodes, help="Node count.")
    parser.add_argument("--steps", type=int, default=PhaseVConfig.steps, help="Simulation steps.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resonance_phase5_outputs"),
        help="Directory for recovery metrics and visualizations.",
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
    config = PhaseVConfig(nodes=args.nodes, steps=args.steps)
    result = run_phase5_experiments(
        config=config,
        output_dir=args.output_dir,
        runs=args.runs,
        seed_base=args.seed_base,
    )
    print(f"\nSaved outputs to: {Path(result['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
