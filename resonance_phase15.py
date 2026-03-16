from __future__ import annotations

import argparse
from pathlib import Path

from resonance_field.phase15_config import PhaseXVConfig
from resonance_field.phase15_experiment import run_phase15_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase XV inverse-triad weak-zone resonance experiments."
    )
    parser.add_argument("--runs", type=int, default=10, help="Runs per scenario.")
    parser.add_argument("--nodes", type=int, default=PhaseXVConfig.nodes, help="Node count.")
    parser.add_argument("--steps", type=int, default=PhaseXVConfig.steps, help="Simulation steps.")
    parser.add_argument(
        "--backend",
        type=str,
        default=PhaseXVConfig.compute_backend,
        choices=["auto", "cpu", "gpu", "cuda"],
        help="Numeric backend selection.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=PhaseXVConfig.backend_gpu_index,
        help="CUDA device index when GPU backend is active.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resonance_phase15_outputs"),
        help="Directory for inverse-triad metrics and visualizations.",
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
    config = PhaseXVConfig(
        nodes=args.nodes,
        steps=args.steps,
        compute_backend=args.backend,
        backend_gpu_index=args.gpu_index,
    )
    result = run_phase15_experiments(
        config=config,
        output_dir=args.output_dir,
        runs=args.runs,
        seed_base=args.seed_base,
    )
    print(f"\nSaved outputs to: {Path(result['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
