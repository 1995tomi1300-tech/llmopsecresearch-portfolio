from __future__ import annotations

import argparse
from pathlib import Path

from resonance_field.phase14_config import PhaseXIVConfig
from resonance_field.phase14_experiment import run_phase14_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase XIV multifocal modulation resonance experiments."
    )
    parser.add_argument("--runs", type=int, default=12, help="Runs per scenario.")
    parser.add_argument("--nodes", type=int, default=PhaseXIVConfig.nodes, help="Node count.")
    parser.add_argument("--steps", type=int, default=PhaseXIVConfig.steps, help="Simulation steps.")
    parser.add_argument(
        "--backend",
        type=str,
        default=PhaseXIVConfig.compute_backend,
        choices=["auto", "cpu", "gpu", "cuda"],
        help="Numeric backend selection.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=PhaseXIVConfig.backend_gpu_index,
        help="CUDA device index when GPU backend is active.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resonance_phase14_outputs"),
        help="Directory for multifocal-training metrics and visualizations.",
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
    config = PhaseXIVConfig(
        nodes=args.nodes,
        steps=args.steps,
        compute_backend=args.backend,
        backend_gpu_index=args.gpu_index,
    )
    result = run_phase14_experiments(
        config=config,
        output_dir=args.output_dir,
        runs=args.runs,
        seed_base=args.seed_base,
    )
    print(f"\nSaved outputs to: {Path(result['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
