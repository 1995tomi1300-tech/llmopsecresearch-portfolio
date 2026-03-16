from __future__ import annotations

import argparse
from pathlib import Path

from resonance_field.phase16_boundary_sweep import run_phase16_boundary_sweep
from resonance_field.phase16_config import PhaseXVIConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase XVI boundary sweep across node count and manipulation strength."
    )
    parser.add_argument(
        "--nodes-list",
        type=str,
        default="1000,2000,5000",
        help="Comma-separated node counts for the direct dense sweep.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Runs per scenario and variant.")
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
        "--max-live-gb",
        type=float,
        default=24.0,
        help="Safety cap for the estimated dense live memory set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resonance_phase16_boundary_outputs"),
        help="Directory for boundary sweep outputs.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=20260314,
        help="Base seed used to derive per-run seeds.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help="Comma-separated boundary variants to run. Empty means all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    node_counts = [int(part.strip()) for part in args.nodes_list.split(",") if part.strip()]
    variant_names = [part.strip() for part in args.variants.split(",") if part.strip()]
    config = PhaseXVIConfig(
        nodes=node_counts[0],
        steps=args.steps,
        compute_backend=args.backend,
        backend_gpu_index=args.gpu_index,
    )
    result = run_phase16_boundary_sweep(
        base_config=config,
        output_dir=args.output_dir,
        runs=args.runs,
        seed_base=args.seed_base,
        node_counts=node_counts,
        max_live_gb=args.max_live_gb,
        variant_names=variant_names or None,
    )
    print(f"\nSaved outputs to: {Path(result['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
