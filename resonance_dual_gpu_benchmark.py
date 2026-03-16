from __future__ import annotations

import argparse
from pathlib import Path

from resonance_field.dual_gpu_benchmark import run_dual_gpu_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark practical dual-GPU execution modes for the SPIRAL engine."
    )
    parser.add_argument("--nodes", type=int, default=256, help="Node count for the benchmark.")
    parser.add_argument("--steps", type=int, default=120, help="Simulation steps per task.")
    parser.add_argument(
        "--runs-per-scenario",
        type=int,
        default=2,
        help="Independent runs per representative scenario.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dual_gpu_benchmark_outputs"),
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=20260320,
        help="Base seed for deterministic task generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_dual_gpu_benchmark(
        output_dir=args.output_dir,
        nodes=args.nodes,
        steps=args.steps,
        runs_per_scenario=args.runs_per_scenario,
        seed_base=args.seed_base,
    )
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")
    print("\nBenchmark summary:")
    for row in result["summary_rows"]:
        print(
            f"- {row['mode']}: runtime={row['runtime_s']:.2f}s, "
            f"gpu0={row['gpu0_util_mean']:.1f}%, gpu1={row['gpu1_util_mean']:.1f}%, "
            f"cpu={row['cpu_util_mean']:.1f}%, stability={row['stability_label']}"
        )


if __name__ == "__main__":
    main()
