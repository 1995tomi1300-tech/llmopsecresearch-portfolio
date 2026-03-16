from __future__ import annotations

import argparse
import json
from pathlib import Path

from resonance_field.dimension_narrowing import build_narrowed_node_list
from resonance_field.phase16_boundary_sweep import run_phase16_boundary_sweep
from resonance_field.phase16_config import PhaseXVIConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase XVI boundary sweep with dimension narrowing."
    )
    parser.add_argument(
        "--logical-nodes-list",
        type=str,
        default="100000,300000,1000000",
        help="Comma-separated logical node counts to compress.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="strong_balanced,critical_push",
        help="Comma-separated boundary variants.",
    )
    parser.add_argument(
        "--exponent",
        type=float,
        default=0.5,
        help="Power-law exponent for dimensional narrowing.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=32.0,
        help="Scale multiplier for dimensional narrowing.",
    )
    parser.add_argument(
        "--min-effective-nodes",
        type=int,
        default=3000,
        help="Lower clamp for effective node count.",
    )
    parser.add_argument(
        "--max-effective-nodes",
        type=int,
        default=17000,
        help="Upper clamp for effective node count.",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument(
        "--backend",
        type=str,
        default="gpu",
        choices=["auto", "cpu", "gpu", "cuda"],
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--max-live-gb", type=float, default=180.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resonance_phase16_dim_narrow_outputs"),
    )
    parser.add_argument("--seed-base", type=int, default=20260314)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logical_nodes = [int(x.strip()) for x in args.logical_nodes_list.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]

    effective_nodes, specs = build_narrowed_node_list(
        logical_nodes,
        exponent=args.exponent,
        scale=args.scale,
        min_effective_nodes=args.min_effective_nodes,
        max_effective_nodes=args.max_effective_nodes,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "narrowing_map.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "logical_nodes_list": logical_nodes,
                "effective_nodes_list": effective_nodes,
                "specs": [spec.__dict__ for spec in specs],
                "narrowing_params": {
                    "exponent": args.exponent,
                    "scale": args.scale,
                    "min_effective_nodes": args.min_effective_nodes,
                    "max_effective_nodes": args.max_effective_nodes,
                },
            },
            handle,
            indent=2,
        )

    base = PhaseXVIConfig(
        nodes=effective_nodes[0],
        steps=args.steps,
        compute_backend=args.backend,
        backend_gpu_index=args.gpu_index,
    )
    result = run_phase16_boundary_sweep(
        base_config=base,
        output_dir=args.output_dir,
        runs=args.runs,
        seed_base=args.seed_base,
        node_counts=effective_nodes,
        max_live_gb=args.max_live_gb,
        variant_names=variants or None,
    )
    print(f"\nSaved outputs to: {Path(result['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
