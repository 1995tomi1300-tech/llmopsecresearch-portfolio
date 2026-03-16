from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from resonance_field.phase16_boundary_sweep import BOUNDARY_SCENARIOS
from resonance_field.phase16_config import PhaseXVIConfig
from resonance_field.phase16_experiment import run_phase16_experiments
from resonance_field.phase16_quantized_pilot import VARIANT_OVERRIDES, apply_variant


MUTATIONS: dict[str, dict[str, float]] = {
    "control_alive": {},
    "receptive_bias": {
        "quantized_receptive_trust_floor": 0.44,
        "quantized_receptive_trust_ceiling": 0.82,
        "quantized_receptive_disturbance_max": 0.42,
        "quantized_drive_scale": 0.62,
        "quantized_vector_phase_gain": 0.045,
        "quantized_vector_freq_gain": 0.028,
    },
    "tight_lock": {
        "quantized_lock_width_bins": 12.0,
        "quantized_drive_scale": 0.55,
        "quantized_vector_phase_gain": 0.050,
        "quantized_vector_freq_gain": 0.032,
        "quantized_vector_geometry_gain": 0.020,
    },
    "shell_expand": {
        "quantized_shell_trust_min": 0.28,
        "quantized_shell_trust_max": 0.84,
        "quantized_shell_disturbance_min": 0.10,
        "quantized_shell_disturbance_max": 0.80,
        "quantized_active_window": 260,
    },
    "repulsive_spike": {
        "quantized_repulsive_scale": 0.85,
        "quantized_repulsive_trust_max": 0.40,
        "quantized_repulsive_disturbance_min": 0.62,
        "quantized_drive_scale": 0.58,
    },
    "memory_bias": {
        "quantized_scalar_memory_weight": 0.28,
        "quantized_scalar_trust_weight": 0.30,
        "quantized_scalar_phase_weight": 0.24,
        "quantized_scalar_freq_weight": 0.18,
        "quantized_reference_blend": 0.68,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase XVI behavior mutation sweep.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/d/resonance_phase16_behavior_mutation_sweep"),
    )
    parser.add_argument("--nodes", type=int, default=120)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=91000)
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_OVERRIDES.keys()),
        default="strong_balanced",
    )
    parser.add_argument("--backend", default="auto")
    return parser.parse_args()


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["scenario"]): row for row in rows}


def classify_behavior(coherence_delta: float, isolation_delta: float, repulsive: float) -> str:
    if coherence_delta >= 0.0 and isolation_delta >= 0.0:
        return "stabilizing"
    if coherence_delta < -0.02 and isolation_delta <= 0.0:
        return "suppressive"
    if repulsive >= 0.15 and coherence_delta < 0.0:
        return "aggressive"
    if isolation_delta > 0.0 and coherence_delta > -0.02:
        return "selective"
    return "neutral"


def render_behavior_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    labels = [f"{row['mutation']}|{row['false_mode']}" for row in rows]
    coherence = [float(row["coherence_delta_vs_baseline"]) for row in rows]
    isolation = [float(row["false_isolation_delta_vs_baseline"]) for row in rows]
    x = range(len(rows))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.plot(list(x), coherence, marker="o", label="coherence delta")
    ax.plot(list(x), isolation, marker="o", label="false isolation delta")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Behavior Mutation Deltas vs Baseline")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_quantized_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    labels = [f"{row['mutation']}|{row['false_mode']}" for row in rows]
    repulsive = [float(row["quantized_repulsive_fraction_late"]) for row in rows]
    lock_fraction = [float(row["quantized_scalar_lock_fraction_late"]) for row in rows]
    drive = [float(row["quantized_vector_drive_mean_late"]) for row in rows]
    x = range(len(rows))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(list(x), repulsive, marker="o", label="repulsive fraction")
    ax.plot(list(x), lock_fraction, marker="o", label="lock fraction")
    ax.plot(list(x), drive, marker="o", label="drive mean")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0.0, max(1.0, max(repulsive + lock_fraction + drive) * 1.15))
    ax.set_title("Behavior Mutation Quantized Metrics")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = PhaseXVIConfig(
        nodes=args.nodes,
        steps=args.steps,
        shock_step=max(1, args.steps // 2),
        late_window=max(40, args.steps // 4),
        benchmark_repeats=2,
        compute_backend=args.backend,
        collect_event_rows=False,
    )
    base = apply_variant(base, args.variant)

    baseline_cfg = replace(base, quantized_scalar_vector_enabled=False)
    baseline_out = args.output_dir / "baseline"
    baseline_result = run_phase16_experiments(
        config=baseline_cfg,
        output_dir=baseline_out,
        runs=args.runs,
        seed_base=args.seed_base,
        scenarios=BOUNDARY_SCENARIOS,
    )
    baseline_map = index_rows(baseline_result["summary_rows"])

    mutation_rows: list[dict[str, Any]] = []
    mutation_configs: list[dict[str, Any]] = []

    for index, (mutation_name, overrides) in enumerate(MUTATIONS.items()):
        mutation_cfg = replace(
            base,
            quantized_scalar_vector_enabled=True,
            **overrides,
        )
        mutation_out = args.output_dir / mutation_name
        result = run_phase16_experiments(
            config=mutation_cfg,
            output_dir=mutation_out,
            runs=args.runs,
            seed_base=args.seed_base + (index + 1) * 1000,
            scenarios=BOUNDARY_SCENARIOS,
        )
        mutation_configs.append(
            {
                "mutation": mutation_name,
                "config": asdict(mutation_cfg),
                "output_dir": str(mutation_out),
            }
        )
        for row in result["summary_rows"]:
            baseline_row = baseline_map[str(row["scenario"])]
            coherence_delta = float(row["late_recovery_coherence_mean"]) - float(
                baseline_row["late_recovery_coherence_mean"]
            )
            isolation_delta = float(row["final_false_isolation_rate_mean"]) - float(
                baseline_row["final_false_isolation_rate_mean"]
            )
            repulsive = float(row["quantized_repulsive_fraction_late_mean"])
            mutation_rows.append(
                {
                    "mutation": mutation_name,
                    "scenario": row["scenario"],
                    "false_mode": row["false_mode"],
                    "late_recovery_coherence": float(row["late_recovery_coherence_mean"]),
                    "baseline_late_recovery_coherence": float(
                        baseline_row["late_recovery_coherence_mean"]
                    ),
                    "coherence_delta_vs_baseline": coherence_delta,
                    "final_false_isolation": float(
                        row["final_false_isolation_rate_mean"]
                    ),
                    "baseline_final_false_isolation": float(
                        baseline_row["final_false_isolation_rate_mean"]
                    ),
                    "false_isolation_delta_vs_baseline": isolation_delta,
                    "quantized_scalar_anchor_gap_late": float(
                        row["quantized_scalar_anchor_gap_late_mean"]
                    ),
                    "quantized_scalar_lock_fraction_late": float(
                        row["quantized_scalar_lock_fraction_late_mean"]
                    ),
                    "quantized_repulsive_fraction_late": repulsive,
                    "quantized_vector_drive_mean_late": float(
                        row["quantized_vector_drive_mean_late_mean"]
                    ),
                    "behavior_class": classify_behavior(
                        coherence_delta,
                        isolation_delta,
                        repulsive,
                    ),
                }
            )

    save_csv(args.output_dir / "mutation_behavior_summary.csv", mutation_rows)
    save_json(
        args.output_dir / "mutation_behavior_summary.json",
        {
            "variant": args.variant,
            "base_config": asdict(base),
            "baseline_config": asdict(baseline_cfg),
            "mutations": mutation_configs,
            "rows": mutation_rows,
        },
    )
    render_behavior_chart(
        mutation_rows,
        args.output_dir / "mutation_behavior_deltas.png",
    )
    render_quantized_chart(
        mutation_rows,
        args.output_dir / "mutation_quantized_metrics.png",
    )


if __name__ == "__main__":
    main()
