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


VARIANT_OVERRIDES: dict[str, dict[str, float]] = {
    "subtle_balanced": {
        "phase_manipulation_phase_shift": 0.18,
        "phase_manipulation_freq_shift": 0.014,
        "phase_manipulation_phase_blend": 0.34,
        "phase_manipulation_freq_blend": 0.18,
        "phase_manipulation_geometry_shift": 0.05,
        "phase_manipulation_helix_shift": 0.12,
    },
    "mid_balanced": {
        "phase_manipulation_phase_shift": 0.28,
        "phase_manipulation_freq_shift": 0.022,
        "phase_manipulation_phase_blend": 0.42,
        "phase_manipulation_freq_blend": 0.24,
        "phase_manipulation_geometry_shift": 0.08,
        "phase_manipulation_helix_shift": 0.18,
    },
    "strong_balanced": {
        "phase_manipulation_phase_shift": 0.36,
        "phase_manipulation_freq_shift": 0.030,
        "phase_manipulation_phase_blend": 0.52,
        "phase_manipulation_freq_blend": 0.30,
        "phase_manipulation_geometry_shift": 0.11,
        "phase_manipulation_helix_shift": 0.24,
    },
    "critical_push": {
        "phase_manipulation_phase_shift": 0.46,
        "phase_manipulation_freq_shift": 0.038,
        "phase_manipulation_phase_blend": 0.62,
        "phase_manipulation_freq_blend": 0.36,
        "phase_manipulation_geometry_shift": 0.14,
        "phase_manipulation_helix_shift": 0.32,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Phase XVI quantized-control pilot.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/d/resonance_phase16_quantized_pilot"),
    )
    parser.add_argument("--nodes", type=int, default=120)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=72000)
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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def apply_variant(base: PhaseXVIConfig, variant_name: str) -> PhaseXVIConfig:
    return replace(base, **VARIANT_OVERRIDES[variant_name])


def index_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["scenario"]): row for row in rows}


def build_comparison_rows(
    baseline_rows: list[dict[str, Any]],
    quantized_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_map = index_summary(baseline_rows)
    quantized_map = index_summary(quantized_rows)
    scenarios = sorted(set(baseline_map) | set(quantized_map))
    comparison: list[dict[str, Any]] = []
    for scenario in scenarios:
        base = baseline_map.get(scenario)
        quant = quantized_map.get(scenario)
        if not base or not quant:
            continue
        comparison.append(
            {
                "scenario": scenario,
                "model_kind": base["model_kind"],
                "bootstrap_mode": base["bootstrap_mode"],
                "false_mode": base["false_mode"],
                "late_recovery_coherence_baseline": float(
                    base["late_recovery_coherence_mean"]
                ),
                "late_recovery_coherence_quantized": float(
                    quant["late_recovery_coherence_mean"]
                ),
                "late_recovery_coherence_delta": float(
                    quant["late_recovery_coherence_mean"]
                )
                - float(base["late_recovery_coherence_mean"]),
                "final_false_isolation_baseline": float(
                    base["final_false_isolation_rate_mean"]
                ),
                "final_false_isolation_quantized": float(
                    quant["final_false_isolation_rate_mean"]
                ),
                "final_false_isolation_delta": float(
                    quant["final_false_isolation_rate_mean"]
                )
                - float(base["final_false_isolation_rate_mean"]),
                "quantized_scalar_anchor_gap_late": float(
                    quant["quantized_scalar_anchor_gap_late_mean"]
                ),
                "quantized_scalar_lock_fraction_late": float(
                    quant["quantized_scalar_lock_fraction_late_mean"]
                ),
                "quantized_repulsive_fraction_late": float(
                    quant["quantized_repulsive_fraction_late_mean"]
                ),
                "quantized_vector_drive_mean_late": float(
                    quant["quantized_vector_drive_mean_late_mean"]
                ),
            }
        )
    return comparison


def render_metric_chart(
    rows: list[dict[str, Any]],
    output_path: Path,
    baseline_key: str,
    quantized_key: str,
    title: str,
) -> None:
    if not rows:
        return
    labels = [str(row["false_mode"]) for row in rows]
    baseline_values = [float(row[baseline_key]) for row in rows]
    quantized_values = [float(row[quantized_key]) for row in rows]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.35
    ax.bar([value - width / 2 for value in x], baseline_values, width=width, label="baseline")
    ax.bar([value + width / 2 for value in x], quantized_values, width=width, label="quantized")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_quantized_chart(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not rows:
        return
    labels = [str(row["false_mode"]) for row in rows]
    lock_values = [float(row["quantized_scalar_lock_fraction_late"]) for row in rows]
    repulse_values = [float(row["quantized_repulsive_fraction_late"]) for row in rows]
    drive_values = [float(row["quantized_vector_drive_mean_late"]) for row in rows]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(list(x), lock_values, marker="o", label="lock fraction")
    ax.plot(list(x), repulse_values, marker="o", label="repulsive fraction")
    ax.plot(list(x), drive_values, marker="o", label="drive mean")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Quantized Control Late Metrics")
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
        benchmark_repeats=4,
        compute_backend=args.backend,
        collect_event_rows=False,
    )
    base = apply_variant(base, args.variant)

    baseline_config = replace(base, quantized_scalar_vector_enabled=False)
    quantized_config = replace(base, quantized_scalar_vector_enabled=True)

    baseline_dir = args.output_dir / "baseline"
    quantized_dir = args.output_dir / "quantized"

    baseline_result = run_phase16_experiments(
        config=baseline_config,
        output_dir=baseline_dir,
        runs=args.runs,
        seed_base=args.seed_base,
        scenarios=BOUNDARY_SCENARIOS,
    )
    quantized_result = run_phase16_experiments(
        config=quantized_config,
        output_dir=quantized_dir,
        runs=args.runs,
        seed_base=args.seed_base,
        scenarios=BOUNDARY_SCENARIOS,
    )

    comparison_rows = build_comparison_rows(
        baseline_result["summary_rows"],
        quantized_result["summary_rows"],
    )
    save_csv(args.output_dir / "quantized_pilot_comparison.csv", comparison_rows)
    save_json(
        args.output_dir / "quantized_pilot_comparison.json",
        {
            "variant": args.variant,
            "base_config": asdict(base),
            "baseline_config": asdict(baseline_config),
            "quantized_config": asdict(quantized_config),
            "comparison_rows": comparison_rows,
        },
    )

    render_metric_chart(
        comparison_rows,
        args.output_dir / "quantized_vs_baseline_coherence.png",
        "late_recovery_coherence_baseline",
        "late_recovery_coherence_quantized",
        "Late Recovery Coherence",
    )
    render_metric_chart(
        comparison_rows,
        args.output_dir / "quantized_vs_baseline_false_isolation.png",
        "final_false_isolation_baseline",
        "final_false_isolation_quantized",
        "Final False Isolation",
    )
    render_quantized_chart(
        comparison_rows,
        args.output_dir / "quantized_late_metrics.png",
    )


if __name__ == "__main__":
    main()
