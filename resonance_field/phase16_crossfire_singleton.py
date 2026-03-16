from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resonance_field.phase16_config import PhaseXVIConfig
from resonance_field.phase16_experiment import run_phase16_experiments
from resonance_field.phase16_quantized_pilot import apply_variant
from resonance_field.phase6_config import PhaseVIScenario


SCENARIOS = {
    "random": PhaseVIScenario(
        model_kind="hybrid_manifold",
        bootstrap_mode="periodic",
        false_mode="random",
        name="hybrid_manifold_periodic_random_crossfire",
        label="Hybrid manifold periodic random | spherical crossfire singleton",
    ),
    "noisy": PhaseVIScenario(
        model_kind="hybrid_manifold",
        bootstrap_mode="periodic",
        false_mode="noisy",
        name="hybrid_manifold_periodic_noisy_crossfire",
        label="Hybrid manifold periodic noisy | spherical crossfire singleton",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single aggressive spherical crossfire training experiment."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/d/resonance_phase16_crossfire_singleton"),
    )
    parser.add_argument("--nodes", type=int, default=120)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=112233)
    parser.add_argument("--backend", default="auto")
    parser.add_argument(
        "--variant",
        default="critical_push",
        choices=("subtle_balanced", "mid_balanced", "strong_balanced", "critical_push"),
    )
    parser.add_argument(
        "--false-mode",
        default="noisy",
        choices=("random", "noisy"),
    )
    return parser.parse_args()


def build_crossfire_config(base: PhaseXVIConfig) -> PhaseXVIConfig:
    return replace(
        base,
        experiment_tag="Phase XVI Crossfire Singleton",
        collect_event_rows=True,
        quantized_scalar_vector_enabled=True,
        quantized_scalar_levels=1000,
        quantized_anchor_scalar=0.53,
        quantized_scalar_phase_weight=0.24,
        quantized_scalar_freq_weight=0.18,
        quantized_scalar_trust_weight=0.30,
        quantized_scalar_memory_weight=0.28,
        quantized_reference_blend=0.68,
        quantized_active_window=base.steps,
        quantized_coherence_trigger=1.05,
        quantized_disturbance_trigger=0.0,
        quantized_core_protect_top_fraction=0.12,
        quantized_shell_trust_min=0.14,
        quantized_shell_trust_max=0.92,
        quantized_shell_disturbance_min=0.0,
        quantized_shell_disturbance_max=1.0,
        quantized_receptive_trust_floor=0.0,
        quantized_receptive_trust_ceiling=0.92,
        quantized_receptive_disturbance_max=1.0,
        quantized_repulsive_trust_max=0.28,
        quantized_repulsive_disturbance_min=0.56,
        quantized_lock_width_bins=10.0,
        quantized_drive_scale=0.46,
        quantized_repulsive_scale=0.78,
        quantized_vector_max_pull=0.95,
        quantized_vector_phase_gain=0.060,
        quantized_vector_freq_gain=0.040,
        quantized_vector_geometry_gain=0.024,
        quantized_vector_helix_gain=0.028,
        modulation_training_enabled=True,
        modulation_active_window=base.steps,
        modulation_interval=1,
        modulation_cooldown=0,
        modulation_focus_fraction=0.58,
        modulation_beam_count=36,
        modulation_angle_spread=math.pi,
        modulation_spatial_sigma=1.45,
        modulation_phase_strength=0.44,
        modulation_freq_blend=0.30,
        modulation_geometry_pull=1.05,
        modulation_memory_boost=0.88,
        modulation_coherence_floor=0.0,
        modulation_cluster_floor=0.0,
        modulation_disturbance_ceiling=1.0,
        modulation_min_trust=0.0,
        modulation_score_threshold=-0.18,
        modulation_ki_base=0.38,
        modulation_ki_gain=1.08,
        modulation_ki_max=1.85,
        phase_manipulation_enabled=True,
        phase_manipulation_active_window=base.steps,
        phase_manipulation_interval=1,
        phase_manipulation_cooldown=0,
        phase_manipulation_max_events=max(24, base.steps // 3),
        phase_manipulation_trust_min=0.10,
        phase_manipulation_trust_max=0.90,
        phase_manipulation_trust_margin=0.12,
        phase_manipulation_degree_min_ratio=0.18,
        phase_manipulation_degree_max_ratio=1.60,
        phase_manipulation_memory_ceiling=0.95,
        phase_manipulation_disturbance_ceiling=1.0,
        phase_manipulation_phase_shift=0.56,
        phase_manipulation_freq_shift=0.046,
        phase_manipulation_phase_blend=0.68,
        phase_manipulation_freq_blend=0.42,
        phase_manipulation_geometry_shift=0.18,
        phase_manipulation_helix_shift=0.38,
        phase_manipulation_geometry_steps=24,
        phase_manipulation_relax_steps=8,
        phase_manipulation_tracking_window=base.steps,
        phase_manipulation_align_threshold=0.18,
        phase_manipulation_freq_threshold=0.055,
        phase_manipulation_recovery_phase_tol=0.32,
        phase_manipulation_recovery_freq_tol=0.022,
    )


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def counter_by_step(rows: list[dict[str, str]]) -> Counter[int]:
    counter: Counter[int] = Counter()
    for row in rows:
        try:
            counter[int(row["step"])] += 1
        except (KeyError, TypeError, ValueError):
            continue
    return counter


def render_event_pressure_chart(
    output_path: Path,
    steps: int,
    modulation_rows: list[dict[str, str]],
    absorber_rows: list[dict[str, str]],
    guardian_rows: list[dict[str, str]],
    phase_rows: list[dict[str, str]],
) -> None:
    modulation = counter_by_step(modulation_rows)
    absorber = counter_by_step(absorber_rows)
    guardian = counter_by_step(guardian_rows)
    phase = counter_by_step(phase_rows)
    xs = list(range(steps))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(xs, [modulation.get(step, 0) for step in xs], label="crossfire modulation")
    ax.plot(xs, [phase.get(step, 0) for step in xs], label="phase manipulation")
    ax.plot(xs, [absorber.get(step, 0) for step in xs], label="absorber reflex")
    ax.plot(xs, [guardian.get(step, 0) for step in xs], label="guardian reflex")
    ax.set_title("Crossfire Pressure vs Immune Reflex")
    ax.set_xlabel("step")
    ax.set_ylabel("event count")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_event_mass_chart(
    output_path: Path,
    modulation_rows: list[dict[str, str]],
    absorber_rows: list[dict[str, str]],
) -> None:
    modulation_targets = [
        float(row.get("target_size", 0.0)) * float(row.get("mean_ki", 0.0))
        for row in modulation_rows
    ]
    absorber_mass = [
        float(row.get("distressed_size", 0.0)) * float(row.get("mean_disturbance", 0.0))
        for row in absorber_rows
    ]

    labels = ["crossfire mass", "absorber load"]
    values = [
        float(sum(modulation_targets)),
        float(sum(absorber_mass)),
    ]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.bar(labels, values, color=["#ef6c00", "#1565c0"])
    ax.set_title("Injected Pressure vs Absorbed Load")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_report(
    output_dir: Path,
    config: PhaseXVIConfig,
    scenario: PhaseVIScenario,
) -> dict[str, Any]:
    summary_rows = load_csv_rows(output_dir / "scenario_summary.csv")
    if not summary_rows:
        raise RuntimeError("scenario_summary.csv was not produced by the run.")
    row = summary_rows[0]

    modulation_rows = load_csv_rows(output_dir / "modulation_events.csv")
    absorber_rows = load_csv_rows(output_dir / "absorber_events.csv")
    guardian_rows = load_csv_rows(output_dir / "guardian_events.csv")
    phase_rows = load_csv_rows(output_dir / "phase_manipulation_events.csv")

    render_event_pressure_chart(
        output_dir / "crossfire_event_pressure.png",
        config.steps,
        modulation_rows,
        absorber_rows,
        guardian_rows,
        phase_rows,
    )
    render_event_mass_chart(
        output_dir / "crossfire_event_mass.png",
        modulation_rows,
        absorber_rows,
    )

    report = {
        "scenario": scenario.name,
        "config": asdict(config),
        "summary_row": row,
        "event_totals": {
            "modulation_events": len(modulation_rows),
            "absorber_events": len(absorber_rows),
            "guardian_events": len(guardian_rows),
            "phase_manipulation_events": len(phase_rows),
        },
        "event_mass": {
            "crossfire_modulation_mass": float(
                sum(
                    float(item.get("target_size", 0.0)) * float(item.get("mean_ki", 0.0))
                    for item in modulation_rows
                )
            ),
            "absorber_load_mass": float(
                sum(
                    float(item.get("distressed_size", 0.0))
                    * float(item.get("mean_disturbance", 0.0))
                    for item in absorber_rows
                )
            ),
        },
    }

    with (output_dir / "crossfire_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    with (output_dir / "crossfire_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "late_recovery_coherence_mean",
                "final_false_isolation_rate_mean",
                "recovery_success_mean",
                "modulation_events",
                "absorber_events",
                "guardian_events",
                "phase_manipulation_events",
                "crossfire_modulation_mass",
                "absorber_load_mass",
                "quantized_scalar_lock_fraction_late_mean",
                "quantized_repulsive_fraction_late_mean",
                "quantized_vector_drive_mean_late_mean",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scenario": scenario.name,
                "late_recovery_coherence_mean": row.get("late_recovery_coherence_mean", 0.0),
                "final_false_isolation_rate_mean": row.get(
                    "final_false_isolation_rate_mean", 0.0
                ),
                "recovery_success_mean": row.get("recovery_success_mean", 0.0),
                "modulation_events": len(modulation_rows),
                "absorber_events": len(absorber_rows),
                "guardian_events": len(guardian_rows),
                "phase_manipulation_events": len(phase_rows),
                "crossfire_modulation_mass": report["event_mass"][
                    "crossfire_modulation_mass"
                ],
                "absorber_load_mass": report["event_mass"]["absorber_load_mass"],
                "quantized_scalar_lock_fraction_late_mean": row.get(
                    "quantized_scalar_lock_fraction_late_mean", 0.0
                ),
                "quantized_repulsive_fraction_late_mean": row.get(
                    "quantized_repulsive_fraction_late_mean", 0.0
                ),
                "quantized_vector_drive_mean_late_mean": row.get(
                    "quantized_vector_drive_mean_late_mean", 0.0
                ),
            }
        )

    return report


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenario = SCENARIOS[args.false_mode]
    base = PhaseXVIConfig(
        nodes=args.nodes,
        steps=args.steps,
        shock_step=max(24, args.steps // 4),
        late_window=max(40, args.steps // 4),
        benchmark_repeats=2,
        compute_backend=args.backend,
        collect_event_rows=True,
    )
    base = apply_variant(base, args.variant)
    crossfire_config = build_crossfire_config(base)

    run_phase16_experiments(
        config=crossfire_config,
        output_dir=args.output_dir,
        runs=1,
        seed_base=args.seed,
        scenarios=[scenario],
    )
    build_report(args.output_dir, crossfire_config, scenario)


if __name__ == "__main__":
    main()
