from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resonance_field.phase16_config import PhaseXVIConfig
from resonance_field.phase16_crossfire_singleton import SCENARIOS, build_crossfire_config
from resonance_field.phase16_experiment import run_phase16_experiments
from resonance_field.phase16_quantized_pilot import apply_variant


PROFILE_STRENGTHS: list[tuple[str, float]] = [
    ("control", 0.00),
    ("seed", 0.18),
    ("pulse", 0.36),
    ("loading", 0.54),
    ("threshold", 0.72),
    ("saturation", 1.00),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map the crossfire threshold from control to saturation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/d/resonance_phase16_crossfire_threshold_map"),
    )
    parser.add_argument("--nodes", type=int, default=120)
    parser.add_argument("--steps", type=int, default=220)
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


def interpolate_value(base: Any, target: Any, strength: float) -> Any:
    if isinstance(base, bool) and isinstance(target, bool):
        return target if strength > 0.0 else base
    if isinstance(base, int) and isinstance(target, int):
        return int(round(base + (target - base) * strength))
    if isinstance(base, float) and isinstance(target, float):
        return float(base + (target - base) * strength)
    return target if strength > 0.0 else base


def build_profile_config(base: PhaseXVIConfig, strength: float) -> PhaseXVIConfig:
    if strength <= 0.0:
        return replace(base, quantized_scalar_vector_enabled=False)

    target = build_crossfire_config(base)
    overrides: dict[str, Any] = {"experiment_tag": f"Phase XVI Crossfire Threshold {strength:.2f}"}
    for field in fields(PhaseXVIConfig):
        name = field.name
        if name == "experiment_tag":
            continue
        base_value = getattr(base, name)
        target_value = getattr(target, name)
        if base_value == target_value:
            continue
        overrides[name] = interpolate_value(base_value, target_value, strength)

    # Keep the intervention alive once strength is non-zero.
    overrides["quantized_scalar_vector_enabled"] = True
    overrides["modulation_training_enabled"] = True
    overrides["phase_manipulation_enabled"] = True
    return replace(base, **overrides)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def classify_zone(
    coherence: float,
    false_isolation: float,
    control_coherence: float,
    control_false_isolation: float,
    contamination: float,
) -> str:
    coherence_ratio = coherence / max(control_coherence, 1e-9)
    isolation_ratio = false_isolation / max(control_false_isolation, 1e-9)
    if coherence_ratio >= 0.92 and isolation_ratio >= 0.82:
        return "resilient"
    if coherence_ratio >= 0.72 and isolation_ratio >= 0.45 and contamination < 0.24:
        return "absorptive"
    if coherence_ratio >= 0.50 and isolation_ratio >= 0.20:
        return "threshold"
    return "saturated"


def render_threshold_lines(rows: list[dict[str, Any]], output_path: Path) -> None:
    xs = [float(row["strength"]) for row in rows]
    coherence = [float(row["late_recovery_coherence_mean"]) for row in rows]
    isolation = [float(row["final_false_isolation_rate_mean"]) for row in rows]
    trust = [float(row["late_trust_mean_mean"]) for row in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(xs, coherence, marker="o", label="late coherence")
    ax.plot(xs, isolation, marker="o", label="false isolation")
    ax.plot(xs, trust, marker="o", label="late trust")
    ax.set_xlabel("crossfire strength")
    ax.set_ylabel("metric value")
    ax.set_title("Crossfire Threshold Map")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_threshold_scatter(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = {
        "resilient": "#2e7d32",
        "absorptive": "#1565c0",
        "threshold": "#ef6c00",
        "saturated": "#c62828",
    }
    for row in rows:
        zone = str(row["zone"])
        ax.scatter(
            float(row["final_false_isolation_rate_mean"]),
            float(row["late_recovery_coherence_mean"]),
            s=55 + 120 * float(row["strength"]),
            color=colors.get(zone, "#424242"),
        )
        ax.text(
            float(row["final_false_isolation_rate_mean"]) + 0.01,
            float(row["late_recovery_coherence_mean"]) + 0.005,
            f"{row['profile']}",
            fontsize=8,
        )
    ax.set_xlabel("false isolation")
    ax.set_ylabel("late coherence")
    ax.set_title("Crossfire Regime Map")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_immune_load(rows: list[dict[str, Any]], output_path: Path) -> None:
    xs = [float(row["strength"]) for row in rows]
    modulation = [float(row["modulation_event_rate_mean"]) for row in rows]
    absorber = [float(row["absorber_event_rate_mean"]) for row in rows]
    contamination = [float(row["phase_manipulation_core_contamination_rate_mean"]) for row in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(xs, modulation, marker="o", label="modulation event rate")
    ax.plot(xs, absorber, marker="o", label="absorber event rate")
    ax.plot(xs, contamination, marker="o", label="core contamination")
    ax.set_xlabel("crossfire strength")
    ax.set_ylabel("load / contamination")
    ax.set_title("Immune Load Under Crossfire")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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

    rows: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []

    control_coherence = None
    control_false_isolation = None

    for index, (profile, strength) in enumerate(PROFILE_STRENGTHS):
        config = build_profile_config(base, strength)
        profile_dir = args.output_dir / profile
        run_phase16_experiments(
            config=config,
            output_dir=profile_dir,
            runs=1,
            seed_base=args.seed + index * 1000,
            scenarios=[scenario],
        )
        summary = load_csv_rows(profile_dir / "scenario_summary.csv")
        if not summary:
            raise RuntimeError(f"No scenario_summary.csv for profile {profile}.")
        row = dict(summary[0])
        row["profile"] = profile
        row["strength"] = strength
        row["output_dir"] = str(profile_dir)

        if control_coherence is None:
            control_coherence = float(row["late_recovery_coherence_mean"])
            control_false_isolation = float(row["final_false_isolation_rate_mean"])

        row["zone"] = classify_zone(
            float(row["late_recovery_coherence_mean"]),
            float(row["final_false_isolation_rate_mean"]),
            float(control_coherence),
            float(control_false_isolation),
            float(row["phase_manipulation_core_contamination_rate_mean"]),
        )
        rows.append(row)
        configs.append({"profile": profile, "strength": strength, "config": asdict(config)})

    csv_path = args.output_dir / "crossfire_threshold_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (args.output_dir / "crossfire_threshold_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "scenario": asdict(scenario),
                "profiles": configs,
                "rows": rows,
            },
            handle,
            indent=2,
        )

    render_threshold_lines(rows, args.output_dir / "crossfire_threshold_lines.png")
    render_threshold_scatter(rows, args.output_dir / "crossfire_threshold_scatter.png")
    render_immune_load(rows, args.output_dir / "crossfire_threshold_immune_load.png")


if __name__ == "__main__":
    main()
