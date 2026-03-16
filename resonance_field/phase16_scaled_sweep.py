from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from resonance_field.phase16_config import PhaseXVIConfig
from resonance_field.phase16_experiment import run_phase16_experiments
from resonance_field.phase6_config import PhaseVIScenario


SCALED_SCENARIOS = [
    PhaseVIScenario(
        model_kind="helix",
        bootstrap_mode="periodic",
        false_mode="random",
        name="helix_periodic_random",
        label="Helix periodic random | scaled manipulation",
    ),
    PhaseVIScenario(
        model_kind="helix",
        bootstrap_mode="periodic",
        false_mode="noisy",
        name="helix_periodic_noisy",
        label="Helix periodic noisy | scaled manipulation",
    ),
    PhaseVIScenario(
        model_kind="hybrid_manifold",
        bootstrap_mode="periodic",
        false_mode="random",
        name="hybrid_manifold_periodic_random",
        label="Hybrid periodic random | scaled manipulation",
    ),
    PhaseVIScenario(
        model_kind="hybrid_manifold",
        bootstrap_mode="periodic",
        false_mode="noisy",
        name="hybrid_manifold_periodic_noisy",
        label="Hybrid periodic noisy | scaled manipulation",
    ),
]


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_variant_configs(base: PhaseXVIConfig) -> list[tuple[str, PhaseXVIConfig]]:
    return [
        (
            "subtle_balanced",
            replace(
                base,
                phase_manipulation_phase_shift=0.18,
                phase_manipulation_freq_shift=0.014,
                phase_manipulation_phase_blend=0.34,
                phase_manipulation_freq_blend=0.18,
                phase_manipulation_geometry_shift=0.05,
                phase_manipulation_helix_shift=0.12,
            ),
        ),
        (
            "mid_balanced",
            replace(
                base,
                phase_manipulation_phase_shift=0.28,
                phase_manipulation_freq_shift=0.022,
                phase_manipulation_phase_blend=0.42,
                phase_manipulation_freq_blend=0.24,
                phase_manipulation_geometry_shift=0.08,
                phase_manipulation_helix_shift=0.18,
            ),
        ),
        (
            "strong_balanced",
            replace(
                base,
                phase_manipulation_phase_shift=0.36,
                phase_manipulation_freq_shift=0.030,
                phase_manipulation_phase_blend=0.52,
                phase_manipulation_freq_blend=0.30,
                phase_manipulation_geometry_shift=0.11,
                phase_manipulation_helix_shift=0.24,
            ),
        ),
    ]


def run_phase16_scaled_sweep(
    base_config: PhaseXVIConfig,
    output_dir: Path,
    runs: int,
    seed_base: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_summary: list[dict[str, Any]] = []
    aggregate_runs: list[dict[str, Any]] = []
    variant_runs: list[dict[str, Any]] = []

    for variant_index, (variant_name, variant_config) in enumerate(build_variant_configs(base_config)):
        variant_dir = output_dir / variant_name
        result = run_phase16_experiments(
            config=variant_config,
            output_dir=variant_dir,
            runs=runs,
            seed_base=seed_base + variant_index * 10000,
            scenarios=SCALED_SCENARIOS,
        )

        summary_rows = list(result["summary_rows"])
        run_rows = list(result["run_rows"])
        for row in summary_rows:
            enriched = {"variant": variant_name, **row}
            aggregate_summary.append(enriched)
        for row in run_rows:
            enriched = {"variant": variant_name, **row}
            aggregate_runs.append(enriched)

        variant_runs.append(
            {
                "variant": variant_name,
                "output_dir": str(variant_dir),
                "config": asdict(variant_config),
            }
        )

    _save_csv(output_dir / "scaled_variant_summary.csv", aggregate_summary)
    _save_csv(output_dir / "scaled_variant_runs.csv", aggregate_runs)
    _save_json(
        output_dir / "scaled_variant_summary.json",
        {
            "base_config": asdict(base_config),
            "variants": variant_runs,
            "summary_rows": aggregate_summary,
            "run_rows": aggregate_runs,
        },
    )

    return {
        "output_dir": output_dir,
        "summary_rows": aggregate_summary,
        "run_rows": aggregate_runs,
        "variants": variant_runs,
    }
