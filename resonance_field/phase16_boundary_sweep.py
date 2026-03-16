from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from resonance_field.phase16_config import PhaseXVIConfig
from resonance_field.phase16_experiment import run_phase16_experiments
from resonance_field.phase6_config import PhaseVIScenario


BOUNDARY_SCENARIOS = [
    PhaseVIScenario(
        model_kind="hybrid_manifold",
        bootstrap_mode="periodic",
        false_mode="random",
        name="hybrid_manifold_periodic_random",
        label="Hybrid periodic random | boundary sweep",
    ),
    PhaseVIScenario(
        model_kind="hybrid_manifold",
        bootstrap_mode="periodic",
        false_mode="noisy",
        name="hybrid_manifold_periodic_noisy",
        label="Hybrid periodic noisy | boundary sweep",
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


def estimate_dense_live_gb(nodes: int) -> float:
    matrix_cells = float(nodes) * float(nodes)
    one_float_bytes = matrix_cells * 8.0
    one_bool_bytes = matrix_cells
    rough_live_bytes = 6.0 * one_float_bytes + 2.0 * one_bool_bytes
    return rough_live_bytes / (1024.0 ** 3)


def build_boundary_variants(base: PhaseXVIConfig) -> list[tuple[str, PhaseXVIConfig]]:
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
        (
            "critical_push",
            replace(
                base,
                phase_manipulation_phase_shift=0.46,
                phase_manipulation_freq_shift=0.038,
                phase_manipulation_phase_blend=0.62,
                phase_manipulation_freq_blend=0.36,
                phase_manipulation_geometry_shift=0.14,
                phase_manipulation_helix_shift=0.32,
            ),
        ),
    ]


def select_boundary_variants(
    base: PhaseXVIConfig,
    variant_names: list[str] | None = None,
) -> list[tuple[str, PhaseXVIConfig]]:
    variants = build_boundary_variants(base)
    if not variant_names:
        return variants
    allowed = {name.strip() for name in variant_names if name.strip()}
    return [(name, cfg) for name, cfg in variants if name in allowed]


def classify_direction(row: dict[str, Any]) -> str:
    upward = float(row.get("phase_manipulation_upward_spread_fraction_mean", 0.0))
    downward = float(row.get("phase_manipulation_downward_spread_fraction_mean", 0.0))
    if upward <= 0.0 and downward <= 0.0:
        return "none"
    if upward > 0.0 and downward > 0.0:
        if abs(upward - downward) <= 0.20:
            return "mixed"
        return "upward" if upward > downward else "downward"
    if upward > 0.0:
        return "upward"
    return "downward"


def classify_regime(row: dict[str, Any]) -> str:
    event_rate = float(row.get("phase_manipulation_event_rate_mean", 0.0))
    contamination = float(row.get("phase_manipulation_core_contamination_rate_mean", 0.0))
    late_coherence = float(row.get("late_recovery_coherence_mean", 0.0))
    pre_shock_coherence = float(row.get("pre_shock_coherence_mean", 1.0))
    recovery_success = float(row.get("recovery_success_mean", 0.0))
    coherence_ratio = late_coherence / max(pre_shock_coherence, 1e-9)

    if event_rate <= 0.0 and coherence_ratio >= 0.88:
        return "rigid_no_effect"
    if contamination >= 0.55 or coherence_ratio < 0.60 or recovery_success < 0.50:
        return "destructive"
    if contamination >= 0.38 or coherence_ratio < 0.82:
        return "boundary"
    return "reinforcing"


def run_phase16_boundary_sweep(
    base_config: PhaseXVIConfig,
    output_dir: Path,
    runs: int,
    seed_base: int,
    node_counts: list[int],
    max_live_gb: float = 24.0,
    variant_names: list[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_summary: list[dict[str, Any]] = []
    aggregate_runs: list[dict[str, Any]] = []
    sweep_manifest: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for node_index, nodes in enumerate(node_counts):
        estimated_live_gb = estimate_dense_live_gb(nodes)
        if estimated_live_gb > max_live_gb:
            skipped.append(
                {
                    "nodes": nodes,
                    "estimated_live_gb": estimated_live_gb,
                    "reason": "estimated dense live set exceeds configured safety budget",
                }
            )
            continue

        node_base = replace(base_config, nodes=nodes)
        node_dir = output_dir / f"nodes_{nodes}"

        selected_variants = select_boundary_variants(node_base, variant_names)
        for variant_index, (variant_name, variant_config) in enumerate(selected_variants):
            variant_dir = node_dir / variant_name
            result = run_phase16_experiments(
                config=variant_config,
                output_dir=variant_dir,
                runs=runs,
                seed_base=seed_base + node_index * 100000 + variant_index * 10000,
                scenarios=BOUNDARY_SCENARIOS,
            )

            for row in result["summary_rows"]:
                enriched = {
                    "nodes": nodes,
                    "estimated_live_gb": estimated_live_gb,
                    "variant": variant_name,
                    "direction_class": classify_direction(row),
                    "regime_class": classify_regime(row),
                    **row,
                }
                aggregate_summary.append(enriched)

            for row in result["run_rows"]:
                aggregate_runs.append(
                    {
                        "nodes": nodes,
                        "estimated_live_gb": estimated_live_gb,
                        "variant": variant_name,
                        **row,
                    }
                )

            sweep_manifest.append(
                {
                    "nodes": nodes,
                    "variant": variant_name,
                    "output_dir": str(variant_dir),
                    "estimated_live_gb": estimated_live_gb,
                    "config": asdict(variant_config),
                }
            )

            # Persist aggregate progress after each completed variant so long
            # sweeps still leave behind analyzable partial outputs.
            _save_csv(output_dir / "boundary_summary.csv", aggregate_summary)
            _save_csv(output_dir / "boundary_runs.csv", aggregate_runs)
            _save_json(
                output_dir / "boundary_summary.json",
                {
                    "base_config": asdict(base_config),
                    "node_counts": node_counts,
                    "max_live_gb": max_live_gb,
                    "variants": sweep_manifest,
                    "skipped": skipped,
                    "summary_rows": aggregate_summary,
                    "run_rows": aggregate_runs,
                    "is_partial_checkpoint": True,
                },
            )

    _save_csv(output_dir / "boundary_summary.csv", aggregate_summary)
    _save_csv(output_dir / "boundary_runs.csv", aggregate_runs)
    _save_json(
        output_dir / "boundary_summary.json",
        {
            "base_config": asdict(base_config),
            "node_counts": node_counts,
            "max_live_gb": max_live_gb,
            "variants": sweep_manifest,
            "skipped": skipped,
            "summary_rows": aggregate_summary,
            "run_rows": aggregate_runs,
        },
    )

    return {
        "output_dir": output_dir,
        "summary_rows": aggregate_summary,
        "run_rows": aggregate_runs,
        "variants": sweep_manifest,
        "skipped": skipped,
    }
