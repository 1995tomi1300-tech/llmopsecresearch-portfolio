from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from resonance_field.numeric_backend import (
    NumericBackend,
    backend_info_dict,
    benchmark_pairwise_kernels,
    resolve_numeric_backend,
)
from resonance_field.phase6_experiment import run_phase6_experiments
from resonance_field.phase13_config import (
    PHASE13_MODEL_LABELS,
    PhaseXIIIConfig,
    build_phase13_scenarios,
)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_phase13_experiments(
    config: PhaseXIIIConfig,
    output_dir: Path,
    runs: int,
    seed_base: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_numeric_backend(config)
    benchmark_rows: list[dict[str, Any]] = [
        benchmark_pairwise_kernels(
            NumericBackend(
                request="cpu",
                mode="cpu",
                library="numpy",
                device="cpu",
                precision="float64",
                note="reference cpu benchmark",
            ),
            nodes=config.benchmark_nodes,
            repeats=config.benchmark_repeats,
            seed=seed_base,
        )
    ]
    if resolved.mode == "gpu":
        benchmark_rows.append(
            benchmark_pairwise_kernels(
                resolved,
                nodes=config.benchmark_nodes,
                repeats=config.benchmark_repeats,
                seed=seed_base,
            )
        )

    _save_json(
        output_dir / "backend_info.json",
        {
            "config": asdict(config),
            "resolved_backend": backend_info_dict(resolved),
            "benchmarks": benchmark_rows,
        },
    )
    _save_csv(output_dir / "backend_benchmark.csv", benchmark_rows)

    return run_phase6_experiments(
        config=config,
        output_dir=output_dir,
        runs=runs,
        seed_base=seed_base,
        scenarios=build_phase13_scenarios(),
        model_labels=PHASE13_MODEL_LABELS,
        experiment_tag=config.experiment_tag,
        baseline_summary_path=Path("/mnt/d/resonance_phase12_outputs/scenario_summary.csv"),
        comparison_filename="phase12_phase13_delta.csv",
        comparison_json_key="phase12_phase13_delta",
    )
