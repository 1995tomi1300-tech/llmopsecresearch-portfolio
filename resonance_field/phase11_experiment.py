from __future__ import annotations

from pathlib import Path
from typing import Any

from resonance_field.phase6_experiment import run_phase6_experiments
from resonance_field.phase11_config import (
    PHASE11_MODEL_LABELS,
    PhaseXIConfig,
    build_phase11_scenarios,
)


def run_phase11_experiments(
    config: PhaseXIConfig,
    output_dir: Path,
    runs: int,
    seed_base: int,
) -> dict[str, Any]:
    return run_phase6_experiments(
        config=config,
        output_dir=output_dir,
        runs=runs,
        seed_base=seed_base,
        scenarios=build_phase11_scenarios(),
        model_labels=PHASE11_MODEL_LABELS,
        experiment_tag=config.experiment_tag,
        baseline_summary_path=Path("/mnt/d/resonance_phase10_outputs/scenario_summary.csv"),
        comparison_filename="phase10_phase11_delta.csv",
        comparison_json_key="phase10_phase11_delta",
    )
