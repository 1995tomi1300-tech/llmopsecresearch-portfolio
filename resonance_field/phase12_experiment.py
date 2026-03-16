from __future__ import annotations

from pathlib import Path
from typing import Any

from resonance_field.phase6_experiment import run_phase6_experiments
from resonance_field.phase12_config import (
    PHASE12_MODEL_LABELS,
    PhaseXIIConfig,
    build_phase12_scenarios,
)


def run_phase12_experiments(
    config: PhaseXIIConfig,
    output_dir: Path,
    runs: int,
    seed_base: int,
) -> dict[str, Any]:
    return run_phase6_experiments(
        config=config,
        output_dir=output_dir,
        runs=runs,
        seed_base=seed_base,
        scenarios=build_phase12_scenarios(),
        model_labels=PHASE12_MODEL_LABELS,
        experiment_tag=config.experiment_tag,
        baseline_summary_path=Path("/mnt/d/resonance_phase11_outputs/scenario_summary.csv"),
        comparison_filename="phase11_phase12_delta.csv",
        comparison_json_key="phase11_phase12_delta",
    )
