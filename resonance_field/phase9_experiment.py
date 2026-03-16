from __future__ import annotations

from pathlib import Path
from typing import Any

from resonance_field.phase6_experiment import run_phase6_experiments
from resonance_field.phase9_config import (
    PHASE9_MODEL_LABELS,
    PhaseIXConfig,
    build_phase9_scenarios,
)


def run_phase9_experiments(
    config: PhaseIXConfig,
    output_dir: Path,
    runs: int,
    seed_base: int,
) -> dict[str, Any]:
    return run_phase6_experiments(
        config=config,
        output_dir=output_dir,
        runs=runs,
        seed_base=seed_base,
        scenarios=build_phase9_scenarios(),
        model_labels=PHASE9_MODEL_LABELS,
        experiment_tag=config.experiment_tag,
        baseline_summary_path=Path("/mnt/d/resonance_phase8_outputs/scenario_summary.csv"),
        comparison_filename="phase8_phase9_delta.csv",
        comparison_json_key="phase8_phase9_delta",
    )
