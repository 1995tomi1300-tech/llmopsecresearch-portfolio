from __future__ import annotations

from pathlib import Path
from typing import Any

from resonance_field.phase6_experiment import run_phase6_experiments
from resonance_field.phase10_config import (
    PHASE10_MODEL_LABELS,
    PhaseXConfig,
    build_phase10_scenarios,
)


def run_phase10_experiments(
    config: PhaseXConfig,
    output_dir: Path,
    runs: int,
    seed_base: int,
) -> dict[str, Any]:
    return run_phase6_experiments(
        config=config,
        output_dir=output_dir,
        runs=runs,
        seed_base=seed_base,
        scenarios=build_phase10_scenarios(),
        model_labels=PHASE10_MODEL_LABELS,
        experiment_tag=config.experiment_tag,
        baseline_summary_path=Path("/mnt/d/resonance_phase8_outputs/scenario_summary.csv"),
        comparison_filename="phase8_phase10_delta.csv",
        comparison_json_key="phase8_phase10_delta",
    )
