from __future__ import annotations

from pathlib import Path
from typing import Any

from resonance_field.phase6_experiment import run_phase6_experiments
from resonance_field.phase7_config import (
    PHASE7_MODEL_LABELS,
    PhaseVIIConfig,
    build_phase7_scenarios,
)


def run_phase7_experiments(
    config: PhaseVIIConfig,
    output_dir: Path,
    runs: int,
    seed_base: int,
) -> dict[str, Any]:
    return run_phase6_experiments(
        config=config,
        output_dir=output_dir,
        runs=runs,
        seed_base=seed_base,
        scenarios=build_phase7_scenarios(),
        model_labels=PHASE7_MODEL_LABELS,
        experiment_tag=config.experiment_tag,
        baseline_summary_path=Path("/mnt/d/resonance_phase6_outputs/scenario_summary.csv"),
        comparison_filename="phase6_phase7_delta.csv",
        comparison_json_key="phase6_phase7_delta",
    )
