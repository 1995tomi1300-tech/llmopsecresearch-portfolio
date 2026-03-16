from __future__ import annotations

from pathlib import Path
from typing import Any

from resonance_field.phase6_experiment import run_phase6_experiments
from resonance_field.phase8_config import (
    PHASE8_MODEL_LABELS,
    PhaseVIIIConfig,
    build_phase8_scenarios,
)


def run_phase8_experiments(
    config: PhaseVIIIConfig,
    output_dir: Path,
    runs: int,
    seed_base: int,
) -> dict[str, Any]:
    return run_phase6_experiments(
        config=config,
        output_dir=output_dir,
        runs=runs,
        seed_base=seed_base,
        scenarios=build_phase8_scenarios(),
        model_labels=PHASE8_MODEL_LABELS,
        experiment_tag=config.experiment_tag,
        baseline_summary_path=Path("/mnt/d/resonance_phase7_outputs/scenario_summary.csv"),
        comparison_filename="phase7_phase8_delta.csv",
        comparison_json_key="phase7_phase8_delta",
    )
