from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resonance_field.phase16_boundary_sweep import BOUNDARY_SCENARIOS
from resonance_field.phase16_experiment import run_phase16_experiments
from resonance_field.phase16_teaching_profile import (
    TEACHING_PROFILE_NAME,
    describe_loading_teaching_profile,
    build_loading_teaching_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase XVI teaching profile.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/d/resonance_phase16_teaching_run"),
    )
    parser.add_argument("--nodes", type=int, default=120)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=150000)
    parser.add_argument("--backend", default="auto")
    parser.add_argument(
        "--variant",
        default="critical_push",
        choices=("subtle_balanced", "mid_balanced", "strong_balanced", "critical_push"),
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_training_report(output_dir: Path) -> dict[str, Any]:
    summary_rows = load_csv_rows(output_dir / "scenario_summary.csv")
    if not summary_rows:
        raise RuntimeError("scenario_summary.csv missing after teaching run.")

    report_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        report_rows.append(
            {
                "scenario": row["scenario"],
                "false_mode": row["false_mode"],
                "late_recovery_coherence_mean": float(row["late_recovery_coherence_mean"]),
                "final_false_isolation_rate_mean": float(
                    row["final_false_isolation_rate_mean"]
                ),
                "late_trust_mean_mean": float(row["late_trust_mean_mean"]),
                "modulation_event_rate_mean": float(row["modulation_event_rate_mean"]),
                "phase_manipulation_event_rate_mean": float(
                    row["phase_manipulation_event_rate_mean"]
                ),
                "phase_manipulation_core_contamination_rate_mean": float(
                    row["phase_manipulation_core_contamination_rate_mean"]
                ),
                "quantized_vector_drive_mean_late_mean": float(
                    row["quantized_vector_drive_mean_late_mean"]
                ),
            }
        )

    payload = {
        "profile_name": TEACHING_PROFILE_NAME,
        "rows": report_rows,
    }
    with (output_dir / "teaching_report.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = build_loading_teaching_profile(
        nodes=args.nodes,
        steps=args.steps,
        backend=args.backend,
        variant=args.variant,
    )
    manifest = describe_loading_teaching_profile(config)
    with (args.output_dir / "teaching_profile_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    run_phase16_experiments(
        config=config,
        output_dir=args.output_dir,
        runs=args.runs,
        seed_base=args.seed_base,
        scenarios=BOUNDARY_SCENARIOS,
    )
    build_training_report(args.output_dir)


if __name__ == "__main__":
    main()
