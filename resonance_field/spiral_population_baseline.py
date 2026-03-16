from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from resonance_field.spiral_dual_gpu_batch import BatchRecord, load_records
from resonance_field.spiral_validation import (
    SpiralSignalVector,
    SpiralValidationEngine,
)


@dataclass(frozen=True)
class PopulationBaselineDecision:
    candidate_id: str
    logical_node_count: float
    trust_score: float
    false_positive_risk: float
    regime_label: str
    rerun_recommendation: str
    anomaly_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "logical_node_count": self.logical_node_count,
            "trust_score": self.trust_score,
            "false_positive_risk": self.false_positive_risk,
            "regime_label": self.regime_label,
            "rerun_recommendation": self.rerun_recommendation,
            "anomaly_count": self.anomaly_count,
        }


def record_to_signal(record: BatchRecord) -> SpiralSignalVector:
    return SpiralSignalVector(
        descriptor_consistency=record.descriptor_consistency,
        conformer_energy_variance=record.conformer_energy_variance,
        RMSD_cluster_density=record.rmsd_cluster_density,
        torsion_entropy=record.torsion_entropy,
        shape_similarity_index=record.shape_similarity_index,
        pharmacophore_deviation=record.pharmacophore_deviation,
        minimization_success_rate=record.minimization_success_rate,
        pipeline_instability_flags=record.effective_instability(),
    )


def scale_logical_node_counts(
    records: list[BatchRecord],
    logical_total: float,
) -> list[BatchRecord]:
    if not records:
        return []
    if logical_total <= 0.0:
        return records
    current_total = sum(max(0.0, record.logical_node_count) for record in records)
    if current_total <= 0.0:
        equal_weight = logical_total / len(records)
        return [
            BatchRecord(
                candidate_id=record.candidate_id,
                logical_node_count=equal_weight,
                descriptor_consistency=record.descriptor_consistency,
                conformer_energy_variance=record.conformer_energy_variance,
                rmsd_cluster_density=record.rmsd_cluster_density,
                torsion_entropy=record.torsion_entropy,
                shape_similarity_index=record.shape_similarity_index,
                pharmacophore_deviation=record.pharmacophore_deviation,
                minimization_success_rate=record.minimization_success_rate,
                pipeline_instability_flags=record.pipeline_instability_flags,
                protein_branch=record.protein_branch,
                missing_signal_count=record.missing_signal_count,
                report_path=record.report_path,
            )
            for record in records
        ]
    scale = logical_total / current_total
    return [
        BatchRecord(
            candidate_id=record.candidate_id,
            logical_node_count=record.logical_node_count * scale,
            descriptor_consistency=record.descriptor_consistency,
            conformer_energy_variance=record.conformer_energy_variance,
            rmsd_cluster_density=record.rmsd_cluster_density,
            torsion_entropy=record.torsion_entropy,
            shape_similarity_index=record.shape_similarity_index,
            pharmacophore_deviation=record.pharmacophore_deviation,
            minimization_success_rate=record.minimization_success_rate,
            pipeline_instability_flags=record.pipeline_instability_flags,
            protein_branch=record.protein_branch,
            missing_signal_count=record.missing_signal_count,
            report_path=record.report_path,
        )
        for record in records
    ]


def evaluate_population_baseline(
    records: list[BatchRecord],
    engine: SpiralValidationEngine | None = None,
) -> dict[str, Any]:
    validation_engine = engine or SpiralValidationEngine()
    decisions: list[PopulationBaselineDecision] = []
    logical_total = 0.0
    stable_mass = 0.0
    questionable_mass = 0.0
    discard_mass = 0.0
    anomaly_mass = 0.0
    trust_total = 0.0
    risk_total = 0.0

    for record in records:
        logical_count = max(0.0, float(record.logical_node_count))
        result = validation_engine.evaluate(
            record.candidate_id,
            record_to_signal(record),
            provenance={
                "protein_branch": record.protein_branch,
                "missing_signals": ["upstream_missing"] * max(0, record.missing_signal_count),
                "report_path": record.report_path,
            },
        )
        decision = PopulationBaselineDecision(
            candidate_id=record.candidate_id,
            logical_node_count=logical_count,
            trust_score=result.trust_score,
            false_positive_risk=result.false_positive_risk,
            regime_label=result.regime_label,
            rerun_recommendation=result.rerun_recommendation,
            anomaly_count=len(result.anomaly_flags),
        )
        decisions.append(decision)
        logical_total += logical_count
        trust_total += logical_count * decision.trust_score
        risk_total += logical_count * decision.false_positive_risk
        if decision.regime_label == "stable":
            stable_mass += logical_count
        elif decision.regime_label == "questionable":
            questionable_mass += logical_count
        else:
            discard_mass += logical_count
        if decision.anomaly_count > 0:
            anomaly_mass += logical_count

    denom = logical_total if logical_total > 0.0 else float(max(1, len(records)))
    return {
        "mode": "population_baseline_no_teaching",
        "cohort_count": len(records),
        "logical_node_total": logical_total,
        "stable_mass_fraction": stable_mass / denom,
        "questionable_mass_fraction": questionable_mass / denom,
        "discard_mass_fraction": discard_mass / denom,
        "anomaly_mass_fraction": anomaly_mass / denom,
        "weighted_mean_trust_score": trust_total / denom,
        "weighted_mean_false_positive_risk": risk_total / denom,
        "decisions": [decision.as_dict() for decision in decisions],
    }


def write_population_baseline(
    inputs: list[str | Path],
    output_path: str | Path,
    logical_total: float | None = None,
) -> dict[str, Any]:
    records = load_records(inputs)
    if logical_total is not None:
        records = scale_logical_node_counts(records, logical_total)
    payload = evaluate_population_baseline(records)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
