from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


VECTOR_FIELDS = (
    "descriptor_consistency",
    "conformer_energy_variance",
    "RMSD_cluster_density",
    "torsion_entropy",
    "shape_similarity_index",
    "pharmacophore_deviation",
    "minimization_success_rate",
    "pipeline_instability_flags",
)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _mean_or_default(values: list[float], default: float = 0.5) -> float:
    return default if not values else float(mean(values))


def _scaled_inverse(value: float, scale: float) -> float:
    return 1.0 / (1.0 + max(0.0, value) / max(scale, 1e-9))


def _normalize_batch_label(label: str, entry_type: str) -> str:
    suffix = " conformers"
    if entry_type == "batch_comparison" and label.endswith(suffix):
        return label[: -len(suffix)]
    return label


@dataclass(frozen=True)
class SpiralSignalVector:
    descriptor_consistency: float
    conformer_energy_variance: float
    RMSD_cluster_density: float
    torsion_entropy: float
    shape_similarity_index: float
    pharmacophore_deviation: float
    minimization_success_rate: float
    pipeline_instability_flags: float

    def as_dict(self) -> dict[str, float]:
        return {
            "descriptor_consistency": self.descriptor_consistency,
            "conformer_energy_variance": self.conformer_energy_variance,
            "RMSD_cluster_density": self.RMSD_cluster_density,
            "torsion_entropy": self.torsion_entropy,
            "shape_similarity_index": self.shape_similarity_index,
            "pharmacophore_deviation": self.pharmacophore_deviation,
            "minimization_success_rate": self.minimization_success_rate,
            "pipeline_instability_flags": self.pipeline_instability_flags,
        }


@dataclass(frozen=True)
class SpiralValidationConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "descriptor_consistency": 0.18,
            "conformer_energy_variance": 0.14,
            "RMSD_cluster_density": 0.14,
            "torsion_entropy": 0.12,
            "shape_similarity_index": 0.12,
            "pharmacophore_deviation": 0.12,
            "minimization_success_rate": 0.10,
            "pipeline_instability_flags": 0.08,
        }
    )
    stable_threshold: float = 0.75
    questionable_threshold: float = 0.45
    critical_anomaly_penalty: float = 0.12
    anomaly_penalty: float = 0.06
    missing_penalty: float = 0.03
    pharmacophore_scale: float = 0.6
    rmsd_scale: float = 0.75
    conformer_variance_scale: float = 0.20


@dataclass(frozen=True)
class SpiralValidationResult:
    candidate_id: str
    trust_score: float
    false_positive_risk: float
    anomaly_flags: list[str]
    rerun_recommendation: str
    regime_label: str
    raw_vector: dict[str, float]
    normalized_vector: dict[str, float]
    score_contributions: dict[str, float]
    provenance: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "trust_score": self.trust_score,
            "false_positive_risk": self.false_positive_risk,
            "anomaly_flags": self.anomaly_flags,
            "rerun_recommendation": self.rerun_recommendation,
            "regime_label": self.regime_label,
            "raw_vector": self.raw_vector,
            "normalized_vector": self.normalized_vector,
            "score_contributions": self.score_contributions,
            "provenance": self.provenance,
        }


class SpiralValidationEngine:
    """Deterministic trust and anomaly layer over upstream scientific signals."""

    def __init__(self, config: SpiralValidationConfig | None = None):
        self.config = config or SpiralValidationConfig()

    def evaluate(
        self,
        candidate_id: str,
        signal_vector: SpiralSignalVector,
        provenance: dict[str, Any] | None = None,
    ) -> SpiralValidationResult:
        raw_vector = signal_vector.as_dict()
        normalized = self._normalize(signal_vector)
        anomaly_flags, critical_count = self._detect_anomalies(
            raw_vector,
            normalized,
            provenance or {},
        )
        contributions = {
            field: self.config.weights[field] * normalized[field]
            for field in VECTOR_FIELDS
        }
        missing_signals = list((provenance or {}).get("missing_signals", []))
        trust_score = _clamp(
            sum(contributions.values())
            - critical_count * self.config.critical_anomaly_penalty
            - max(0, len(anomaly_flags) - critical_count) * self.config.anomaly_penalty
            - len(missing_signals) * self.config.missing_penalty
        )
        anomaly_pressure = _clamp(
            0.2 * len(anomaly_flags) + 0.4 * critical_count + 0.4 * raw_vector["pipeline_instability_flags"]
        )
        false_positive_risk = _clamp(
            0.60 * (1.0 - trust_score)
            + 0.25 * anomaly_pressure
            + 0.15 * raw_vector["pipeline_instability_flags"]
        )
        regime_label = self._classify_regime(trust_score, critical_count, raw_vector)
        rerun_recommendation = self._recommend_rerun(
            regime_label,
            raw_vector,
            anomaly_flags,
        )
        return SpiralValidationResult(
            candidate_id=candidate_id,
            trust_score=round(trust_score, 4),
            false_positive_risk=round(false_positive_risk, 4),
            anomaly_flags=anomaly_flags,
            rerun_recommendation=rerun_recommendation,
            regime_label=regime_label,
            raw_vector={key: round(value, 4) for key, value in raw_vector.items()},
            normalized_vector={key: round(value, 4) for key, value in normalized.items()},
            score_contributions={key: round(value, 4) for key, value in contributions.items()},
            provenance=provenance or {},
        )

    def _normalize(self, signal_vector: SpiralSignalVector) -> dict[str, float]:
        raw = signal_vector.as_dict()
        return {
            "descriptor_consistency": _clamp(raw["descriptor_consistency"]),
            "conformer_energy_variance": _clamp(
                _scaled_inverse(raw["conformer_energy_variance"], self.config.conformer_variance_scale)
            ),
            "RMSD_cluster_density": _clamp(raw["RMSD_cluster_density"]),
            "torsion_entropy": _clamp(1.0 - raw["torsion_entropy"]),
            "shape_similarity_index": _clamp(raw["shape_similarity_index"]),
            "pharmacophore_deviation": _clamp(
                _scaled_inverse(raw["pharmacophore_deviation"], self.config.pharmacophore_scale)
            ),
            "minimization_success_rate": _clamp(raw["minimization_success_rate"]),
            "pipeline_instability_flags": _clamp(1.0 - raw["pipeline_instability_flags"]),
        }

    def _detect_anomalies(
        self,
        raw: dict[str, float],
        normalized: dict[str, float],
        provenance: dict[str, Any],
    ) -> tuple[list[str], int]:
        flags: list[str] = []
        critical = 0

        if raw["RMSD_cluster_density"] >= 0.65 and raw["torsion_entropy"] >= 0.55:
            flags.append("low_rmsd_density_with_high_torsion_entropy")
        if raw["minimization_success_rate"] >= 0.8 and raw["pharmacophore_deviation"] >= 0.55:
            flags.append("stable_minimization_with_high_pharmacophore_deviation")
        if raw["descriptor_consistency"] >= 0.75 and self._structural_stability(normalized) <= 0.45:
            flags.append("descriptor_consistency_with_structural_instability")
            critical += 1
        if raw["conformer_energy_variance"] >= 0.55 and raw["RMSD_cluster_density"] >= 0.65:
            flags.append("energy_variance_inconsistent_with_conformer_clustering")
        if raw["minimization_success_rate"] >= 0.7 and raw["pipeline_instability_flags"] >= 0.5:
            flags.append("execution_success_with_instability_flags")
        if raw["shape_similarity_index"] >= 0.8 and raw["pharmacophore_deviation"] >= 0.6:
            flags.append("shape_similarity_with_pharmacophore_mismatch")
        missing_signals = list(provenance.get("missing_signals", []))
        if missing_signals:
            flags.append("missing_validation_evidence")
        if provenance.get("protein_branch", False):
            flags.append("protein_signals_not_ground_truth")

        return flags, critical

    def _structural_stability(self, normalized: dict[str, float]) -> float:
        return _mean_or_default(
            [
                normalized["RMSD_cluster_density"],
                normalized["torsion_entropy"],
                normalized["shape_similarity_index"],
                normalized["pharmacophore_deviation"],
                normalized["conformer_energy_variance"],
            ]
        )

    def _classify_regime(
        self,
        trust_score: float,
        critical_count: int,
        raw: dict[str, float],
    ) -> str:
        if critical_count > 0 or raw["pipeline_instability_flags"] >= 0.75:
            return "discard"
        if trust_score >= self.config.stable_threshold:
            return "stable"
        if trust_score >= self.config.questionable_threshold:
            return "questionable"
        return "discard"

    def _recommend_rerun(
        self,
        regime_label: str,
        raw: dict[str, float],
        anomaly_flags: list[str],
    ) -> str:
        if regime_label == "stable" and not anomaly_flags:
            return "accept"
        if raw["pipeline_instability_flags"] >= 0.6:
            return "rerun_pipeline_stage"
        if raw["minimization_success_rate"] < 0.45:
            return "rerun_minimization"
        if raw["torsion_entropy"] >= 0.6 or raw["conformer_energy_variance"] >= 0.6:
            return "rerun_conformer_generation"
        if raw["descriptor_consistency"] < 0.45:
            return "rerun_descriptor_validation"
        if regime_label == "discard":
            return "discard_candidate"
        return "manual_review"


def build_signal_vector_from_report_entries(
    entries: list[dict[str, Any]],
) -> tuple[SpiralSignalVector, dict[str, Any]]:
    by_type: dict[str, dict[str, Any]] = {}
    for entry in entries:
        by_type.setdefault(entry["type"], entry)

    missing_signals: list[str] = []
    source_notes: list[str] = []

    descriptor_consistency = 0.5
    descriptor_snapshot: dict[str, Any] = {}
    descriptor_entry = by_type.get("drug_likeness")
    if descriptor_entry:
        descriptor_data = descriptor_entry.get("data", {})
        admet_data = descriptor_entry.get("admet", {})
        lipinski_score = 1.0 if descriptor_data.get("lipinski_pass") else _clamp(
            1.0 - 0.25 * descriptor_data.get("lipinski_violations", 0)
        )
        veber_score = 1.0 if descriptor_data.get("veber_pass") else 0.0
        lead_score = 1.0 if descriptor_data.get("lead_like") else 0.5
        filter_quality = _mean_or_default([lipinski_score, veber_score, lead_score])
        if admet_data:
            admet_quality = _mean_or_default(
                [
                    admet_data.get("overall_druglike", 0.5),
                    admet_data.get("absorption_score", 0.5),
                    1.0 - admet_data.get("hepatotoxicity_risk", 0.5),
                ]
            )
            descriptor_consistency = _clamp(1.0 - abs(filter_quality - admet_quality))
        else:
            descriptor_consistency = _clamp(filter_quality)
            missing_signals.append("admet_proxy_outputs")
        descriptor_snapshot = {
            "filter_quality": round(filter_quality, 4),
            "has_admet": bool(admet_data),
        }
    else:
        missing_signals.append("descriptor_layer")

    conformer_energy_variance = 0.5
    rmsd_cluster_density = 0.5
    torsion_entropy = 0.5
    batch_entry = by_type.get("batch_comparison")
    if batch_entry:
        batch_data = batch_entry.get("data", {})
        individual = batch_data.get("individual", [])
        if individual:
            rmsd_values = [float(item["rmsd_aligned"]) for item in individual if "rmsd_aligned" in item]
            torsion_values = [
                _clamp(float(item["torsion_similarity"])) for item in individual if "torsion_similarity" in item
            ]
            batch_scores = [float(item["overall_score"]) for item in individual if "overall_score" in item]
            if rmsd_values:
                rmsd_cluster_density = _clamp(
                    mean(max(0.0, 1.0 - value / 0.75) for value in rmsd_values)
                )
            if torsion_values:
                torsion_entropy = _clamp(mean(1.0 - value for value in torsion_values))
            if batch_scores:
                conformer_energy_variance = _clamp(pstdev(batch_scores) / 0.15)
                source_notes.append("conformer_energy_variance derived from batch score spread proxy")
        else:
            missing_signals.append("batch_conformer_metrics")
    else:
        missing_signals.append("conformer_batch_metrics")

    shape_similarity_index = 0.5
    pharmacophore_deviation = 0.5
    structure_entry = by_type.get("structure_comparison")
    if structure_entry:
        structure_data = structure_entry.get("data", {})
        shape_similarity_index = _clamp(structure_data.get("shape_tanimoto", 0.5))
        pharmacophore_deviation = _clamp(
            float(structure_data.get("pharmacophore_rmsd", 0.5))
        )
        if "torsion_similarity" in structure_data and "batch_comparison" not in by_type:
            torsion_entropy = _clamp(1.0 - float(structure_data["torsion_similarity"]))
    else:
        missing_signals.append("structure_comparison_metrics")

    minimization_success_rate = 0.5
    pipeline_instability_flags = 0.0
    missing_signals.append("minimization_execution_metrics")

    provenance = {
        "missing_signals": missing_signals,
        "source_notes": source_notes,
        "entry_types": sorted(by_type.keys()),
        "descriptor_snapshot": descriptor_snapshot,
        "protein_branch": False,
    }
    return (
        SpiralSignalVector(
            descriptor_consistency=descriptor_consistency,
            conformer_energy_variance=conformer_energy_variance,
            RMSD_cluster_density=rmsd_cluster_density,
            torsion_entropy=torsion_entropy,
            shape_similarity_index=shape_similarity_index,
            pharmacophore_deviation=pharmacophore_deviation,
            minimization_success_rate=minimization_success_rate,
            pipeline_instability_flags=pipeline_instability_flags,
        ),
        provenance,
    )


def load_report_candidates(path: str | Path) -> list[tuple[str, SpiralSignalVector, dict[str, Any]]]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in report.get("entries", []):
        label = _normalize_batch_label(entry.get("label", "unknown"), entry.get("type", "custom"))
        grouped[label].append(entry)

    candidates: list[tuple[str, SpiralSignalVector, dict[str, Any]]] = []
    for candidate_id in sorted(grouped.keys()):
        vector, provenance = build_signal_vector_from_report_entries(grouped[candidate_id])
        provenance["report_path"] = str(Path(path))
        candidates.append((candidate_id, vector, provenance))
    return candidates


def evaluate_report(
    path: str | Path,
    engine: SpiralValidationEngine | None = None,
) -> list[SpiralValidationResult]:
    validation_engine = engine or SpiralValidationEngine()
    results: list[SpiralValidationResult] = []
    for candidate_id, vector, provenance in load_report_candidates(path):
        results.append(validation_engine.evaluate(candidate_id, vector, provenance))
    return results
