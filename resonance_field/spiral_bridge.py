from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from resonance_field.spiral_validation import (
    SpiralSignalVector,
    load_report_candidates,
)


BRIDGE_VERSION = "1.0"


@dataclass(frozen=True)
class SpiralBridgeRecord:
    candidate_id: str
    descriptor_consistency: float
    conformer_energy_variance: float
    rmsd_cluster_density: float
    torsion_entropy: float
    shape_similarity_index: float
    pharmacophore_deviation: float
    minimization_success_rate: float
    pipeline_instability_flags: float
    protein_branch: bool
    missing_signal_count: int
    report_path: str
    missing_signals: list[str]
    entry_types: list[str]
    source_notes: list[str]

    @classmethod
    def from_candidate(
        cls,
        candidate_id: str,
        signal_vector: SpiralSignalVector,
        provenance: dict[str, Any],
    ) -> "SpiralBridgeRecord":
        raw = signal_vector.as_dict()
        return cls(
            candidate_id=candidate_id,
            descriptor_consistency=raw["descriptor_consistency"],
            conformer_energy_variance=raw["conformer_energy_variance"],
            rmsd_cluster_density=raw["RMSD_cluster_density"],
            torsion_entropy=raw["torsion_entropy"],
            shape_similarity_index=raw["shape_similarity_index"],
            pharmacophore_deviation=raw["pharmacophore_deviation"],
            minimization_success_rate=raw["minimization_success_rate"],
            pipeline_instability_flags=raw["pipeline_instability_flags"],
            protein_branch=bool(provenance.get("protein_branch", False)),
            missing_signal_count=len(provenance.get("missing_signals", [])),
            report_path=str(provenance.get("report_path", "")),
            missing_signals=list(provenance.get("missing_signals", [])),
            entry_types=list(provenance.get("entry_types", [])),
            source_notes=list(provenance.get("source_notes", [])),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "descriptor_consistency": round(self.descriptor_consistency, 6),
            "conformer_energy_variance": round(self.conformer_energy_variance, 6),
            "rmsd_cluster_density": round(self.rmsd_cluster_density, 6),
            "torsion_entropy": round(self.torsion_entropy, 6),
            "shape_similarity_index": round(self.shape_similarity_index, 6),
            "pharmacophore_deviation": round(self.pharmacophore_deviation, 6),
            "minimization_success_rate": round(self.minimization_success_rate, 6),
            "pipeline_instability_flags": round(self.pipeline_instability_flags, 6),
            "protein_branch": self.protein_branch,
            "missing_signal_count": self.missing_signal_count,
            "report_path": self.report_path,
            "missing_signals": self.missing_signals,
            "entry_types": self.entry_types,
            "source_notes": self.source_notes,
        }


def export_bridge_records(report_paths: list[str | Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for report_path in report_paths:
        for candidate_id, signal_vector, provenance in load_report_candidates(report_path):
            records.append(
                SpiralBridgeRecord.from_candidate(
                    candidate_id,
                    signal_vector,
                    provenance,
                ).as_dict()
            )
    return {
        "bridge_version": BRIDGE_VERSION,
        "record_count": len(records),
        "records": records,
    }


def write_bridge_payload(report_paths: list[str | Path], output_path: str | Path) -> dict[str, Any]:
    payload = export_bridge_records(report_paths)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
