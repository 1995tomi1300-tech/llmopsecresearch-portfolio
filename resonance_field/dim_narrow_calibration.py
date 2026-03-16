from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METRICS = [
    "late_recovery_coherence_mean",
    "final_false_isolation_rate_mean",
    "phase_manipulation_event_rate_mean",
    "phase_manipulation_core_contamination_rate_mean",
]


@dataclass(frozen=True)
class RefKey:
    scenario: str
    variant: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_reference(
    reference_root: Path,
) -> tuple[dict[RefKey, dict[str, float]], list[dict[str, Any]]]:
    by_key: dict[RefKey, list[dict[str, float]]] = {}
    rows_out: list[dict[str, Any]] = []

    for summary_path in sorted(reference_root.rglob("scenario_summary.csv")):
        parts = summary_path.parts
        # expected: .../nodes_<n>/<variant>/scenario_summary.csv
        if len(parts) < 3:
            continue
        variant = summary_path.parent.name
        parent = summary_path.parent.parent.name
        if not parent.startswith("nodes_"):
            continue
        nodes = int(parent.split("_", 1)[1])
        for row in _read_csv(summary_path):
            key = RefKey(scenario=row["scenario"], variant=variant)
            values = {m: _float(row.get(m, 0.0)) for m in METRICS}
            values["nodes"] = float(nodes)
            by_key.setdefault(key, []).append(values)
            rows_out.append(
                {
                    "nodes": nodes,
                    "variant": variant,
                    "scenario": row["scenario"],
                    **{m: values[m] for m in METRICS},
                }
            )

    aggregated: dict[RefKey, dict[str, float]] = {}
    for key, rows in by_key.items():
        agg: dict[str, float] = {}
        for metric in METRICS:
            agg[metric] = sum(item[metric] for item in rows) / max(len(rows), 1)
        agg["sample_count"] = float(len(rows))
        aggregated[key] = agg
    return aggregated, rows_out


def load_narrowing_specs(narrowing_map_path: Path) -> dict[int, list[int]]:
    payload = json.loads(narrowing_map_path.read_text(encoding="utf-8"))
    mapped: dict[int, list[int]] = {}
    for item in payload.get("specs", []):
        eff = int(item["effective_nodes"])
        logical = int(item["logical_nodes"])
        mapped.setdefault(eff, []).append(logical)
    return mapped


def calibrate_dim_narrow(
    reference_root: Path,
    dim_root: Path,
    output_dir: Path,
    reference_weight: float = 0.60,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dim_summary = dim_root / "boundary_summary.csv"
    narrowing_map = dim_root / "narrowing_map.json"

    if not dim_summary.exists():
        raise FileNotFoundError(f"Missing dim summary: {dim_summary}")
    if not narrowing_map.exists():
        raise FileNotFoundError(f"Missing narrowing map: {narrowing_map}")

    ref_agg, ref_rows = load_reference(reference_root)
    eff_to_logical = load_narrowing_specs(narrowing_map)
    dim_rows = _read_csv(dim_summary)

    comparison_rows: list[dict[str, Any]] = []
    logical_rows: list[dict[str, Any]] = []

    for row in dim_rows:
        scenario = row["scenario"]
        variant = row["variant"]
        nodes = int(row["nodes"])
        key = RefKey(scenario=scenario, variant=variant)
        ref = ref_agg.get(key)
        if ref is None:
            continue

        comp: dict[str, Any] = {
            "effective_nodes": nodes,
            "variant": variant,
            "scenario": scenario,
        }
        for metric in METRICS:
            dim_value = _float(row.get(metric, 0.0))
            ref_value = _float(ref.get(metric, 0.0))
            delta = dim_value - ref_value
            ratio = dim_value / ref_value if abs(ref_value) > 1e-9 else 1.0
            blended = reference_weight * ref_value + (1.0 - reference_weight) * dim_value
            comp[f"{metric}_reference"] = ref_value
            comp[f"{metric}_dim"] = dim_value
            comp[f"{metric}_delta"] = delta
            comp[f"{metric}_ratio"] = ratio
            comp[f"{metric}_blended"] = blended
        comparison_rows.append(comp)

        for logical_nodes in eff_to_logical.get(nodes, []):
            logical_row = {
                "logical_nodes": logical_nodes,
                "effective_nodes": nodes,
                "variant": variant,
                "scenario": scenario,
            }
            for metric in METRICS:
                logical_row[metric] = comp[f"{metric}_blended"]
            logical_rows.append(logical_row)

    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    _write_csv(output_dir / "reference_samples.csv", ref_rows)
    _write_csv(output_dir / "calibration_comparison.csv", comparison_rows)
    _write_csv(output_dir / "logical_calibrated_estimates.csv", logical_rows)

    payload = {
        "reference_root": str(reference_root),
        "dim_root": str(dim_root),
        "output_dir": str(output_dir),
        "reference_weight": reference_weight,
        "reference_rows": len(ref_rows),
        "comparison_rows": len(comparison_rows),
        "logical_rows": len(logical_rows),
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload
