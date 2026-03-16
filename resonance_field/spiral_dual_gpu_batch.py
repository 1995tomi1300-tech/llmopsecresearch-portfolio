from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from resonance_field.spiral_bridge import export_bridge_records


IMAGE = "modular/max-nvidia-full:latest"
DISTRO = "Ubuntu"
WORKSPACE = Path("/mnt/d")
OUTPUT_ROOT = WORKSPACE / "spiral_validation_outputs" / "dual_gpu_batch"
GPU0_WRAPPER = WORKSPACE / "_spiral_gpu0_batch.mojo"
GPU1_WRAPPER = WORKSPACE / "_spiral_gpu1_batch.mojo"


@dataclass(frozen=True)
class BatchRecord:
    candidate_id: str
    logical_node_count: float
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

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BatchRecord":
        return cls(
            candidate_id=str(raw["candidate_id"]),
            logical_node_count=float(raw.get("logical_node_count", 1.0)),
            descriptor_consistency=float(raw["descriptor_consistency"]),
            conformer_energy_variance=float(raw["conformer_energy_variance"]),
            rmsd_cluster_density=float(raw["rmsd_cluster_density"]),
            torsion_entropy=float(raw["torsion_entropy"]),
            shape_similarity_index=float(raw["shape_similarity_index"]),
            pharmacophore_deviation=float(raw["pharmacophore_deviation"]),
            minimization_success_rate=float(raw["minimization_success_rate"]),
            pipeline_instability_flags=float(raw["pipeline_instability_flags"]),
            protein_branch=bool(raw.get("protein_branch", False)),
            missing_signal_count=int(raw.get("missing_signal_count", 0)),
            report_path=str(raw.get("report_path", "")),
        )

    @property
    def workload_weight(self) -> float:
        return (
            1.0
            + float(self.missing_signal_count)
            + (1.5 if self.protein_branch else 0.0)
        )

    def effective_instability(self) -> float:
        penalty = 0.03 * float(self.missing_signal_count)
        if self.protein_branch:
            penalty += 0.05
        return min(1.0, self.pipeline_instability_flags + penalty)


def _escape_mojo_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _fmt_float(value: float) -> str:
    return f"{value:.12g}"


def load_records_from_reports(report_paths: list[str | Path]) -> list[BatchRecord]:
    payload = export_bridge_records(report_paths)
    return [BatchRecord.from_dict(record) for record in payload["records"]]


def load_records_from_payload(payload_path: str | Path) -> list[BatchRecord]:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "bridge_version" not in payload or "records" not in payload:
        raise ValueError(f"{payload_path} is not a SPIRAL bridge payload")
    return [BatchRecord.from_dict(record) for record in payload["records"]]


def load_records(inputs: list[str | Path]) -> list[BatchRecord]:
    records: list[BatchRecord] = []
    report_paths: list[str | Path] = []

    for path_like in inputs:
        path = Path(path_like)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            report_paths.append(path)
            continue

        if isinstance(payload, dict) and "bridge_version" in payload and "records" in payload:
            records.extend(BatchRecord.from_dict(record) for record in payload["records"])
        else:
            report_paths.append(path)

    if report_paths:
        records.extend(load_records_from_reports(report_paths))

    return records


def split_records(records: list[BatchRecord]) -> tuple[list[BatchRecord], list[BatchRecord]]:
    gpu0: list[BatchRecord] = []
    gpu1: list[BatchRecord] = []
    gpu0_weight = 0.0
    gpu1_weight = 0.0

    for record in sorted(records, key=lambda item: item.workload_weight, reverse=True):
        if gpu0_weight <= gpu1_weight:
            gpu0.append(record)
            gpu0_weight += record.workload_weight
        else:
            gpu1.append(record)
            gpu1_weight += record.workload_weight

    return gpu0, gpu1


def render_wrapper(records: list[BatchRecord], shard_name: str) -> str:
    lines = [
        'from std.collections import List',
        (
            "from spiral_validation import "
            "SpiralSignalVector, SpiralCohortSignal, SpiralAIEngine, "
            "SpiralPopulationEngine, regime_label, rerun_label"
        ),
        "",
        "def main():",
        "    var engine = SpiralAIEngine()",
        "    var population_engine = SpiralPopulationEngine()",
        "    var cohorts = List[SpiralCohortSignal]()",
        f'    print("SHARD|{_escape_mojo_string(shard_name)}")',
    ]

    for index, record in enumerate(records):
        instab = _fmt_float(record.effective_instability())
        lines.extend(
            [
                (
                    "    var signal_{0} = SpiralSignalVector({1}, {2}, {3}, {4}, "
                    "{5}, {6}, {7}, {8})"
                ).format(
                    index,
                    _fmt_float(record.descriptor_consistency),
                    _fmt_float(record.conformer_energy_variance),
                    _fmt_float(record.rmsd_cluster_density),
                    _fmt_float(record.torsion_entropy),
                    _fmt_float(record.shape_similarity_index),
                    _fmt_float(record.pharmacophore_deviation),
                    _fmt_float(record.minimization_success_rate),
                    instab,
                ),
                f"    var decision_{index} = engine.infer(signal_{index})",
                (
                    '    print("RECORD|{0}|{1}|" + String(decision_{0}.trust_score) + "|" '
                    '+ String(decision_{0}.false_positive_risk) + "|" '
                    '+ regime_label(decision_{0}.regime_code) + "|" '
                    '+ rerun_label(decision_{0}.rerun_code) + "|" '
                    '+ String(decision_{0}.anomaly_mask) + "|" '
                    '+ String(decision_{0}.anomaly_count) + "|" '
                    '+ String(decision_{0}.cohort_alignment) + "|" '
                    '+ String(decision_{0}.memory_size))'
                ).format(index, _escape_mojo_string(record.candidate_id)),
                (
                    "    cohorts.append(SpiralCohortSignal({0}, {1}, {2}, {3}, {4}, "
                    "{5}, {6}, {7}, {8}))"
                ).format(
                    _fmt_float(record.logical_node_count),
                    _fmt_float(record.descriptor_consistency),
                    _fmt_float(record.conformer_energy_variance),
                    _fmt_float(record.rmsd_cluster_density),
                    _fmt_float(record.torsion_entropy),
                    _fmt_float(record.shape_similarity_index),
                    _fmt_float(record.pharmacophore_deviation),
                    _fmt_float(record.minimization_success_rate),
                    instab,
                ),
            ]
        )

    lines.extend(
        [
            "    var summary = population_engine.audit_batch(cohorts)",
            (
                '    print("SUMMARY|" + String(summary.cohort_count) + "|" '
                '+ String(summary.stable_count) + "|" '
                '+ String(summary.questionable_count) + "|" '
                '+ String(summary.discard_count) + "|" '
                '+ String(summary.anomaly_count) + "|" '
                '+ String(summary.logical_node_total) + "|" '
                '+ String(summary.mean_trust_score) + "|" '
                '+ String(summary.mean_false_positive_risk) + "|" '
                '+ String(summary.mean_cohort_alignment))'
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_wrappers(
    gpu0_records: list[BatchRecord],
    gpu1_records: list[BatchRecord],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu0_path = GPU0_WRAPPER
    gpu1_path = GPU1_WRAPPER
    gpu0_path.write_text(render_wrapper(gpu0_records, "gpu0"), encoding="utf-8")
    gpu1_path.write_text(render_wrapper(gpu1_records, "gpu1"), encoding="utf-8")
    return gpu0_path, gpu1_path


def _docker_run_command(gpu_index: int, wrapper_path: Path) -> str:
    workspace_path = WORKSPACE.as_posix()
    wrapper_in_workspace = f"/workspace/{wrapper_path.relative_to(WORKSPACE).as_posix()}"
    return (
        "docker run --rm "
        f'--gpus "device={gpu_index}" '
        f"-e NVIDIA_VISIBLE_DEVICES={gpu_index} "
        f"-e CUDA_VISIBLE_DEVICES={gpu_index} "
        "--entrypoint /bin/bash "
        f"-v {workspace_path}:/workspace -w /workspace "
        f"{IMAGE} "
        f'-lc "mojo {wrapper_in_workspace}"'
    )


def start_wrapper(gpu_index: int, wrapper_path: Path) -> tuple[str, subprocess.Popen[str]]:
    command = _docker_run_command(gpu_index, wrapper_path)
    process = subprocess.Popen(
        ["wsl.exe", "-d", DISTRO, "-u", "root", "--", "bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return command, process


def run_wrapper(gpu_index: int, wrapper_path: Path) -> subprocess.CompletedProcess[str]:
    command = _docker_run_command(gpu_index, wrapper_path)
    return subprocess.run(
        ["wsl.exe", "-d", DISTRO, "-u", "root", "--", "bash", "-lc", command],
        check=False,
        capture_output=True,
        text=True,
    )


RECORD_RE = re.compile(
    r"^RECORD\|(?P<index>\d+)\|(?P<candidate>[^|]+)\|(?P<trust>[^|]+)\|"
    r"(?P<risk>[^|]+)\|(?P<regime>[^|]+)\|(?P<rerun>[^|]+)\|"
    r"(?P<mask>[^|]+)\|(?P<count>[^|]+)\|(?P<alignment>[^|]+)\|(?P<memory>[^|]+)$"
)
SUMMARY_RE = re.compile(
    r"^SUMMARY\|(?P<cohort_count>[^|]+)\|(?P<stable_count>[^|]+)\|"
    r"(?P<questionable_count>[^|]+)\|(?P<discard_count>[^|]+)\|"
    r"(?P<anomaly_count>[^|]+)\|(?P<logical_node_total>[^|]+)\|"
    r"(?P<mean_trust_score>[^|]+)\|(?P<mean_false_positive_risk>[^|]+)\|"
    r"(?P<mean_cohort_alignment>[^|]+)$"
)


def parse_output(output: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    shard = ""
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("SHARD|"):
            shard = line.split("|", 1)[1]
            continue
        record_match = RECORD_RE.match(line)
        if record_match:
            data = record_match.groupdict()
            records.append(
                {
                    "record_index": int(data["index"]),
                    "candidate_id": data["candidate"],
                    "trust_score": float(data["trust"]),
                    "false_positive_risk": float(data["risk"]),
                    "regime_label": data["regime"],
                    "rerun_recommendation": data["rerun"],
                    "anomaly_mask": int(float(data["mask"])),
                    "anomaly_count": int(float(data["count"])),
                    "cohort_alignment": float(data["alignment"]),
                    "memory_size": int(float(data["memory"])),
                }
            )
            continue
        summary_match = SUMMARY_RE.match(line)
        if summary_match:
            data = summary_match.groupdict()
            summary = {
                "cohort_count": int(float(data["cohort_count"])),
                "stable_count": int(float(data["stable_count"])),
                "questionable_count": int(float(data["questionable_count"])),
                "discard_count": int(float(data["discard_count"])),
                "anomaly_count": int(float(data["anomaly_count"])),
                "logical_node_total": float(data["logical_node_total"]),
                "mean_trust_score": float(data["mean_trust_score"]),
                "mean_false_positive_risk": float(data["mean_false_positive_risk"]),
                "mean_cohort_alignment": float(data["mean_cohort_alignment"]),
            }
    return {"shard": shard, "records": records, "summary": summary}


def _build_result_payload(
    records: list[BatchRecord],
    output_dir: Path,
    gpu0_records: list[BatchRecord],
    gpu1_records: list[BatchRecord],
    gpu0_path: Path,
    gpu1_path: Path,
    gpu0_process: subprocess.CompletedProcess[str],
    gpu1_process: subprocess.CompletedProcess[str],
    runtime_s: float,
    mode: str,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "image": IMAGE,
        "distro": DISTRO,
        "workspace": str(WORKSPACE),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "runtime_s": runtime_s,
        "gpu0": {
            "wrapper_path": str(gpu0_path),
            "records": [record.candidate_id for record in gpu0_records],
            "parsed": parse_output(gpu0_process.stdout),
            "stderr": gpu0_process.stderr,
            "returncode": gpu0_process.returncode,
        },
        "gpu1": {
            "wrapper_path": str(gpu1_path),
            "records": [record.candidate_id for record in gpu1_records],
            "parsed": parse_output(gpu1_process.stdout),
            "stderr": gpu1_process.stderr,
            "returncode": gpu1_process.returncode,
        },
    }


def run_single_gpu_batch_records(
    records: list[BatchRecord],
    output_dir: Path = OUTPUT_ROOT,
    gpu_index: int = 0,
    result_filename: str = "single_gpu_batch_result.json",
) -> dict[str, Any]:
    gpu0_records = list(records)
    gpu1_records: list[BatchRecord] = []
    gpu0_path, gpu1_path = write_wrappers(gpu0_records, gpu1_records, output_dir)

    start = time.perf_counter()
    process = run_wrapper(gpu_index, gpu0_path)
    runtime_s = time.perf_counter() - start

    if process.returncode != 0:
        raise RuntimeError(
            "Single-GPU Mojo batch failed",
            {
                "gpu_index": gpu_index,
                "returncode": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            },
        )

    empty_process = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="",
        stderr="",
    )
    result = _build_result_payload(
        records=records,
        output_dir=output_dir,
        gpu0_records=gpu0_records,
        gpu1_records=gpu1_records,
        gpu0_path=gpu0_path,
        gpu1_path=gpu1_path,
        gpu0_process=process if gpu_index == 0 else empty_process,
        gpu1_process=process if gpu_index == 1 else empty_process,
        runtime_s=runtime_s,
        mode=f"single_gpu_{gpu_index}",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / result_filename).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def run_dual_gpu_batch(
    inputs: list[str | Path], output_dir: Path = OUTPUT_ROOT
) -> dict[str, Any]:
    records = load_records(inputs)
    return run_dual_gpu_batch_records(records, output_dir)


def run_dual_gpu_batch_records(
    records: list[BatchRecord],
    output_dir: Path = OUTPUT_ROOT,
    result_filename: str = "dual_gpu_batch_result.json",
) -> dict[str, Any]:
    gpu0_records, gpu1_records = split_records(records)
    gpu0_path, gpu1_path = write_wrappers(gpu0_records, gpu1_records, output_dir)

    start = time.perf_counter()
    gpu0_command, gpu0_live = start_wrapper(0, gpu0_path)
    gpu1_command, gpu1_live = start_wrapper(1, gpu1_path)
    gpu0_stdout, gpu0_stderr = gpu0_live.communicate()
    gpu1_stdout, gpu1_stderr = gpu1_live.communicate()
    runtime_s = time.perf_counter() - start
    gpu0_process = subprocess.CompletedProcess(
        args=gpu0_command,
        returncode=gpu0_live.returncode or 0,
        stdout=gpu0_stdout,
        stderr=gpu0_stderr,
    )
    gpu1_process = subprocess.CompletedProcess(
        args=gpu1_command,
        returncode=gpu1_live.returncode or 0,
        stdout=gpu1_stdout,
        stderr=gpu1_stderr,
    )

    if gpu0_process.returncode != 0 or gpu1_process.returncode != 0:
        raise RuntimeError(
            "Dual-GPU Mojo batch failed",
            {
                "gpu0_returncode": gpu0_process.returncode,
                "gpu0_stdout": gpu0_process.stdout,
                "gpu0_stderr": gpu0_process.stderr,
                "gpu1_returncode": gpu1_process.returncode,
                "gpu1_stdout": gpu1_process.stdout,
                "gpu1_stderr": gpu1_process.stderr,
            },
        )

    result = _build_result_payload(
        records=records,
        output_dir=output_dir,
        gpu0_records=gpu0_records,
        gpu1_records=gpu1_records,
        gpu0_path=gpu0_path,
        gpu1_path=gpu1_path,
        gpu0_process=gpu0_process,
        gpu1_process=gpu1_process,
        runtime_s=runtime_s,
        mode="dual_gpu_parallel",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / result_filename).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result
