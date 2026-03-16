from __future__ import annotations

import argparse
import csv
import json
import subprocess
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Any

from resonance_field.spiral_dual_gpu_batch import (
    BatchRecord,
    load_records,
    run_dual_gpu_batch_records,
    run_single_gpu_batch_records,
)


def _all_records(mode_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard_name in ("gpu0", "gpu1"):
        rows.extend(mode_result[shard_name]["parsed"]["records"])
    return rows


def _aggregate_result(mode_result: dict[str, Any]) -> dict[str, Any]:
    records = _all_records(mode_result)
    stable_count = sum(1 for row in records if row["regime_label"] == "stable")
    questionable_count = sum(1 for row in records if row["regime_label"] == "questionable")
    discard_count = sum(1 for row in records if row["regime_label"] == "discard")
    anomaly_count = sum(int(row["anomaly_count"]) for row in records)
    return {
        "stable_count": stable_count,
        "questionable_count": questionable_count,
        "discard_count": discard_count,
        "anomaly_count": anomaly_count,
        "mean_trust_score": mean(float(row["trust_score"]) for row in records) if records else 0.0,
        "mean_false_positive_risk": (
            mean(float(row["false_positive_risk"]) for row in records) if records else 0.0
        ),
    }


def _stability_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[float, int]:
    baseline_records = {row["candidate_id"]: row for row in _all_records(baseline)}
    candidate_records = {row["candidate_id"]: row for row in _all_records(candidate)}
    keys = sorted(set(baseline_records) | set(candidate_records))
    deltas: list[float] = []
    mismatches = 0

    for key in keys:
        left = baseline_records.get(key)
        right = candidate_records.get(key)
        if left is None or right is None:
            mismatches += 1
            deltas.append(1.0)
            continue
        deltas.append(abs(float(left["trust_score"]) - float(right["trust_score"])))
        deltas.append(abs(float(left["false_positive_risk"]) - float(right["false_positive_risk"])))
        if left["regime_label"] != right["regime_label"]:
            mismatches += 1
            deltas.append(1.0)
        if left["rerun_recommendation"] != right["rerun_recommendation"]:
            mismatches += 1
            deltas.append(1.0)
        if int(left["anomaly_count"]) != int(right["anomaly_count"]):
            mismatches += 1
            deltas.append(1.0)

    return (mean(deltas) if deltas else 0.0, mismatches)


def _read_cpu_stat() -> tuple[int, int]:
    with open("/proc/stat", "r", encoding="utf-8") as handle:
        first = handle.readline().strip().split()
    values = [int(value) for value in first[1:]]
    idle = values[3] + values[4]
    total = sum(values)
    return idle, total


class SystemMonitor:
    def __init__(self, sample_interval_s: float = 0.5) -> None:
        self.sample_interval_s = sample_interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        prev_idle, prev_total = _read_cpu_stat()
        while not self._stop.is_set():
            time.sleep(self.sample_interval_s)
            idle, total = _read_cpu_stat()
            delta_idle = idle - prev_idle
            delta_total = total - prev_total
            prev_idle, prev_total = idle, total
            if delta_total <= 0:
                cpu_util = 0.0
            else:
                cpu_util = 100.0 * (1.0 - (delta_idle / delta_total))

            sample = {"cpu_util": cpu_util}
            for gpu_index in (0, 1):
                sample[f"gpu{gpu_index}_util"] = 0.0
                sample[f"gpu{gpu_index}_mem"] = 0.0

            try:
                raw = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
            except Exception:
                raw = ""

            if raw:
                for line in raw.splitlines():
                    index_str, util_str, mem_str = [part.strip() for part in line.split(",")]
                    gpu_index = int(index_str)
                    if gpu_index in (0, 1):
                        sample[f"gpu{gpu_index}_util"] = float(util_str)
                        sample[f"gpu{gpu_index}_mem"] = float(mem_str)

            self.samples.append(sample)

    def summarize(self) -> dict[str, float]:
        if not self.samples:
            return {
                "cpu_util_mean": 0.0,
                "gpu0_util_mean": 0.0,
                "gpu1_util_mean": 0.0,
                "gpu0_mem_peak_mb": 0.0,
                "gpu1_mem_peak_mb": 0.0,
            }
        return {
            "cpu_util_mean": mean(sample["cpu_util"] for sample in self.samples),
            "gpu0_util_mean": mean(sample["gpu0_util"] for sample in self.samples),
            "gpu1_util_mean": mean(sample["gpu1_util"] for sample in self.samples),
            "gpu0_mem_peak_mb": max(sample["gpu0_mem"] for sample in self.samples),
            "gpu1_mem_peak_mb": max(sample["gpu1_mem"] for sample in self.samples),
        }


def replicate_records(records: list[BatchRecord], repeat_inputs: int) -> list[BatchRecord]:
    repeated: list[BatchRecord] = []
    for repeat_index in range(repeat_inputs):
        for record in records:
            repeated.append(
                replace(
                    record,
                    candidate_id=f"{record.candidate_id}__r{repeat_index}",
                )
            )
    return repeated


def summarize_mode(
    mode: str,
    mode_result: dict[str, Any],
    monitor_stats: dict[str, float],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_runtime = None if baseline is None else float(baseline["runtime_s"])
    runtime_s = float(mode_result["runtime_s"])
    throughput = float(mode_result["record_count"]) / max(runtime_s, 1e-9)
    aggregate = _aggregate_result(mode_result)
    if baseline is None:
        stability_delta = 0.0
        mismatch_count = 0
    else:
        stability_delta, mismatch_count = _stability_delta(baseline, mode_result)

    return {
        "mode": mode,
        "record_count": int(mode_result["record_count"]),
        **aggregate,
        "runtime_s": runtime_s,
        "throughput_records_per_s": throughput,
        "speedup_vs_single_gpu0": (
            1.0 if baseline_runtime in (None, 0.0) else baseline_runtime / max(runtime_s, 1e-9)
        ),
        "cpu_util_mean": monitor_stats["cpu_util_mean"],
        "gpu0_util_mean": monitor_stats["gpu0_util_mean"],
        "gpu1_util_mean": monitor_stats["gpu1_util_mean"],
        "gpu0_mem_peak_mb": monitor_stats["gpu0_mem_peak_mb"],
        "gpu1_mem_peak_mb": monitor_stats["gpu1_mem_peak_mb"],
        "stability_mean_abs_delta": stability_delta,
        "decision_mismatch_count": mismatch_count,
        "stability_label": (
            "match"
            if stability_delta < 1e-9 and mismatch_count == 0
            else f"delta={stability_delta:.6f}/mismatch={mismatch_count}"
        ),
    }


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_mode(
    mode: str,
    records: list[BatchRecord],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, float]]:
    monitor = SystemMonitor()
    monitor.start()
    try:
        if mode == "single_gpu0":
            result = run_single_gpu_batch_records(
                records,
                output_dir=output_dir,
                gpu_index=0,
                result_filename="single_gpu0_result.json",
            )
        elif mode == "single_gpu1":
            result = run_single_gpu_batch_records(
                records,
                output_dir=output_dir,
                gpu_index=1,
                result_filename="single_gpu1_result.json",
            )
        elif mode == "dual_gpu_parallel":
            result = run_dual_gpu_batch_records(
                records,
                output_dir=output_dir,
                result_filename="dual_gpu_parallel_result.json",
            )
        else:
            raise ValueError(f"Unsupported benchmark mode: {mode}")
    finally:
        monitor.stop()
    return result, monitor.summarize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the real SPIRAL Mojo single-GPU and dual-GPU batch paths."
    )
    parser.add_argument("inputs", nargs="+", help="SPIRAL report or bridge payload JSON files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/d/spiral_validation_outputs/batch_benchmark"),
        help="Directory for result JSON and summary tables.",
    )
    parser.add_argument(
        "--repeat-inputs",
        type=int,
        default=4,
        help="Repeat the logical batch this many times to create a measurable workload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_records = load_records(args.inputs)
    records = replicate_records(base_records, max(1, args.repeat_inputs))

    mode_rows: list[dict[str, Any]] = []
    raw_results: dict[str, Any] = {}
    baseline_result: dict[str, Any] | None = None

    for mode in ("single_gpu0", "single_gpu1", "dual_gpu_parallel"):
        mode_output_dir = args.output_dir / mode
        mode_output_dir.mkdir(parents=True, exist_ok=True)
        result, monitor_stats = _run_mode(mode, records, mode_output_dir)
        raw_results[mode] = result
        if mode == "single_gpu0":
            baseline_result = result
        mode_rows.append(
            summarize_mode(
                mode=mode,
                mode_result=result,
                monitor_stats=monitor_stats,
                baseline=baseline_result,
            )
        )

    summary_payload = {
        "inputs": [str(Path(item)) for item in args.inputs],
        "repeat_inputs": args.repeat_inputs,
        "logical_record_count": len(base_records),
        "benchmark_record_count": len(records),
        "summary": mode_rows,
        "results": raw_results,
    }

    (args.output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    _save_csv(args.output_dir / "benchmark_summary.csv", mode_rows)

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
