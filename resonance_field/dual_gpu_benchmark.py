from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import queue
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import psutil

from resonance_field.phase13_config import PhaseXIIIConfig, build_phase13_scenarios
from resonance_field.phase6_experiment import run_single_phase6


REPRESENTATIVE_SCENARIOS = [
    "baseline_periodic_random",
    "triangle_periodic_noisy",
    "helix_single_noisy",
    "hybrid_manifold_periodic_noisy",
]

TOPOLOGY_WEIGHTS = {
    "baseline": 1.0,
    "triangle": 1.4,
    "helix": 1.8,
    "hybrid_manifold": 1.3,
}


def _model_kind_from_scenario(scenario_name: str) -> str:
    for prefix in ("hybrid_manifold", "baseline", "triangle", "helix"):
        if scenario_name.startswith(f"{prefix}_"):
            return prefix
    raise KeyError(f"Unknown scenario prefix: {scenario_name}")


@dataclass(frozen=True)
class BenchmarkTask:
    scenario_name: str
    run_index: int
    seed: int


@dataclass(frozen=True)
class ModeSummary:
    mode: str
    runtime_s: float
    throughput_runs_per_s: float
    throughput_node_steps_per_s: float
    gpu0_util_mean: float
    gpu1_util_mean: float
    gpu0_mem_peak_mb: float
    gpu1_mem_peak_mb: float
    cpu_util_mean: float
    stability_mean_abs_delta: float
    stability_label: str


class SystemMonitor:
    def __init__(self, sample_interval: float = 0.5) -> None:
        self.sample_interval = sample_interval
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        psutil.cpu_percent(interval=None)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            row = {"cpu_util": float(psutil.cpu_percent(interval=None))}
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
            except Exception:
                out = ""
            for idx in range(2):
                row[f"gpu{idx}_util"] = 0.0
                row[f"gpu{idx}_mem"] = 0.0
            if out:
                for line in out.splitlines():
                    index_str, util_str, mem_str = [part.strip() for part in line.split(",")]
                    index = int(index_str)
                    if index in (0, 1):
                        row[f"gpu{index}_util"] = float(util_str)
                        row[f"gpu{index}_mem"] = float(mem_str)
            self.samples.append(row)
            time.sleep(self.sample_interval)

    def summarize(self) -> dict[str, float]:
        if not self.samples:
            return {
                "gpu0_util_mean": 0.0,
                "gpu1_util_mean": 0.0,
                "gpu0_mem_peak_mb": 0.0,
                "gpu1_mem_peak_mb": 0.0,
                "cpu_util_mean": 0.0,
            }
        return {
            "gpu0_util_mean": mean(sample["gpu0_util"] for sample in self.samples),
            "gpu1_util_mean": mean(sample["gpu1_util"] for sample in self.samples),
            "gpu0_mem_peak_mb": max(sample["gpu0_mem"] for sample in self.samples),
            "gpu1_mem_peak_mb": max(sample["gpu1_mem"] for sample in self.samples),
            "cpu_util_mean": mean(sample["cpu_util"] for sample in self.samples),
        }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _worker_main(
    gpu_index: int,
    config_payload: dict[str, Any],
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_index)
    except Exception:
        pass

    payload = dict(config_payload)
    payload["backend_gpu_index"] = gpu_index
    payload["compute_backend"] = "gpu"
    config = PhaseXIIIConfig(**payload)
    scenario_map = {scenario.name: scenario for scenario in build_phase13_scenarios()}

    while True:
        try:
            payload = task_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if payload is None:
            break
        task = BenchmarkTask(**payload)
        start = time.perf_counter()
        metrics, *_ = run_single_phase6(
            config=config,
            scenario=scenario_map[task.scenario_name],
            run_index=task.run_index,
            seed=task.seed,
        )
        elapsed = time.perf_counter() - start
        result_queue.put(
            {
                "gpu_index": gpu_index,
                "task": asdict(task),
                "runtime_s": elapsed,
                "scenario": metrics.scenario,
                "model_kind": metrics.model_kind,
                "coherence_recovery_ratio": metrics.coherence_recovery_ratio,
                "cluster_recovery_ratio": metrics.cluster_recovery_ratio,
                "recovery_success": metrics.recovery_success,
            }
        )


def build_benchmark_tasks(runs_per_scenario: int, seed_base: int) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    for scenario_offset, scenario_name in enumerate(REPRESENTATIVE_SCENARIOS):
        for run_index in range(runs_per_scenario):
            tasks.append(
                BenchmarkTask(
                    scenario_name=scenario_name,
                    run_index=run_index,
                    seed=seed_base + scenario_offset * 100 + run_index,
                )
            )
    return tasks


def assign_mode_tasks(mode: str, tasks: list[BenchmarkTask]) -> dict[int, list[BenchmarkTask]]:
    if mode == "single_gpu":
        return {0: list(tasks), 1: []}

    if mode == "dual_gpu_batch_split":
        assigned = {0: [], 1: []}
        for task in tasks:
            assigned[task.run_index % 2].append(task)
        return assigned

    if mode == "dual_gpu_topology_split":
        assigned = {0: [], 1: []}
        for task in tasks:
            if task.scenario_name.startswith(("helix_", "triangle_")):
                assigned[0].append(task)
            else:
                assigned[1].append(task)
        return assigned

    if mode == "dual_gpu_scenario_parallel":
        scenario_groups: dict[str, list[BenchmarkTask]] = {}
        for task in tasks:
            scenario_groups.setdefault(task.scenario_name, []).append(task)
        loads = {0: 0.0, 1: 0.0}
        assigned = {0: [], 1: []}
        ordered = sorted(
            scenario_groups.items(),
            key=lambda item: TOPOLOGY_WEIGHTS[_model_kind_from_scenario(item[0])],
            reverse=True,
        )
        for scenario_name, group in ordered:
            model_kind = _model_kind_from_scenario(scenario_name)
            gpu_index = 0 if loads[0] <= loads[1] else 1
            assigned[gpu_index].extend(group)
            loads[gpu_index] += TOPOLOGY_WEIGHTS[model_kind] * len(group)
        return assigned

    raise ValueError(f"Unsupported mode: {mode}")


def _run_mode(
    mode: str,
    config: PhaseXIIIConfig,
    tasks: list[BenchmarkTask],
) -> tuple[ModeSummary, list[dict[str, Any]]]:
    assignments = assign_mode_tasks(mode, tasks)
    result_queue: mp.Queue = mp.Queue()
    task_queues = {0: mp.Queue(), 1: mp.Queue()}
    processes: list[mp.Process] = []

    for gpu_index in (0, 1):
        if mode == "single_gpu" and gpu_index == 1:
            continue
        process = mp.Process(
            target=_worker_main,
            args=(gpu_index, asdict(config), task_queues[gpu_index], result_queue),
            daemon=True,
        )
        process.start()
        processes.append(process)

    for gpu_index, gpu_tasks in assignments.items():
        for task in gpu_tasks:
            task_queues[gpu_index].put(asdict(task))
    for gpu_index in (0, 1):
        if mode == "single_gpu" and gpu_index == 1:
            continue
        task_queues[gpu_index].put(None)

    monitor = SystemMonitor()
    monitor.start()
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    while len(results) < len(tasks):
        results.append(result_queue.get())
    runtime_s = time.perf_counter() - start
    monitor.stop()

    for process in processes:
        process.join(timeout=10.0)

    monitor_stats = monitor.summarize()
    throughput_runs = len(tasks) / max(runtime_s, 1e-6)
    throughput_node_steps = len(tasks) * config.nodes * config.steps / max(runtime_s, 1e-6)

    summary = ModeSummary(
        mode=mode,
        runtime_s=runtime_s,
        throughput_runs_per_s=throughput_runs,
        throughput_node_steps_per_s=throughput_node_steps,
        gpu0_util_mean=monitor_stats["gpu0_util_mean"],
        gpu1_util_mean=monitor_stats["gpu1_util_mean"],
        gpu0_mem_peak_mb=monitor_stats["gpu0_mem_peak_mb"],
        gpu1_mem_peak_mb=monitor_stats["gpu1_mem_peak_mb"],
        cpu_util_mean=monitor_stats["cpu_util_mean"],
        stability_mean_abs_delta=0.0,
        stability_label="baseline",
    )
    return summary, results


def _stability_against(
    baseline_results: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
) -> tuple[float, str]:
    baseline_map = {
        (row["scenario"], row["task"]["run_index"]): row for row in baseline_results
    }
    deltas: list[float] = []
    for row in candidate_results:
        key = (row["scenario"], row["task"]["run_index"])
        ref = baseline_map[key]
        deltas.extend(
            [
                abs(float(row["coherence_recovery_ratio"]) - float(ref["coherence_recovery_ratio"])),
                abs(float(row["cluster_recovery_ratio"]) - float(ref["cluster_recovery_ratio"])),
                abs(float(row["recovery_success"]) - float(ref["recovery_success"])),
            ]
        )
    mean_delta = float(mean(deltas)) if deltas else 0.0
    label = "match" if mean_delta < 1e-9 else f"mean_abs_delta={mean_delta:.3e}"
    return mean_delta, label


def run_dual_gpu_benchmark(
    output_dir: Path,
    nodes: int = 256,
    steps: int = 120,
    runs_per_scenario: int = 2,
    seed_base: int = 20260320,
) -> dict[str, Any]:
    mp.set_start_method("spawn", force=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PhaseXIIIConfig(
        nodes=nodes,
        steps=steps,
        compute_backend="gpu",
        benchmark_nodes=max(1024, nodes),
        benchmark_repeats=6,
        controller_interval=4,
        cluster_refresh_interval=4,
        collect_event_rows=False,
    )
    tasks = build_benchmark_tasks(runs_per_scenario=runs_per_scenario, seed_base=seed_base)
    modes = [
        "single_gpu",
        "dual_gpu_scenario_parallel",
        "dual_gpu_batch_split",
        "dual_gpu_topology_split",
    ]

    mode_summaries: list[ModeSummary] = []
    mode_results: dict[str, list[dict[str, Any]]] = {}
    for mode in modes:
        summary, results = _run_mode(mode, config, tasks)
        mode_summaries.append(summary)
        mode_results[mode] = results

    baseline = mode_results["single_gpu"]
    mode_summaries = [
        ModeSummary(
            **{
                **asdict(summary),
                "stability_mean_abs_delta": _stability_against(baseline, mode_results[summary.mode])[0],
                "stability_label": _stability_against(baseline, mode_results[summary.mode])[1],
            }
        )
        for summary in mode_summaries
    ]

    summary_rows = [asdict(item) for item in mode_summaries]
    _save_csv(output_dir / "benchmark_summary.csv", summary_rows)
    _save_json(
        output_dir / "benchmark_summary.json",
        {
            "config": asdict(config),
            "representative_scenarios": REPRESENTATIVE_SCENARIOS,
            "tasks": [asdict(task) for task in tasks],
            "summary": summary_rows,
            "results": mode_results,
        },
    )
    _save_json(
        output_dir / "execution_plan.json",
        {
            mode: {
                str(gpu_index): [asdict(task) for task in gpu_tasks]
                for gpu_index, gpu_tasks in assign_mode_tasks(mode, tasks).items()
            }
            for mode in modes
        },
    )
    return {
        "config": config,
        "summary_rows": summary_rows,
        "results": mode_results,
    }
