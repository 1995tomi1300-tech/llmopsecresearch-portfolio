from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from resonance_field.phase3_config import (
    PHASE3_MODEL_LABELS,
    PhaseIIIConfig,
    PhaseIIIScenario,
    build_phase3_scenarios,
)
from resonance_field.phase3_models import PHASE3_MODEL_REGISTRY
from resonance_field.topologies import bootstrap_kick


@dataclass
class PhaseIIIRunMetrics:
    scenario: str
    label: str
    model_kind: str
    bootstrap_mode: str
    false_mode: str
    run_index: int
    seed: int
    final_coherence: float
    late_coherence: float
    largest_cluster_fraction: float
    cluster_count: int
    mean_degree: float
    false_isolation_rate: float
    resonance_energy: float
    cluster_persistence: float
    topological_memory_index: float
    triangle_density: float


def connected_components(adjacency: np.ndarray) -> tuple[np.ndarray, list[int]]:
    node_count = adjacency.shape[0]
    seen = np.zeros(node_count, dtype=bool)
    labels = -np.ones(node_count, dtype=int)
    sizes: list[int] = []
    component_id = 0

    for start in range(node_count):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        labels[start] = component_id
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = np.flatnonzero(adjacency[node] & ~seen)
            if neighbors.size:
                seen[neighbors] = True
                labels[neighbors] = component_id
                stack.extend(neighbors.tolist())
        sizes.append(size)
        component_id += 1
    return labels, sizes


def cluster_sets(
    adjacency: np.ndarray,
    threshold: int,
) -> tuple[list[set[int]], set[int], list[int]]:
    labels, sizes = connected_components(adjacency)
    clusters: list[set[int]] = []
    member_union: set[int] = set()
    cluster_sizes: list[int] = []
    for component_id, size in enumerate(sizes):
        if size < threshold:
            continue
        members = set(np.flatnonzero(labels == component_id).tolist())
        clusters.append(members)
        member_union.update(members)
        cluster_sizes.append(size)
    return clusters, member_union, cluster_sizes


def jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def cluster_persistence_rows(
    scenario: PhaseIIIScenario,
    run_index: int,
    seed: int,
    sampled_memberships: list[tuple[int, set[int]]],
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    values: list[float] = []
    for (time_start, members_start), (time_end, members_end) in zip(
        sampled_memberships,
        sampled_memberships[1:],
    ):
        if not members_start:
            persistence = 0.0
        else:
            persistence = len(members_start & members_end) / len(members_start)
        values.append(float(persistence))
        rows.append(
            {
                "scenario": scenario.name,
                "model_kind": scenario.model_kind,
                "bootstrap_mode": scenario.bootstrap_mode,
                "false_mode": scenario.false_mode,
                "run_index": run_index,
                "seed": seed,
                "time_start": time_start,
                "time_end": time_end,
                "cluster_persistence": float(persistence),
            }
        )
    return rows, float(np.mean(values)) if values else 0.0


def topological_memory_index(
    sampled_clusters: list[list[set[int]]],
    threshold: float,
) -> float:
    if len(sampled_clusters) < 3:
        return 0.0

    events = 0
    candidates = 0
    for index, clusters in enumerate(sampled_clusters[:-2]):
        immediate_next = sampled_clusters[index + 1]
        later_clusters = sampled_clusters[index + 2 :]
        for cluster in clusters:
            candidates += 1
            persists_immediately = any(
                jaccard(cluster, next_cluster) >= threshold
                for next_cluster in immediate_next
            )
            if persists_immediately:
                continue
            if any(
                jaccard(cluster, later_cluster) >= threshold
                for snapshot_clusters in later_clusters
                for later_cluster in snapshot_clusters
            ):
                events += 1
    return events / candidates if candidates else 0.0


def initialize_state(
    config: PhaseIIIConfig,
    scenario: PhaseIIIScenario,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    model = PHASE3_MODEL_REGISTRY[scenario.model_kind]
    state = model.initialize_state(config, rng, scenario.false_mode)
    if scenario.bootstrap_mode in {"single", "periodic"}:
        state["theta"] = bootstrap_kick(
            state["theta"],
            config.bootstrap_anchor,
            config.bootstrap_strength,
        )
    return state


def finalize_run(
    scenario: PhaseIIIScenario,
    config: PhaseIIIConfig,
    run_index: int,
    seed: int,
    state: dict[str, np.ndarray],
    final_adjacency: np.ndarray,
    coherence_history: list[float],
    resonance_energy_history: list[float],
    triangle_density_history: list[float],
    sampled_memberships: list[tuple[int, set[int]]],
    sampled_clusters: list[list[set[int]]],
) -> tuple[PhaseIIIRunMetrics, list[dict[str, Any]]]:
    degrees = final_adjacency.sum(axis=1).astype(float)
    _, _, cluster_sizes = cluster_sets(final_adjacency, config.cluster_threshold)
    largest_cluster = max(cluster_sizes) if cluster_sizes else 0

    false_mask = state["false_mask"]
    if np.any(false_mask):
        false_isolation_rate = float(np.mean(degrees[false_mask] == 0))
    else:
        false_isolation_rate = 0.0

    persistence_rows, persistence_mean = cluster_persistence_rows(
        scenario=scenario,
        run_index=run_index,
        seed=seed,
        sampled_memberships=sampled_memberships,
    )
    memory_index = topological_memory_index(
        sampled_clusters,
        config.memory_similarity_threshold,
    )
    late_values = coherence_history[-min(len(coherence_history), config.late_window) :]

    metrics = PhaseIIIRunMetrics(
        scenario=scenario.name,
        label=scenario.label,
        model_kind=scenario.model_kind,
        bootstrap_mode=scenario.bootstrap_mode,
        false_mode=scenario.false_mode,
        run_index=run_index,
        seed=seed,
        final_coherence=float(coherence_history[-1]),
        late_coherence=float(np.mean(late_values)),
        largest_cluster_fraction=largest_cluster / config.nodes,
        cluster_count=len(cluster_sizes),
        mean_degree=float(np.mean(degrees)),
        false_isolation_rate=false_isolation_rate,
        resonance_energy=float(resonance_energy_history[-1]),
        cluster_persistence=persistence_mean,
        topological_memory_index=memory_index,
        triangle_density=float(triangle_density_history[-1]),
    )
    return metrics, persistence_rows


def run_single_phase3(
    config: PhaseIIIConfig,
    scenario: PhaseIIIScenario,
    run_index: int,
    seed: int,
) -> tuple[PhaseIIIRunMetrics, list[dict[str, Any]], np.ndarray]:
    model = PHASE3_MODEL_REGISTRY[scenario.model_kind]
    state = initialize_state(config, scenario, seed)
    rng = np.random.default_rng(seed + 1)

    coherence_history: list[float] = []
    resonance_energy_history: list[float] = []
    triangle_density_history: list[float] = []
    energy_timeline = np.zeros(config.steps, dtype=float)

    sampled_memberships: list[tuple[int, set[int]]] = []
    sampled_clusters: list[list[set[int]]] = []
    final_adjacency = np.zeros((config.nodes, config.nodes), dtype=bool)

    for step in range(config.steps):
        if (
            scenario.bootstrap_mode == "periodic"
            and step > 0
            and step % config.bootstrap_period == 0
        ):
            state["theta"] = bootstrap_kick(
                state["theta"],
                config.bootstrap_anchor,
                config.bootstrap_strength,
            )

        if scenario.false_mode == "random" and np.any(state["false_mask"]):
            state["freq"][state["false_mask"]] += rng.normal(
                loc=0.0,
                scale=config.false_frequency_drift_sigma * np.sqrt(config.dt),
                size=int(np.sum(state["false_mask"])),
            )
            state["freq"][state["false_mask"]] = np.clip(
                state["freq"][state["false_mask"]],
                0.5,
                1.5,
            )

        step_result = model.step(state, config)
        state["theta"] = (
            state["theta"] + (state["freq"] + step_result.phase_drive) * config.dt
        ) % (2.0 * np.pi)

        if scenario.false_mode == "noisy" and np.any(state["false_mask"]):
            state["theta"][state["false_mask"]] += rng.normal(
                loc=0.0,
                scale=config.false_phase_noise_sigma * np.sqrt(config.dt),
                size=int(np.sum(state["false_mask"])),
            )
            state["theta"] %= 2.0 * np.pi

        coherence_history.append(float(np.abs(np.mean(np.exp(1j * state["theta"])))))
        resonance_energy_history.append(step_result.resonance_energy)
        triangle_density_history.append(step_result.triangle_density)
        energy_timeline[step] = step_result.resonance_energy
        final_adjacency = step_result.validated_adjacency

        if (step + 1) % config.persistence_delta == 0:
            clusters, _, _ = cluster_sets(final_adjacency, config.cluster_threshold)
            primary_cluster = max(clusters, key=len, default=set())
            sampled_memberships.append((step + 1, primary_cluster))
            sampled_clusters.append(clusters)

    metrics, persistence_rows = finalize_run(
        scenario=scenario,
        config=config,
        run_index=run_index,
        seed=seed,
        state=state,
        final_adjacency=final_adjacency,
        coherence_history=coherence_history,
        resonance_energy_history=resonance_energy_history,
        triangle_density_history=triangle_density_history,
        sampled_memberships=sampled_memberships,
        sampled_clusters=sampled_clusters,
    )
    return metrics, persistence_rows, energy_timeline


def summarize_runs(
    scenarios: list[PhaseIIIScenario],
    run_metrics: list[PhaseIIIRunMetrics],
) -> list[dict[str, float | str]]:
    grouped: dict[str, list[PhaseIIIRunMetrics]] = {}
    for item in run_metrics:
        grouped.setdefault(item.scenario, []).append(item)

    metric_fields = [
        "final_coherence",
        "late_coherence",
        "largest_cluster_fraction",
        "cluster_count",
        "mean_degree",
        "false_isolation_rate",
        "resonance_energy",
        "cluster_persistence",
        "topological_memory_index",
        "triangle_density",
    ]

    rows: list[dict[str, float | str]] = []
    for scenario in scenarios:
        items = grouped.get(scenario.name, [])
        if not items:
            continue
        row: dict[str, float | str] = {
            "scenario": scenario.name,
            "label": scenario.label,
            "model_kind": scenario.model_kind,
            "bootstrap_mode": scenario.bootstrap_mode,
            "false_mode": scenario.false_mode,
            "runs": len(items),
        }
        for field in metric_fields:
            values = np.array([float(getattr(item, field)) for item in items], dtype=float)
            row[f"{field}_mean"] = float(np.mean(values))
            row[f"{field}_std"] = float(np.std(values))
        rows.append(row)
    return rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def render_grouped_metric(
    summary_rows: list[dict[str, float | str]],
    output_path: Path,
    metric_key: str,
    title: str,
) -> None:
    row_map = {
        (str(row["model_kind"]), str(row["bootstrap_mode"]), str(row["false_mode"])): row
        for row in summary_rows
    }
    model_order = ("baseline", "triangle", "helix", "hybrid_manifold")
    false_modes = ("none", "random", "noisy")
    bootstrap_modes = ("single", "periodic")
    x = np.arange(len(false_modes))
    width = 0.34

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    for axis, model_kind in zip(axes.flat, model_order):
        for offset, bootstrap_mode in enumerate(bootstrap_modes):
            means = []
            stds = []
            for false_mode in false_modes:
                row = row_map[(model_kind, bootstrap_mode, false_mode)]
                means.append(float(row[f"{metric_key}_mean"]))
                stds.append(float(row[f"{metric_key}_std"]))
            axis.bar(
                x + (offset - 0.5) * width,
                means,
                width=width,
                yerr=stds,
                capsize=4,
                label=bootstrap_mode if model_kind == "baseline" else None,
                alpha=0.88,
            )
        axis.set_title(PHASE3_MODEL_LABELS[model_kind])
        axis.set_xticks(x)
        axis.set_xticklabels(false_modes)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(bottom=0.0)
    axes[0, 0].legend(title="bootstrap", loc="upper left")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_energy_timeline(
    scenarios: list[PhaseIIIScenario],
    timeline_means: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    model_order = ("baseline", "triangle", "helix", "hybrid_manifold")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    x = np.arange(len(next(iter(timeline_means.values()))))

    for axis, model_kind in zip(axes.flat, model_order):
        for scenario in [item for item in scenarios if item.model_kind == model_kind]:
            axis.plot(
                x,
                timeline_means[scenario.name],
                linewidth=1.4,
                label=f"{scenario.bootstrap_mode}/{scenario.false_mode}",
            )
        axis.set_title(PHASE3_MODEL_LABELS[model_kind])
        axis.set_xlabel("step")
        axis.set_ylabel("energy")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)

    fig.suptitle("Resonance Energy Timeline", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_console_summary(summary_rows: list[dict[str, float | str]]) -> None:
    print("\nPhase III summary (late coherence / persistence / memory):")
    for row in summary_rows:
        print(
            f"- {row['scenario']}: "
            f"coherence={row['late_coherence_mean']:.3f}, "
            f"persistence={row['cluster_persistence_mean']:.3f}, "
            f"memory={row['topological_memory_index_mean']:.3f}"
        )


def run_phase3_experiments(
    config: PhaseIIIConfig,
    output_dir: Path,
    runs: int,
    seed_base: int,
) -> dict[str, Any]:
    scenarios = build_phase3_scenarios()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_metrics: list[PhaseIIIRunMetrics] = []
    persistence_rows: list[dict[str, Any]] = []
    timeline_sums: dict[str, np.ndarray] = {
        scenario.name: np.zeros(config.steps, dtype=float) for scenario in scenarios
    }

    for scenario_index, scenario in enumerate(scenarios):
        for run_index in range(runs):
            seed = seed_base + scenario_index * 100 + run_index
            metrics, cluster_rows, energy_timeline = run_single_phase3(
                config=config,
                scenario=scenario,
                run_index=run_index,
                seed=seed,
            )
            run_metrics.append(metrics)
            persistence_rows.extend(cluster_rows)
            timeline_sums[scenario.name] += energy_timeline

    timeline_means = {
        scenario.name: timeline_sums[scenario.name] / runs for scenario in scenarios
    }
    run_rows = [asdict(item) for item in run_metrics]
    summary_rows = summarize_runs(scenarios, run_metrics)

    save_csv(output_dir / "run_metrics.csv", run_rows)
    save_csv(output_dir / "cluster_persistence.csv", persistence_rows)
    save_csv(output_dir / "scenario_summary.csv", summary_rows)
    save_json(
        output_dir / "scenario_summary.json",
        {
            "config": asdict(config),
            "runs_per_scenario": runs,
            "scenarios": [asdict(item) for item in scenarios],
            "run_metrics": run_rows,
            "cluster_persistence_rows": persistence_rows,
            "summary": summary_rows,
            "energy_timelines_mean": {
                key: value.tolist() for key, value in timeline_means.items()
            },
        },
    )

    render_grouped_metric(
        summary_rows,
        output_dir / "coherence_comparison.png",
        "late_coherence",
        "Late Coherence Comparison",
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "cluster_persistence_comparison.png",
        "cluster_persistence",
        "Cluster Persistence vs Topology",
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "false_isolation_comparison.png",
        "false_isolation_rate",
        "False Isolation Comparison",
    )
    render_energy_timeline(
        scenarios,
        timeline_means,
        output_dir / "resonance_energy_timeline.png",
    )
    print_console_summary(summary_rows)
    return {
        "output_dir": output_dir,
        "summary_rows": summary_rows,
        "run_rows": run_rows,
        "cluster_persistence_rows": persistence_rows,
    }
