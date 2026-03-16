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

from resonance_field.phase4_config import (
    PHASE4_MODEL_LABELS,
    PhaseIVConfig,
    PhaseIVScenario,
    build_phase4_scenarios,
)
from resonance_field.phase4_models import PHASE4_MODEL_REGISTRY
from resonance_field.topologies import bootstrap_kick


@dataclass
class PhaseIVRunMetrics:
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
    memory_edge_mass: float
    recall_event_rate: float
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
) -> tuple[list[set[int]], list[int]]:
    labels, sizes = connected_components(adjacency)
    clusters: list[set[int]] = []
    cluster_sizes: list[int] = []
    for component_id, size in enumerate(sizes):
        if size < threshold:
            continue
        members = set(np.flatnonzero(labels == component_id).tolist())
        clusters.append(members)
        cluster_sizes.append(size)
    return clusters, cluster_sizes


def jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def primary_cluster(clusters: list[set[int]]) -> set[int]:
    return max(clusters, key=len, default=set())


def reinforce_memory(
    memory_matrix: np.ndarray,
    cluster: set[int],
    config: PhaseIVConfig,
    boost: float = 1.0,
) -> None:
    if len(cluster) < config.cluster_threshold:
        return
    indices = np.array(sorted(cluster), dtype=int)
    memory_matrix[np.ix_(indices, indices)] += config.memory_learning_rate * boost
    np.fill_diagonal(memory_matrix, 0.0)
    np.clip(memory_matrix, 0.0, config.memory_max, out=memory_matrix)


def cluster_persistence_rows(
    scenario: PhaseIVScenario,
    run_index: int,
    seed: int,
    sampled_primary_clusters: list[tuple[int, set[int]]],
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    values: list[float] = []
    for (time_start, cluster_start), (time_end, cluster_end) in zip(
        sampled_primary_clusters,
        sampled_primary_clusters[1:],
    ):
        persistence = len(cluster_start & cluster_end) / len(cluster_start) if cluster_start else 0.0
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


def initialize_state(
    config: PhaseIVConfig,
    scenario: PhaseIVScenario,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    model = PHASE4_MODEL_REGISTRY[scenario.model_kind]
    state = model.initialize_state(config, rng, scenario.false_mode)
    if scenario.bootstrap_mode in {"single", "periodic"}:
        state["theta"] = bootstrap_kick(
            state["theta"],
            config.bootstrap_anchor,
            config.bootstrap_strength,
        )
    return state


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
        axis.set_title(PHASE4_MODEL_LABELS[model_kind])
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
    scenarios: list[PhaseIVScenario],
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
        axis.set_title(PHASE4_MODEL_LABELS[model_kind])
        axis.set_xlabel("step")
        axis.set_ylabel("energy")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)

    fig.suptitle("Phase IV Resonance Energy Timeline", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    if not rows and not fieldnames:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def summarize_runs(
    scenarios: list[PhaseIVScenario],
    run_metrics: list[PhaseIVRunMetrics],
) -> list[dict[str, float | str]]:
    grouped: dict[str, list[PhaseIVRunMetrics]] = {}
    for item in run_metrics:
        grouped.setdefault(item.scenario, []).append(item)

    fields = [
        "final_coherence",
        "late_coherence",
        "largest_cluster_fraction",
        "cluster_count",
        "mean_degree",
        "false_isolation_rate",
        "resonance_energy",
        "cluster_persistence",
        "topological_memory_index",
        "memory_edge_mass",
        "recall_event_rate",
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
        for field in fields:
            values = np.array([float(getattr(item, field)) for item in items], dtype=float)
            row[f"{field}_mean"] = float(np.mean(values))
            row[f"{field}_std"] = float(np.std(values))
        rows.append(row)
    return rows


def run_single_phase4(
    config: PhaseIVConfig,
    scenario: PhaseIVScenario,
    run_index: int,
    seed: int,
) -> tuple[
    PhaseIVRunMetrics,
    list[dict[str, Any]],
    list[dict[str, Any]],
    np.ndarray,
]:
    model = PHASE4_MODEL_REGISTRY[scenario.model_kind]
    state = initialize_state(config, scenario, seed)
    rng = np.random.default_rng(seed + 1)

    memory_matrix = np.zeros((config.nodes, config.nodes), dtype=float)
    recall_cooldown = 0
    recall_events = 0
    memory_events = 0
    memory_supported_samples = 0
    pending_recall_target: set[int] | None = None

    stored_prototypes: list[set[int]] = []
    previous_primary = set()

    coherence_history: list[float] = []
    resonance_energy_history: list[float] = []
    triangle_density_history: list[float] = []
    memory_mass_history: list[float] = []
    energy_timeline = np.zeros(config.steps, dtype=float)

    persistence_rows: list[dict[str, Any]] = []
    memory_event_rows: list[dict[str, Any]] = []
    sampled_primary_clusters: list[tuple[int, set[int]]] = []

    final_adjacency = np.zeros((config.nodes, config.nodes), dtype=bool)

    for step in range(config.steps):
        memory_matrix *= config.memory_decay
        np.fill_diagonal(memory_matrix, 0.0)

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

        base_layer = model.base_layer(state, config)
        memory_mask = (
            (memory_matrix > config.memory_activation_threshold)
            & (base_layer.pair_distance < config.memory_radius_factor * config.radius)
        )
        np.fill_diagonal(memory_mask, False)

        effective_weights = base_layer.validated_weights + (
            config.memory_influence * memory_matrix * memory_mask
        )
        effective_adjacency = base_layer.validated_adjacency | memory_mask

        phase_diff = state["theta"][None, :] - state["theta"][:, None]
        phase_drive = config.phase_coupling * np.sum(
            effective_weights * np.sin(phase_diff),
            axis=1,
        )
        state["theta"] = (state["theta"] + (state["freq"] + phase_drive) * config.dt) % (
            2.0 * np.pi
        )

        if scenario.false_mode == "noisy" and np.any(state["false_mask"]):
            state["theta"][state["false_mask"]] += rng.normal(
                loc=0.0,
                scale=config.false_phase_noise_sigma * np.sqrt(config.dt),
                size=int(np.sum(state["false_mask"])),
            )
            state["theta"] %= 2.0 * np.pi

        phase_diff = state["theta"][None, :] - state["theta"][:, None]
        model.apply_motion(
            state=state,
            effective_weights=effective_weights,
            phase_diff=phase_diff,
            pair_distance=base_layer.pair_distance,
            geometry_data=base_layer.geometry_data,
            config=config,
        )

        coherence_history.append(float(np.abs(np.mean(np.exp(1j * state["theta"])))))
        resonance_energy_history.append(float(0.5 * np.sum(np.cos(phase_diff) * effective_weights)))
        triangle_density_history.append(base_layer.triangle_density)
        memory_mass_history.append(float(np.mean(np.triu(memory_matrix, k=1))))
        energy_timeline[step] = resonance_energy_history[-1]
        final_adjacency = effective_adjacency

        if recall_cooldown > 0:
            recall_cooldown -= 1

        if (step + 1) % config.persistence_delta == 0:
            clusters, cluster_sizes = cluster_sets(effective_adjacency, config.cluster_threshold)
            current_primary = primary_cluster(clusters)
            sampled_primary_clusters.append((step + 1, current_primary))

            persistence = jaccard(previous_primary, current_primary) if previous_primary else 0.0
            if current_primary:
                current_indices = np.array(sorted(current_primary), dtype=int)
                component_memory = memory_mask[np.ix_(current_indices, current_indices)]
                upper = np.triu(component_memory, k=1)
                memory_supported = bool(np.any(upper)) and any(
                    jaccard(current_primary, prototype) >= config.memory_similarity_threshold
                    for prototype in stored_prototypes
                )
                if memory_supported:
                    memory_supported_samples += 1
            if current_primary and persistence >= config.stability_gate:
                reinforce_memory(memory_matrix, current_primary, config)
                if not any(
                    jaccard(current_primary, prototype) >= config.memory_similarity_threshold
                    for prototype in stored_prototypes
                ):
                    stored_prototypes.append(set(current_primary))

            if (
                pending_recall_target
                and current_primary
                and jaccard(current_primary, pending_recall_target) >= config.memory_similarity_threshold
                and jaccard(previous_primary, current_primary) < config.memory_similarity_threshold
            ):
                memory_events += 1
                memory_event_rows.append(
                    {
                        "scenario": scenario.name,
                        "model_kind": scenario.model_kind,
                        "bootstrap_mode": scenario.bootstrap_mode,
                        "false_mode": scenario.false_mode,
                        "run_index": run_index,
                        "seed": seed,
                        "step": step + 1,
                        "memory_event": 1,
                        "cluster_size": len(current_primary),
                    }
                )
                pending_recall_target = None
            elif current_primary and previous_primary:
                returned = (
                    jaccard(previous_primary, current_primary) < config.memory_similarity_threshold
                    and any(
                        jaccard(current_primary, prototype) >= config.memory_similarity_threshold
                        for prototype in stored_prototypes
                    )
                )
                if returned:
                    memory_events += 1
                    memory_event_rows.append(
                        {
                            "scenario": scenario.name,
                            "model_kind": scenario.model_kind,
                            "bootstrap_mode": scenario.bootstrap_mode,
                            "false_mode": scenario.false_mode,
                            "run_index": run_index,
                            "seed": seed,
                            "step": step + 1,
                            "memory_event": 1,
                            "cluster_size": len(current_primary),
                        }
                    )

            recall_candidates = [
                prototype
                for prototype in stored_prototypes
                if jaccard(prototype, current_primary) < config.memory_similarity_threshold
            ]
            if (
                recall_cooldown == 0
                and recall_candidates
                and (
                    not current_primary
                    or persistence < config.stability_gate
                    or (step + 1) % (2 * config.persistence_delta) == 0
                )
            ):
                prototype = max(recall_candidates, key=len)
                indices = np.array(sorted(prototype), dtype=int)
                state["theta"][indices] += config.recall_phase_strength * np.sin(
                    config.bootstrap_anchor - state["theta"][indices]
                )
                state["theta"] %= 2.0 * np.pi
                reinforce_memory(memory_matrix, prototype, config, boost=0.5)
                recall_cooldown = config.recall_cooldown
                recall_events += 1
                pending_recall_target = set(prototype)
                memory_event_rows.append(
                    {
                        "scenario": scenario.name,
                        "model_kind": scenario.model_kind,
                        "bootstrap_mode": scenario.bootstrap_mode,
                        "false_mode": scenario.false_mode,
                        "run_index": run_index,
                        "seed": seed,
                        "step": step + 1,
                        "memory_event": 0,
                        "cluster_size": len(prototype),
                    }
                )

            previous_primary = set(current_primary)

    persistence_rows, persistence_mean = cluster_persistence_rows(
        scenario=scenario,
        run_index=run_index,
        seed=seed,
        sampled_primary_clusters=sampled_primary_clusters,
    )

    degrees = final_adjacency.sum(axis=1).astype(float)
    _, final_cluster_sizes = cluster_sets(final_adjacency, config.cluster_threshold)
    largest_cluster = max(final_cluster_sizes) if final_cluster_sizes else 0
    false_mask = state["false_mask"]
    false_isolation_rate = float(np.mean(degrees[false_mask] == 0)) if np.any(false_mask) else 0.0

    late_values = coherence_history[-min(len(coherence_history), config.late_window) :]
    metrics = PhaseIVRunMetrics(
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
        cluster_count=len(final_cluster_sizes),
        mean_degree=float(np.mean(degrees)),
        false_isolation_rate=false_isolation_rate,
        resonance_energy=float(resonance_energy_history[-1]),
        cluster_persistence=persistence_mean,
        topological_memory_index=memory_supported_samples / max(1, len(sampled_primary_clusters)),
        memory_edge_mass=float(np.mean(memory_mass_history)) if memory_mass_history else 0.0,
        recall_event_rate=recall_events / max(1, len(sampled_primary_clusters)),
        triangle_density=float(triangle_density_history[-1]) if triangle_density_history else 0.0,
    )
    return metrics, persistence_rows, memory_event_rows, energy_timeline


def run_phase4_experiments(
    config: PhaseIVConfig,
    output_dir: Path,
    runs: int,
    seed_base: int,
) -> dict[str, Any]:
    scenarios = build_phase4_scenarios()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_metrics: list[PhaseIVRunMetrics] = []
    persistence_rows: list[dict[str, Any]] = []
    memory_event_rows: list[dict[str, Any]] = []
    timeline_sums: dict[str, np.ndarray] = {
        scenario.name: np.zeros(config.steps, dtype=float) for scenario in scenarios
    }

    for scenario_index, scenario in enumerate(scenarios):
        for run_index in range(runs):
            seed = seed_base + scenario_index * 100 + run_index
            metrics, cluster_rows, event_rows, energy_timeline = run_single_phase4(
                config=config,
                scenario=scenario,
                run_index=run_index,
                seed=seed,
            )
            run_metrics.append(metrics)
            persistence_rows.extend(cluster_rows)
            memory_event_rows.extend(event_rows)
            timeline_sums[scenario.name] += energy_timeline

    timeline_means = {
        scenario.name: timeline_sums[scenario.name] / runs for scenario in scenarios
    }
    run_rows = [asdict(item) for item in run_metrics]
    summary_rows = summarize_runs(scenarios, run_metrics)

    save_csv(output_dir / "run_metrics.csv", run_rows)
    save_csv(output_dir / "cluster_persistence.csv", persistence_rows)
    save_csv(
        output_dir / "memory_events.csv",
        memory_event_rows,
        fieldnames=[
            "scenario",
            "model_kind",
            "bootstrap_mode",
            "false_mode",
            "run_index",
            "seed",
            "step",
            "memory_event",
            "cluster_size",
        ],
    )
    save_csv(output_dir / "scenario_summary.csv", summary_rows)
    save_json(
        output_dir / "scenario_summary.json",
        {
            "config": asdict(config),
            "runs_per_scenario": runs,
            "scenarios": [asdict(item) for item in scenarios],
            "run_metrics": run_rows,
            "cluster_persistence_rows": persistence_rows,
            "memory_event_rows": memory_event_rows,
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
        "Phase IV Late Coherence Comparison",
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "cluster_persistence_comparison.png",
        "cluster_persistence",
        "Phase IV Cluster Persistence Comparison",
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "memory_index_comparison.png",
        "topological_memory_index",
        "Phase IV Topological Memory Comparison",
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "false_isolation_comparison.png",
        "false_isolation_rate",
        "Phase IV False Isolation Comparison",
    )
    render_energy_timeline(
        scenarios,
        timeline_means,
        output_dir / "resonance_energy_timeline.png",
    )

    print("\nPhase IV summary (late coherence / persistence / memory / recall):")
    for row in summary_rows:
        print(
            f"- {row['scenario']}: "
            f"coherence={row['late_coherence_mean']:.3f}, "
            f"persistence={row['cluster_persistence_mean']:.3f}, "
            f"memory={row['topological_memory_index_mean']:.3f}, "
            f"recall={row['recall_event_rate_mean']:.3f}"
        )

    return {
        "output_dir": output_dir,
        "summary_rows": summary_rows,
        "run_rows": run_rows,
        "cluster_persistence_rows": persistence_rows,
        "memory_event_rows": memory_event_rows,
    }
