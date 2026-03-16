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

from resonance_field.config import ExperimentConfig, MODEL_LABELS, Scenario, build_scenarios
from resonance_field.topologies import (
    TOPOLOGY_REGISTRY,
    bootstrap_kick,
    build_false_frequencies,
)


@dataclass
class RunMetrics:
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
    cluster_membership_fraction: float
    cluster_count: int
    mean_degree: float
    false_isolation_rate: float
    false_mean_degree: float
    normal_mean_degree: float
    resonance_energy: float
    topological_persistence: float
    triangle_density: float
    resonance_channels: float
    spatial_clustering: float


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


def compute_topological_persistence(samples: list[np.ndarray]) -> float:
    if len(samples) < 2:
        return 1.0

    scores: list[float] = []
    for previous, current in zip(samples, samples[1:]):
        prev_edges = np.triu(previous, k=1)
        curr_edges = np.triu(current, k=1)
        union = np.logical_or(prev_edges, curr_edges).sum()
        if union == 0:
            scores.append(1.0)
            continue
        intersection = np.logical_and(prev_edges, curr_edges).sum()
        scores.append(float(intersection / union))
    return float(np.mean(scores))


def compute_channel_count(
    manifold: np.ndarray | None,
    labels: np.ndarray,
    sizes: list[int],
    support: np.ndarray,
    config: ExperimentConfig,
) -> float:
    if manifold is None:
        return 0.0

    span_threshold = config.channel_turn_threshold * 2.0 * np.pi
    count = 0
    for component_id, size in enumerate(sizes):
        if size < config.cluster_threshold:
            continue
        member_indices = np.flatnonzero(labels == component_id)
        members = manifold[member_indices]
        if not members.size:
            continue
        component_support = support[np.ix_(member_indices, member_indices)]
        component_delta = np.abs(members[None, :] - members[:, None])
        inter_turn_support = np.triu(
            component_support & (component_delta >= span_threshold),
            k=1,
        )
        if float(np.max(members) - np.min(members)) >= span_threshold or np.any(inter_turn_support):
            count += 1
    return float(count)


def initialize_state(
    config: ExperimentConfig,
    scenario: Scenario,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    topology = TOPOLOGY_REGISTRY[scenario.model_kind]
    state = topology.initialize_state(config, rng)
    state["theta"] = rng.uniform(0.0, 2.0 * np.pi, size=config.nodes)
    state["freq"] = rng.uniform(0.95, 1.05, size=config.nodes)
    state["false_mask"] = np.zeros(config.nodes, dtype=bool)

    false_count = int(round(config.nodes * config.false_ratio)) if scenario.false_mode != "none" else 0
    if false_count:
        false_indices = rng.choice(config.nodes, size=false_count, replace=False)
        state["false_mask"][false_indices] = True
        state["freq"][state["false_mask"]] = build_false_frequencies(rng, false_count)

    if scenario.bootstrap_mode in {"single", "periodic"}:
        state["theta"] = bootstrap_kick(
            state["theta"],
            config.bootstrap_anchor,
            config.bootstrap_strength,
        )

    state["rng_seed"] = np.array([seed], dtype=int)
    return state


def finalize_metrics(
    scenario: Scenario,
    config: ExperimentConfig,
    run_index: int,
    seed: int,
    state: dict[str, np.ndarray],
    final_support: np.ndarray,
    coherence_history: list[float],
    persistence_samples: list[np.ndarray],
    resonance_energy_history: list[float],
    triangle_density_history: list[float],
    spatial_clustering_history: list[float],
) -> RunMetrics:
    degrees = final_support.sum(axis=1).astype(float)
    labels, sizes = connected_components(final_support)
    cluster_sizes = [size for size in sizes if size >= config.cluster_threshold]
    largest_cluster = max(sizes) if sizes else 0
    cluster_membership = sum(cluster_sizes)

    false_mask = state["false_mask"]
    if false_mask.any():
        false_degrees = degrees[false_mask]
        false_isolation_rate = float(np.mean(false_degrees == 0))
        false_mean_degree = float(np.mean(false_degrees))
    else:
        false_isolation_rate = 0.0
        false_mean_degree = 0.0

    normal_mask = ~false_mask
    normal_mean_degree = float(np.mean(degrees[normal_mask])) if normal_mask.any() else 0.0
    late_values = coherence_history[-min(len(coherence_history), config.late_window) :]

    topology = TOPOLOGY_REGISTRY[scenario.model_kind]
    resonance_channels = compute_channel_count(
        topology.manifold_coordinate(state),
        labels,
        sizes,
        final_support,
        config,
    )

    return RunMetrics(
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
        cluster_membership_fraction=cluster_membership / config.nodes,
        cluster_count=len(cluster_sizes),
        mean_degree=float(np.mean(degrees)),
        false_isolation_rate=false_isolation_rate,
        false_mean_degree=false_mean_degree,
        normal_mean_degree=normal_mean_degree,
        resonance_energy=float(resonance_energy_history[-1]),
        topological_persistence=compute_topological_persistence(persistence_samples),
        triangle_density=float(triangle_density_history[-1]),
        resonance_channels=resonance_channels,
        spatial_clustering=float(spatial_clustering_history[-1]),
    )


def run_single(
    config: ExperimentConfig,
    scenario: Scenario,
    run_index: int,
    seed: int,
) -> tuple[RunMetrics, dict[str, Any]]:
    topology = TOPOLOGY_REGISTRY[scenario.model_kind]
    state = initialize_state(config, scenario, seed)
    rng = np.random.default_rng(seed + 1)

    coherence_history: list[float] = []
    persistence_samples: list[np.ndarray] = []
    resonance_energy_history: list[float] = []
    triangle_density_history: list[float] = []
    spatial_clustering_history: list[float] = []
    final_support = np.zeros((config.nodes, config.nodes), dtype=bool)

    for step in range(config.steps):
        if (
            scenario.bootstrap_mode == "periodic"
            and step > 0
            and step % config.bootstrap_period == 0
        ):
            state["theta"] = bootstrap_kick(
                state["theta"],
                config.bootstrap_anchor,
                config.periodic_strength,
            )

        if scenario.false_mode == "noisy" and np.any(state["false_mask"]):
            state["freq"][state["false_mask"]] += rng.normal(
                loc=0.0,
                scale=config.noisy_false_sigma * np.sqrt(config.dt),
                size=int(np.sum(state["false_mask"])),
            )
            state["freq"][state["false_mask"]] = np.clip(
                state["freq"][state["false_mask"]],
                0.5,
                1.5,
            )

        step_result = topology.step(state, config)
        state["theta"] = (
            state["theta"] + (state["freq"] + step_result.phase_drive) * config.dt
        ) % (2.0 * np.pi)

        coherence_history.append(float(np.abs(np.mean(np.exp(1j * state["theta"])))))
        resonance_energy_history.append(step_result.resonance_energy)
        triangle_density_history.append(step_result.triangle_density)
        spatial_clustering_history.append(step_result.spatial_clustering)
        final_support = step_result.support

        if step % config.persistence_stride == 0 or step == config.steps - 1:
            persistence_samples.append(step_result.support.copy())

    metrics = finalize_metrics(
        scenario=scenario,
        config=config,
        run_index=run_index,
        seed=seed,
        state=state,
        final_support=final_support,
        coherence_history=coherence_history,
        persistence_samples=persistence_samples,
        resonance_energy_history=resonance_energy_history,
        triangle_density_history=triangle_density_history,
        spatial_clustering_history=spatial_clustering_history,
    )

    snapshot = topology.snapshot(state, config)
    snapshot["theta"] = state["theta"].copy()
    snapshot["false_mask"] = state["false_mask"].copy()
    snapshot["degrees"] = final_support.sum(axis=1).astype(float)
    return metrics, snapshot


def summarize_runs(
    scenarios: list[Scenario],
    metrics: list[RunMetrics],
) -> list[dict[str, float | str]]:
    scenario_lookup = {scenario.name: scenario for scenario in scenarios}
    grouped: dict[str, list[RunMetrics]] = {}
    for item in metrics:
        grouped.setdefault(item.scenario, []).append(item)

    fields = [
        "final_coherence",
        "late_coherence",
        "largest_cluster_fraction",
        "cluster_membership_fraction",
        "cluster_count",
        "mean_degree",
        "false_isolation_rate",
        "false_mean_degree",
        "normal_mean_degree",
        "resonance_energy",
        "topological_persistence",
        "triangle_density",
        "resonance_channels",
        "spatial_clustering",
    ]

    summary_rows: list[dict[str, float | str]] = []
    for scenario in scenarios:
        rows = grouped.get(scenario.name, [])
        if not rows:
            continue
        item: dict[str, float | str] = {
            "scenario": scenario.name,
            "label": scenario.label,
            "model_kind": scenario.model_kind,
            "bootstrap_mode": scenario.bootstrap_mode,
            "false_mode": scenario.false_mode,
            "runs": len(rows),
        }
        for field in fields:
            values = np.array([float(getattr(row, field)) for row in rows], dtype=float)
            item[f"{field}_mean"] = float(np.mean(values))
            item[f"{field}_std"] = float(np.std(values))
        summary_rows.append(item)

    return summary_rows


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


def render_metric_comparison(
    summary_rows: list[dict[str, float | str]],
    output_path: Path,
    metric_key: str,
    title: str,
) -> None:
    row_map = {
        (str(row["model_kind"]), str(row["bootstrap_mode"]), str(row["false_mode"])): row
        for row in summary_rows
    }
    model_order = ("baseline", "triangle", "helix", "hybrid")
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
        axis.set_title(MODEL_LABELS[model_kind])
        axis.set_xticks(x)
        axis.set_xticklabels(false_modes)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(bottom=0.0)
    axes[0, 0].legend(title="bootstrap", loc="upper left")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_degree_distribution(
    scenario_samples: dict[str, dict[str, Any]],
    scenarios: list[Scenario],
    output_path: Path,
) -> None:
    model_order = ("baseline", "triangle", "helix", "hybrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, model_kind in zip(axes.flat, model_order):
        model_scenarios = [scenario for scenario in scenarios if scenario.model_kind == model_kind]
        for scenario in model_scenarios:
            sample = scenario_samples[scenario.name]
            axis.hist(
                sample["degrees"],
                bins=14,
                histtype="step",
                linewidth=1.4,
                alpha=0.85,
                label=f"{scenario.bootstrap_mode}/{scenario.false_mode}",
            )
        axis.set_title(MODEL_LABELS[model_kind])
        axis.set_xlabel("degree")
        axis.set_ylabel("nodes")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Final Degree Distribution by Scenario Sample", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_snapshot(
    scenario: Scenario,
    sample: dict[str, Any],
    output_path: Path,
) -> None:
    if sample["projection"] == "3d":
        fig = plt.figure(figsize=(7, 6))
        axis = fig.add_subplot(111, projection="3d")
        normal_mask = ~sample["false_mask"]
        axis.scatter(
            sample["x"][normal_mask],
            sample["y"][normal_mask],
            sample["z"][normal_mask],
            c=sample["theta"][normal_mask],
            cmap="twilight",
            s=22,
            alpha=0.85,
        )
        if np.any(sample["false_mask"]):
            axis.scatter(
                sample["x"][sample["false_mask"]],
                sample["y"][sample["false_mask"]],
                sample["z"][sample["false_mask"]],
                c="#111111",
                s=28,
                marker="x",
                linewidths=1.3,
            )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
    else:
        fig, axis = plt.subplots(figsize=(6, 6))
        normal_mask = ~sample["false_mask"]
        axis.scatter(
            sample["x"][normal_mask],
            sample["y"][normal_mask],
            c=sample["theta"][normal_mask],
            cmap="twilight",
            s=26,
            alpha=0.85,
        )
        if np.any(sample["false_mask"]):
            axis.scatter(
                sample["x"][sample["false_mask"]],
                sample["y"][sample["false_mask"]],
                c="#111111",
                s=34,
                marker="x",
                linewidths=1.2,
            )
        axis.set_xlim(np.min(sample["x"]) - 0.05, np.max(sample["x"]) + 0.05)
        axis.set_ylim(np.min(sample["y"]) - 0.05, np.max(sample["y"]) + 0.05)
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    fig.suptitle(scenario.label, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_console_summary(summary_rows: list[dict[str, float | str]]) -> None:
    print("\nPhase II summary (late coherence / false isolation / persistence):")
    for row in summary_rows:
        print(
            f"- {row['scenario']}: "
            f"coherence={row['late_coherence_mean']:.3f}, "
            f"false_isolation={row['false_isolation_rate_mean']:.3f}, "
            f"persistence={row['topological_persistence_mean']:.3f}"
        )


def run_experiments(
    config: ExperimentConfig,
    output_dir: Path,
    runs: int,
    seed_base: int,
) -> dict[str, Any]:
    scenarios = build_scenarios()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    run_metrics: list[RunMetrics] = []
    scenario_samples: dict[str, dict[str, Any]] = {}

    for scenario_index, scenario in enumerate(scenarios):
        for run_index in range(runs):
            seed = seed_base + scenario_index * 100 + run_index
            metrics, snapshot = run_single(config, scenario, run_index, seed)
            run_metrics.append(metrics)

            current = scenario_samples.get(scenario.name)
            if current is None or metrics.late_coherence > float(current["score"]):
                scenario_samples[scenario.name] = {
                    **snapshot,
                    "score": float(metrics.late_coherence),
                }

    run_rows = [asdict(item) for item in run_metrics]
    summary_rows = summarize_runs(scenarios, run_metrics)

    save_csv(output_dir / "run_metrics.csv", run_rows)
    save_csv(output_dir / "scenario_summary.csv", summary_rows)
    save_json(
        output_dir / "scenario_summary.json",
        {
            "config": asdict(config),
            "runs_per_scenario": runs,
            "scenarios": [asdict(item) for item in scenarios],
            "run_metrics": run_rows,
            "summary": summary_rows,
        },
    )

    render_metric_comparison(
        summary_rows,
        output_dir / "coherence_comparison.png",
        "late_coherence",
        "Late Coherence Comparison",
    )
    render_metric_comparison(
        summary_rows,
        output_dir / "cluster_size_comparison.png",
        "largest_cluster_fraction",
        "Largest Cluster Fraction Comparison",
    )
    render_metric_comparison(
        summary_rows,
        output_dir / "false_isolation_comparison.png",
        "false_isolation_rate",
        "False Isolation Comparison",
    )
    render_degree_distribution(scenario_samples, scenarios, output_dir / "degree_distribution.png")

    for scenario in scenarios:
        render_snapshot(
            scenario,
            scenario_samples[scenario.name],
            snapshot_dir / f"{scenario.name}.png",
        )

    print_console_summary(summary_rows)
    return {
        "summary_rows": summary_rows,
        "run_rows": run_rows,
        "scenario_samples": scenario_samples,
        "output_dir": output_dir,
    }
