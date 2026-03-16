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

from resonance_field.phase4_experiment import (
    cluster_sets,
    jaccard,
    primary_cluster,
    reinforce_memory,
)
from resonance_field.phase4_models import PHASE4_MODEL_REGISTRY
from resonance_field.phase5_config import (
    PHASE5_MODEL_LABELS,
    PhaseVConfig,
    PhaseVScenario,
    build_phase5_scenarios,
)
from resonance_field.topologies import bootstrap_kick


@dataclass
class PhaseVRunMetrics:
    scenario: str
    label: str
    model_kind: str
    bootstrap_mode: str
    false_mode: str
    run_index: int
    seed: int
    pre_shock_coherence: float
    shock_min_coherence: float
    late_recovery_coherence: float
    coherence_recovery_ratio: float
    pre_shock_cluster_fraction: float
    shock_min_cluster_fraction: float
    late_cluster_fraction: float
    cluster_recovery_ratio: float
    final_false_isolation_rate: float
    recovery_time_steps: float
    recovery_success: float
    pre_shock_memory_mass: float
    late_memory_mass: float
    memory_recovery_ratio: float
    topological_memory_index: float
    recall_event_rate: float
    resonance_energy_late: float


def initialize_state(
    config: PhaseVConfig,
    scenario: PhaseVScenario,
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


def apply_shock(
    state: dict[str, np.ndarray],
    memory_matrix: np.ndarray,
    model_kind: str,
    config: PhaseVConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    node_count = config.nodes
    phase_count = max(1, int(round(node_count * config.shock_phase_fraction)))
    geometry_count = max(1, int(round(node_count * config.shock_geometry_fraction)))

    phase_nodes = rng.choice(node_count, size=phase_count, replace=False)
    geometry_nodes = rng.choice(node_count, size=geometry_count, replace=False)

    state["theta"][phase_nodes] = rng.uniform(0.0, 2.0 * np.pi, size=phase_count)
    state["freq"][phase_nodes] += rng.normal(
        loc=0.0,
        scale=config.shock_frequency_sigma,
        size=phase_count,
    )
    state["freq"] = np.clip(state["freq"], 0.5, 1.5)

    if "pos" in state:
        state["pos"][geometry_nodes] = rng.random((geometry_count, 2))
        state["vel"][geometry_nodes] = 0.0
    if "helix_t" in state:
        max_t = config.helix_turns * 2.0 * np.pi
        state["helix_t"][geometry_nodes] = rng.uniform(0.0, max_t, size=geometry_count)
        state["helix_vel"][geometry_nodes] = 0.0

    memory_matrix *= config.shock_memory_retention
    np.fill_diagonal(memory_matrix, 0.0)

    return {
        "phase_nodes": phase_count,
        "geometry_nodes": geometry_count,
        "memory_retention": config.shock_memory_retention,
    }


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if values.size < window:
        return np.full(values.shape, np.mean(values) if values.size else 0.0)
    kernel = np.ones(window, dtype=float) / window
    valid = np.convolve(values, kernel, mode="valid")
    prefix = np.full(window - 1, valid[0])
    return np.concatenate([prefix, valid])


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
        axis.set_title(PHASE5_MODEL_LABELS[model_kind])
        axis.set_xticks(x)
        axis.set_xticklabels(false_modes)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(bottom=0.0)
    axes[0, 0].legend(title="bootstrap", loc="upper left")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_recovery_timeline(
    scenarios: list[PhaseVScenario],
    timeline_means: dict[str, np.ndarray],
    config: PhaseVConfig,
    output_path: Path,
) -> None:
    model_order = ("baseline", "triangle", "helix", "hybrid_manifold")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    shock_step = min(config.shock_step, max(1, config.steps // 2))
    x = np.arange(-shock_step, config.steps - shock_step)

    for axis, model_kind in zip(axes.flat, model_order):
        for scenario in [item for item in scenarios if item.model_kind == model_kind]:
            axis.plot(
                x,
                timeline_means[scenario.name],
                linewidth=1.35,
                label=f"{scenario.bootstrap_mode}/{scenario.false_mode}",
            )
        axis.axvline(0, color="#222222", linestyle="--", linewidth=1.0)
        axis.set_title(PHASE5_MODEL_LABELS[model_kind])
        axis.set_xlabel("steps from shock")
        axis.set_ylabel("coherence")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)

    fig.suptitle("Shock Response and Recovery Timeline", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_runs(
    scenarios: list[PhaseVScenario],
    run_metrics: list[PhaseVRunMetrics],
) -> list[dict[str, float | str]]:
    grouped: dict[str, list[PhaseVRunMetrics]] = {}
    for item in run_metrics:
        grouped.setdefault(item.scenario, []).append(item)

    fields = [
        "pre_shock_coherence",
        "shock_min_coherence",
        "late_recovery_coherence",
        "coherence_recovery_ratio",
        "pre_shock_cluster_fraction",
        "shock_min_cluster_fraction",
        "late_cluster_fraction",
        "cluster_recovery_ratio",
        "final_false_isolation_rate",
        "recovery_time_steps",
        "recovery_success",
        "pre_shock_memory_mass",
        "late_memory_mass",
        "memory_recovery_ratio",
        "topological_memory_index",
        "recall_event_rate",
        "resonance_energy_late",
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


def run_single_phase5(
    config: PhaseVConfig,
    scenario: PhaseVScenario,
    run_index: int,
    seed: int,
) -> tuple[PhaseVRunMetrics, list[dict[str, Any]], np.ndarray]:
    shock_step = min(config.shock_step, max(1, config.steps // 2))
    model = PHASE4_MODEL_REGISTRY[scenario.model_kind]
    state = initialize_state(config, scenario, seed)
    rng = np.random.default_rng(seed + 1)

    memory_matrix = np.zeros((config.nodes, config.nodes), dtype=float)
    recall_cooldown = 0
    recall_events = 0
    memory_supported_samples = 0
    pending_recall_target: set[int] | None = None
    stored_prototypes: list[set[int]] = []
    previous_primary = set()

    coherence_history = np.zeros(config.steps, dtype=float)
    cluster_fraction_history = np.zeros(config.steps, dtype=float)
    memory_mass_history = np.zeros(config.steps, dtype=float)
    energy_history = np.zeros(config.steps, dtype=float)

    shock_rows: list[dict[str, Any]] = []
    final_adjacency = np.zeros((config.nodes, config.nodes), dtype=bool)

    for step in range(config.steps):
        memory_matrix *= config.memory_decay
        np.fill_diagonal(memory_matrix, 0.0)

        if step == shock_step:
            shock_info = apply_shock(state, memory_matrix, scenario.model_kind, config, rng)
            shock_rows.append(
                {
                    "scenario": scenario.name,
                    "model_kind": scenario.model_kind,
                    "bootstrap_mode": scenario.bootstrap_mode,
                    "false_mode": scenario.false_mode,
                    "run_index": run_index,
                    "seed": seed,
            "shock_step": step,
                    **shock_info,
                }
            )

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

        coherence_history[step] = float(np.abs(np.mean(np.exp(1j * state["theta"]))))
        energy_history[step] = float(0.5 * np.sum(np.cos(phase_diff) * effective_weights))
        memory_mass_history[step] = float(np.mean(np.triu(memory_matrix, k=1)))
        final_adjacency = effective_adjacency

        if recall_cooldown > 0:
            recall_cooldown -= 1

        if (step + 1) % config.persistence_delta == 0:
            clusters, cluster_sizes = cluster_sets(effective_adjacency, config.cluster_threshold)
            current_primary = primary_cluster(clusters)
            persistence = jaccard(previous_primary, current_primary) if previous_primary else 0.0
            cluster_fraction_history[step] = (
                len(current_primary) / config.nodes if current_primary else 0.0
            )

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
                pending_recall_target = None

            recall_candidates = [
                prototype
                for prototype in stored_prototypes
                if jaccard(prototype, current_primary) < config.memory_similarity_threshold
            ]
            if (
                step >= shock_step
                and recall_cooldown == 0
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

            previous_primary = set(current_primary)
        elif step > 0:
            cluster_fraction_history[step] = cluster_fraction_history[step - 1]

    pre_start = max(0, shock_step - config.shock_window)
    pre_slice = slice(pre_start, shock_step)
    post_end = min(config.steps, shock_step + config.shock_window)
    post_slice = slice(shock_step, post_end)
    late_slice = slice(max(0, config.steps - config.late_window), config.steps)

    pre_shock_coherence = float(np.mean(coherence_history[pre_slice]))
    shock_window_values = coherence_history[post_slice]
    if shock_window_values.size == 0:
        shock_window_values = coherence_history[shock_step:]
    shock_min_coherence = float(np.min(shock_window_values))
    late_recovery_coherence = float(np.mean(coherence_history[late_slice]))
    pre_shock_cluster_fraction = float(np.mean(cluster_fraction_history[pre_slice]))
    cluster_window_values = cluster_fraction_history[post_slice]
    if cluster_window_values.size == 0:
        cluster_window_values = cluster_fraction_history[shock_step:]
    shock_min_cluster_fraction = float(np.min(cluster_window_values))
    late_cluster_fraction = float(np.mean(cluster_fraction_history[late_slice]))
    pre_shock_memory_mass = float(np.mean(memory_mass_history[pre_slice]))
    late_memory_mass = float(np.mean(memory_mass_history[late_slice]))

    recovery_target = config.recovery_threshold * pre_shock_coherence
    cluster_target = config.cluster_recovery_threshold * max(pre_shock_cluster_fraction, 1e-6)
    coherence_recovery_curve = rolling_mean(coherence_history, config.response_window)
    cluster_recovery_curve = rolling_mean(cluster_fraction_history, config.response_window)

    recovery_time = -1
    for step in range(shock_step, config.steps):
        if (
            coherence_recovery_curve[step] >= recovery_target
            and cluster_recovery_curve[step] >= cluster_target
        ):
            recovery_time = step - shock_step
            break

    false_mask = state["false_mask"]
    degrees = final_adjacency.sum(axis=1).astype(float)
    final_false_isolation_rate = (
        float(np.mean(degrees[false_mask] == 0)) if np.any(false_mask) else 0.0
    )

    metrics = PhaseVRunMetrics(
        scenario=scenario.name,
        label=scenario.label,
        model_kind=scenario.model_kind,
        bootstrap_mode=scenario.bootstrap_mode,
        false_mode=scenario.false_mode,
        run_index=run_index,
        seed=seed,
        pre_shock_coherence=pre_shock_coherence,
        shock_min_coherence=shock_min_coherence,
        late_recovery_coherence=late_recovery_coherence,
        coherence_recovery_ratio=late_recovery_coherence / max(pre_shock_coherence, 1e-6),
        pre_shock_cluster_fraction=pre_shock_cluster_fraction,
        shock_min_cluster_fraction=shock_min_cluster_fraction,
        late_cluster_fraction=late_cluster_fraction,
        cluster_recovery_ratio=late_cluster_fraction / max(pre_shock_cluster_fraction, 1e-6),
        final_false_isolation_rate=final_false_isolation_rate,
        recovery_time_steps=float(recovery_time if recovery_time >= 0 else config.steps),
        recovery_success=float(recovery_time >= 0),
        pre_shock_memory_mass=pre_shock_memory_mass,
        late_memory_mass=late_memory_mass,
        memory_recovery_ratio=late_memory_mass / max(pre_shock_memory_mass, 0.01),
        topological_memory_index=memory_supported_samples / max(1, config.steps // config.persistence_delta),
        recall_event_rate=recall_events / max(1, config.steps // config.persistence_delta),
        resonance_energy_late=float(np.mean(energy_history[late_slice])),
    )
    return metrics, shock_rows, coherence_history


def run_phase5_experiments(
    config: PhaseVConfig,
    output_dir: Path,
    runs: int,
    seed_base: int,
) -> dict[str, Any]:
    scenarios = build_phase5_scenarios()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_metrics: list[PhaseVRunMetrics] = []
    shock_rows: list[dict[str, Any]] = []
    timeline_sums: dict[str, np.ndarray] = {
        scenario.name: np.zeros(config.steps, dtype=float) for scenario in scenarios
    }

    for scenario_index, scenario in enumerate(scenarios):
        for run_index in range(runs):
            seed = seed_base + scenario_index * 100 + run_index
            metrics, current_shock_rows, coherence_history = run_single_phase5(
                config=config,
                scenario=scenario,
                run_index=run_index,
                seed=seed,
            )
            run_metrics.append(metrics)
            shock_rows.extend(current_shock_rows)
            timeline_sums[scenario.name] += coherence_history

    timeline_means = {
        scenario.name: timeline_sums[scenario.name] / runs for scenario in scenarios
    }
    run_rows = [asdict(item) for item in run_metrics]
    summary_rows = summarize_runs(scenarios, run_metrics)

    save_csv(output_dir / "recovery_metrics.csv", run_rows)
    save_csv(
        output_dir / "shock_events.csv",
        shock_rows,
        fieldnames=[
            "scenario",
            "model_kind",
            "bootstrap_mode",
            "false_mode",
            "run_index",
            "seed",
            "shock_step",
            "phase_nodes",
            "geometry_nodes",
            "memory_retention",
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
            "shock_events": shock_rows,
            "summary": summary_rows,
            "coherence_timelines_mean": {
                key: value.tolist() for key, value in timeline_means.items()
            },
        },
    )

    render_grouped_metric(
        summary_rows,
        output_dir / "coherence_recovery_comparison.png",
        "coherence_recovery_ratio",
        "Coherence Recovery Ratio",
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "recovery_time_comparison.png",
        "recovery_time_steps",
        "Recovery Time After Shock",
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "cluster_recovery_comparison.png",
        "cluster_recovery_ratio",
        "Cluster Recovery Ratio",
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "memory_recovery_comparison.png",
        "memory_recovery_ratio",
        "Memory Recovery Ratio",
    )
    render_recovery_timeline(
        scenarios,
        timeline_means,
        config,
        output_dir / "shock_response_timeline.png",
    )

    print("\nPhase V summary (recovery / cluster / memory / success):")
    for row in summary_rows:
        print(
            f"- {row['scenario']}: "
            f"coh_recovery={row['coherence_recovery_ratio_mean']:.3f}, "
            f"cluster_recovery={row['cluster_recovery_ratio_mean']:.3f}, "
            f"memory_recovery={row['memory_recovery_ratio_mean']:.3f}, "
            f"success={row['recovery_success_mean']:.3f}"
        )

    return {
        "output_dir": output_dir,
        "summary_rows": summary_rows,
        "run_rows": run_rows,
        "shock_rows": shock_rows,
    }
