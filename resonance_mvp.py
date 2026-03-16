from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    nodes: int = 120
    steps: int = 480
    dt: float = 0.03
    radius: float = 0.16
    epsilon: float = 0.05
    phase_coupling: float = 0.48
    spatial_coupling: float = 0.012
    drag: float = 0.92
    speed_cap: float = 0.04
    false_ratio: float = 0.18
    bootstrap_strength: float = 0.9
    periodic_strength: float = 0.24
    bootstrap_period: int = 90
    bootstrap_anchor: float = 0.0
    noisy_false_sigma: float = 0.12
    late_window: int = 80


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    bootstrap_mode: str
    false_mode: str


@dataclass
class RunMetrics:
    scenario: str
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


def build_scenarios() -> list[Scenario]:
    return [
        Scenario("clean_single", "Clean + Single Pulse", "single", "none"),
        Scenario("clean_periodic", "Clean + Periodic Pulse", "periodic", "none"),
        Scenario("random_false_single", "Stable False + Single Pulse", "single", "random"),
        Scenario("random_false_periodic", "Stable False + Periodic Pulse", "periodic", "random"),
        Scenario("noisy_false_single", "Noisy False + Single Pulse", "single", "noisy"),
        Scenario("noisy_false_periodic", "Noisy False + Periodic Pulse", "periodic", "noisy"),
    ]


def build_false_frequencies(rng: np.random.Generator, count: int) -> np.ndarray:
    side = rng.random(count) < 0.5
    low_band = rng.uniform(0.62, 0.82, size=count)
    high_band = rng.uniform(1.18, 1.38, size=count)
    return np.where(side, low_band, high_band)


def bootstrap_kick(theta: np.ndarray, anchor: float, strength: float) -> np.ndarray:
    return theta + strength * np.sin(anchor - theta)


def toroidal_delta(pos: np.ndarray) -> np.ndarray:
    delta = pos[None, :, :] - pos[:, None, :]
    return (delta + 0.5) % 1.0 - 0.5


def capped_velocity(vel: np.ndarray, speed_cap: float) -> np.ndarray:
    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    scale = np.minimum(1.0, speed_cap / np.maximum(speed, 1e-9))
    return vel * scale


def connected_components(adjacency: np.ndarray) -> list[int]:
    node_count = adjacency.shape[0]
    seen = np.zeros(node_count, dtype=bool)
    sizes: list[int] = []

    for start in range(node_count):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = np.flatnonzero(adjacency[node] & ~seen)
            if neighbors.size:
                seen[neighbors] = True
                stack.extend(neighbors.tolist())
        sizes.append(size)
    return sizes


def compute_metrics(
    scenario: Scenario,
    run_index: int,
    seed: int,
    coherence_history: list[float],
    adjacency: np.ndarray,
    false_mask: np.ndarray,
    late_window: int,
) -> RunMetrics:
    degrees = adjacency.sum(axis=1).astype(float)
    component_sizes = connected_components(adjacency)
    cluster_sizes = [size for size in component_sizes if size >= 3]
    largest_cluster = max(component_sizes) if component_sizes else 0
    cluster_membership = sum(cluster_sizes)

    if false_mask.any():
        false_degrees = degrees[false_mask]
        false_isolation_rate = float(np.mean(false_degrees == 0))
        false_mean_degree = float(np.mean(false_degrees))
    else:
        false_isolation_rate = 0.0
        false_mean_degree = 0.0

    normal_mask = ~false_mask
    normal_mean_degree = float(np.mean(degrees[normal_mask])) if normal_mask.any() else 0.0
    late_values = coherence_history[-min(len(coherence_history), late_window) :]

    return RunMetrics(
        scenario=scenario.name,
        run_index=run_index,
        seed=seed,
        final_coherence=float(coherence_history[-1]),
        late_coherence=float(np.mean(late_values)),
        largest_cluster_fraction=largest_cluster / adjacency.shape[0],
        cluster_membership_fraction=cluster_membership / adjacency.shape[0],
        cluster_count=len(cluster_sizes),
        mean_degree=float(np.mean(degrees)),
        false_isolation_rate=false_isolation_rate,
        false_mean_degree=false_mean_degree,
        normal_mean_degree=normal_mean_degree,
    )


def simulate(
    config: ModelConfig,
    scenario: Scenario,
    run_index: int,
    seed: int,
) -> tuple[RunMetrics, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    node_count = config.nodes
    false_count = int(round(node_count * config.false_ratio)) if scenario.false_mode != "none" else 0

    pos = rng.random((node_count, 2))
    vel = np.zeros((node_count, 2), dtype=float)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=node_count)
    freq = rng.uniform(0.95, 1.05, size=node_count)
    false_mask = np.zeros(node_count, dtype=bool)

    if false_count:
        false_indices = rng.choice(node_count, size=false_count, replace=False)
        false_mask[false_indices] = True
        freq[false_mask] = build_false_frequencies(rng, false_count)

    if scenario.bootstrap_mode in {"single", "periodic"}:
        theta = bootstrap_kick(theta, config.bootstrap_anchor, config.bootstrap_strength)

    coherence_history: list[float] = []
    adjacency = np.zeros((node_count, node_count), dtype=bool)

    for step in range(config.steps):
        if (
            scenario.bootstrap_mode == "periodic"
            and step > 0
            and step % config.bootstrap_period == 0
        ):
            theta = bootstrap_kick(theta, config.bootstrap_anchor, config.periodic_strength)

        if scenario.false_mode == "noisy" and false_mask.any():
            freq[false_mask] += rng.normal(
                loc=0.0,
                scale=config.noisy_false_sigma * np.sqrt(config.dt),
                size=false_mask.sum(),
            )
            freq[false_mask] = np.clip(freq[false_mask], 0.5, 1.5)

        delta = toroidal_delta(pos)
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, np.inf)

        freq_gap = np.abs(freq[:, None] - freq[None, :])
        adjacency = (dist < config.radius) & (freq_gap < config.epsilon)

        phase_diff = theta[None, :] - theta[:, None]
        dtheta = config.phase_coupling * np.sum(np.sin(phase_diff) * adjacency, axis=1)
        theta = (theta + (freq + dtheta) * config.dt) % (2.0 * np.pi)

        unit_delta = delta / (dist[..., None] + 1e-6)
        attraction = np.cos(phase_diff) * adjacency
        force = config.spatial_coupling * np.sum(attraction[:, :, None] * unit_delta, axis=1)

        vel = config.drag * vel + force * config.dt
        vel = capped_velocity(vel, config.speed_cap)
        pos = (pos + vel * config.dt) % 1.0

        coherence = np.abs(np.mean(np.exp(1j * theta)))
        coherence_history.append(float(coherence))

    metrics = compute_metrics(
        scenario=scenario,
        run_index=run_index,
        seed=seed,
        coherence_history=coherence_history,
        adjacency=adjacency,
        false_mask=false_mask,
        late_window=config.late_window,
    )
    snapshot = {
        "pos": pos,
        "theta": theta,
        "false_mask": false_mask,
    }
    return metrics, snapshot


def summarize_runs(metrics: Iterable[RunMetrics]) -> list[dict[str, float | str]]:
    metric_rows = [asdict(item) for item in metrics]
    grouped: dict[str, list[dict[str, float | str]]] = {}
    for row in metric_rows:
        grouped.setdefault(str(row["scenario"]), []).append(row)

    summary: list[dict[str, float | str]] = []
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
    ]

    for scenario_name, rows in grouped.items():
        item: dict[str, float | str] = {"scenario": scenario_name, "runs": len(rows)}
        for field in fields:
            values = np.array([float(row[field]) for row in rows], dtype=float)
            item[f"{field}_mean"] = float(np.mean(values))
            item[f"{field}_std"] = float(np.std(values))
        summary.append(item)

    summary.sort(key=lambda row: str(row["scenario"]))
    return summary


def save_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def render_summary_chart(
    summary_rows: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    labels = [str(row["scenario"]) for row in summary_rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [
        ("late_coherence_mean", "late_coherence_std", "Late Coherence"),
        ("largest_cluster_fraction_mean", "largest_cluster_fraction_std", "Largest Cluster Fraction"),
        ("mean_degree_mean", "mean_degree_std", "Mean Degree"),
        ("false_isolation_rate_mean", "false_isolation_rate_std", "False Isolation Rate"),
    ]

    for axis, (mean_key, std_key, title) in zip(axes.flat, panels):
        means = [float(row[mean_key]) for row in summary_rows]
        stds = [float(row[std_key]) for row in summary_rows]
        axis.bar(x, means, yerr=stds, color="#2d6a4f", alpha=0.85, capsize=4)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(bottom=0.0)

    fig.suptitle("Resonance MVP Summary Across Scenarios", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_scenario_snapshots(
    scenarios: list[Scenario],
    scenario_samples: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    cols = 2
    rows = int(np.ceil(len(scenarios) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes_array = np.atleast_1d(axes).reshape(rows, cols)

    for axis in axes_array.flat:
        axis.set_visible(False)

    for axis, scenario in zip(axes_array.flat, scenarios):
        sample = scenario_samples[scenario.name]
        pos = sample["pos"]
        theta = sample["theta"]
        false_mask = sample["false_mask"]

        axis.set_visible(True)
        normal_mask = ~false_mask
        axis.scatter(
            pos[normal_mask, 0],
            pos[normal_mask, 1],
            c=theta[normal_mask],
            s=28,
            cmap="twilight",
            alpha=0.85,
        )
        if false_mask.any():
            axis.scatter(
                pos[false_mask, 0],
                pos[false_mask, 1],
                c="#111111",
                s=36,
                marker="x",
                linewidths=1.2,
                label="false",
            )
            axis.legend(loc="upper right")
        axis.set_title(scenario.label)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_aspect("equal")

    fig.suptitle("Sample Final States by Scenario", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_console_summary(summary_rows: list[dict[str, float | str]]) -> None:
    print("\nScenario summary (mean +/- std across runs):")
    for row in summary_rows:
        print(
            f"- {row['scenario']}: "
            f"coherence={row['late_coherence_mean']:.3f} +/- {row['late_coherence_std']:.3f}, "
            f"largest_cluster={row['largest_cluster_fraction_mean']:.3f} +/- {row['largest_cluster_fraction_std']:.3f}, "
            f"false_isolation={row['false_isolation_rate_mean']:.3f} +/- {row['false_isolation_rate_std']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the resonance-field MVP across multiple scenarios."
    )
    parser.add_argument("--runs", type=int, default=10, help="Runs per scenario.")
    parser.add_argument("--nodes", type=int, default=ModelConfig.nodes, help="Node count.")
    parser.add_argument("--steps", type=int, default=ModelConfig.steps, help="Simulation steps.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resonance_outputs"),
        help="Directory for CSV, JSON, and plot outputs.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=20260311,
        help="Base seed used to derive run seeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ModelConfig(nodes=args.nodes, steps=args.steps)
    scenarios = build_scenarios()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[RunMetrics] = []
    scenario_samples: dict[str, dict[str, object]] = {}

    for scenario_index, scenario in enumerate(scenarios):
        for run_index in range(args.runs):
            seed = args.seed_base + scenario_index * 100 + run_index
            metrics, snapshot = simulate(config, scenario, run_index, seed)
            all_metrics.append(metrics)

            current = scenario_samples.get(scenario.name)
            if current is None or metrics.late_coherence > float(current["score"]):
                scenario_samples[scenario.name] = {
                    "pos": snapshot["pos"],
                    "theta": snapshot["theta"],
                    "false_mask": snapshot["false_mask"],
                    "score": float(metrics.late_coherence),
                }

    run_rows = [asdict(item) for item in all_metrics]
    summary_rows = summarize_runs(all_metrics)

    save_csv(output_dir / "run_metrics.csv", run_rows)
    save_csv(output_dir / "scenario_summary.csv", summary_rows)
    save_json(
        output_dir / "scenario_summary.json",
        {
            "config": asdict(config),
            "scenarios": [asdict(item) for item in scenarios],
            "runs": run_rows,
            "summary": summary_rows,
        },
    )

    render_summary_chart(summary_rows, output_dir / "summary_metrics.png")
    render_scenario_snapshots(scenarios, scenario_samples, output_dir / "scenario_snapshots.png")
    print_console_summary(summary_rows)
    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
