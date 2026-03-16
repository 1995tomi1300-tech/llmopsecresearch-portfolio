from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resonance_field.phase16_curricula import apply_curriculum
from resonance_field.phase16_config import PhaseXVIConfig
from resonance_field.phase16_experiment import run_phase16_experiments
from resonance_field.phase16_teaching_profile import build_crossfire_profile
from resonance_field.phase16_quantized_pilot import apply_variant
from resonance_field.phase6_config import PhaseVIScenario

DEFAULT_FIVE_CURRICULA = [
    "dialectic_literary",
    "mathematics_philosophy",
    "psychological_extrapolation",
    "biological_synthesizer",
    "ontology_void_finitude",
]


@dataclass
class ArenaAgent:
    agent_id: str
    position: np.ndarray
    strength: float
    anchor_scalar: float
    reference_blend: float
    curriculum: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-agent Phase XVI virtual arena.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "resonance_phase16_virtual_arena",
    )
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--snapshot-interval", type=int, default=1, help="Save image every N rounds")
    parser.add_argument("--nodes", type=int, default=120)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--interaction-window", type=int, default=2, help="Consecutive rounds where alanyok interact")
    parser.add_argument("--cooldown-window", type=int, default=1, help="Consecutive rounds of communication cooldown")
    parser.add_argument("--seed-base", type=int, default=180000)
    parser.add_argument("--backend", default="auto")
    parser.add_argument(
        "--variant",
        default="critical_push",
        choices=("subtle_balanced", "mid_balanced", "strong_balanced", "critical_push"),
    )
    parser.add_argument(
        "--false-mode",
        default="noisy",
        choices=("random", "noisy"),
    )
    parser.add_argument(
        "--strengths",
        default="",
        help="Comma-separated list of crossfire profile strengths, one per model.",
    )
    parser.add_argument(
        "--anchors",
        default="",
        help="Comma-separated list of quantized anchor scalars, one per model.",
    )
    parser.add_argument(
        "--blends",
        default="",
        help="Comma-separated list of quantized reference blends, one per model.",
    )
    parser.add_argument(
        "--curricula",
        default="",
        help="Comma-separated curriculum names, one per model.",
    )
    return parser.parse_args()



def build_scenario(false_mode: str) -> PhaseVIScenario:
    return PhaseVIScenario(
        model_kind="hybrid_manifold",
        bootstrap_mode="periodic",
        false_mode=false_mode,
        name=f"hybrid_manifold_periodic_{false_mode}_arena",
        label=f"Hybrid manifold periodic {false_mode} | virtual arena",
    )


def circle_points(count: int) -> list[np.ndarray]:
    if count <= 1:
        return [np.array([0.0, 1.0, 0.0], dtype=float)]
    points: list[np.ndarray] = []
    for index in range(count):
        theta = 2.0 * math.pi * index / count
        x = math.cos(theta)
        y = math.sin(theta)
        z = 0.0
        points.append(np.array([x, y, z], dtype=float))
    return points


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def make_agents(
    count: int,
    strengths_override: list[float] | None = None,
    anchors_override: list[float] | None = None,
    blends_override: list[float] | None = None,
    curricula_override: list[str] | None = None,
) -> list[ArenaAgent]:
    positions = circle_points(count)
    if strengths_override:
        strengths = strengths_override
    elif count == 1:
        strengths = [0.54]
    else:
        strengths = np.linspace(0.36, 0.72, count).tolist()
    anchors = anchors_override or np.linspace(0.47, 0.59, count).tolist()
    blends = blends_override or np.linspace(0.58, 0.68, count).tolist()
    if curricula_override:
        curricula = curricula_override
    elif count == 5:
        curricula = list(DEFAULT_FIVE_CURRICULA)
    else:
        curricula = ["neutral_loading"] * count
    return [
        ArenaAgent(
            agent_id=f"alany_{index+1:02d}",
            position=positions[index],
            strength=float(strengths[index]),
            anchor_scalar=float(anchors[index]),
            reference_blend=float(blends[index]),
            curriculum=str(curricula[index]),
        )
        for index in range(count)
    ]


def load_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows[0]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def interaction_weights(agents: list[ArenaAgent]) -> np.ndarray:
    count = len(agents)
    weights = np.zeros((count, count), dtype=float)
    for i, source in enumerate(agents):
        for j, target in enumerate(agents):
            if i == j:
                continue
            dist = float(np.linalg.norm(source.position - target.position))
            weights[i, j] = 1.0 / max(dist, 1e-6)
        row_sum = float(np.sum(weights[i]))
        if row_sum > 0.0:
            weights[i] /= row_sum
    return weights


def build_agent_config(
    base: PhaseXVIConfig,
    agent: ArenaAgent,
) -> PhaseXVIConfig:
    config = build_crossfire_profile(base, agent.strength)
    config = replace(
        config,
        quantized_scalar_vector_enabled=True,
        quantized_anchor_scalar=agent.anchor_scalar,
        quantized_reference_blend=agent.reference_blend,
        experiment_tag=f"Phase XVI Virtual Arena {agent.agent_id}",
    )
    return apply_curriculum(config, agent.curriculum)


def update_agents(
    agents: list[ArenaAgent],
    round_rows: list[dict[str, Any]],
) -> None:
    weights = interaction_weights(agents)
    coherences = np.array([float(row["late_recovery_coherence_mean"]) for row in round_rows], dtype=float)
    isolations = np.array(
        [float(row["final_false_isolation_rate_mean"]) for row in round_rows], dtype=float
    )
    trusts = np.array([float(row["late_trust_mean_mean"]) for row in round_rows], dtype=float)
    contamination = np.array(
        [float(row["phase_manipulation_core_contamination_rate_mean"]) for row in round_rows],
        dtype=float,
    )
    strengths = np.array([agent.strength for agent in agents], dtype=float)
    anchors = np.array([agent.anchor_scalar for agent in agents], dtype=float)

    # 300k Anchor Reference Target (Theoretical Sweet Spot)
    TARGET_STRENGTH = 0.618
    TARGET_ANCHOR = 0.500
    TARGET_BLEND = 0.618

    for index, agent in enumerate(agents):
        neighbor_strength = float(np.dot(weights[index], strengths))
        neighbor_anchor = float(np.dot(weights[index], anchors))
        neighbor_coherence = float(np.dot(weights[index], coherences))
        neighbor_isolation = float(np.dot(weights[index], isolations))
        neighbor_trust = float(np.dot(weights[index], trusts))
        neighbor_contamination = float(np.dot(weights[index], contamination))

        # Attractor pull based on instability (the more unstable, the stronger the 300k anchor pulls it)
        instability = 1.0 - clamp(neighbor_coherence, 0.0, 1.0) + clamp(neighbor_contamination, 0.0, 1.0)
        pull_factor = clamp(instability * 0.15, 0.05, 0.40)

        strength_push = (
            0.35 * neighbor_strength
            + 0.20 * clamp(neighbor_coherence, 0.0, 1.0)
            + 0.10 * clamp(neighbor_trust, 0.0, 1.0)
            - 0.25 * clamp(0.35 - neighbor_isolation, 0.0, 0.35)
            - 0.20 * clamp(neighbor_contamination - 0.20, 0.0, 0.40)
        )
        # Apply 300k anchor pull vector
        strength_target = (1.0 - pull_factor) * strength_push + pull_factor * TARGET_STRENGTH
        agent.strength = clamp(0.62 * agent.strength + 0.38 * strength_target, 0.18, 0.78)

        anchor_push = (
            0.55 * neighbor_anchor
            + 0.15 * clamp(neighbor_coherence, 0.0, 1.0)
            - 0.10 * clamp(0.30 - neighbor_isolation, 0.0, 0.30)
        )
        anchor_target = (1.0 - pull_factor) * anchor_push + pull_factor * TARGET_ANCHOR
        agent.anchor_scalar = clamp(0.68 * agent.anchor_scalar + 0.32 * anchor_target, 0.42, 0.64)

        blend_push = 0.58 + 0.12 * clamp(neighbor_trust, 0.0, 1.0) - 0.10 * clamp(
            neighbor_contamination - 0.18, 0.0, 0.32
        )
        blend_target = (1.0 - pull_factor) * blend_push + pull_factor * TARGET_BLEND
        agent.reference_blend = clamp(0.72 * agent.reference_blend + 0.28 * blend_target, 0.52, 0.72)



def render_strength_timeline(rows: list[dict[str, Any]], output_path: Path) -> None:
    agents = sorted({row["agent_id"] for row in rows})
    rounds = sorted({int(row["round_index"]) for row in rows})
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent_id in agents:
        series = [row for row in rows if row["agent_id"] == agent_id]
        series.sort(key=lambda item: int(item["round_index"]))
        ax.plot(
            [int(row["round_index"]) for row in series],
            [float(row["agent_strength"]) for row in series],
            marker="o",
            label=agent_id,
        )
    ax.set_xticks(rounds)
    ax.set_xlabel("interaction round")
    ax.set_ylabel("profile strength")
    ax.set_title("Arena Agent Strength Timeline")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_metric_timeline(rows: list[dict[str, Any]], output_path: Path) -> None:
    rounds = sorted({int(row["round_index"]) for row in rows})
    mean_coherence = []
    mean_isolation = []
    mean_trust = []
    for round_index in rounds:
        sample = [row for row in rows if int(row["round_index"]) == round_index]
        mean_coherence.append(float(np.mean([float(row["late_recovery_coherence_mean"]) for row in sample])))
        mean_isolation.append(
            float(np.mean([float(row["final_false_isolation_rate_mean"]) for row in sample]))
        )
        mean_trust.append(float(np.mean([float(row["late_trust_mean_mean"]) for row in sample])))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, mean_coherence, marker="o", label="mean coherence")
    ax.plot(rounds, mean_isolation, marker="o", label="mean false isolation")
    ax.plot(rounds, mean_trust, marker="o", label="mean trust")
    ax.set_xlabel("interaction round")
    ax.set_ylabel("metric value")
    ax.set_title("Arena Mean Metrics by Round")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_snapshot(agents: list[ArenaAgent], rows: list[dict[str, Any]], target_round: int, output_path: Path, in_interaction: bool) -> None:
    sample = {row["agent_id"]: row for row in rows if int(row["round_index"]) == target_round}
    if not sample:
        return
    fig = plt.figure(figsize=(8.4, 6.2))
    ax = fig.add_subplot(111)
    
    # Draw interactions
    if in_interaction:
        count = len(agents)
        for i in range(count):
            for j in range(i + 1, count):
                dist = np.linalg.norm(agents[i].position - agents[j].position)
                if dist > 0:
                    weight = 1.0 / dist
                    ax.plot(
                        [agents[i].position[0], agents[j].position[0]],
                        [agents[i].position[1], agents[j].position[1]],
                        color="gray",
                        alpha=min(1.0, float(weight) * 0.3),
                        linestyle="--",
                        zorder=1
                    )
            
    for agent in agents:
        if agent.agent_id not in sample:
            continue
        row = sample[agent.agent_id]
        coherence = float(row["late_recovery_coherence_mean"])
        size = 80 + 220 * float(row["final_false_isolation_rate_mean"])
        ax.scatter(
            agent.position[0],
            agent.position[1],
            s=size,
            c=[[coherence, 0.2, max(0.0, 1.0 - coherence)]],
            zorder=3
        )
        ax.text(
            agent.position[0],
            agent.position[1],
            agent.agent_id,
            fontsize=8,
            zorder=4
        )
    status_text = "Interaction Active" if in_interaction else "Cooldown (No Comm)"
    ax.set_title(f"2D Virtual Arena Snapshot - Round {target_round} [{status_text}]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    strengths_override = parse_float_list(args.strengths)
    anchors_override = parse_float_list(args.anchors)
    blends_override = parse_float_list(args.blends)
    curricula_override = [item.strip() for item in args.curricula.split(",") if item.strip()]
    model_count = args.agents
    if strengths_override:
        model_count = len(strengths_override)
    for name, values in (
        ("anchors", anchors_override),
        ("blends", blends_override),
    ):
        if values and len(values) != model_count:
            raise ValueError(
                f"{name} count ({len(values)}) must match model count ({model_count})."
            )
    if curricula_override:
        if len(curricula_override) > model_count:
            raise ValueError(
                f"curricula count ({len(curricula_override)}) cannot exceed model count ({model_count})."
            )
        curricula_override = curricula_override + ["neutral_loading"] * (
            model_count - len(curricula_override)
        )

    base = PhaseXVIConfig(
        nodes=args.nodes,
        steps=args.steps,
        shock_step=max(24, args.steps // 4),
        late_window=max(40, args.steps // 4),
        benchmark_repeats=2,
        compute_backend=args.backend,
        collect_event_rows=True,
    )
    base = apply_variant(base, args.variant)
    scenario = build_scenario(args.false_mode)

    agents = make_agents(
        model_count,
        strengths_override=strengths_override or None,
        anchors_override=anchors_override or None,
        blends_override=blends_override or None,
        curricula_override=curricula_override or None,
    )
    all_rows: list[dict[str, Any]] = []

    round_index = 0
    while True:
        if args.rounds > 0 and round_index >= args.rounds:
            break
        round_rows: list[dict[str, Any]] = []
        round_dir = args.output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        for agent_index, agent in enumerate(agents):
            config = build_agent_config(base, agent)
            agent_dir = round_dir / agent.agent_id
            run_phase16_experiments(
                config=config,
                output_dir=agent_dir,
                runs=1,
                seed_base=args.seed_base + round_index * 10000 + agent_index * 1000,
                scenarios=[scenario],
            )
            row = load_csv_row(agent_dir / "scenario_summary.csv")
            enriched = {
                **row,
                "round_index": round_index,
                "agent_id": agent.agent_id,
                "agent_strength": agent.strength,
                "agent_anchor_scalar": agent.anchor_scalar,
                "agent_reference_blend": agent.reference_blend,
                "agent_curriculum": agent.curriculum,
                "agent_x": float(agent.position[0]),
                "agent_y": float(agent.position[1]),
                "agent_z": float(agent.position[2]),
                "output_dir": str(agent_dir),
            }
            round_rows.append(enriched)
            all_rows.append(enriched)
        
        cycle_length = args.interaction_window + args.cooldown_window
        in_interaction = (round_index % cycle_length) < args.interaction_window
        
        if (round_index % args.snapshot_interval) == 0:
            render_snapshot(agents, all_rows, round_index, round_dir / f"arena_snapshot_round_{round_index:02d}.png", in_interaction)
        
        if in_interaction:
            update_agents(agents, round_rows)

        summary_path = args.output_dir / "arena_summary.csv"
        file_exists = summary_path.exists()
        with summary_path.open("a" if file_exists else "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(round_rows[0].keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerows(round_rows)
            
        render_strength_timeline(all_rows, args.output_dir / "arena_strength_timeline.png")
        render_metric_timeline(all_rows, args.output_dir / "arena_metric_timeline.png")
        
        print(f"Round {round_index} completed. Target Field Pull Active: {in_interaction}")
        round_index += 1


    manifest = {
        "scenario": asdict(scenario),
        "agents": [
            {
                "agent_id": agent.agent_id,
                "position": agent.position.tolist(),
                "final_strength": agent.strength,
                "final_anchor_scalar": agent.anchor_scalar,
                "final_reference_blend": agent.reference_blend,
                "curriculum": agent.curriculum,
            }
            for agent in agents
        ],
        "rounds": args.rounds,
        "rows": all_rows,
    }
    with (args.output_dir / "arena_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    render_strength_timeline(all_rows, args.output_dir / "arena_strength_timeline.png")
    render_metric_timeline(all_rows, args.output_dir / "arena_metric_timeline.png")
    # Per-round snapshots are generated during the loop

    try:
        from nvda_virtualis_ter_lidar.synthetic_lidar_bridge import SpatialScenario, write_bridge_payload
        scenarios = []
        for agent in agents:
            topology = "sphere" if agent.strength > 0.5 else "helix"
            pharmacophore_bias = (agent.reference_blend - 0.5) * 0.2
            shape_bias = (agent.anchor_scalar - 0.5) * 0.2
            scenario = SpatialScenario(
                name=f"arena_{agent.agent_id}_{agent.curriculum}",
                topology=topology,
                point_count=1200 + int(agent.strength * 800),
                seed=abs(hash(agent.agent_id)) % 100000,
                radius=1.0 + agent.anchor_scalar,
                turns=1.0 + (agent.strength * 2.0),
                intensity_noise=0.05 + max(0, 0.3 - agent.reference_blend),
                occlusion_rate=0.05 + max(0, 1.0 - agent.strength) * 0.2,
                fragmentation=0.05 + max(0, 0.8 - agent.anchor_scalar) * 0.2,
                shell_jitter=0.04 + max(0, 1.0 - agent.reference_blend) * 0.1,
                vertical_jitter=0.03 + max(0, 1.0 - agent.strength) * 0.1,
                descriptor_bias=agent.strength * 0.2,
                shape_bias=shape_bias,
                pharmacophore_bias=pharmacophore_bias,
            )
            scenarios.append(scenario)
        
        bridge_output = ROOT / "nvda_virtualis_ter_lidar" / "outputs" / "arena_bridge_payload.json"
        write_bridge_payload(bridge_output, scenarios)
        print(f"Exported 5 agents to NVDA Virtual Space LiDAR bridge: {bridge_output}")
    except Exception as e:
        print(f"Failed to export to NVDA Virtual Space: {e}")


if __name__ == "__main__":
    main()
