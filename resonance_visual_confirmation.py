from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
import numpy as np


DEFAULT_SUMMARY = Path("/mnt/d/resonance_phase6_outputs/scenario_summary.json")
DEFAULT_OUTPUT_DIR = Path("/mnt/d/resonance_phase6_outputs/visual_confirmation")


@dataclass(frozen=True)
class ScenarioBundle:
    scenario: str
    summary_row: dict[str, Any]
    coherence_timeline: np.ndarray
    shock_step: int
    guardian_steps: list[int]
    nodes: int
    steps: int


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def choose_default_scenario(summary_rows: list[dict[str, Any]]) -> str:
    def score(row: dict[str, Any]) -> float:
        return (
            _to_float(row.get("coherence_recovery_ratio_mean"))
            + _to_float(row.get("cluster_recovery_ratio_mean"))
            + _to_float(row.get("memory_recovery_ratio_mean"))
            + 0.5 * _to_float(row.get("guardian_event_rate_mean"))
        )

    best = max(summary_rows, key=score)
    return str(best["scenario"])


def load_bundle(summary_path: Path, scenario_name: str | None) -> ScenarioBundle:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_rows = payload["summary"]
    chosen = scenario_name or choose_default_scenario(summary_rows)
    summary_row = next(row for row in summary_rows if row["scenario"] == chosen)
    coherence = np.asarray(payload["coherence_timelines_mean"][chosen], dtype=float)

    shock_rows = [row for row in payload.get("shock_events", []) if row["scenario"] == chosen]
    if shock_rows:
        shock_step = int(round(np.mean([_to_float(row["shock_step"]) for row in shock_rows])))
    else:
        shock_step = len(coherence) // 2

    guardian_rows = [row for row in payload.get("guardian_events", []) if row["scenario"] == chosen]
    guardian_steps = sorted({int(_to_float(row["step"])) for row in guardian_rows})

    config = payload["config"]
    return ScenarioBundle(
        scenario=chosen,
        summary_row=summary_row,
        coherence_timeline=coherence,
        shock_step=shock_step,
        guardian_steps=guardian_steps,
        nodes=int(config["nodes"]),
        steps=int(config["steps"]),
    )


def smoothstep(value: np.ndarray) -> np.ndarray:
    return value * value * (3.0 - 2.0 * value)


def make_recovery_envelope(
    steps: int,
    shock_step: int,
    pre_value: float,
    shock_value: float,
    late_value: float,
) -> np.ndarray:
    series = np.empty(steps, dtype=float)
    pre_end = max(1, shock_step)
    post_end = max(pre_end + 1, steps)
    pre_curve = np.linspace(0.0, 1.0, pre_end, endpoint=False)
    pre_curve = smoothstep(pre_curve)
    series[:pre_end] = pre_value * (0.92 + 0.08 * pre_curve)

    drop_span = max(6, int(0.06 * steps))
    drop_end = min(steps, shock_step + drop_span)
    if drop_end > pre_end:
        drop_curve = np.linspace(0.0, 1.0, drop_end - pre_end)
        drop_curve = smoothstep(drop_curve)
        series[pre_end:drop_end] = pre_value + (shock_value - pre_value) * drop_curve

    recover_start = drop_end
    if recover_start < post_end:
        recover_curve = np.linspace(0.0, 1.0, post_end - recover_start)
        recover_curve = smoothstep(recover_curve)
        series[recover_start:post_end] = shock_value + (late_value - shock_value) * recover_curve

    if drop_end < steps and recover_start >= steps:
        series[drop_end:] = shock_value
    return series


def guardian_pulse_series(steps: int, guardian_steps: list[int]) -> np.ndarray:
    signal = np.zeros(steps, dtype=float)
    if not guardian_steps:
        return signal
    indices = np.arange(steps, dtype=float)
    for step in guardian_steps:
        signal += np.exp(-((indices - float(step)) / 10.0) ** 2)
    if np.max(signal) > 0:
        signal /= np.max(signal)
    return signal


def deterministic_jitter(count: int, frame_value: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(count, dtype=float)
    x = np.sin(idx * 0.37 + frame_value * 0.11) + 0.5 * np.cos(idx * 0.13 - frame_value * 0.07)
    y = np.cos(idx * 0.29 - frame_value * 0.09) + 0.5 * np.sin(idx * 0.17 + frame_value * 0.05)
    return x, y


def build_manifold_points(
    node_count: int,
    frame_ratio: float,
    coherence_norm: float,
    cluster_level: float,
    memory_level: float,
    guardian_level: float,
    shock_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    turns = 4.0
    t = np.linspace(0.0, turns * 2.0 * np.pi, node_count, endpoint=False)
    compression = np.clip(
        0.18 + 0.55 * coherence_norm + 0.18 * cluster_level + 0.12 * memory_level - 0.28 * shock_level,
        0.02,
        1.0,
    )
    radius = 1.15 - 0.72 * compression
    pitch = 0.22 + 0.56 * (1.0 - compression) + 0.08 * shock_level
    jitter_strength = 0.18 * (1.0 - coherence_norm) + 0.15 * shock_level - 0.05 * guardian_level
    jitter_strength = max(0.01, jitter_strength)

    base_x = radius * np.cos(t + frame_ratio * 1.1)
    base_y = pitch * (t / (2.0 * np.pi) - turns / 2.0) + 0.20 * radius * np.sin(t * 0.5)
    jitter_x, jitter_y = deterministic_jitter(node_count, frame_ratio * 100.0)
    x = base_x + jitter_strength * jitter_x
    y = base_y + jitter_strength * jitter_y

    phase_color = 0.5 + 0.5 * np.cos(t + frame_ratio * 2.2 + guardian_level)
    color_mix = np.clip(0.45 * phase_color + 0.35 * coherence_norm + 0.20 * memory_level, 0.0, 1.0)
    return x, y, color_mix


def line_pairs(node_count: int, density: float) -> list[tuple[int, int]]:
    stride = max(3, int(round(9 - 5 * density)))
    pairs = []
    for index in range(0, node_count - stride, stride):
        pairs.append((index, index + stride))
    return pairs


def build_frames(bundle: ScenarioBundle) -> dict[str, np.ndarray]:
    coherence = bundle.coherence_timeline
    steps = len(coherence)
    coherence_norm = (coherence - float(np.min(coherence))) / max(1e-9, float(np.max(coherence) - np.min(coherence)))

    summary = bundle.summary_row
    cluster_series = make_recovery_envelope(
        steps,
        bundle.shock_step,
        _to_float(summary["pre_shock_cluster_fraction_mean"]),
        _to_float(summary["shock_min_cluster_fraction_mean"]),
        _to_float(summary["late_cluster_fraction_mean"]),
    )
    memory_series = make_recovery_envelope(
        steps,
        bundle.shock_step,
        _to_float(summary["pre_shock_memory_mass_mean"]),
        min(
            _to_float(summary["pre_shock_memory_mass_mean"]),
            _to_float(summary["late_memory_mass_mean"]) * 0.35,
        ),
        _to_float(summary["late_memory_mass_mean"]),
    )
    max_memory = max(1e-9, float(np.max(memory_series)))
    memory_norm = np.clip(memory_series / max_memory, 0.0, 1.0)
    guardian = guardian_pulse_series(steps, bundle.guardian_steps)

    indices = np.arange(steps, dtype=float)
    shock_wave = np.exp(-((indices - float(bundle.shock_step)) / max(4.0, 0.04 * steps)) ** 2)

    return {
        "step": indices,
        "coherence": coherence,
        "coherence_norm": coherence_norm,
        "cluster": cluster_series,
        "cluster_norm": np.clip(cluster_series / max(1e-9, float(np.max(cluster_series))), 0.0, 1.0),
        "memory_norm": memory_norm,
        "guardian": guardian,
        "shock_wave": shock_wave,
    }


def render_visual_confirmation(
    bundle: ScenarioBundle,
    output_gif: Path,
    output_png: Path,
    output_manifest: Path,
    target_nodes: int,
    fps: int,
) -> dict[str, Any]:
    frames = build_frames(bundle)
    steps = bundle.steps
    sample_steps = np.linspace(0, steps - 1, min(150, steps), dtype=int)
    frame_count = len(sample_steps)

    fig = plt.figure(figsize=(14, 10), facecolor="#081018")
    grid = GridSpec(2, 2, height_ratios=[2.3, 1.0], width_ratios=[1.0, 1.0], hspace=0.18, wspace=0.12)
    ax_obs = fig.add_subplot(grid[0, 0])
    ax_ext = fig.add_subplot(grid[0, 1])
    ax_timeline = fig.add_subplot(grid[1, 0])
    ax_state = fig.add_subplot(grid[1, 1])

    for axis in (ax_obs, ax_ext, ax_timeline, ax_state):
        axis.set_facecolor("#0d1621")

    ax_obs.set_title(f"Observed manifold ({bundle.nodes} nodes)", color="white", fontsize=13)
    ax_ext.set_title(f"Scaled visual extrapolation ({target_nodes} nodes)", color="white", fontsize=13)
    for axis in (ax_obs, ax_ext):
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlim(-1.5, 1.5)
        axis.set_ylim(-2.2, 2.2)
        for spine in axis.spines.values():
            spine.set_color("#304050")

    time_axis = np.arange(steps)
    ax_timeline.plot(time_axis, frames["coherence"], color="#77d7ff", linewidth=2.0, label="coherence")
    ax_timeline.plot(
        time_axis,
        0.15 + 0.75 * frames["cluster_norm"],
        color="#9ae66e",
        linewidth=1.4,
        alpha=0.85,
        label="cluster envelope",
    )
    ax_timeline.plot(
        time_axis,
        0.10 + 0.70 * frames["memory_norm"],
        color="#ffd166",
        linewidth=1.2,
        alpha=0.9,
        label="memory envelope",
    )
    ax_timeline.axvline(bundle.shock_step, color="#ff5d73", linestyle="--", linewidth=1.4, label="shock")
    for guardian_step in bundle.guardian_steps:
        ax_timeline.axvline(guardian_step, color="#64ffb0", linewidth=0.5, alpha=0.22)
    cursor = ax_timeline.axvline(0, color="white", linewidth=1.0)
    ax_timeline.set_title("Recovered collapse trace", color="white", fontsize=12)
    ax_timeline.tick_params(colors="#c7d2de")
    for spine in ax_timeline.spines.values():
        spine.set_color("#304050")
    ax_timeline.grid(alpha=0.18)
    ax_timeline.legend(loc="upper left", fontsize=8, facecolor="#0d1621", edgecolor="#304050", labelcolor="white")

    state_labels = ["coherence", "compression", "crystallization", "guardian"]
    state_colors = ["#77d7ff", "#d4ff72", "#ffd166", "#64ffb0"]
    bars = ax_state.barh(state_labels, [0, 0, 0, 0], color=state_colors, alpha=0.88)
    ax_state.set_xlim(0, 1)
    ax_state.set_title("State reconstruction", color="white", fontsize=12)
    ax_state.tick_params(colors="#c7d2de")
    for spine in ax_state.spines.values():
        spine.set_color("#304050")
    ax_state.grid(axis="x", alpha=0.18)
    state_text = ax_state.text(
        0.02,
        -0.75,
        "",
        color="white",
        fontsize=10,
        va="top",
        ha="left",
        transform=ax_state.transData,
    )
    fig.text(
        0.5,
        0.02,
        "Visual reconstruction inferred from measured collapse/recovery summaries, not raw node trace.",
        color="#b2c0cf",
        ha="center",
        fontsize=9,
    )

    obs_pairs = line_pairs(bundle.nodes, 0.6)
    ext_pairs = line_pairs(target_nodes, 0.72)

    def draw_manifold(axis, node_count: int, pairs: list[tuple[int, int]], step_index: int) -> None:
        axis.cla()
        axis.set_facecolor("#0d1621")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlim(-1.5, 1.5)
        axis.set_ylim(-2.2, 2.2)
        for spine in axis.spines.values():
            spine.set_color("#304050")

        coherence_norm = float(frames["coherence_norm"][step_index])
        cluster_norm = float(frames["cluster_norm"][step_index])
        memory_norm = float(frames["memory_norm"][step_index])
        guardian_norm = float(frames["guardian"][step_index])
        shock_norm = float(frames["shock_wave"][step_index])
        x, y, color_mix = build_manifold_points(
            node_count=node_count,
            frame_ratio=step_index / max(1, steps - 1),
            coherence_norm=coherence_norm,
            cluster_level=cluster_norm,
            memory_level=memory_norm,
            guardian_level=guardian_norm,
            shock_level=shock_norm,
        )
        crystallization = np.clip(0.30 * coherence_norm + 0.35 * cluster_norm + 0.25 * memory_norm + 0.10 * guardian_norm, 0.0, 1.0)
        line_alpha = 0.08 + 0.35 * crystallization
        for i, j in pairs:
            axis.plot(
                [x[i], x[j]],
                [y[i], y[j]],
                color=(0.48, 0.88, 1.0, line_alpha),
                linewidth=0.55 + 0.5 * crystallization,
            )
        axis.scatter(
            x,
            y,
            c=color_mix,
            cmap="viridis",
            s=max(6, 2400 / node_count),
            alpha=0.92,
            edgecolors="none",
        )
        if shock_norm > 0.08:
            axis.add_patch(
                plt.Circle((0.0, 0.0), 1.25 + 0.35 * shock_norm, color="#ff5d73", fill=False, linewidth=2.0, alpha=0.18 + 0.25 * shock_norm)
            )
        if guardian_norm > 0.08:
            axis.add_patch(
                plt.Circle((0.0, 0.0), 0.45 + 0.25 * guardian_norm, color="#64ffb0", fill=False, linewidth=1.8, alpha=0.18 + 0.32 * guardian_norm)
            )

    def update(frame_idx: int):
        step_index = int(sample_steps[frame_idx])
        draw_manifold(ax_obs, bundle.nodes, obs_pairs, step_index)
        draw_manifold(ax_ext, target_nodes, ext_pairs, step_index)
        ax_obs.set_title(f"Observed manifold ({bundle.nodes} nodes)", color="white", fontsize=13)
        ax_ext.set_title(f"Scaled visual extrapolation ({target_nodes} nodes)", color="white", fontsize=13)

        cursor.set_xdata([step_index, step_index])
        coherence_now = float(frames["coherence_norm"][step_index])
        cluster_now = float(frames["cluster_norm"][step_index])
        memory_now = float(frames["memory_norm"][step_index])
        guardian_now = float(frames["guardian"][step_index])
        shock_now = float(frames["shock_wave"][step_index])
        compression_now = np.clip(0.18 + 0.55 * coherence_now + 0.18 * cluster_now + 0.12 * memory_now - 0.28 * shock_now, 0.0, 1.0)
        crystallization_now = np.clip(0.30 * coherence_now + 0.35 * cluster_now + 0.25 * memory_now + 0.10 * guardian_now, 0.0, 1.0)
        values = [coherence_now, compression_now, crystallization_now, guardian_now]
        for bar, value in zip(bars, values):
            bar.set_width(value)

        state_text.set_text(
            "\n".join(
                [
                    f"scenario: {bundle.scenario}",
                    f"step: {step_index}/{steps - 1}",
                    f"shock step: {bundle.shock_step}",
                    f"guardian pulses: {len(bundle.guardian_steps)}",
                    f"shock wave: {shock_now:.2f}",
                ]
            )
        )
        return [cursor, *bars, state_text]

    anim = animation.FuncAnimation(fig, update, frames=frame_count, interval=1000 / max(1, fps), blit=False)
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_gif, writer=writer)
    update(frame_count - 1)
    fig.savefig(output_png, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    manifest = {
        "source_summary": str(DEFAULT_SUMMARY),
        "scenario": bundle.scenario,
        "nodes_observed": bundle.nodes,
        "nodes_extrapolated": target_nodes,
        "shock_step": bundle.shock_step,
        "guardian_steps": bundle.guardian_steps,
        "frames": frame_count,
        "fps": fps,
        "gif_path": str(output_gif),
        "png_path": str(output_png),
        "notes": "Data-driven visual reconstruction from phase summary metrics and coherence timeline.",
    }
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a data-driven visual confirmation from SPIRAL collapse/recovery summaries."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Path to scenario_summary.json")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario name. Defaults to the strongest recovery trace.")
    parser.add_argument("--target-nodes", type=int, default=512, help="Visual extrapolation node count.")
    parser.add_argument("--fps", type=int, default=18, help="GIF frame rate.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.summary, args.scenario)
    stem = bundle.scenario
    output_gif = args.output_dir / f"{stem}_visual_confirmation.gif"
    output_png = args.output_dir / f"{stem}_visual_confirmation.png"
    output_manifest = args.output_dir / f"{stem}_visual_confirmation.json"
    manifest = render_visual_confirmation(
        bundle=bundle,
        output_gif=output_gif,
        output_png=output_png,
        output_manifest=output_manifest,
        target_nodes=args.target_nodes,
        fps=args.fps,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
