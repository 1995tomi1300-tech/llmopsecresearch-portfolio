from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
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
    persistence_stride: int = 12
    cluster_threshold: int = 3
    helix_radius: float = 0.33
    helix_pitch: float = 0.02
    helix_turns: float = 4.0
    channel_turn_threshold: float = 0.4


@dataclass(frozen=True)
class Scenario:
    model_kind: str
    bootstrap_mode: str
    false_mode: str
    name: str
    label: str


MODEL_LABELS = {
    "baseline": "Baseline Graph",
    "triangle": "Triangle Simplicial",
    "helix": "Helix Field",
    "hybrid": "Hybrid Helix+Triangle",
}


def build_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    for model_kind in ("baseline", "triangle", "helix", "hybrid"):
        for bootstrap_mode in ("single", "periodic"):
            for false_mode in ("none", "random", "noisy"):
                name = f"{model_kind}_{bootstrap_mode}_{false_mode}"
                label = (
                    f"{MODEL_LABELS[model_kind]} | "
                    f"{bootstrap_mode} pulse | {false_mode} false"
                )
                scenarios.append(
                    Scenario(
                        model_kind=model_kind,
                        bootstrap_mode=bootstrap_mode,
                        false_mode=false_mode,
                        name=name,
                        label=label,
                    )
                )
    return scenarios
