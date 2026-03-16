from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseIVConfig:
    nodes: int = 120
    steps: int = 600
    dt: float = 0.03
    radius: float = 0.16
    epsilon: float = 0.05
    phase_coupling: float = 0.48
    spatial_coupling: float = 0.012
    drag: float = 0.92
    speed_cap: float = 0.04
    false_ratio: float = 0.18
    bootstrap_strength: float = 0.25
    bootstrap_period: int = 120
    bootstrap_anchor: float = 0.0
    late_window: int = 100
    cluster_threshold: int = 3
    persistence_delta: int = 50
    helix_radius: float = 0.33
    helix_pitch: float = 0.25
    helix_turns: float = 4.0
    triangle_reinforce: float = 0.85
    triangle_weaken: float = 0.25
    activation_threshold: float = 0.5
    false_frequency_drift_sigma: float = 0.08
    false_phase_noise_sigma: float = 0.16
    memory_decay: float = 0.995
    memory_learning_rate: float = 0.22
    memory_influence: float = 0.8
    memory_activation_threshold: float = 0.28
    memory_radius_factor: float = 1.35
    memory_max: float = 1.5
    stability_gate: float = 0.7
    memory_similarity_threshold: float = 0.6
    recall_phase_strength: float = 0.12
    recall_cooldown: int = 40


@dataclass(frozen=True)
class PhaseIVScenario:
    model_kind: str
    bootstrap_mode: str
    false_mode: str
    name: str
    label: str


PHASE4_MODEL_LABELS = {
    "baseline": "Baseline+Memory",
    "triangle": "Triangle+Memory",
    "helix": "Helix+Memory",
    "hybrid_manifold": "Hybrid Manifold+Memory",
}


def build_phase4_scenarios() -> list[PhaseIVScenario]:
    scenarios: list[PhaseIVScenario] = []
    for model_kind in ("baseline", "triangle", "helix", "hybrid_manifold"):
        for bootstrap_mode in ("single", "periodic"):
            for false_mode in ("none", "random", "noisy"):
                name = f"{model_kind}_{bootstrap_mode}_{false_mode}"
                label = (
                    f"{PHASE4_MODEL_LABELS[model_kind]} | "
                    f"{bootstrap_mode} pulse | {false_mode} false"
                )
                scenarios.append(
                    PhaseIVScenario(
                        model_kind=model_kind,
                        bootstrap_mode=bootstrap_mode,
                        false_mode=false_mode,
                        name=name,
                        label=label,
                    )
                )
    return scenarios
