from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseVIConfig:
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
    shock_step: int = 300
    shock_phase_fraction: float = 0.72
    shock_geometry_fraction: float = 0.62
    shock_memory_retention: float = 0.2
    shock_frequency_sigma: float = 0.06
    shock_window: int = 80
    recovery_threshold: float = 0.85
    cluster_recovery_threshold: float = 0.75
    response_window: int = 20
    trust_decay: float = 0.92
    trust_gain: float = 0.22
    trust_floor: float = 0.26
    quarantine_threshold: float = 0.42
    quarantine_strength: float = 0.45
    alignment_floor: float = 0.30
    core_memory_decay: float = 0.998
    core_memory_learning_rate: float = 0.18
    core_memory_influence: float = 1.35
    core_memory_activation_threshold: float = 0.18
    core_memory_radius_factor: float = 1.75
    core_memory_max: float = 1.8
    shock_core_retention: float = 0.84
    guardian_coherence_floor: float = 0.68
    guardian_cluster_floor: float = 0.58
    guardian_pulse_strength: float = 0.24
    guardian_memory_boost: float = 1.0
    guardian_cooldown: int = 14
    guardian_trust_fraction: float = 0.22


@dataclass(frozen=True)
class PhaseVIScenario:
    model_kind: str
    bootstrap_mode: str
    false_mode: str
    name: str
    label: str


PHASE6_MODEL_LABELS = {
    "baseline": "Baseline+Hardened",
    "triangle": "Triangle+Hardened",
    "helix": "Helix+Hardened",
    "hybrid_manifold": "Hybrid Manifold+Hardened",
}


def build_phase6_scenarios() -> list[PhaseVIScenario]:
    scenarios: list[PhaseVIScenario] = []
    for model_kind in ("baseline", "triangle", "helix", "hybrid_manifold"):
        for bootstrap_mode in ("single", "periodic"):
            for false_mode in ("none", "random", "noisy"):
                name = f"{model_kind}_{bootstrap_mode}_{false_mode}"
                label = (
                    f"{PHASE6_MODEL_LABELS[model_kind]} | "
                    f"{bootstrap_mode} pulse | {false_mode} false"
                )
                scenarios.append(
                    PhaseVIScenario(
                        model_kind=model_kind,
                        bootstrap_mode=bootstrap_mode,
                        false_mode=false_mode,
                        name=name,
                        label=label,
                    )
                )
    return scenarios
