from __future__ import annotations

from dataclasses import dataclass

from resonance_field.phase6_config import PhaseVIConfig, PhaseVIScenario


@dataclass(frozen=True)
class PhaseVIIConfig(PhaseVIConfig):
    experiment_tag: str = "Phase VII"
    absorber_coherence_floor: float = 0.78
    absorber_cluster_floor: float = 0.70
    absorber_disturbance_threshold: float = 0.42
    absorber_distressed_ratio_floor: float = 0.28
    absorber_recovery_margin: float = 1.05
    absorber_active_window: int = 160
    absorber_phase_strength: float = 0.34
    absorber_freq_blend: float = 0.18
    absorber_velocity_damp: float = 0.45
    absorber_memory_boost: float = 0.60
    absorber_cooldown: int = 18


PHASE7_MODEL_LABELS = {
    "baseline": "Baseline+Absorber",
    "triangle": "Triangle+Absorber",
    "helix": "Helix+Absorber",
    "hybrid_manifold": "Hybrid Manifold+Absorber",
}


def build_phase7_scenarios() -> list[PhaseVIScenario]:
    scenarios: list[PhaseVIScenario] = []
    for model_kind in ("baseline", "triangle", "helix", "hybrid_manifold"):
        for bootstrap_mode in ("single", "periodic"):
            for false_mode in ("none", "random", "noisy"):
                name = f"{model_kind}_{bootstrap_mode}_{false_mode}"
                label = (
                    f"{PHASE7_MODEL_LABELS[model_kind]} | "
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
