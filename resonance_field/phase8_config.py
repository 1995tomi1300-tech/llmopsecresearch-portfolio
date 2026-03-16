from __future__ import annotations

from dataclasses import dataclass

from resonance_field.phase6_config import PhaseVIScenario
from resonance_field.phase7_config import PhaseVIIConfig


@dataclass(frozen=True)
class PhaseVIIIConfig(PhaseVIIConfig):
    experiment_tag: str = "Phase VIII"
    scaffold_model_kinds: tuple[str, ...] = ("baseline",)
    scaffold_decay: float = 0.9985
    scaffold_learning_rate: float = 0.16
    scaffold_max: float = 1.8
    scaffold_influence: float = 0.82
    scaffold_activation_threshold: float = 0.14
    scaffold_radius_factor: float = 2.10
    scaffold_trust_floor: float = 0.72
    shock_scaffold_retention: float = 0.88
    scaffold_cluster_floor: float = 0.92
    scaffold_recruit_fraction: float = 0.16
    scaffold_affinity_threshold: float = 0.04
    scaffold_phase_strength: float = 0.26
    scaffold_freq_blend: float = 0.16
    scaffold_velocity_pull: float = 0.10
    scaffold_event_boost: float = 0.65
    scaffold_cooldown: int = 24
    scaffold_active_window: int = 220


PHASE8_MODEL_LABELS = {
    "baseline": "Baseline+Scaffold",
    "triangle": "Triangle+Scaffold",
    "helix": "Helix+Scaffold",
    "hybrid_manifold": "Hybrid Manifold+Scaffold",
}


def build_phase8_scenarios() -> list[PhaseVIScenario]:
    scenarios: list[PhaseVIScenario] = []
    for model_kind in ("baseline", "triangle", "helix", "hybrid_manifold"):
        for bootstrap_mode in ("single", "periodic"):
            for false_mode in ("none", "random", "noisy"):
                name = f"{model_kind}_{bootstrap_mode}_{false_mode}"
                label = (
                    f"{PHASE8_MODEL_LABELS[model_kind]} | "
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
