from __future__ import annotations

from dataclasses import dataclass

from resonance_field.phase6_config import PhaseVIScenario
from resonance_field.phase13_config import PhaseXIIIConfig


@dataclass(frozen=True)
class PhaseXIVConfig(PhaseXIIIConfig):
    experiment_tag: str = "Phase XIV"
    modulation_training_enabled: bool = True
    modulation_active_window: int = 240
    modulation_interval: int = 4
    modulation_cooldown: int = 8
    modulation_focus_fraction: float = 0.18
    modulation_beam_count: int = 5
    modulation_angle_spread: float = 0.90
    modulation_spatial_sigma: float = 0.85
    modulation_phase_strength: float = 0.24
    modulation_freq_blend: float = 0.16
    modulation_geometry_pull: float = 0.68
    modulation_memory_boost: float = 0.58
    modulation_coherence_floor: float = 0.82
    modulation_cluster_floor: float = 0.58
    modulation_disturbance_ceiling: float = 0.48
    modulation_min_trust: float = 0.20
    modulation_score_threshold: float = 0.00
    modulation_ki_base: float = 0.18
    modulation_ki_gain: float = 0.84
    modulation_ki_max: float = 1.30


PHASE14_MODEL_LABELS = {
    "baseline": "Baseline+Multifocal Training",
    "triangle": "Triangle+Multifocal Training",
    "helix": "Helix+Multifocal Training",
    "hybrid_manifold": "Hybrid Manifold+Multifocal Training",
}


def build_phase14_scenarios() -> list[PhaseVIScenario]:
    scenarios: list[PhaseVIScenario] = []
    for model_kind in ("baseline", "triangle", "helix", "hybrid_manifold"):
        for bootstrap_mode in ("single", "periodic"):
            for false_mode in ("none", "random", "noisy"):
                name = f"{model_kind}_{bootstrap_mode}_{false_mode}"
                label = (
                    f"{PHASE14_MODEL_LABELS[model_kind]} | "
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
