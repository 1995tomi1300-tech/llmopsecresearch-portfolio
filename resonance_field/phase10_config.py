from __future__ import annotations

from dataclasses import dataclass

from resonance_field.phase6_config import PhaseVIScenario
from resonance_field.phase8_config import PhaseVIIIConfig


@dataclass(frozen=True)
class PhaseXConfig(PhaseVIIIConfig):
    experiment_tag: str = "Phase X"
    controller_enabled: bool = True
    controller_active_window: int = 220
    controller_coherence_critical: float = 0.82
    controller_coherence_release: float = 0.96
    controller_cluster_priority: float = 0.88
    controller_cluster_release: float = 0.96
    controller_scaffold_coherence_floor: float = 0.84
    controller_disturbance_priority: float = 0.22
    controller_min_dwell: int = 18


PHASE10_MODEL_LABELS = {
    "baseline": "Baseline+Adaptive Controller",
    "triangle": "Triangle+Adaptive Controller",
    "helix": "Helix+Adaptive Controller",
    "hybrid_manifold": "Hybrid Manifold+Adaptive Controller",
}


def build_phase10_scenarios() -> list[PhaseVIScenario]:
    scenarios: list[PhaseVIScenario] = []
    for model_kind in ("baseline", "triangle", "helix", "hybrid_manifold"):
        for bootstrap_mode in ("single", "periodic"):
            for false_mode in ("none", "random", "noisy"):
                name = f"{model_kind}_{bootstrap_mode}_{false_mode}"
                label = (
                    f"{PHASE10_MODEL_LABELS[model_kind]} | "
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
