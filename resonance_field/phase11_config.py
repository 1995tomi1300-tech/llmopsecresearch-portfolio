from __future__ import annotations

from dataclasses import dataclass

from resonance_field.phase6_config import PhaseVIScenario
from resonance_field.phase10_config import PhaseXConfig


@dataclass(frozen=True)
class PhaseXIConfig(PhaseXConfig):
    experiment_tag: str = "Phase XI"
    policy_learning_enabled: bool = True
    policy_learning_rate: float = 0.030
    policy_restore_rate: float = 0.012
    policy_coherence_critical_max: float = 0.96
    policy_coherence_release_max: float = 1.06
    policy_cluster_priority_max: float = 1.04
    policy_cluster_release_max: float = 1.08
    policy_scaffold_floor_min: float = 0.76
    policy_disturbance_min: float = 0.12
    policy_min_dwell_min: float = 10.0
    policy_min_dwell_max: float = 28.0


PHASE11_MODEL_LABELS = {
    "baseline": "Baseline+Policy Learner",
    "triangle": "Triangle+Policy Learner",
    "helix": "Helix+Policy Learner",
    "hybrid_manifold": "Hybrid Manifold+Policy Learner",
}


def build_phase11_scenarios() -> list[PhaseVIScenario]:
    scenarios: list[PhaseVIScenario] = []
    for model_kind in ("baseline", "triangle", "helix", "hybrid_manifold"):
        for bootstrap_mode in ("single", "periodic"):
            for false_mode in ("none", "random", "noisy"):
                name = f"{model_kind}_{bootstrap_mode}_{false_mode}"
                label = (
                    f"{PHASE11_MODEL_LABELS[model_kind]} | "
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
