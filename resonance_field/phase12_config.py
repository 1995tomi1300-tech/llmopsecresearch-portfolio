from __future__ import annotations

from dataclasses import dataclass

from resonance_field.phase6_config import PhaseVIScenario
from resonance_field.phase11_config import PhaseXIConfig


@dataclass(frozen=True)
class PhaseXIIConfig(PhaseXIConfig):
    experiment_tag: str = "Phase XII"
    policy_learning_enabled: bool = False
    reward_controller_enabled: bool = True
    reward_learning_rate: float = 0.28
    reward_ucb_scale: float = 0.12
    reward_switch_cost: float = 0.08
    reward_min_dwell: int = 10
    reward_coherence_gain: float = 2.40
    reward_cluster_gain: float = 1.80
    reward_disturbance_gain: float = 1.20
    reward_trust_gain: float = 0.80
    reward_observe_cost: float = 0.00
    reward_absorber_cost: float = 0.06
    reward_scaffold_cost: float = 0.04
    reward_idle_penalty: float = 0.22
    reward_clip: float = 2.50
    reward_initial_observe_value: float = 0.08
    reward_initial_absorber_value: float = 0.02
    reward_initial_scaffold_value: float = 0.08
    reward_disabled_scaffold_value: float = -0.30
    reward_observe_prior_gain: float = 0.82
    reward_absorber_prior_gain: float = 0.70
    reward_scaffold_prior_gain: float = 1.05
    reward_observe_penalty: float = 0.32
    reward_absorber_overuse_penalty: float = 0.26
    reward_scaffold_penalty: float = 0.45
    reward_observe_coherence_floor: float = 0.96
    reward_observe_cluster_floor: float = 0.92
    reward_observe_disturbance_ceiling: float = 0.18
    reward_scaffold_coherence_floor: float = 0.84


PHASE12_MODEL_LABELS = {
    "baseline": "Baseline+Reward Controller",
    "triangle": "Triangle+Reward Controller",
    "helix": "Helix+Reward Controller",
    "hybrid_manifold": "Hybrid Manifold+Reward Controller",
}


def build_phase12_scenarios() -> list[PhaseVIScenario]:
    scenarios: list[PhaseVIScenario] = []
    for model_kind in ("baseline", "triangle", "helix", "hybrid_manifold"):
        for bootstrap_mode in ("single", "periodic"):
            for false_mode in ("none", "random", "noisy"):
                name = f"{model_kind}_{bootstrap_mode}_{false_mode}"
                label = (
                    f"{PHASE12_MODEL_LABELS[model_kind]} | "
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
