from __future__ import annotations

from dataclasses import dataclass

from resonance_field.phase6_config import PhaseVIScenario
from resonance_field.phase8_config import PhaseVIIIConfig


@dataclass(frozen=True)
class PhaseIXConfig(PhaseVIIIConfig):
    experiment_tag: str = "Phase IX"
    scaffold_recruit_floor: float = 0.48
    scaffold_phase_compatibility_min: float = 0.42
    scaffold_phase_guard_gain: float = 0.24
    scaffold_candidate_trust_min: float = 0.30
    scaffold_pull_floor: float = 0.42
    scaffold_phase_boost_gain: float = 0.42


PHASE9_MODEL_LABELS = {
    "baseline": "Baseline+Coherent Scaffold",
    "triangle": "Triangle+Coherent Scaffold",
    "helix": "Helix+Coherent Scaffold",
    "hybrid_manifold": "Hybrid Manifold+Coherent Scaffold",
}


def build_phase9_scenarios() -> list[PhaseVIScenario]:
    scenarios: list[PhaseVIScenario] = []
    for model_kind in ("baseline", "triangle", "helix", "hybrid_manifold"):
        for bootstrap_mode in ("single", "periodic"):
            for false_mode in ("none", "random", "noisy"):
                name = f"{model_kind}_{bootstrap_mode}_{false_mode}"
                label = (
                    f"{PHASE9_MODEL_LABELS[model_kind]} | "
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
