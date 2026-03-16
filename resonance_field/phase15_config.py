from __future__ import annotations

from dataclasses import dataclass

from resonance_field.phase6_config import PhaseVIScenario
from resonance_field.phase14_config import PhaseXIVConfig


@dataclass(frozen=True)
class PhaseXVConfig(PhaseXIVConfig):
    experiment_tag: str = "Phase XV"
    collect_event_rows: bool = True
    inverse_triad_enabled: bool = True
    inverse_triad_active_window: int = 220
    inverse_triad_interval: int = 8
    inverse_triad_cooldown: int = 20
    inverse_triad_max_events: int = 1
    inverse_triad_trust_min: float = 0.35
    inverse_triad_trust_max: float = 0.60
    inverse_triad_memory_ceiling: float = 0.18
    inverse_triad_disturbance_ceiling: float = 0.58
    inverse_triad_degree_ratio_ceiling: float = 0.95
    inverse_triad_candidate_pool: int = 18
    inverse_triad_min_edges: int = 2
    inverse_triad_core_top_fraction: float = 0.18
    inverse_triad_min_core_bridge: float = 0.01
    inverse_triad_pair_distance_factor: float = 1.35
    inverse_triad_lift: float = 0.18
    inverse_triad_skew: float = 0.12
    inverse_triad_phase_strength: float = 0.22
    inverse_triad_freq_shift: float = 0.018
    inverse_triad_geometry_steps: int = 12
    inverse_triad_relax_steps: int = 36
    inverse_triad_tracking_window: int = 120
    inverse_triad_recovery_gain: float = 0.18
    inverse_triad_core_disturbance_threshold: float = 0.42
    inverse_triad_helix_scale: float = 0.35


PHASE15_MODEL_LABELS = {
    "triangle": "Triangle+Inverse Triad",
    "helix": "Helix+Inverse Triad",
    "hybrid_manifold": "Hybrid Manifold+Inverse Triad",
}


def build_phase15_scenarios() -> list[PhaseVIScenario]:
    scenarios: list[PhaseVIScenario] = []
    for model_kind in ("triangle", "helix", "hybrid_manifold"):
        for bootstrap_mode in ("single", "periodic"):
            for false_mode in ("random", "noisy"):
                name = f"{model_kind}_{bootstrap_mode}_{false_mode}"
                label = (
                    f"{PHASE15_MODEL_LABELS[model_kind]} | "
                    f"{bootstrap_mode} pulse | {false_mode} false | inverse triad"
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
