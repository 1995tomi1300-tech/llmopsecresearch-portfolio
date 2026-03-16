from __future__ import annotations

from dataclasses import dataclass

from resonance_field.phase6_config import PhaseVIScenario
from resonance_field.phase14_config import PhaseXIVConfig


@dataclass(frozen=True)
class PhaseXVIConfig(PhaseXIVConfig):
    experiment_tag: str = "Phase XVI"
    inverse_triad_enabled: bool = False
    phase_manipulation_enabled: bool = True
    collect_event_rows: bool = True
    quantized_scalar_vector_enabled: bool = False
    quantized_scalar_levels: int = 1000
    quantized_anchor_scalar: float = 0.53
    quantized_scalar_phase_weight: float = 0.34
    quantized_scalar_freq_weight: float = 0.22
    quantized_scalar_trust_weight: float = 0.26
    quantized_scalar_memory_weight: float = 0.18
    quantized_reference_blend: float = 0.55
    quantized_active_window: int = 220
    quantized_coherence_trigger: float = 0.98
    quantized_disturbance_trigger: float = 0.18
    quantized_core_protect_top_fraction: float = 0.18
    quantized_shell_trust_min: float = 0.34
    quantized_shell_trust_max: float = 0.78
    quantized_shell_disturbance_min: float = 0.14
    quantized_shell_disturbance_max: float = 0.72
    quantized_freq_span: float = 0.18
    quantized_receptive_trust_floor: float = 0.48
    quantized_receptive_trust_ceiling: float = 0.78
    quantized_receptive_disturbance_max: float = 0.36
    quantized_repulsive_trust_max: float = 0.32
    quantized_repulsive_disturbance_min: float = 0.72
    quantized_lock_width_bins: float = 24.0
    quantized_drive_scale: float = 0.72
    quantized_repulsive_scale: float = 0.42
    quantized_vector_max_pull: float = 0.65
    quantized_vector_phase_gain: float = 0.035
    quantized_vector_freq_gain: float = 0.022
    quantized_vector_geometry_gain: float = 0.015
    quantized_vector_helix_gain: float = 0.018
    phase_manipulation_active_window: int = 220
    phase_manipulation_interval: int = 8
    phase_manipulation_cooldown: int = 20
    phase_manipulation_max_events: int = 1
    phase_manipulation_trust_min: float = 0.45
    phase_manipulation_trust_max: float = 0.72
    phase_manipulation_trust_margin: float = 0.08
    phase_manipulation_degree_min_ratio: float = 0.35
    phase_manipulation_degree_max_ratio: float = 1.35
    phase_manipulation_memory_ceiling: float = 0.35
    phase_manipulation_disturbance_ceiling: float = 0.72
    phase_manipulation_phase_shift: float = 0.28
    phase_manipulation_freq_shift: float = 0.022
    phase_manipulation_phase_blend: float = 0.42
    phase_manipulation_freq_blend: float = 0.24
    phase_manipulation_geometry_shift: float = 0.08
    phase_manipulation_helix_shift: float = 0.18
    phase_manipulation_geometry_steps: int = 16
    phase_manipulation_relax_steps: int = 48
    phase_manipulation_tracking_window: int = 120
    phase_manipulation_align_threshold: float = 0.30
    phase_manipulation_freq_threshold: float = 0.035
    phase_manipulation_recovery_phase_tol: float = 0.20
    phase_manipulation_recovery_freq_tol: float = 0.015
    phase_manipulation_core_disturbance_threshold: float = 0.42


PHASE16_MODEL_LABELS = {
    "triangle": "Triangle+Phase Manipulation",
    "helix": "Helix+Phase Manipulation",
    "hybrid_manifold": "Hybrid Manifold+Phase Manipulation",
}


def build_phase16_scenarios() -> list[PhaseVIScenario]:
    scenarios: list[PhaseVIScenario] = []
    for model_kind in ("triangle", "helix", "hybrid_manifold"):
        for bootstrap_mode in ("single", "periodic"):
            for false_mode in ("random", "noisy"):
                name = f"{model_kind}_{bootstrap_mode}_{false_mode}"
                label = (
                    f"{PHASE16_MODEL_LABELS[model_kind]} | "
                    f"{bootstrap_mode} pulse | {false_mode} false | mid-node phase manipulation"
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
