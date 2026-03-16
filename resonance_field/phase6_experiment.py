from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from resonance_field.numeric_backend import (
    backend_info_dict,
    benchmark_pairwise_kernels,
    phase_kernels,
    resolve_numeric_backend,
)
from resonance_field.phase4_experiment import (
    cluster_sets,
    jaccard,
    primary_cluster,
    reinforce_memory,
)
from resonance_field.phase4_models import PHASE4_MODEL_REGISTRY
from resonance_field.phase5_experiment import apply_shock, rolling_mean
from resonance_field.phase6_config import (
    PHASE6_MODEL_LABELS,
    PhaseVIConfig,
    PhaseVIScenario,
    build_phase6_scenarios,
)
from resonance_field.topologies import bootstrap_kick


@dataclass
class PhaseVIRunMetrics:
    scenario: str
    label: str
    model_kind: str
    bootstrap_mode: str
    false_mode: str
    run_index: int
    seed: int
    pre_shock_coherence: float
    shock_min_coherence: float
    late_recovery_coherence: float
    coherence_recovery_ratio: float
    pre_shock_cluster_fraction: float
    shock_min_cluster_fraction: float
    late_cluster_fraction: float
    cluster_recovery_ratio: float
    final_false_isolation_rate: float
    recovery_time_steps: float
    recovery_success: float
    pre_shock_memory_mass: float
    late_memory_mass: float
    memory_recovery_ratio: float
    pre_shock_core_memory_mass: float
    late_core_memory_mass: float
    core_memory_recovery_ratio: float
    topological_memory_index: float
    recall_event_rate: float
    guardian_event_rate: float
    absorber_event_rate: float
    scaffold_event_rate: float
    modulation_event_rate: float
    modulation_target_fraction: float
    modulation_mean_ki: float
    scaffold_edge_mass: float
    controller_absorber_rate: float
    controller_scaffold_rate: float
    controller_observe_rate: float
    controller_switch_rate: float
    controller_reward_mean: float
    learned_coherence_critical: float
    learned_cluster_priority: float
    learned_disturbance_priority: float
    learned_action_value_observe: float
    learned_action_value_absorber: float
    learned_action_value_scaffold: float
    resonance_energy_late: float
    late_trust_mean: float
    inverse_triad_event_rate: float
    inverse_triad_mean_anchor_trust: float
    inverse_triad_mean_bridge_strength: float
    inverse_triad_mean_shell_score: float
    inverse_triad_propagation_fraction: float
    inverse_triad_boundary_recovery_steps: float
    inverse_triad_core_stability_delta: float
    inverse_triad_core_contamination_rate: float
    phase_manipulation_event_rate: float
    phase_manipulation_mean_target_trust: float
    phase_manipulation_mean_bidirectional_balance: float
    phase_manipulation_upward_spread_fraction: float
    phase_manipulation_downward_spread_fraction: float
    phase_manipulation_recovery_steps: float
    phase_manipulation_core_stability_delta: float
    phase_manipulation_core_contamination_rate: float
    quantized_scalar_anchor_gap_late: float
    quantized_scalar_lock_fraction_late: float
    quantized_repulsive_fraction_late: float
    quantized_vector_drive_mean_late: float


def initialize_state(
    config: PhaseVIConfig,
    scenario: PhaseVIScenario,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    model = PHASE4_MODEL_REGISTRY[scenario.model_kind]
    state = model.initialize_state(config, rng, scenario.false_mode)
    if scenario.bootstrap_mode in {"single", "periodic"}:
        state["theta"] = bootstrap_kick(
            state["theta"],
            config.bootstrap_anchor,
            config.bootstrap_strength,
        )
    return state


def save_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    if not rows and not fieldnames:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def reinforce_core_memory(
    core_memory_matrix: np.ndarray,
    cluster: set[int],
    config: PhaseVIConfig,
    boost: float = 1.0,
) -> None:
    if len(cluster) < config.cluster_threshold:
        return
    indices = np.array(sorted(cluster), dtype=int)
    core_memory_matrix[np.ix_(indices, indices)] += config.core_memory_learning_rate * boost
    np.fill_diagonal(core_memory_matrix, 0.0)
    np.clip(core_memory_matrix, 0.0, config.core_memory_max, out=core_memory_matrix)


def reinforce_scaffold(
    scaffold_matrix: np.ndarray,
    cluster: set[int],
    config: Any,
    boost: float = 1.0,
) -> None:
    if len(cluster) < config.cluster_threshold:
        return
    indices = np.array(sorted(cluster), dtype=int)
    scaffold_matrix[np.ix_(indices, indices)] += config.scaffold_learning_rate * boost
    np.fill_diagonal(scaffold_matrix, 0.0)
    np.clip(scaffold_matrix, 0.0, config.scaffold_max, out=scaffold_matrix)


def apply_scaffold_geometry_pull(
    state: dict[str, np.ndarray],
    recruit_indices: np.ndarray,
    anchor_indices: np.ndarray,
    config: Any,
    scale: float = 1.0,
) -> None:
    if recruit_indices.size == 0 or anchor_indices.size == 0:
        return
    pull = float(getattr(config, "scaffold_velocity_pull", 0.12)) * scale
    if "pos" in state:
        anchor_center = np.mean(state["pos"][anchor_indices], axis=0)
        delta = anchor_center - state["pos"][recruit_indices]
        state["vel"][recruit_indices] += pull * delta
    if "helix_t" in state:
        anchor_t = float(np.mean(state["helix_t"][anchor_indices]))
        state["helix_vel"][recruit_indices] += pull * (anchor_t - state["helix_t"][recruit_indices])


def circular_mean_angle(phases: np.ndarray) -> float:
    if phases.size == 0:
        return 0.0
    return float(np.angle(np.mean(np.exp(1j * phases))))


def quantize_unit_interval(values: np.ndarray | float, levels: int) -> np.ndarray | float:
    levels = max(2, int(levels))
    step = 1.0 / float(levels - 1)
    quantized = np.round(np.clip(values, 0.0, 1.0) / step) * step
    return np.clip(quantized, 0.0, 1.0)


def apply_quantized_scalar_vector_control(
    state: dict[str, np.ndarray],
    trust_scores: np.ndarray,
    memory_support: np.ndarray,
    disturbance: np.ndarray,
    anchor_indices: np.ndarray,
    config: Any,
) -> dict[str, float]:
    if not getattr(config, "quantized_scalar_vector_enabled", False):
        return {
            "anchor_gap_mean": 0.0,
            "lock_fraction": 0.0,
            "repulsive_fraction": 0.0,
            "drive_mean": 0.0,
        }

    node_count = int(state["theta"].shape[0])
    if anchor_indices.size == 0:
        anchor_indices = np.arange(node_count, dtype=int)

    anchor_phase = circular_mean_angle(state["theta"][anchor_indices])
    anchor_freq = float(np.mean(state["freq"][anchor_indices]))
    levels = int(getattr(config, "quantized_scalar_levels", 1000))
    anchor_scalar_target = float(
        quantize_unit_interval(
            float(getattr(config, "quantized_anchor_scalar", 0.53)),
            levels,
        )
    )
    bin_width = 1.0 / float(max(1, levels - 1))
    lock_width = float(getattr(config, "quantized_lock_width_bins", 1.5)) * bin_width

    phase_alignment = 0.5 + 0.5 * np.cos(state["theta"] - anchor_phase)
    freq_span = max(float(getattr(config, "quantized_freq_span", 0.18)), 1e-6)
    freq_alignment = np.clip(1.0 - np.abs(state["freq"] - anchor_freq) / freq_span, 0.0, 1.0)

    scalar_raw = (
        float(getattr(config, "quantized_scalar_phase_weight", 0.34)) * phase_alignment
        + float(getattr(config, "quantized_scalar_freq_weight", 0.22)) * freq_alignment
        + float(getattr(config, "quantized_scalar_trust_weight", 0.26)) * trust_scores
        + float(getattr(config, "quantized_scalar_memory_weight", 0.18)) * memory_support
    )
    scalar_quantized = np.asarray(quantize_unit_interval(scalar_raw, levels), dtype=float)
    local_anchor_scalar = float(np.mean(scalar_quantized[anchor_indices]))
    anchor_scalar = float(
        quantize_unit_interval(
            float(getattr(config, "quantized_reference_blend", 0.55)) * anchor_scalar_target
            + (1.0 - float(getattr(config, "quantized_reference_blend", 0.55)))
            * local_anchor_scalar,
            levels,
        )
    )
    scalar_gap = np.abs(anchor_scalar - scalar_quantized)

    protected_mask = np.zeros(node_count, dtype=bool)
    protected_mask[anchor_indices] = True
    protect_count = max(
        int(anchor_indices.size),
        int(
            round(
                node_count
                * float(getattr(config, "quantized_core_protect_top_fraction", 0.18))
            )
        ),
    )
    protected_mask[np.argsort(trust_scores)[-protect_count:]] = True

    shell_mask = (~protected_mask) & (
        (
            trust_scores
            >= float(getattr(config, "quantized_shell_trust_min", 0.34))
        )
        & (
            trust_scores
            <= float(getattr(config, "quantized_shell_trust_max", 0.78))
        )
    )
    shell_mask |= (~protected_mask) & (
        (
            disturbance
            >= float(getattr(config, "quantized_shell_disturbance_min", 0.14))
        )
        & (
            disturbance
            <= float(getattr(config, "quantized_shell_disturbance_max", 0.72))
        )
    )

    receptive_mask = (
        shell_mask
        & (~state["false_mask"])
        & (trust_scores >= float(getattr(config, "quantized_receptive_trust_floor", 0.52)))
        & (trust_scores <= float(getattr(config, "quantized_receptive_trust_ceiling", 0.78)))
        & (
            disturbance
            <= float(getattr(config, "quantized_receptive_disturbance_max", 0.42))
        )
    )
    repulsive_mask = (
        shell_mask
        & (
            state["false_mask"]
            | (trust_scores <= float(getattr(config, "quantized_repulsive_trust_max", 0.44)))
            | (
                disturbance
                >= float(getattr(config, "quantized_repulsive_disturbance_min", 0.58))
            )
        )
    )

    passive_lock_mask = (
        protected_mask
        | (trust_scores <= float(getattr(config, "quantized_repulsive_trust_max", 0.44)))
    )

    mode = np.zeros(node_count, dtype=float)
    mode[receptive_mask] = 1.0
    mode[repulsive_mask] = -1.0

    drive = np.clip(
        scalar_gap / max(float(getattr(config, "quantized_drive_scale", 0.72)), lock_width, 1e-6),
        0.0,
        float(getattr(config, "quantized_vector_max_pull", 1.0)),
    ) * mode
    drive[repulsive_mask] *= float(getattr(config, "quantized_repulsive_scale", 0.42))
    drive[passive_lock_mask] = 0.0

    phase_gain = float(getattr(config, "quantized_vector_phase_gain", 0.10))
    state["theta"] = (
        state["theta"] + phase_gain * drive * np.sin(anchor_phase - state["theta"])
    ) % (2.0 * np.pi)

    freq_gain = float(getattr(config, "quantized_vector_freq_gain", 0.08))
    state["freq"] = np.clip(
        state["freq"] + freq_gain * drive * (anchor_freq - state["freq"]),
        0.5,
        1.5,
    )

    geom_gain = float(getattr(config, "quantized_vector_geometry_gain", 0.045))
    if "pos" in state and "vel" in state:
        anchor_center = np.mean(state["pos"][anchor_indices], axis=0)
        delta = anchor_center - state["pos"]
        delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
        direction = np.divide(
            delta,
            np.maximum(delta_norm, 1e-6),
            out=np.zeros_like(delta),
            where=delta_norm > 0.0,
        )
        state["vel"] += geom_gain * drive[:, None] * direction

    helix_gain = float(getattr(config, "quantized_vector_helix_gain", 0.055))
    if "helix_t" in state and "helix_vel" in state:
        anchor_t = float(np.mean(state["helix_t"][anchor_indices]))
        state["helix_vel"] += helix_gain * drive * (anchor_t - state["helix_t"])

    return {
        "anchor_gap_mean": float(np.mean(scalar_gap)),
        "lock_fraction": float(np.mean(scalar_gap <= lock_width)),
        "repulsive_fraction": float(np.mean(drive < 0.0)),
        "drive_mean": float(np.mean(np.abs(drive))),
    }


def riemann_straighten_coordinates(
    state: dict[str, np.ndarray],
    anchor_indices: np.ndarray,
) -> tuple[np.ndarray, float]:
    node_count = int(state["theta"].shape[0])
    if anchor_indices.size == 0:
        return np.zeros(node_count, dtype=float), 1.0

    # Use a local tangent-like coordinate so multifocal beams can act on a linearized
    # strand even when the underlying manifold is curved.
    if "helix_t" in state:
        anchor_t = float(np.mean(state["helix_t"][anchor_indices]))
        straight = state["helix_t"] - anchor_t
        span = max(float(np.std(straight[anchor_indices])), 0.25)
        return straight, span

    if "pos" in state:
        anchor_pos = state["pos"][anchor_indices]
        center = np.mean(anchor_pos, axis=0)
        centered = anchor_pos - center
        if anchor_pos.shape[0] >= 2:
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                tangent = vh[0]
            except np.linalg.LinAlgError:
                tangent = np.array([1.0, 0.0], dtype=float)
        else:
            tangent = np.array([1.0, 0.0], dtype=float)
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-6)
        straight = (state["pos"] - center) @ tangent
        span = max(float(np.std(straight[anchor_indices])), 0.15)
        return straight, span

    return np.zeros(node_count, dtype=float), 1.0


def choose_multifocal_modulation_targets(
    state: dict[str, np.ndarray],
    anchor_indices: np.ndarray,
    trust_scores: np.ndarray,
    memory_support: np.ndarray,
    disturbance: np.ndarray,
    config: Any,
) -> dict[str, Any] | None:
    if anchor_indices.size == 0:
        return None

    beam_count = max(1, int(getattr(config, "modulation_beam_count", 5)))
    beam_angle_spread = float(getattr(config, "modulation_angle_spread", 0.80))
    beam_angles = np.linspace(-beam_angle_spread, beam_angle_spread, beam_count)
    anchor_phase = circular_mean_angle(state["theta"][anchor_indices])
    beam_phases = anchor_phase + beam_angles

    straight_coord, straight_span = riemann_straighten_coordinates(state, anchor_indices)
    line_offsets = np.linspace(-straight_span, straight_span, beam_count)
    sigma = max(float(getattr(config, "modulation_spatial_sigma", 0.75)) * straight_span, 1e-3)

    phase_profile = 0.5 + 0.5 * np.cos(state["theta"][:, None] - beam_phases[None, :])
    spatial_profile = np.exp(
        -0.5 * ((straight_coord[:, None] - line_offsets[None, :]) / sigma) ** 2
    )
    beam_profile = phase_profile * spatial_profile
    beam_response = np.max(beam_profile, axis=1)

    phase_error = np.abs(np.angle(np.exp(1j * (state["theta"] - anchor_phase))))
    phase_compatibility = np.clip(1.0 - phase_error / np.pi, 0.0, 1.0)
    ki_base = float(getattr(config, "modulation_ki_base", 0.20))
    ki_gain = float(getattr(config, "modulation_ki_gain", 0.80))
    ki_max = float(getattr(config, "modulation_ki_max", 1.25))
    ki = np.clip(
        ki_base
        + ki_gain
        * (
            0.45 * beam_response
            + 0.30 * phase_compatibility
            + 0.15 * trust_scores
            + 0.10 * memory_support
        ),
        0.0,
        ki_max,
    )

    score = (
        beam_response * (0.45 + 0.35 * phase_compatibility + 0.20 * trust_scores)
        - 0.55 * disturbance
    )
    score[anchor_indices] = -1.0
    min_trust = float(getattr(config, "modulation_min_trust", 0.30))
    score[trust_scores < min_trust] = -1.0

    recruit_count = max(
        int(getattr(config, "cluster_threshold", 3)),
        int(round(state["theta"].shape[0] * float(getattr(config, "modulation_focus_fraction", 0.14)))),
    )
    score_threshold = float(getattr(config, "modulation_score_threshold", 0.12))
    candidate_indices = np.argsort(score)[-recruit_count:]
    candidate_indices = candidate_indices[score[candidate_indices] > score_threshold]
    if candidate_indices.size == 0:
        return None

    return {
        "anchor_phase": anchor_phase,
        "anchor_freq": float(np.mean(state["freq"][anchor_indices])),
        "beam_phases": beam_phases,
        "candidate_indices": candidate_indices,
        "ki": ki,
        "beam_response": beam_response,
    }


def component_fraction_for_indices(adjacency: np.ndarray, seed_indices: np.ndarray) -> float:
    if seed_indices.size == 0 or adjacency.size == 0:
        return 0.0
    seen = np.zeros(adjacency.shape[0], dtype=bool)
    stack = seed_indices.tolist()
    seen[seed_indices] = True
    while stack:
        node = stack.pop()
        neighbors = np.flatnonzero(adjacency[node] & ~seen)
        if neighbors.size:
            seen[neighbors] = True
            stack.extend(neighbors.tolist())
    return float(np.mean(seen))


def choose_inverse_triad_targets(
    state: dict[str, np.ndarray],
    effective_adjacency: np.ndarray,
    effective_weights: np.ndarray,
    pair_distance: np.ndarray,
    trust_scores: np.ndarray,
    memory_support: np.ndarray,
    disturbance: np.ndarray,
    current_primary: set[int],
    config: Any,
) -> dict[str, Any] | None:
    if not getattr(config, "inverse_triad_enabled", False):
        return None

    node_count = int(state["theta"].shape[0])
    degrees = effective_adjacency.sum(axis=1).astype(float)
    mean_degree = max(float(np.mean(degrees)), 1e-6)
    degree_norm = max(float(np.max(degrees)), 1.0)

    core_indices = (
        np.array(sorted(current_primary), dtype=int)
        if current_primary
        else np.argsort(trust_scores)[-max(config.cluster_threshold, int(round(node_count * float(getattr(config, "inverse_triad_core_top_fraction", 0.18))))):]
    )

    bridge_strength = np.zeros(node_count, dtype=float)
    if core_indices.size:
        bridge_strength = np.mean(effective_weights[:, core_indices], axis=1)

    per_node_shell_score = (
        0.40 * (1.0 - trust_scores)
        + 0.20 * (1.0 - np.clip(memory_support, 0.0, 1.0))
        + 0.15 * (1.0 - np.clip(degrees / degree_norm, 0.0, 1.0))
        + 0.10 * (1.0 - np.clip(disturbance, 0.0, 1.0))
        + 0.15 * np.clip(bridge_strength, 0.0, 1.0)
    )

    def try_select(
        trust_min: float,
        trust_max: float,
        memory_ceiling: float,
        disturbance_ceiling: float,
        degree_ratio_ceiling: float,
        min_edges: int,
        pair_distance_factor: float,
        min_bridge: float,
        pool_boost: int = 0,
    ) -> dict[str, Any] | None:
        candidate_mask = ~state["false_mask"]
        candidate_mask &= trust_scores >= trust_min
        candidate_mask &= trust_scores <= trust_max
        candidate_mask &= degrees > 0.0
        candidate_mask &= degrees <= mean_degree * degree_ratio_ceiling
        candidate_mask &= memory_support <= memory_ceiling
        candidate_mask &= disturbance <= disturbance_ceiling
        if core_indices.size:
            candidate_mask[core_indices] = False

        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size < config.cluster_threshold:
            return None

        pool_size = min(
            candidate_indices.size,
            max(
                config.cluster_threshold + 3,
                int(getattr(config, "inverse_triad_candidate_pool", 18)) + pool_boost,
            ),
        )
        shortlist = candidate_indices[
            np.argsort(per_node_shell_score[candidate_indices])[-pool_size:]
        ]

        best_payload: dict[str, Any] | None = None
        best_score = -np.inf
        max_pair_distance = float(pair_distance_factor) * float(config.radius)

        for triad in combinations(shortlist.tolist(), config.cluster_threshold):
            triad_indices = np.array(triad, dtype=int)
            triad_adjacency = effective_adjacency[np.ix_(triad_indices, triad_indices)]
            edge_count = int(np.sum(np.triu(triad_adjacency, k=1)))
            if edge_count < min_edges:
                continue

            triad_pair_distance = pair_distance[np.ix_(triad_indices, triad_indices)]
            if np.any(np.triu(triad_pair_distance, k=1) > max_pair_distance):
                continue

            triad_bridge = (
                float(np.mean(bridge_strength[triad_indices])) if core_indices.size else 0.0
            )
            if core_indices.size and triad_bridge < min_bridge:
                continue

            triad_trust = float(np.mean(trust_scores[triad_indices]))
            triad_memory = float(np.mean(memory_support[triad_indices]))
            triad_degree = float(np.mean(degrees[triad_indices] / degree_norm))
            triad_disturbance = float(np.mean(disturbance[triad_indices]))
            shell_score = (
                0.42 * (1.0 - triad_trust)
                + 0.22 * (1.0 - triad_memory)
                + 0.14 * (1.0 - triad_degree)
                + 0.08 * (1.0 - triad_disturbance)
                + 0.14 * np.clip(triad_bridge, 0.0, 1.0)
            )
            selection_score = shell_score + 0.05 * edge_count
            if selection_score > best_score:
                best_score = selection_score
                best_payload = {
                    "triad_indices": triad_indices,
                    "core_indices": core_indices.copy(),
                    "edge_count": edge_count,
                    "mean_trust": triad_trust,
                    "mean_memory_support": triad_memory,
                    "mean_bridge_strength": triad_bridge,
                    "mean_disturbance": triad_disturbance,
                    "mean_degree_norm": triad_degree,
                    "shell_score": shell_score,
                }

        return best_payload

    strict_payload = try_select(
        trust_min=float(getattr(config, "inverse_triad_trust_min", 0.35)),
        trust_max=float(getattr(config, "inverse_triad_trust_max", 0.60)),
        memory_ceiling=float(getattr(config, "inverse_triad_memory_ceiling", 0.18)),
        disturbance_ceiling=float(getattr(config, "inverse_triad_disturbance_ceiling", 0.58)),
        degree_ratio_ceiling=float(getattr(config, "inverse_triad_degree_ratio_ceiling", 0.95)),
        min_edges=int(getattr(config, "inverse_triad_min_edges", 2)),
        pair_distance_factor=float(getattr(config, "inverse_triad_pair_distance_factor", 1.35)),
        min_bridge=float(getattr(config, "inverse_triad_min_core_bridge", 0.01)),
    )
    if strict_payload is not None:
        return strict_payload

    relaxed_payload = try_select(
        trust_min=max(config.trust_floor, float(getattr(config, "inverse_triad_trust_min", 0.35)) - 0.10),
        trust_max=min(0.82, float(getattr(config, "inverse_triad_trust_max", 0.60)) + 0.12),
        memory_ceiling=min(1.0, float(getattr(config, "inverse_triad_memory_ceiling", 0.18)) + 0.18),
        disturbance_ceiling=min(1.0, float(getattr(config, "inverse_triad_disturbance_ceiling", 0.58)) + 0.18),
        degree_ratio_ceiling=float(getattr(config, "inverse_triad_degree_ratio_ceiling", 0.95)) + 0.35,
        min_edges=1,
        pair_distance_factor=float(getattr(config, "inverse_triad_pair_distance_factor", 1.35)) + 0.40,
        min_bridge=0.0,
        pool_boost=10,
    )
    if relaxed_payload is not None:
        return relaxed_payload

    fallback_pool = np.flatnonzero(
        (~state["false_mask"])
        & (
            trust_scores
            <= float(getattr(config, "inverse_triad_fallback_trust_max", 0.82))
        )
    )
    if core_indices.size:
        fallback_pool = fallback_pool[~np.isin(fallback_pool, core_indices)]
    if fallback_pool.size < config.cluster_threshold:
        return None
    fallback_pool = fallback_pool[np.argsort(per_node_shell_score[fallback_pool])[-12:]]
    if fallback_pool.size < config.cluster_threshold:
        return None
    triad_indices = np.array(fallback_pool[-config.cluster_threshold :], dtype=int)
    triad_trust = float(np.mean(trust_scores[triad_indices]))
    if triad_trust > float(getattr(config, "inverse_triad_fallback_trust_max", 0.82)):
        return None
    return {
        "triad_indices": triad_indices,
        "core_indices": core_indices.copy(),
        "edge_count": int(np.sum(np.triu(effective_adjacency[np.ix_(triad_indices, triad_indices)], k=1))),
        "mean_trust": triad_trust,
        "mean_memory_support": float(np.mean(memory_support[triad_indices])),
        "mean_bridge_strength": float(np.mean(bridge_strength[triad_indices])) if core_indices.size else 0.0,
        "mean_disturbance": float(np.mean(disturbance[triad_indices])),
        "mean_degree_norm": float(np.mean(degrees[triad_indices] / degree_norm)),
        "shell_score": float(np.mean(per_node_shell_score[triad_indices])),
    }


def apply_inverse_triad_transform(
    state: dict[str, np.ndarray],
    triad_indices: np.ndarray,
    config: Any,
    scale: float = 1.0,
) -> None:
    if triad_indices.size == 0 or scale <= 0.0:
        return

    order = np.argsort(state["freq"][triad_indices])
    ordered = triad_indices[order]
    pattern = np.array([-1.0, 0.0, 1.0], dtype=float)
    lift = float(getattr(config, "inverse_triad_lift", 0.18)) * scale
    skew = float(getattr(config, "inverse_triad_skew", 0.12)) * scale
    phase_strength = float(getattr(config, "inverse_triad_phase_strength", 0.22)) * scale
    freq_shift = float(getattr(config, "inverse_triad_freq_shift", 0.018)) * scale

    state["theta"][ordered] = (
        state["theta"][ordered] + phase_strength * (pattern + 0.35 * np.roll(pattern, 1))
    ) % (2.0 * np.pi)
    state["freq"][ordered] = np.clip(
        state["freq"][ordered] + freq_shift * (pattern + 0.25 * np.roll(pattern, -1)),
        0.5,
        1.5,
    )

    if "pos" in state:
        points = state["pos"][ordered]
        center = np.mean(points, axis=0)
        radial = points - center
        radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
        fallback = np.array(
            [[1.0, 0.0], [-0.5, 0.8660254], [-0.5, -0.8660254]],
            dtype=float,
        )
        radial = np.where(radial_norm > 1e-6, radial / np.maximum(radial_norm, 1e-6), fallback)
        perp = np.stack([-radial[:, 1], radial[:, 0]], axis=1)
        delta = lift * radial + skew * pattern[:, None] * perp
        state["pos"][ordered] = (state["pos"][ordered] + delta) % 1.0
        if "vel" in state:
            state["vel"][ordered] += 0.6 * delta

    if "helix_t" in state:
        helix_scale = float(getattr(config, "inverse_triad_helix_scale", 0.35)) * np.pi
        offsets = helix_scale * (lift * pattern + 0.5 * skew * np.roll(pattern, 1))
        state["helix_t"][ordered] += offsets
        max_t = config.helix_turns * 2.0 * np.pi
        state["helix_t"][ordered] = np.clip(state["helix_t"][ordered], 0.0, max_t)
        if "helix_vel" in state:
            state["helix_vel"][ordered] += 0.45 * offsets


def update_inverse_triad_trackers(
    trackers: list[dict[str, Any]],
    effective_adjacency: np.ndarray,
    trust_scores: np.ndarray,
    distressed: np.ndarray,
    step: int,
    config: Any,
) -> None:
    tracking_window = int(getattr(config, "inverse_triad_tracking_window", 120))
    recovery_gain = float(getattr(config, "inverse_triad_recovery_gain", 0.18))
    contamination_threshold = float(
        getattr(config, "inverse_triad_core_disturbance_threshold", 0.42)
    )

    for tracker in trackers:
        if tracker.get("finalized", False) or step < tracker["trigger_step"]:
            continue
        triad_indices = tracker["triad_indices"]
        core_indices = tracker["core_indices"]
        age = step - tracker["trigger_step"]

        triad_mean_trust = float(np.mean(trust_scores[triad_indices]))
        recovery_target = tracker["pre_triad_trust"] + recovery_gain * (
            1.0 - tracker["pre_triad_trust"]
        )
        if tracker["recovered_step"] is None and triad_mean_trust >= recovery_target:
            tracker["recovered_step"] = age

        tracker["max_component_fraction"] = max(
            tracker["max_component_fraction"],
            component_fraction_for_indices(effective_adjacency, triad_indices),
        )
        if core_indices.size:
            core_distressed_fraction = float(
                np.mean(distressed[core_indices] >= contamination_threshold)
            )
            tracker["peak_core_contamination"] = max(
                tracker["peak_core_contamination"],
                core_distressed_fraction,
            )
            tracker["last_core_trust"] = float(np.mean(trust_scores[core_indices]))

        if age >= tracking_window:
            tracker["finalized"] = True


def choose_phase_manipulation_target(
    state: dict[str, np.ndarray],
    effective_adjacency: np.ndarray,
    effective_weights: np.ndarray,
    trust_scores: np.ndarray,
    memory_support: np.ndarray,
    disturbance: np.ndarray,
    current_primary: set[int],
    config: Any,
) -> dict[str, Any] | None:
    if not getattr(config, "phase_manipulation_enabled", False):
        return None

    node_count = int(state["theta"].shape[0])
    degrees = effective_adjacency.sum(axis=1).astype(float)
    mean_degree = max(float(np.mean(degrees)), 1e-6)
    core_indices = (
        np.array(sorted(current_primary), dtype=int)
        if current_primary
        else np.array([], dtype=int)
    )

    candidate_mask = ~state["false_mask"]
    candidate_mask &= trust_scores >= float(
        getattr(config, "phase_manipulation_trust_min", 0.45)
    )
    candidate_mask &= trust_scores <= float(
        getattr(config, "phase_manipulation_trust_max", 0.72)
    )
    candidate_mask &= degrees >= mean_degree * float(
        getattr(config, "phase_manipulation_degree_min_ratio", 0.35)
    )
    candidate_mask &= degrees <= mean_degree * float(
        getattr(config, "phase_manipulation_degree_max_ratio", 1.35)
    )
    candidate_mask &= memory_support <= float(
        getattr(config, "phase_manipulation_memory_ceiling", 0.35)
    )
    candidate_mask &= disturbance <= float(
        getattr(config, "phase_manipulation_disturbance_ceiling", 0.72)
    )
    if core_indices.size:
        candidate_mask[core_indices] = False

    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        return None

    best_payload: dict[str, Any] | None = None
    best_score = -np.inf
    trust_margin = float(getattr(config, "phase_manipulation_trust_margin", 0.08))

    for node in candidate_indices.tolist():
        higher = np.flatnonzero(
            effective_adjacency[node] & (trust_scores >= trust_scores[node] + trust_margin)
        )
        lower = np.flatnonzero(
            effective_adjacency[node] & (trust_scores <= trust_scores[node] - trust_margin)
        )
        if higher.size == 0 and lower.size == 0:
            continue

        higher_strength = (
            float(np.mean(effective_weights[node, higher])) if higher.size else 0.0
        )
        lower_strength = (
            float(np.mean(effective_weights[node, lower])) if lower.size else 0.0
        )
        total_strength = higher_strength + lower_strength
        balance = (
            2.0 * min(higher_strength, lower_strength) / max(total_strength, 1e-6)
            if total_strength > 0.0
            else 0.0
        )
        node_score = (
            0.34 * balance
            + 0.22 * np.clip(total_strength, 0.0, 1.5)
            + 0.18 * (1.0 - abs(trust_scores[node] - 0.58))
            + 0.14 * (1.0 - np.clip(memory_support[node], 0.0, 1.0))
            + 0.12 * (1.0 - np.clip(disturbance[node], 0.0, 1.0))
        )
        if node_score > best_score:
            best_score = node_score
            best_payload = {
                "node_index": int(node),
                "higher_indices": higher.astype(int),
                "lower_indices": lower.astype(int),
                "target_trust": float(trust_scores[node]),
                "balance": float(balance),
                "higher_strength": float(higher_strength),
                "lower_strength": float(lower_strength),
                "score": float(node_score),
                "core_indices": core_indices.copy(),
            }

    return best_payload


def apply_phase_manipulation_transform(
    state: dict[str, np.ndarray],
    tracker: dict[str, Any],
    config: Any,
    scale: float = 1.0,
) -> None:
    if scale <= 0.0:
        return

    idx = int(tracker["node_index"])
    phase_blend = float(getattr(config, "phase_manipulation_phase_blend", 0.42)) * scale
    freq_blend = float(getattr(config, "phase_manipulation_freq_blend", 0.24)) * scale

    phase_error = float(
        np.angle(np.exp(1j * (tracker["target_phase"] - state["theta"][idx])))
    )
    state["theta"][idx] = (state["theta"][idx] + phase_blend * phase_error) % (2.0 * np.pi)
    state["freq"][idx] = np.clip(
        (1.0 - freq_blend) * state["freq"][idx] + freq_blend * tracker["target_freq"],
        0.5,
        1.5,
    )

    geom_scale = float(getattr(config, "phase_manipulation_geometry_shift", 0.08)) * scale
    if "pos" in state:
        state["pos"][idx, 0] = (state["pos"][idx, 0] + geom_scale) % 1.0
        if "vel" in state:
            state["vel"][idx, 0] += 0.4 * geom_scale
    if "helix_t" in state:
        delta_t = float(getattr(config, "phase_manipulation_helix_shift", 0.18)) * np.pi * scale
        state["helix_t"][idx] = np.clip(
            state["helix_t"][idx] + delta_t,
            0.0,
            config.helix_turns * 2.0 * np.pi,
        )
        if "helix_vel" in state:
            state["helix_vel"][idx] += 0.35 * delta_t


def update_phase_manipulation_trackers(
    trackers: list[dict[str, Any]],
    state: dict[str, np.ndarray],
    effective_adjacency: np.ndarray,
    trust_scores: np.ndarray,
    distressed: np.ndarray,
    step: int,
    config: Any,
) -> None:
    tracking_window = int(getattr(config, "phase_manipulation_tracking_window", 120))
    align_threshold = float(getattr(config, "phase_manipulation_align_threshold", 0.30))
    freq_threshold = float(getattr(config, "phase_manipulation_freq_threshold", 0.035))
    recovery_phase_tol = float(
        getattr(config, "phase_manipulation_recovery_phase_tol", 0.20)
    )
    recovery_freq_tol = float(
        getattr(config, "phase_manipulation_recovery_freq_tol", 0.015)
    )
    contamination_threshold = float(
        getattr(config, "phase_manipulation_core_disturbance_threshold", 0.42)
    )

    for tracker in trackers:
        if tracker.get("finalized", False) or step < tracker["trigger_step"]:
            continue
        idx = int(tracker["node_index"])
        age = step - tracker["trigger_step"]
        phase_gap = np.abs(np.angle(np.exp(1j * (state["theta"] - tracker["target_phase"]))))
        freq_gap = np.abs(state["freq"] - tracker["target_freq"])
        influenced = (phase_gap <= align_threshold) & (freq_gap <= freq_threshold)

        if tracker["higher_indices"].size:
            tracker["max_upward_fraction"] = max(
                tracker["max_upward_fraction"],
                float(np.mean(influenced[tracker["higher_indices"]])),
            )
        if tracker["lower_indices"].size:
            tracker["max_downward_fraction"] = max(
                tracker["max_downward_fraction"],
                float(np.mean(influenced[tracker["lower_indices"]])),
            )

        target_phase_error = float(
            np.abs(np.angle(np.exp(1j * (state["theta"][idx] - tracker["pre_phase"]))))
        )
        target_freq_error = float(np.abs(state["freq"][idx] - tracker["pre_freq"]))
        if (
            tracker["recovered_step"] is None
            and target_phase_error <= recovery_phase_tol
            and target_freq_error <= recovery_freq_tol
        ):
            tracker["recovered_step"] = age

        if tracker["core_indices"].size:
            core_distressed_fraction = float(
                np.mean(distressed[tracker["core_indices"]] >= contamination_threshold)
            )
            tracker["peak_core_contamination"] = max(
                tracker["peak_core_contamination"],
                core_distressed_fraction,
            )
            tracker["last_core_trust"] = float(np.mean(trust_scores[tracker["core_indices"]]))

        if age >= tracking_window:
            tracker["finalized"] = True


def choose_guardian_target(
    stored_prototypes: list[set[int]],
    trust_scores: np.ndarray,
    core_memory_matrix: np.ndarray,
    config: PhaseVIConfig,
) -> set[int]:
    if stored_prototypes:
        def prototype_score(prototype: set[int]) -> float:
            indices = np.array(sorted(prototype), dtype=int)
            core_mass = float(np.mean(core_memory_matrix[np.ix_(indices, indices)]))
            return (
                0.55 * float(np.mean(trust_scores[indices]))
                + 0.30 * (len(prototype) / config.nodes)
                + 0.15 * core_mass
            )

        return set(max(stored_prototypes, key=prototype_score))

    guardian_count = max(1, int(round(config.nodes * config.guardian_trust_fraction)))
    indices = np.argsort(trust_scores)[-guardian_count:]
    return set(indices.tolist())


def choose_controller_mode(
    current_mode: str,
    steps_in_mode: int,
    scaffold_enabled: bool,
    coherence_ratio: float,
    cluster_ratio: float,
    distressed_ratio: float,
    config: Any,
    policy: dict[str, float] | None = None,
) -> str:
    if not getattr(config, "controller_enabled", False):
        return current_mode

    critical = float(
        (policy or {}).get(
            "coherence_critical",
            getattr(config, "controller_coherence_critical", 0.78),
        )
    )
    release = float(
        (policy or {}).get(
            "coherence_release",
            getattr(config, "controller_coherence_release", 0.94),
        )
    )
    cluster_priority = float(
        (policy or {}).get(
            "cluster_priority",
            getattr(config, "controller_cluster_priority", 0.86),
        )
    )
    cluster_release = float(
        (policy or {}).get(
            "cluster_release",
            getattr(config, "controller_cluster_release", 0.96),
        )
    )
    scaffold_coherence_floor = float(
        (policy or {}).get(
            "scaffold_coherence_floor",
            getattr(config, "controller_scaffold_coherence_floor", 0.82),
        )
    )
    disturbance_priority = float(
        (policy or {}).get(
            "disturbance_priority",
            getattr(config, "controller_disturbance_priority", 0.22),
        )
    )
    min_dwell = int((policy or {}).get("min_dwell", getattr(config, "controller_min_dwell", 18)))

    if current_mode == "absorber":
        if steps_in_mode < min_dwell and (
            coherence_ratio < release or distressed_ratio > disturbance_priority
        ):
            return "absorber"
        if scaffold_enabled and cluster_ratio < cluster_priority and coherence_ratio >= scaffold_coherence_floor:
            return "scaffold"
        if coherence_ratio < release or distressed_ratio > disturbance_priority:
            return "absorber"
        return "observe"

    if current_mode == "scaffold":
        if steps_in_mode < min_dwell and cluster_ratio < cluster_release:
            return "scaffold"
        if coherence_ratio < critical or distressed_ratio > disturbance_priority:
            return "absorber"
        if scaffold_enabled and cluster_ratio < cluster_release and coherence_ratio >= scaffold_coherence_floor:
            return "scaffold"
        return "observe"

    if coherence_ratio < critical or distressed_ratio > disturbance_priority:
        return "absorber"
    if scaffold_enabled and cluster_ratio < cluster_priority and coherence_ratio >= scaffold_coherence_floor:
        return "scaffold"
    return "observe"


def build_controller_policy(config: Any) -> dict[str, float]:
    return {
        "coherence_critical": float(getattr(config, "controller_coherence_critical", 0.78)),
        "coherence_release": float(getattr(config, "controller_coherence_release", 0.94)),
        "cluster_priority": float(getattr(config, "controller_cluster_priority", 0.86)),
        "cluster_release": float(getattr(config, "controller_cluster_release", 0.96)),
        "scaffold_coherence_floor": float(
            getattr(config, "controller_scaffold_coherence_floor", 0.82)
        ),
        "disturbance_priority": float(
            getattr(config, "controller_disturbance_priority", 0.22)
        ),
        "min_dwell": float(getattr(config, "controller_min_dwell", 18)),
    }


def update_controller_policy(
    policy: dict[str, float],
    coherence_ratio: float,
    cluster_ratio: float,
    distressed_ratio: float,
    mode: str,
    config: Any,
) -> None:
    if not getattr(config, "policy_learning_enabled", False):
        return

    lr = float(getattr(config, "policy_learning_rate", 0.025))
    restore = float(getattr(config, "policy_restore_rate", 0.010))
    base = build_controller_policy(config)

    # Strong coherence failure: bias toward absorber sooner.
    if coherence_ratio < 0.92:
        policy["coherence_critical"] = min(
            float(getattr(config, "policy_coherence_critical_max", 0.95)),
            policy["coherence_critical"] + lr * (0.92 - coherence_ratio),
        )
        policy["coherence_release"] = min(
            float(getattr(config, "policy_coherence_release_max", 1.05)),
            policy["coherence_release"] + 0.5 * lr * (0.92 - coherence_ratio),
        )
        policy["disturbance_priority"] = max(
            float(getattr(config, "policy_disturbance_min", 0.12)),
            policy["disturbance_priority"] - 0.5 * lr * max(0.0, distressed_ratio - 0.18),
        )

    # Weak structural regrowth while coherence is acceptable: bias toward scaffold.
    if cluster_ratio < 0.90 and coherence_ratio >= 0.88:
        policy["cluster_priority"] = min(
            float(getattr(config, "policy_cluster_priority_max", 1.02)),
            policy["cluster_priority"] + lr * (0.90 - cluster_ratio),
        )
        policy["cluster_release"] = min(
            float(getattr(config, "policy_cluster_release_max", 1.06)),
            policy["cluster_release"] + 0.5 * lr * (0.90 - cluster_ratio),
        )
        policy["scaffold_coherence_floor"] = max(
            float(getattr(config, "policy_scaffold_floor_min", 0.76)),
            policy["scaffold_coherence_floor"] - 0.5 * lr * (0.90 - cluster_ratio),
        )

    # When both are healthy, drift back toward simpler defaults.
    if coherence_ratio >= 1.0 and cluster_ratio >= 1.0 and distressed_ratio < 0.16:
        for key in (
            "coherence_critical",
            "coherence_release",
            "cluster_priority",
            "cluster_release",
            "scaffold_coherence_floor",
            "disturbance_priority",
        ):
            policy[key] += restore * (base[key] - policy[key])

    # Gentle anti-thrashing if controller keeps switching.
    if mode == "observe" and coherence_ratio >= 0.98 and cluster_ratio >= 0.95:
        policy["min_dwell"] = max(
            float(getattr(config, "policy_min_dwell_min", 10)),
            policy["min_dwell"] - restore,
        )
    else:
        policy["min_dwell"] = min(
            float(getattr(config, "policy_min_dwell_max", 28)),
            policy["min_dwell"] + 0.25 * restore,
        )


def build_reward_controller_state(config: Any, scaffold_enabled: bool) -> dict[str, Any]:
    return {
        "values": {
            "observe": float(getattr(config, "reward_initial_observe_value", 0.08)),
            "absorber": float(getattr(config, "reward_initial_absorber_value", 0.02)),
            "scaffold": float(
                getattr(config, "reward_initial_scaffold_value", 0.04)
                if scaffold_enabled
                else getattr(config, "reward_disabled_scaffold_value", -0.30)
            ),
        },
        "counts": {"observe": 0, "absorber": 0, "scaffold": 0},
        "pending_action": None,
        "pending_snapshot": None,
        "reward_total": 0.0,
        "reward_updates": 0,
    }


def settle_reward_controller(
    reward_state: dict[str, Any],
    coherence_ratio: float,
    cluster_ratio: float,
    distressed_ratio: float,
    trust_mean: float,
    config: Any,
) -> None:
    pending_action = reward_state.get("pending_action")
    pending_snapshot = reward_state.get("pending_snapshot")
    if pending_action is None or pending_snapshot is None:
        return

    coherence_gain = coherence_ratio - float(pending_snapshot["coherence_ratio"])
    cluster_gain = cluster_ratio - float(pending_snapshot["cluster_ratio"])
    disturbance_gain = float(pending_snapshot["distressed_ratio"]) - distressed_ratio
    trust_gain = trust_mean - float(pending_snapshot["trust_mean"])

    reward = (
        float(getattr(config, "reward_coherence_gain", 2.40)) * coherence_gain
        + float(getattr(config, "reward_cluster_gain", 1.80)) * cluster_gain
        + float(getattr(config, "reward_disturbance_gain", 1.20)) * disturbance_gain
        + float(getattr(config, "reward_trust_gain", 0.80)) * trust_gain
    )

    if pending_action == "observe":
        reward -= float(getattr(config, "reward_observe_cost", 0.00))
        if (
            coherence_ratio < float(getattr(config, "reward_observe_coherence_floor", 0.96))
            or cluster_ratio < float(getattr(config, "reward_observe_cluster_floor", 0.92))
            or distressed_ratio > float(getattr(config, "reward_observe_disturbance_ceiling", 0.18))
        ):
            reward -= float(getattr(config, "reward_idle_penalty", 0.22))
    elif pending_action == "absorber":
        reward -= float(getattr(config, "reward_absorber_cost", 0.05))
        if disturbance_gain > 0.0:
            reward += 0.20 * disturbance_gain
    elif pending_action == "scaffold":
        reward -= float(getattr(config, "reward_scaffold_cost", 0.06))
        if cluster_gain < 0.0:
            reward += 0.35 * cluster_gain
        if coherence_ratio < float(getattr(config, "reward_scaffold_coherence_floor", 0.84)):
            reward -= 0.12

    clip = float(getattr(config, "reward_clip", 2.50))
    reward = float(np.clip(reward, -clip, clip))

    counts = reward_state["counts"]
    values = reward_state["values"]
    count = counts[pending_action]
    alpha = float(getattr(config, "reward_learning_rate", 0.28)) / np.sqrt(count + 1.0)
    values[pending_action] += alpha * (reward - values[pending_action])
    counts[pending_action] += 1
    reward_state["reward_total"] += reward
    reward_state["reward_updates"] += 1
    reward_state["pending_action"] = None
    reward_state["pending_snapshot"] = None


def prime_reward_controller(
    reward_state: dict[str, Any],
    action: str,
    coherence_ratio: float,
    cluster_ratio: float,
    distressed_ratio: float,
    trust_mean: float,
) -> None:
    reward_state["pending_action"] = action
    reward_state["pending_snapshot"] = {
        "coherence_ratio": float(coherence_ratio),
        "cluster_ratio": float(cluster_ratio),
        "distressed_ratio": float(distressed_ratio),
        "trust_mean": float(trust_mean),
    }


def choose_reward_controller_mode(
    current_mode: str,
    steps_in_mode: int,
    scaffold_enabled: bool,
    coherence_ratio: float,
    cluster_ratio: float,
    distressed_ratio: float,
    reward_state: dict[str, Any],
    config: Any,
) -> str:
    valid_actions = ["observe", "absorber"]
    if scaffold_enabled:
        valid_actions.append("scaffold")

    values = reward_state["values"]
    counts = reward_state["counts"]
    total_updates = max(1, int(reward_state.get("reward_updates", 0)))
    prior_scale = 1.0 / np.sqrt(1.0 + 0.03 * total_updates)

    coherence_deficit = max(0.0, 1.0 - coherence_ratio)
    cluster_deficit = max(0.0, 1.0 - cluster_ratio)
    healthy_margin = max(0.0, min(coherence_ratio - 0.97, cluster_ratio - 0.95))
    observe_disturbance_ceiling = float(
        getattr(config, "reward_observe_disturbance_ceiling", 0.18)
    )
    mild_coherence_gap = max(0.0, coherence_deficit - 0.03)
    urgent_coherence_gap = max(0.0, coherence_deficit - 0.08)
    mild_cluster_gap = max(0.0, cluster_deficit - 0.04)
    urgent_cluster_gap = max(0.0, cluster_deficit - 0.10)
    disturbance_pressure = max(0.0, distressed_ratio - observe_disturbance_ceiling)
    scaffold_floor = float(getattr(config, "reward_scaffold_coherence_floor", 0.84))
    observe_coherence_floor = float(getattr(config, "reward_observe_coherence_floor", 0.96))
    observe_cluster_floor = float(getattr(config, "reward_observe_cluster_floor", 0.92))
    min_dwell = int(getattr(config, "reward_min_dwell", 12))
    switch_cost = float(getattr(config, "reward_switch_cost", 0.10))
    ucb_scale = float(getattr(config, "reward_ucb_scale", 0.14))

    scores: dict[str, float] = {}
    for action in valid_actions:
        explore = ucb_scale * np.sqrt(np.log(total_updates + 2.0) / (counts[action] + 1.0))
        score = float(values[action]) + float(explore)

        if action == "observe":
            score += prior_scale * float(getattr(config, "reward_observe_prior_gain", 0.70)) * (
                1.20 * healthy_margin
                - 0.55 * mild_coherence_gap
                - 0.35 * mild_cluster_gap
                - 0.65 * disturbance_pressure
            )
            if (
                coherence_ratio < observe_coherence_floor
                or cluster_ratio < observe_cluster_floor
                or distressed_ratio > observe_disturbance_ceiling
            ):
                score -= float(getattr(config, "reward_observe_penalty", 0.32))
        elif action == "absorber":
            score += prior_scale * float(getattr(config, "reward_absorber_prior_gain", 0.92)) * (
                1.55 * urgent_coherence_gap
                + 1.35 * max(0.0, distressed_ratio - 0.22)
                + 0.15 * mild_cluster_gap
            )
            if coherence_ratio > 0.94 and cluster_ratio > 0.92 and distressed_ratio < 0.24:
                score -= float(getattr(config, "reward_absorber_overuse_penalty", 0.18))
        elif action == "scaffold":
            score += prior_scale * float(getattr(config, "reward_scaffold_prior_gain", 0.82)) * (
                1.55 * urgent_cluster_gap
                + 0.20 * max(0.0, coherence_ratio - scaffold_floor)
                - 0.20 * max(0.0, distressed_ratio - 0.28)
            )
            if coherence_ratio < scaffold_floor:
                score -= float(getattr(config, "reward_scaffold_penalty", 0.45))
            if cluster_ratio > 0.98:
                score -= 0.12

        if action != current_mode and steps_in_mode < min_dwell:
            score -= switch_cost * (min_dwell - steps_in_mode) / max(1, min_dwell)

        scores[action] = score

    return max(valid_actions, key=lambda action: scores[action])


def render_grouped_metric(
    summary_rows: list[dict[str, float | str]],
    output_path: Path,
    metric_key: str,
    title: str,
    model_labels: dict[str, str],
) -> None:
    row_map = {
        (str(row["model_kind"]), str(row["bootstrap_mode"]), str(row["false_mode"])): row
        for row in summary_rows
    }
    model_order = ("baseline", "triangle", "helix", "hybrid_manifold")
    false_modes = ("none", "random", "noisy")
    bootstrap_modes = ("single", "periodic")
    x = np.arange(len(false_modes))
    width = 0.34

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    for axis, model_kind in zip(axes.flat, model_order):
        for offset, bootstrap_mode in enumerate(bootstrap_modes):
            means = []
            stds = []
            for false_mode in false_modes:
                row = row_map.get((model_kind, bootstrap_mode, false_mode))
                if row is None:
                    means.append(np.nan)
                    stds.append(0.0)
                    continue
                means.append(float(row[f"{metric_key}_mean"]))
                stds.append(float(row[f"{metric_key}_std"]))
            axis.bar(
                x + (offset - 0.5) * width,
                means,
                width=width,
                yerr=stds,
                capsize=4,
                label=bootstrap_mode if model_kind == "baseline" else None,
                alpha=0.88,
            )
        axis.set_title(model_labels.get(model_kind, model_kind))
        axis.set_xticks(x)
        axis.set_xticklabels(false_modes)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(bottom=0.0)
    axes[0, 0].legend(title="bootstrap", loc="upper left")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_recovery_timeline(
    scenarios: list[PhaseVIScenario],
    timeline_means: dict[str, np.ndarray],
    config: PhaseVIConfig,
    output_path: Path,
    model_labels: dict[str, str],
    experiment_tag: str,
) -> None:
    model_order = ("baseline", "triangle", "helix", "hybrid_manifold")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    shock_step = min(config.shock_step, max(1, config.steps // 2))
    x = np.arange(-shock_step, config.steps - shock_step)

    for axis, model_kind in zip(axes.flat, model_order):
        for scenario in [item for item in scenarios if item.model_kind == model_kind]:
            axis.plot(
                x,
                timeline_means[scenario.name],
                linewidth=1.35,
                label=f"{scenario.bootstrap_mode}/{scenario.false_mode}",
            )
        axis.axvline(0, color="#222222", linestyle="--", linewidth=1.0)
        axis.set_title(model_labels.get(model_kind, model_kind))
        axis.set_xlabel("steps from shock")
        axis.set_ylabel("coherence")
        axis.grid(alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)

    fig.suptitle(f"{experiment_tag} Shock Response", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_runs(
    scenarios: list[PhaseVIScenario],
    run_metrics: list[PhaseVIRunMetrics],
) -> list[dict[str, float | str]]:
    grouped: dict[str, list[PhaseVIRunMetrics]] = {}
    for item in run_metrics:
        grouped.setdefault(item.scenario, []).append(item)

    fields = [
        "pre_shock_coherence",
        "shock_min_coherence",
        "late_recovery_coherence",
        "coherence_recovery_ratio",
        "pre_shock_cluster_fraction",
        "shock_min_cluster_fraction",
        "late_cluster_fraction",
        "cluster_recovery_ratio",
        "final_false_isolation_rate",
        "recovery_time_steps",
        "recovery_success",
        "pre_shock_memory_mass",
        "late_memory_mass",
        "memory_recovery_ratio",
        "pre_shock_core_memory_mass",
        "late_core_memory_mass",
        "core_memory_recovery_ratio",
        "topological_memory_index",
        "recall_event_rate",
        "guardian_event_rate",
        "absorber_event_rate",
        "scaffold_event_rate",
        "modulation_event_rate",
        "modulation_target_fraction",
        "modulation_mean_ki",
        "scaffold_edge_mass",
        "controller_absorber_rate",
        "controller_scaffold_rate",
        "controller_observe_rate",
        "controller_switch_rate",
        "controller_reward_mean",
        "learned_coherence_critical",
        "learned_cluster_priority",
        "learned_disturbance_priority",
        "learned_action_value_observe",
        "learned_action_value_absorber",
        "learned_action_value_scaffold",
        "resonance_energy_late",
        "late_trust_mean",
        "inverse_triad_event_rate",
        "inverse_triad_mean_anchor_trust",
        "inverse_triad_mean_bridge_strength",
        "inverse_triad_mean_shell_score",
        "inverse_triad_propagation_fraction",
        "inverse_triad_boundary_recovery_steps",
        "inverse_triad_core_stability_delta",
        "inverse_triad_core_contamination_rate",
        "phase_manipulation_event_rate",
        "phase_manipulation_mean_target_trust",
        "phase_manipulation_mean_bidirectional_balance",
        "phase_manipulation_upward_spread_fraction",
        "phase_manipulation_downward_spread_fraction",
        "phase_manipulation_recovery_steps",
        "phase_manipulation_core_stability_delta",
        "phase_manipulation_core_contamination_rate",
        "quantized_scalar_anchor_gap_late",
        "quantized_scalar_lock_fraction_late",
        "quantized_repulsive_fraction_late",
        "quantized_vector_drive_mean_late",
    ]

    rows: list[dict[str, float | str]] = []
    for scenario in scenarios:
        items = grouped.get(scenario.name, [])
        if not items:
            continue
        row: dict[str, float | str] = {
            "scenario": scenario.name,
            "label": scenario.label,
            "model_kind": scenario.model_kind,
            "bootstrap_mode": scenario.bootstrap_mode,
            "false_mode": scenario.false_mode,
            "runs": len(items),
        }
        for field in fields:
            values = np.array([float(getattr(item, field)) for item in items], dtype=float)
            row[f"{field}_mean"] = float(np.mean(values))
            row[f"{field}_std"] = float(np.std(values))
        rows.append(row)
    return rows


def load_summary_csv(path: Path) -> dict[str, dict[str, float | str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return {str(row["scenario"]): row for row in rows}


def build_phase_comparison(
    phase5_summary_path: Path,
    phase6_summary_rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    phase5_map = load_summary_csv(phase5_summary_path)
    if not phase5_map:
        return []

    compare_fields = [
        "coherence_recovery_ratio",
        "cluster_recovery_ratio",
        "memory_recovery_ratio",
        "final_false_isolation_rate",
        "recovery_success",
        "recovery_time_steps",
        "late_recovery_coherence",
    ]

    rows: list[dict[str, float | str]] = []
    for row in phase6_summary_rows:
        phase5_row = phase5_map.get(str(row["scenario"]))
        if not phase5_row:
            continue
        delta_row: dict[str, float | str] = {
            "scenario": row["scenario"],
            "model_kind": row["model_kind"],
            "bootstrap_mode": row["bootstrap_mode"],
            "false_mode": row["false_mode"],
        }
        for field in compare_fields:
            phase6_value = float(row[f"{field}_mean"])
            phase5_value = float(phase5_row[f"{field}_mean"])
            delta_row[f"{field}_phase5"] = phase5_value
            delta_row[f"{field}_phase6"] = phase6_value
            delta_row[f"{field}_delta"] = phase6_value - phase5_value
        rows.append(delta_row)
    return rows


def checkpoint_phase6_progress(
    *,
    output_dir: Path,
    config: PhaseVIConfig,
    scenarios: list[PhaseVIScenario],
    runs: int,
    experiment_tag: str,
    baseline_summary_path: Path | None,
    comparison_filename: str,
    comparison_json_key: str,
    run_metrics: list[PhaseVIRunMetrics],
    shock_rows: list[dict[str, Any]],
    guardian_rows: list[dict[str, Any]],
    absorber_rows: list[dict[str, Any]],
    scaffold_rows: list[dict[str, Any]],
    modulation_rows: list[dict[str, Any]],
    inverse_triad_rows: list[dict[str, Any]],
    phase_manipulation_rows: list[dict[str, Any]],
    completed_count: int,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]], list[dict[str, Any]]]:
    completed_scenarios = scenarios[:completed_count]
    run_rows = [asdict(item) for item in run_metrics]
    summary_rows = summarize_runs(completed_scenarios, run_metrics)
    comparison_rows = build_phase_comparison(
        baseline_summary_path or Path("/mnt/d/resonance_phase5_outputs/scenario_summary.csv"),
        summary_rows,
    )

    save_csv(output_dir / "recovery_metrics.csv", run_rows)
    save_csv(output_dir / "scenario_summary.csv", summary_rows)
    save_csv(output_dir / comparison_filename, comparison_rows)
    save_json(
        output_dir / "scenario_summary.json",
        {
            "config": asdict(config),
            "runs_per_scenario": runs,
            "scenarios": [asdict(item) for item in completed_scenarios],
            "completed_scenarios": completed_count,
            "total_scenarios": len(scenarios),
            "is_partial_checkpoint": completed_count < len(scenarios),
            "run_metrics": run_rows,
            "shock_events": shock_rows,
            "guardian_events": guardian_rows,
            "absorber_events": absorber_rows,
            "scaffold_events": scaffold_rows,
            "modulation_events": modulation_rows,
            "inverse_triad_events": inverse_triad_rows,
            "phase_manipulation_events": phase_manipulation_rows,
            "summary": summary_rows,
            comparison_json_key: comparison_rows,
            "experiment_tag": experiment_tag,
        },
    )
    return summary_rows, comparison_rows, run_rows


def run_single_phase6(
    config: PhaseVIConfig,
    scenario: PhaseVIScenario,
    run_index: int,
    seed: int,
) -> tuple[
    PhaseVIRunMetrics,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    np.ndarray,
]:
    shock_step = min(config.shock_step, max(1, config.steps // 2))
    model = PHASE4_MODEL_REGISTRY[scenario.model_kind]
    compute_backend = resolve_numeric_backend(config)
    scaffold_enabled = hasattr(config, "scaffold_activation_threshold") and (
        not hasattr(config, "scaffold_model_kinds")
        or scenario.model_kind in getattr(config, "scaffold_model_kinds")
    )
    state = initialize_state(config, scenario, seed)
    rng = np.random.default_rng(seed + 1)

    memory_matrix = np.zeros((config.nodes, config.nodes), dtype=float)
    core_memory_matrix = np.zeros((config.nodes, config.nodes), dtype=float)
    scaffold_matrix = np.zeros((config.nodes, config.nodes), dtype=float)
    trust_scores = np.ones(config.nodes, dtype=float)
    recall_cooldown = 0
    guardian_cooldown = 0
    recall_events = 0
    guardian_events = 0
    absorber_events = 0
    scaffold_events = 0
    modulation_events = 0
    controller_absorber_steps = 0
    controller_scaffold_steps = 0
    controller_observe_steps = 0
    controller_switches = 0
    memory_supported_samples = 0
    stored_prototypes: list[set[int]] = []
    previous_primary = set()
    current_primary: set[int] = set()
    current_cluster_fraction = 0.0
    pre_shock_reference_coherence: float | None = None
    pre_shock_reference_cluster: float | None = None
    absorber_cooldown = 0
    scaffold_cooldown = 0
    modulation_cooldown = 0
    inverse_triad_cooldown = 0
    phase_manipulation_cooldown = 0
    controller_mode = "observe"
    controller_mode_start = shock_step
    controller_policy = build_controller_policy(config)
    reward_controller = build_reward_controller_state(config, scaffold_enabled)
    cluster_refresh_interval = max(1, int(getattr(config, "cluster_refresh_interval", 1)))
    controller_interval = max(1, int(getattr(config, "controller_interval", 1)))
    collect_event_rows = bool(getattr(config, "collect_event_rows", True))

    coherence_history = np.zeros(config.steps, dtype=float)
    cluster_fraction_history = np.zeros(config.steps, dtype=float)
    memory_mass_history = np.zeros(config.steps, dtype=float)
    core_memory_mass_history = np.zeros(config.steps, dtype=float)
    scaffold_mass_history = np.zeros(config.steps, dtype=float)
    energy_history = np.zeros(config.steps, dtype=float)
    trust_mean_history = np.zeros(config.steps, dtype=float)
    quantized_anchor_gap_history = np.zeros(config.steps, dtype=float)
    quantized_lock_fraction_history = np.zeros(config.steps, dtype=float)
    quantized_repulsive_fraction_history = np.zeros(config.steps, dtype=float)
    quantized_vector_drive_history = np.zeros(config.steps, dtype=float)

    shock_rows: list[dict[str, Any]] = []
    guardian_rows: list[dict[str, Any]] = []
    absorber_rows: list[dict[str, Any]] = []
    scaffold_rows: list[dict[str, Any]] = []
    modulation_rows: list[dict[str, Any]] = []
    inverse_triad_rows: list[dict[str, Any]] = []
    phase_manipulation_rows: list[dict[str, Any]] = []
    phase_manipulation_rows: list[dict[str, Any]] = []
    final_adjacency = np.zeros((config.nodes, config.nodes), dtype=bool)
    modulation_target_total = 0
    modulation_ki_total = 0.0
    inverse_triad_events = 0
    inverse_triad_anchor_trust_total = 0.0
    inverse_triad_bridge_total = 0.0
    inverse_triad_shell_total = 0.0
    inverse_triad_trackers: list[dict[str, Any]] = []
    inverse_triad_max_events = int(getattr(config, "inverse_triad_max_events", 1))
    phase_manipulation_events = 0
    phase_manipulation_target_trust_total = 0.0
    phase_manipulation_balance_total = 0.0
    phase_manipulation_trackers: list[dict[str, Any]] = []
    phase_manipulation_max_events = int(getattr(config, "phase_manipulation_max_events", 1))

    for step in range(config.steps):
        memory_matrix *= config.memory_decay
        core_memory_matrix *= config.core_memory_decay
        if scaffold_enabled:
            scaffold_matrix *= config.scaffold_decay
        np.fill_diagonal(memory_matrix, 0.0)
        np.fill_diagonal(core_memory_matrix, 0.0)
        np.fill_diagonal(scaffold_matrix, 0.0)

        if inverse_triad_cooldown > 0:
            inverse_triad_cooldown -= 1
        if getattr(config, "inverse_triad_enabled", False):
            geom_steps = int(getattr(config, "inverse_triad_geometry_steps", 12))
            relax_steps = int(getattr(config, "inverse_triad_relax_steps", 36))
            total_steps = geom_steps + relax_steps
            for tracker in inverse_triad_trackers:
                age = step - tracker["trigger_step"]
                if age < 0 or age >= total_steps:
                    continue
                if age < geom_steps:
                    scale = 1.0
                else:
                    scale = max(0.0, 1.0 - (age - geom_steps) / max(1, relax_steps))
                apply_inverse_triad_transform(
                    state,
                    tracker["triad_indices"],
                    config,
                    scale=scale,
                )
        if phase_manipulation_cooldown > 0:
            phase_manipulation_cooldown -= 1
        if getattr(config, "phase_manipulation_enabled", False):
            geom_steps = int(getattr(config, "phase_manipulation_geometry_steps", 16))
            relax_steps = int(getattr(config, "phase_manipulation_relax_steps", 48))
            total_steps = geom_steps + relax_steps
            for tracker in phase_manipulation_trackers:
                age = step - tracker["trigger_step"]
                if age < 0 or age >= total_steps:
                    continue
                if age < geom_steps:
                    scale = 1.0
                else:
                    scale = max(0.0, 1.0 - (age - geom_steps) / max(1, relax_steps))
                apply_phase_manipulation_transform(
                    state,
                    tracker,
                    config,
                    scale=scale,
                )

        if step == shock_step:
            if shock_step > 0:
                start = max(0, shock_step - config.shock_window)
                pre_shock_reference_coherence = float(np.mean(coherence_history[start:shock_step]))
                pre_shock_reference_cluster = float(
                    np.mean(cluster_fraction_history[start:shock_step])
                )
            shock_info = apply_shock(state, memory_matrix, scenario.model_kind, config, rng)
            core_memory_matrix *= config.shock_core_retention
            if scaffold_enabled:
                scaffold_matrix *= config.shock_scaffold_retention
            shock_rows.append(
                {
                    "scenario": scenario.name,
                    "model_kind": scenario.model_kind,
                    "bootstrap_mode": scenario.bootstrap_mode,
                    "false_mode": scenario.false_mode,
                    "run_index": run_index,
                    "seed": seed,
                    "shock_step": step,
                    **shock_info,
                    "core_memory_retention": config.shock_core_retention,
                    "scaffold_retention": float(
                        config.shock_scaffold_retention if scaffold_enabled else 0.0
                    ),
                }
            )

        if (
            scenario.bootstrap_mode == "periodic"
            and step > 0
            and step % config.bootstrap_period == 0
        ):
            state["theta"] = bootstrap_kick(
                state["theta"],
                config.bootstrap_anchor,
                config.bootstrap_strength,
            )

        if scenario.false_mode == "random" and np.any(state["false_mask"]):
            state["freq"][state["false_mask"]] += rng.normal(
                loc=0.0,
                scale=config.false_frequency_drift_sigma * np.sqrt(config.dt),
                size=int(np.sum(state["false_mask"])),
            )
            state["freq"][state["false_mask"]] = np.clip(
                state["freq"][state["false_mask"]],
                0.5,
                1.5,
            )

        base_layer = model.base_layer(state, config, backend=compute_backend)
        phase_state = phase_kernels(state["theta"], compute_backend)
        phase_diff = phase_state.phase_diff
        phase_alignment = phase_state.positive_alignment

        memory_mask = (
            (memory_matrix > config.memory_activation_threshold)
            & (base_layer.pair_distance < config.memory_radius_factor * config.radius)
        )
        core_memory_mask = (
            (core_memory_matrix > config.core_memory_activation_threshold)
            & (base_layer.pair_distance < config.core_memory_radius_factor * config.radius)
        )
        scaffold_mask = np.zeros_like(memory_mask)
        if scaffold_enabled:
            scaffold_mask = (
                (scaffold_matrix > config.scaffold_activation_threshold)
                & (
                    base_layer.pair_distance
                    < getattr(config, "scaffold_radius_factor", 1.85) * config.radius
                )
            )
        np.fill_diagonal(memory_mask, False)
        np.fill_diagonal(core_memory_mask, False)
        np.fill_diagonal(scaffold_mask, False)

        pair_trust = np.sqrt(np.outer(trust_scores, trust_scores))
        quarantine_scale = np.ones(config.nodes, dtype=float)
        quarantine_scale[trust_scores < config.quarantine_threshold] = config.quarantine_strength
        pair_quarantine = np.sqrt(np.outer(quarantine_scale, quarantine_scale))

        alignment_gate = config.alignment_floor + (1.0 - config.alignment_floor) * phase_alignment
        trust_gate = 0.55 + 0.45 * pair_trust
        gated_base_weights = (
            base_layer.validated_weights * alignment_gate * trust_gate * pair_quarantine
        )

        memory_pair_gain = 0.70 + 0.30 * pair_trust
        effective_weights = gated_base_weights + (
            config.memory_influence * memory_matrix * memory_mask * memory_pair_gain
        ) + (
            config.core_memory_influence
            * core_memory_matrix
            * core_memory_mask
            * np.maximum(memory_pair_gain, 0.85)
        ) + (
            (config.scaffold_influence if scaffold_enabled else 0.0)
            * scaffold_matrix
            * scaffold_mask
            * np.maximum(memory_pair_gain, config.scaffold_trust_floor if scaffold_enabled else 0.72)
        )

        base_active = gated_base_weights > 0.12
        effective_adjacency = base_active | memory_mask | core_memory_mask | scaffold_mask

        phase_drive = config.phase_coupling * np.sum(
            effective_weights * phase_state.sin_phase,
            axis=1,
        )
        state["theta"] = (state["theta"] + (state["freq"] + phase_drive) * config.dt) % (
            2.0 * np.pi
        )

        if scenario.false_mode == "noisy" and np.any(state["false_mask"]):
            state["theta"][state["false_mask"]] += rng.normal(
                loc=0.0,
                scale=config.false_phase_noise_sigma * np.sqrt(config.dt),
                size=int(np.sum(state["false_mask"])),
            )
            state["theta"] %= 2.0 * np.pi

        phase_state = phase_kernels(state["theta"], compute_backend)
        phase_diff = phase_state.phase_diff
        model.apply_motion(
            state=state,
            effective_weights=effective_weights,
            phase_diff=phase_diff,
            pair_distance=base_layer.pair_distance,
            geometry_data=base_layer.geometry_data,
            config=config,
            backend=compute_backend,
        )

        coherence_history[step] = float(np.abs(np.mean(np.exp(1j * state["theta"]))))
        energy_history[step] = float(0.5 * np.sum(phase_state.cos_phase * effective_weights))
        memory_mass_history[step] = float(np.mean(np.triu(memory_matrix, k=1)))
        core_memory_mass_history[step] = float(np.mean(np.triu(core_memory_matrix, k=1)))
        scaffold_mass_history[step] = float(np.mean(np.triu(scaffold_matrix, k=1)))
        trust_mean_history[step] = float(np.mean(trust_scores))
        final_adjacency = effective_adjacency

        if recall_cooldown > 0:
            recall_cooldown -= 1
        if guardian_cooldown > 0:
            guardian_cooldown -= 1
        if absorber_cooldown > 0:
            absorber_cooldown -= 1
        if scaffold_cooldown > 0:
            scaffold_cooldown -= 1
        if modulation_cooldown > 0:
            modulation_cooldown -= 1

        should_refresh_clusters = (
            step == 0
            or step % cluster_refresh_interval == 0
            or (step + 1) % config.persistence_delta == 0
        )
        if should_refresh_clusters:
            current_clusters, _ = cluster_sets(effective_adjacency, config.cluster_threshold)
            current_primary = primary_cluster(current_clusters)
            current_cluster_fraction = (
                len(current_primary) / config.nodes if current_primary else 0.0
            )
        degrees = final_adjacency.sum(axis=1).astype(float)
        aligned_sum = np.sum(phase_state.positive_alignment * effective_adjacency, axis=1)
        local_alignment = np.divide(
            aligned_sum,
            degrees,
            out=np.zeros(config.nodes, dtype=float),
            where=degrees > 0.0,
        )
        degree_norm = max(float(np.max(degrees)), 1.0)
        degree_support = degrees / degree_norm
        memory_support = np.mean(memory_mask | core_memory_mask | scaffold_mask, axis=1)
        target_trust = 0.60 * local_alignment + 0.20 * degree_support + 0.20 * memory_support
        trust_scores = np.clip(
            config.trust_decay * trust_scores + config.trust_gain * target_trust,
            config.trust_floor,
            1.0,
        )

        coherence_ref = max(pre_shock_reference_coherence or coherence_history[step], 1e-6)
        cluster_ref = max(pre_shock_reference_cluster or current_cluster_fraction or 1e-6, 1e-6)
        emitter_core = choose_guardian_target(
            stored_prototypes,
            trust_scores,
            core_memory_matrix,
            config,
        )
        emitter_indices = np.array(sorted(emitter_core), dtype=int)
        emitter_phase = float(np.angle(np.mean(np.exp(1j * state["theta"][emitter_indices]))))
        phase_error = np.abs(np.angle(np.exp(1j * (state["theta"] - emitter_phase))))
        disturbance = (
            0.45 * (1.0 - trust_scores)
            + 0.35 * (1.0 - local_alignment)
            + 0.20 * np.clip(phase_error / np.pi, 0.0, 1.0)
        )
        distressed = disturbance > getattr(config, "absorber_disturbance_threshold", 0.42)
        distressed_ratio = float(np.mean(distressed))

        quantized_stats = {
            "anchor_gap_mean": 0.0,
            "lock_fraction": 0.0,
            "repulsive_fraction": 0.0,
            "drive_mean": 0.0,
        }
        quantized_active_window = int(getattr(config, "quantized_active_window", 220))
        quantized_in_window = shock_step <= step <= shock_step + quantized_active_window
        quantized_early_response = step <= shock_step + max(1, int(getattr(config, "response_window", 20)))
        quantized_coherence_ratio = float(
            np.clip(coherence_history[step] / coherence_ref, 0.0, 2.0)
        )
        quantized_should_activate = (
            getattr(config, "quantized_scalar_vector_enabled", False)
            and quantized_in_window
            and (
                quantized_early_response
                or quantized_coherence_ratio
                <= float(getattr(config, "quantized_coherence_trigger", 0.98))
                or distressed_ratio
                >= float(getattr(config, "quantized_disturbance_trigger", 0.18))
            )
        )
        if quantized_should_activate:
            anchor_cluster = emitter_core if emitter_core else current_primary
            anchor_indices = (
                np.array(sorted(anchor_cluster), dtype=int)
                if anchor_cluster
                else np.argsort(trust_scores)[
                    -max(config.cluster_threshold, int(round(config.nodes * 0.18))) :
                ]
            )
            quantized_stats = apply_quantized_scalar_vector_control(
                state=state,
                trust_scores=trust_scores,
                memory_support=memory_support,
                disturbance=disturbance,
                anchor_indices=anchor_indices.astype(int),
                config=config,
            )
        quantized_anchor_gap_history[step] = float(quantized_stats["anchor_gap_mean"])
        quantized_lock_fraction_history[step] = float(quantized_stats["lock_fraction"])
        quantized_repulsive_fraction_history[step] = float(
            quantized_stats["repulsive_fraction"]
        )
        quantized_vector_drive_history[step] = float(quantized_stats["drive_mean"])

        if getattr(config, "inverse_triad_enabled", False):
            update_inverse_triad_trackers(
                trackers=inverse_triad_trackers,
                effective_adjacency=effective_adjacency,
                trust_scores=trust_scores,
                distressed=distressed,
                step=step,
                config=config,
            )

            active_window = int(getattr(config, "inverse_triad_active_window", 220))
            trigger_interval = max(1, int(getattr(config, "inverse_triad_interval", 8)))
            within_window = shock_step <= step <= shock_step + active_window
            should_trigger_inverse = (
                inverse_triad_cooldown == 0
                and inverse_triad_events < inverse_triad_max_events
                and within_window
                and step % trigger_interval == 0
            )
            if should_trigger_inverse:
                inverse_payload = choose_inverse_triad_targets(
                    state=state,
                    effective_adjacency=effective_adjacency,
                    effective_weights=effective_weights,
                    pair_distance=base_layer.pair_distance,
                    trust_scores=trust_scores,
                    memory_support=memory_support,
                    disturbance=disturbance,
                    current_primary=current_primary,
                    config=config,
                )
                if inverse_payload is not None:
                    triad_indices = inverse_payload["triad_indices"]
                    apply_inverse_triad_transform(
                        state,
                        triad_indices,
                        config,
                        scale=1.0,
                    )
                    inverse_triad_events += 1
                    inverse_triad_anchor_trust_total += float(inverse_payload["mean_trust"])
                    inverse_triad_bridge_total += float(inverse_payload["mean_bridge_strength"])
                    inverse_triad_shell_total += float(inverse_payload["shell_score"])
                    inverse_triad_cooldown = int(
                        getattr(config, "inverse_triad_cooldown", 20)
                    )
                    tracker = {
                        "trigger_step": step,
                        "triad_indices": triad_indices.copy(),
                        "core_indices": inverse_payload["core_indices"].copy(),
                        "pre_triad_trust": float(inverse_payload["mean_trust"]),
                        "pre_core_trust": float(
                            np.mean(trust_scores[inverse_payload["core_indices"]])
                        )
                        if inverse_payload["core_indices"].size
                        else float(np.mean(trust_scores)),
                        "mean_bridge_strength": float(inverse_payload["mean_bridge_strength"]),
                        "shell_score": float(inverse_payload["shell_score"]),
                        "max_component_fraction": component_fraction_for_indices(
                            effective_adjacency,
                            triad_indices,
                        ),
                        "peak_core_contamination": 0.0,
                        "last_core_trust": float(
                            np.mean(trust_scores[inverse_payload["core_indices"]])
                        )
                        if inverse_payload["core_indices"].size
                        else float(np.mean(trust_scores)),
                        "recovered_step": None,
                        "finalized": False,
                    }
                    inverse_triad_trackers.append(tracker)
                    if collect_event_rows:
                        inverse_triad_rows.append(
                            {
                                "scenario": scenario.name,
                                "model_kind": scenario.model_kind,
                                "bootstrap_mode": scenario.bootstrap_mode,
                                "false_mode": scenario.false_mode,
                                "run_index": run_index,
                                "seed": seed,
                                "step": step + 1,
                                "zone": "outer_weak",
                                "triad_indices": " ".join(map(str, triad_indices.tolist())),
                                "core_size": int(inverse_payload["core_indices"].size),
                                "triad_edge_count": int(inverse_payload["edge_count"]),
                                "mean_trust": float(inverse_payload["mean_trust"]),
                                "mean_memory_support": float(
                                    inverse_payload["mean_memory_support"]
                                ),
                                "mean_bridge_strength": float(
                                    inverse_payload["mean_bridge_strength"]
                                ),
                                "mean_disturbance": float(inverse_payload["mean_disturbance"]),
                                "shell_score": float(inverse_payload["shell_score"]),
                            }
                        )

        if getattr(config, "phase_manipulation_enabled", False):
            update_phase_manipulation_trackers(
                trackers=phase_manipulation_trackers,
                state=state,
                effective_adjacency=effective_adjacency,
                trust_scores=trust_scores,
                distressed=distressed,
                step=step,
                config=config,
            )

            active_window = int(getattr(config, "phase_manipulation_active_window", 220))
            trigger_interval = max(1, int(getattr(config, "phase_manipulation_interval", 8)))
            within_window = shock_step <= step <= shock_step + active_window
            should_trigger_phase_manipulation = (
                phase_manipulation_cooldown == 0
                and phase_manipulation_events < phase_manipulation_max_events
                and within_window
                and step % trigger_interval == 0
            )
            if should_trigger_phase_manipulation:
                manipulation = choose_phase_manipulation_target(
                    state=state,
                    effective_adjacency=effective_adjacency,
                    effective_weights=effective_weights,
                    trust_scores=trust_scores,
                    memory_support=memory_support,
                    disturbance=disturbance,
                    current_primary=current_primary,
                    config=config,
                )
                if manipulation is not None:
                    node_index = int(manipulation["node_index"])
                    phase_shift = float(
                        getattr(config, "phase_manipulation_phase_shift", 0.28)
                    )
                    freq_shift = float(
                        getattr(config, "phase_manipulation_freq_shift", 0.022)
                    )
                    tracker = {
                        "trigger_step": step,
                        "node_index": node_index,
                        "higher_indices": manipulation["higher_indices"].copy(),
                        "lower_indices": manipulation["lower_indices"].copy(),
                        "pre_phase": float(state["theta"][node_index]),
                        "pre_freq": float(state["freq"][node_index]),
                        "target_phase": float(
                            (state["theta"][node_index] + phase_shift) % (2.0 * np.pi)
                        ),
                        "target_freq": float(
                            np.clip(state["freq"][node_index] + freq_shift, 0.5, 1.5)
                        ),
                        "balance": float(manipulation["balance"]),
                        "target_trust": float(manipulation["target_trust"]),
                        "core_indices": manipulation["core_indices"].copy(),
                        "pre_core_trust": float(
                            np.mean(trust_scores[manipulation["core_indices"]])
                        )
                        if manipulation["core_indices"].size
                        else float(np.mean(trust_scores)),
                        "last_core_trust": float(
                            np.mean(trust_scores[manipulation["core_indices"]])
                        )
                        if manipulation["core_indices"].size
                        else float(np.mean(trust_scores)),
                        "max_upward_fraction": 0.0,
                        "max_downward_fraction": 0.0,
                        "peak_core_contamination": 0.0,
                        "recovered_step": None,
                        "finalized": False,
                    }
                    phase_manipulation_trackers.append(tracker)
                    apply_phase_manipulation_transform(
                        state,
                        tracker,
                        config,
                        scale=1.0,
                    )
                    phase_manipulation_events += 1
                    phase_manipulation_target_trust_total += float(
                        manipulation["target_trust"]
                    )
                    phase_manipulation_balance_total += float(manipulation["balance"])
                    phase_manipulation_cooldown = int(
                        getattr(config, "phase_manipulation_cooldown", 20)
                    )
                    if collect_event_rows:
                        phase_manipulation_rows.append(
                            {
                                "scenario": scenario.name,
                                "model_kind": scenario.model_kind,
                                "bootstrap_mode": scenario.bootstrap_mode,
                                "false_mode": scenario.false_mode,
                                "run_index": run_index,
                                "seed": seed,
                                "step": step + 1,
                                "node_index": node_index,
                                "target_trust": float(manipulation["target_trust"]),
                                "balance": float(manipulation["balance"]),
                                "higher_size": int(manipulation["higher_indices"].size),
                                "lower_size": int(manipulation["lower_indices"].size),
                                "higher_strength": float(manipulation["higher_strength"]),
                                "lower_strength": float(manipulation["lower_strength"]),
                                "score": float(manipulation["score"]),
                            }
                        )

        if getattr(config, "controller_enabled", False) and step >= shock_step:
            active_window = int(getattr(config, "controller_active_window", 2 * config.shock_window))
            coherence_ratio = float(np.clip(coherence_history[step] / coherence_ref, 0.0, 2.0))
            cluster_ratio = float(np.clip(current_cluster_fraction / cluster_ref, 0.0, 2.0))
            within_controller_window = step <= shock_step + active_window
            should_evaluate_controller = (
                step == shock_step
                or step % controller_interval == 0
                or not within_controller_window
            )

            if should_evaluate_controller:
                if getattr(config, "reward_controller_enabled", False):
                    settle_reward_controller(
                        reward_state=reward_controller,
                        coherence_ratio=coherence_ratio,
                        cluster_ratio=cluster_ratio,
                        distressed_ratio=distressed_ratio,
                        trust_mean=trust_mean_history[step],
                        config=config,
                    )
                    if within_controller_window:
                        next_mode = choose_reward_controller_mode(
                            current_mode=controller_mode,
                            steps_in_mode=step - controller_mode_start,
                            scaffold_enabled=scaffold_enabled,
                            coherence_ratio=coherence_ratio,
                            cluster_ratio=cluster_ratio,
                            distressed_ratio=distressed_ratio,
                            reward_state=reward_controller,
                            config=config,
                        )
                        prime_reward_controller(
                            reward_state=reward_controller,
                            action=next_mode,
                            coherence_ratio=coherence_ratio,
                            cluster_ratio=cluster_ratio,
                            distressed_ratio=distressed_ratio,
                            trust_mean=trust_mean_history[step],
                        )
                    else:
                        reward_controller["pending_action"] = None
                        reward_controller["pending_snapshot"] = None
                        next_mode = "observe"
                else:
                    if within_controller_window:
                        next_mode = choose_controller_mode(
                            current_mode=controller_mode,
                            steps_in_mode=step - controller_mode_start,
                            scaffold_enabled=scaffold_enabled,
                            coherence_ratio=coherence_ratio,
                            cluster_ratio=cluster_ratio,
                            distressed_ratio=distressed_ratio,
                            config=config,
                            policy=controller_policy,
                        )
                    else:
                        next_mode = "observe"

                if next_mode != controller_mode:
                    controller_mode = next_mode
                    controller_mode_start = step
                    controller_switches += 1

            if controller_mode == "absorber":
                controller_absorber_steps += 1
            elif controller_mode == "scaffold":
                controller_scaffold_steps += 1
            else:
                controller_observe_steps += 1

            if should_evaluate_controller and not getattr(config, "reward_controller_enabled", False):
                update_controller_policy(
                    policy=controller_policy,
                    coherence_ratio=coherence_ratio,
                    cluster_ratio=cluster_ratio,
                    distressed_ratio=distressed_ratio,
                    mode=controller_mode,
                    config=config,
                )

        if hasattr(config, "absorber_phase_strength") and step >= shock_step:
            if getattr(config, "controller_enabled", False) and controller_mode != "absorber":
                pass
            else:
                low_coherence = coherence_history[step] < getattr(
                    config,
                    "absorber_coherence_floor",
                    0.78,
                ) * coherence_ref
                low_cluster = current_cluster_fraction < getattr(
                    config,
                    "absorber_cluster_floor",
                    0.70,
                ) * cluster_ref
                active_window = int(getattr(config, "absorber_active_window", 2 * config.shock_window))
                within_absorber_window = step <= shock_step + active_window
                recovery_margin = float(getattr(config, "absorber_recovery_margin", 1.05))
                ratio_floor = float(getattr(config, "absorber_distressed_ratio_floor", 0.28))
                needs_disturbance_control = (
                    within_absorber_window
                    and distressed_ratio > ratio_floor
                    and coherence_history[step] < recovery_margin * coherence_ref
                )
                if absorber_cooldown == 0 and (low_coherence or low_cluster or needs_disturbance_control):
                    target_indices = np.flatnonzero(distressed)
                    if target_indices.size == 0:
                        target_indices = np.array([int(np.argmax(disturbance))], dtype=int)
                    phase_strength = getattr(config, "absorber_phase_strength", 0.34)
                    state["theta"][target_indices] += (
                        phase_strength
                        * disturbance[target_indices]
                        * np.sin(emitter_phase - state["theta"][target_indices])
                    )
                    state["theta"] %= 2.0 * np.pi
                    core_freq = float(np.mean(state["freq"][emitter_indices]))
                    blend = getattr(config, "absorber_freq_blend", 0.18)
                    state["freq"][target_indices] = (
                        (1.0 - blend) * state["freq"][target_indices] + blend * core_freq
                    )
                    damp = getattr(config, "absorber_velocity_damp", 0.45)
                    if "vel" in state:
                        state["vel"][target_indices] *= damp
                    if "helix_vel" in state:
                        state["helix_vel"][target_indices] *= damp
                    memory_boost = getattr(config, "absorber_memory_boost", 0.60)
                    reinforce_memory(memory_matrix, emitter_core, config, boost=memory_boost)
                    reinforce_core_memory(core_memory_matrix, emitter_core, config, boost=0.5 * memory_boost)
                    absorber_events += 1
                    absorber_cooldown = int(getattr(config, "absorber_cooldown", 10))
                    if collect_event_rows:
                        absorber_rows.append(
                            {
                                "scenario": scenario.name,
                                "model_kind": scenario.model_kind,
                                "bootstrap_mode": scenario.bootstrap_mode,
                                "false_mode": scenario.false_mode,
                                "run_index": run_index,
                                "seed": seed,
                                "step": step + 1,
                                "emitter_size": len(emitter_core),
                                "distressed_size": int(target_indices.size),
                                "distressed_ratio": distressed_ratio,
                                "mean_disturbance": float(np.mean(disturbance[target_indices])),
                            }
                        )

        if scaffold_enabled and step >= shock_step:
            if getattr(config, "controller_enabled", False) and controller_mode != "scaffold":
                pass
            else:
                scaffold_window = int(getattr(config, "scaffold_active_window", 2 * config.shock_window))
                within_scaffold_window = step <= shock_step + scaffold_window
                low_cluster_regrowth = (
                    current_cluster_fraction
                    < getattr(config, "scaffold_cluster_floor", 0.82) * cluster_ref
                )
                if scaffold_cooldown == 0 and within_scaffold_window and low_cluster_regrowth:
                    anchor_cluster = choose_guardian_target(
                        stored_prototypes,
                        trust_scores,
                        scaffold_matrix if np.any(scaffold_matrix) else core_memory_matrix,
                        config,
                    )
                    anchor_indices = np.array(sorted(anchor_cluster), dtype=int)
                    if anchor_indices.size:
                        affinity = np.mean(scaffold_matrix[:, anchor_indices], axis=1)
                        affinity[anchor_indices] = -1.0
                        anchor_phase = float(
                            np.angle(np.mean(np.exp(1j * state["theta"][anchor_indices])))
                        )
                        scaffold_phase_error = np.abs(
                            np.angle(np.exp(1j * (state["theta"] - anchor_phase)))
                        )
                        phase_compatibility = np.clip(1.0 - scaffold_phase_error / np.pi, 0.0, 1.0)
                        coherence_ratio = np.clip(coherence_history[step] / coherence_ref, 0.0, 1.2)
                        recruit_scale = (
                            getattr(config, "scaffold_recruit_floor", 0.55)
                            + (1.0 - getattr(config, "scaffold_recruit_floor", 0.55))
                            * min(coherence_ratio, 1.0)
                        )
                        recruit_count = max(
                            config.cluster_threshold,
                            int(
                                round(
                                    config.nodes
                                    * getattr(config, "scaffold_recruit_fraction", 0.14)
                                    * recruit_scale
                                )
                            ),
                        )
                        phase_min = (
                            getattr(config, "scaffold_phase_compatibility_min", 0.40)
                            + getattr(config, "scaffold_phase_guard_gain", 0.22)
                            * max(0.0, 1.0 - min(coherence_ratio, 1.0))
                        )
                        score = (
                            affinity
                            * (0.50 + 0.35 * phase_compatibility + 0.15 * trust_scores)
                        )
                        score[phase_compatibility < phase_min] = -1.0
                        score[trust_scores < getattr(config, "scaffold_candidate_trust_min", 0.28)] = -1.0
                        candidate_indices = np.argsort(score)[-recruit_count:]
                        candidate_indices = candidate_indices[score[candidate_indices] > getattr(
                            config,
                            "scaffold_affinity_threshold",
                            0.08,
                        )]
                        if candidate_indices.size:
                            phase_strength = float(getattr(config, "scaffold_phase_strength", 0.22)) * (
                                1.0
                                + getattr(config, "scaffold_phase_boost_gain", 0.35)
                                * max(0.0, 1.0 - min(coherence_ratio, 1.0))
                            )
                            state["theta"][candidate_indices] += phase_strength * np.sin(
                                anchor_phase - state["theta"][candidate_indices]
                            )
                            state["theta"] %= 2.0 * np.pi
                            freq_blend = float(getattr(config, "scaffold_freq_blend", 0.12))
                            anchor_freq = float(np.mean(state["freq"][anchor_indices]))
                            state["freq"][candidate_indices] = (
                                (1.0 - freq_blend) * state["freq"][candidate_indices]
                                + freq_blend * anchor_freq
                            )
                            apply_scaffold_geometry_pull(
                                state,
                                candidate_indices,
                                anchor_indices,
                                config,
                                scale=(
                                    getattr(config, "scaffold_pull_floor", 0.55)
                                    + (1.0 - getattr(config, "scaffold_pull_floor", 0.55))
                                    * min(coherence_ratio, 1.0)
                                ),
                            )
                            union_cluster = set(anchor_cluster) | set(candidate_indices.tolist())
                            reinforce_scaffold(
                                scaffold_matrix,
                                union_cluster,
                                config,
                                boost=getattr(config, "scaffold_event_boost", 0.75),
                            )
                            reinforce_core_memory(
                                core_memory_matrix,
                                union_cluster,
                                config,
                                boost=0.35,
                            )
                            scaffold_events += 1
                            scaffold_cooldown = int(getattr(config, "scaffold_cooldown", 24))
                            if collect_event_rows:
                                scaffold_rows.append(
                                    {
                                        "scenario": scenario.name,
                                        "model_kind": scenario.model_kind,
                                        "bootstrap_mode": scenario.bootstrap_mode,
                                        "false_mode": scenario.false_mode,
                                        "run_index": run_index,
                                        "seed": seed,
                                        "step": step + 1,
                                        "anchor_size": int(anchor_indices.size),
                                        "recruit_size": int(candidate_indices.size),
                                        "cluster_fraction": float(current_cluster_fraction),
                                        "mean_affinity": float(np.mean(affinity[candidate_indices])),
                                    }
                                )

        if getattr(config, "modulation_training_enabled", False) and step >= shock_step:
            active_window = int(getattr(config, "modulation_active_window", 2 * config.shock_window))
            within_modulation_window = step <= shock_step + active_window
            modulation_interval = max(1, int(getattr(config, "modulation_interval", 6)))
            coherence_floor = float(getattr(config, "modulation_coherence_floor", 0.90))
            cluster_floor = float(getattr(config, "modulation_cluster_floor", 0.74))
            disturbance_ceiling = float(getattr(config, "modulation_disturbance_ceiling", 0.34))
            coherence_ready = coherence_history[step] >= coherence_floor * coherence_ref
            cluster_ready = current_cluster_fraction >= cluster_floor * cluster_ref
            disturbance_ready = distressed_ratio <= disturbance_ceiling
            early_window = step <= shock_step + max(1, int(getattr(config, "response_window", 20)))

            if (
                modulation_cooldown == 0
                and within_modulation_window
                and step % modulation_interval == 0
                and (coherence_ready or cluster_ready or disturbance_ready or early_window)
            ):
                anchor_cluster = choose_guardian_target(
                    stored_prototypes,
                    trust_scores,
                    scaffold_matrix if np.any(scaffold_matrix) else core_memory_matrix,
                    config,
                )
                anchor_indices = np.array(sorted(anchor_cluster), dtype=int)
                if anchor_indices.size:
                    modulation = choose_multifocal_modulation_targets(
                        state=state,
                        anchor_indices=anchor_indices,
                        trust_scores=trust_scores,
                        memory_support=memory_support,
                        disturbance=disturbance,
                        config=config,
                    )
                    if modulation is not None:
                        candidate_indices = modulation["candidate_indices"]
                        ki = modulation["ki"][candidate_indices]
                        beam_phases = modulation["beam_phases"]
                        steering = np.mean(
                            np.sin(beam_phases[None, :] - state["theta"][candidate_indices, None]),
                            axis=1,
                        )
                        phase_strength = float(getattr(config, "modulation_phase_strength", 0.18))
                        state["theta"][candidate_indices] += phase_strength * ki * steering
                        state["theta"] %= 2.0 * np.pi

                        freq_blend = float(getattr(config, "modulation_freq_blend", 0.14))
                        blend_weight = np.clip(freq_blend * ki, 0.0, 0.92)
                        state["freq"][candidate_indices] = (
                            (1.0 - blend_weight) * state["freq"][candidate_indices]
                            + blend_weight * modulation["anchor_freq"]
                        )

                        apply_scaffold_geometry_pull(
                            state,
                            candidate_indices,
                            anchor_indices,
                            config,
                            scale=float(getattr(config, "modulation_geometry_pull", 0.60))
                            * float(np.mean(ki)),
                        )

                        union_cluster = set(anchor_cluster) | set(candidate_indices.tolist())
                        memory_boost = float(getattr(config, "modulation_memory_boost", 0.50))
                        reinforce_memory(memory_matrix, union_cluster, config, boost=memory_boost)
                        reinforce_core_memory(
                            core_memory_matrix,
                            union_cluster,
                            config,
                            boost=0.5 * memory_boost,
                        )
                        if scaffold_enabled:
                            reinforce_scaffold(
                                scaffold_matrix,
                                union_cluster,
                                config,
                                boost=0.65 * memory_boost,
                            )

                        modulation_events += 1
                        modulation_target_total += int(candidate_indices.size)
                        modulation_ki_total += float(np.sum(ki))
                        modulation_cooldown = int(getattr(config, "modulation_cooldown", 10))
                        if collect_event_rows:
                            modulation_rows.append(
                                {
                                    "scenario": scenario.name,
                                    "model_kind": scenario.model_kind,
                                    "bootstrap_mode": scenario.bootstrap_mode,
                                    "false_mode": scenario.false_mode,
                                    "run_index": run_index,
                                    "seed": seed,
                                    "step": step + 1,
                                    "anchor_size": int(anchor_indices.size),
                                    "target_size": int(candidate_indices.size),
                                    "beam_count": int(len(beam_phases)),
                                    "mean_ki": float(np.mean(ki)),
                                    "mean_beam_response": float(
                                        np.mean(modulation["beam_response"][candidate_indices])
                                    ),
                                }
                            )

        if (step + 1) % config.persistence_delta == 0:
            persistence = jaccard(previous_primary, current_primary) if previous_primary else 0.0
            cluster_fraction_history[step] = (
                len(current_primary) / config.nodes if current_primary else 0.0
            )

            if current_primary:
                current_indices = np.array(sorted(current_primary), dtype=int)
                support_matrix = (memory_mask | core_memory_mask | scaffold_mask)[
                    np.ix_(current_indices, current_indices)
                ]
                upper = np.triu(support_matrix, k=1)
                memory_supported = bool(np.any(upper)) and any(
                    jaccard(current_primary, prototype) >= config.memory_similarity_threshold
                    for prototype in stored_prototypes
                )
                if memory_supported:
                    memory_supported_samples += 1

            if current_primary and persistence >= config.stability_gate:
                reinforce_memory(memory_matrix, current_primary, config)
                reinforce_core_memory(core_memory_matrix, current_primary, config)
                if scaffold_enabled:
                    reinforce_scaffold(scaffold_matrix, current_primary, config)
                if not any(
                    jaccard(current_primary, prototype) >= config.memory_similarity_threshold
                    for prototype in stored_prototypes
                ):
                    stored_prototypes.append(set(current_primary))

            guardian_triggered = False
            if step >= shock_step and stored_prototypes and guardian_cooldown == 0:
                low_coherence = coherence_history[step] < config.guardian_coherence_floor * coherence_ref
                low_cluster = current_cluster_fraction < config.guardian_cluster_floor * cluster_ref
                if low_coherence or low_cluster or persistence < config.stability_gate:
                    target_cluster = choose_guardian_target(
                        stored_prototypes,
                        trust_scores,
                        core_memory_matrix,
                        config,
                    )
                    support_count = max(
                        1,
                        int(round(config.nodes * config.guardian_trust_fraction)),
                    )
                    support_indices = np.argsort(trust_scores)[-support_count:]
                    combined_target = set(target_cluster) | set(support_indices.tolist())
                    indices = np.array(sorted(combined_target), dtype=int)
                    state["theta"][indices] += config.guardian_pulse_strength * np.sin(
                        config.bootstrap_anchor - state["theta"][indices]
                    )
                    target_freq = float(np.mean(state["freq"][indices]))
                    state["freq"][indices] = 0.90 * state["freq"][indices] + 0.10 * target_freq
                    state["theta"] %= 2.0 * np.pi
                    reinforce_memory(
                        memory_matrix,
                        target_cluster,
                        config,
                        boost=config.guardian_memory_boost,
                    )
                    reinforce_core_memory(
                        core_memory_matrix,
                        target_cluster,
                        config,
                        boost=0.5 * config.guardian_memory_boost,
                    )
                    guardian_events += 1
                    guardian_cooldown = config.guardian_cooldown
                    guardian_triggered = True
                    if collect_event_rows:
                        guardian_rows.append(
                            {
                                "scenario": scenario.name,
                                "model_kind": scenario.model_kind,
                                "bootstrap_mode": scenario.bootstrap_mode,
                                "false_mode": scenario.false_mode,
                                "run_index": run_index,
                                "seed": seed,
                                "step": step + 1,
                                "guardian_size": len(combined_target),
                                "coherence": coherence_history[step],
                                "cluster_fraction": current_cluster_fraction,
                            }
                        )

            recall_candidates = [
                prototype
                for prototype in stored_prototypes
                if jaccard(prototype, current_primary) < config.memory_similarity_threshold
            ]
            if (
                step >= shock_step
                and not guardian_triggered
                and recall_cooldown == 0
                and recall_candidates
                and (
                    not current_primary
                    or persistence < config.stability_gate
                    or (step + 1) % (2 * config.persistence_delta) == 0
                )
            ):
                prototype = max(recall_candidates, key=len)
                indices = np.array(sorted(prototype), dtype=int)
                state["theta"][indices] += config.recall_phase_strength * np.sin(
                    config.bootstrap_anchor - state["theta"][indices]
                )
                state["theta"] %= 2.0 * np.pi
                reinforce_memory(memory_matrix, prototype, config, boost=0.5)
                recall_cooldown = config.recall_cooldown
                recall_events += 1

            previous_primary = set(current_primary)
        elif step > 0:
            cluster_fraction_history[step] = cluster_fraction_history[step - 1]

    pre_start = max(0, shock_step - config.shock_window)
    pre_slice = slice(pre_start, shock_step)
    post_end = min(config.steps, shock_step + config.shock_window)
    post_slice = slice(shock_step, post_end)
    late_slice = slice(max(0, config.steps - config.late_window), config.steps)

    pre_shock_coherence = float(np.mean(coherence_history[pre_slice]))
    shock_window_values = coherence_history[post_slice]
    if shock_window_values.size == 0:
        shock_window_values = coherence_history[shock_step:]
    shock_min_coherence = float(np.min(shock_window_values))
    late_recovery_coherence = float(np.mean(coherence_history[late_slice]))
    pre_shock_cluster_fraction = float(np.mean(cluster_fraction_history[pre_slice]))
    cluster_window_values = cluster_fraction_history[post_slice]
    if cluster_window_values.size == 0:
        cluster_window_values = cluster_fraction_history[shock_step:]
    shock_min_cluster_fraction = float(np.min(cluster_window_values))
    late_cluster_fraction = float(np.mean(cluster_fraction_history[late_slice]))
    pre_shock_memory_mass = float(np.mean(memory_mass_history[pre_slice]))
    late_memory_mass = float(np.mean(memory_mass_history[late_slice]))
    pre_shock_core_memory_mass = float(np.mean(core_memory_mass_history[pre_slice]))
    late_core_memory_mass = float(np.mean(core_memory_mass_history[late_slice]))
    quantized_scalar_anchor_gap_late = float(np.mean(quantized_anchor_gap_history[late_slice]))
    quantized_scalar_lock_fraction_late = float(
        np.mean(quantized_lock_fraction_history[late_slice])
    )
    quantized_repulsive_fraction_late = float(
        np.mean(quantized_repulsive_fraction_history[late_slice])
    )
    quantized_vector_drive_mean_late = float(
        np.mean(quantized_vector_drive_history[late_slice])
    )

    recovery_target = config.recovery_threshold * pre_shock_coherence
    cluster_target = config.cluster_recovery_threshold * max(pre_shock_cluster_fraction, 1e-6)
    coherence_recovery_curve = rolling_mean(coherence_history, config.response_window)
    cluster_recovery_curve = rolling_mean(cluster_fraction_history, config.response_window)

    recovery_time = -1
    for step in range(shock_step, config.steps):
        if (
            coherence_recovery_curve[step] >= recovery_target
            and cluster_recovery_curve[step] >= cluster_target
        ):
            recovery_time = step - shock_step
            break

    false_mask = state["false_mask"]
    final_degrees = final_adjacency.sum(axis=1).astype(float)
    final_false_isolation_rate = (
        float(np.mean(final_degrees[false_mask] == 0)) if np.any(false_mask) else 0.0
    )
    controller_total_steps = max(1, controller_absorber_steps + controller_scaffold_steps + controller_observe_steps)

    for tracker in inverse_triad_trackers:
        if tracker.get("finalized", False):
            continue
        if tracker["core_indices"].size:
            tracker["last_core_trust"] = float(np.mean(trust_scores[tracker["core_indices"]]))
            tracker["peak_core_contamination"] = max(
                tracker["peak_core_contamination"],
                float(np.mean(distressed[tracker["core_indices"]])),
            )
        tracker["max_component_fraction"] = max(
            tracker["max_component_fraction"],
            component_fraction_for_indices(final_adjacency, tracker["triad_indices"]),
        )
        tracker["finalized"] = True

    if inverse_triad_trackers:
        inverse_triad_propagation_fraction = float(
            np.mean([item["max_component_fraction"] for item in inverse_triad_trackers])
        )
        inverse_triad_boundary_recovery_steps = float(
            np.mean(
                [
                    (
                        item["recovered_step"]
                        if item["recovered_step"] is not None
                        else getattr(config, "inverse_triad_tracking_window", 120)
                    )
                    for item in inverse_triad_trackers
                ]
            )
        )
        inverse_triad_core_stability_delta = float(
            np.mean(
                [
                    item["last_core_trust"] - item["pre_core_trust"]
                    for item in inverse_triad_trackers
                ]
            )
        )
        inverse_triad_core_contamination_rate = float(
            np.mean([item["peak_core_contamination"] for item in inverse_triad_trackers])
        )
    else:
        inverse_triad_propagation_fraction = 0.0
        inverse_triad_boundary_recovery_steps = float(
            getattr(config, "inverse_triad_tracking_window", 120)
        )
        inverse_triad_core_stability_delta = 0.0
        inverse_triad_core_contamination_rate = 0.0

    for tracker in phase_manipulation_trackers:
        if tracker.get("finalized", False):
            continue
        if tracker["core_indices"].size:
            tracker["last_core_trust"] = float(np.mean(trust_scores[tracker["core_indices"]]))
            tracker["peak_core_contamination"] = max(
                tracker["peak_core_contamination"],
                float(np.mean(distressed[tracker["core_indices"]])),
            )
        tracker["finalized"] = True

    if phase_manipulation_trackers:
        phase_manipulation_upward_spread_fraction = float(
            np.mean([item["max_upward_fraction"] for item in phase_manipulation_trackers])
        )
        phase_manipulation_downward_spread_fraction = float(
            np.mean([item["max_downward_fraction"] for item in phase_manipulation_trackers])
        )
        phase_manipulation_recovery_steps = float(
            np.mean(
                [
                    (
                        item["recovered_step"]
                        if item["recovered_step"] is not None
                        else getattr(config, "phase_manipulation_tracking_window", 120)
                    )
                    for item in phase_manipulation_trackers
                ]
            )
        )
        phase_manipulation_core_stability_delta = float(
            np.mean(
                [
                    item["last_core_trust"] - item["pre_core_trust"]
                    for item in phase_manipulation_trackers
                ]
            )
        )
        phase_manipulation_core_contamination_rate = float(
            np.mean([item["peak_core_contamination"] for item in phase_manipulation_trackers])
        )
    else:
        phase_manipulation_upward_spread_fraction = 0.0
        phase_manipulation_downward_spread_fraction = 0.0
        phase_manipulation_recovery_steps = float(
            getattr(config, "phase_manipulation_tracking_window", 120)
        )
        phase_manipulation_core_stability_delta = 0.0
        phase_manipulation_core_contamination_rate = 0.0

    metrics = PhaseVIRunMetrics(
        scenario=scenario.name,
        label=scenario.label,
        model_kind=scenario.model_kind,
        bootstrap_mode=scenario.bootstrap_mode,
        false_mode=scenario.false_mode,
        run_index=run_index,
        seed=seed,
        pre_shock_coherence=pre_shock_coherence,
        shock_min_coherence=shock_min_coherence,
        late_recovery_coherence=late_recovery_coherence,
        coherence_recovery_ratio=late_recovery_coherence / max(pre_shock_coherence, 1e-6),
        pre_shock_cluster_fraction=pre_shock_cluster_fraction,
        shock_min_cluster_fraction=shock_min_cluster_fraction,
        late_cluster_fraction=late_cluster_fraction,
        cluster_recovery_ratio=late_cluster_fraction / max(pre_shock_cluster_fraction, 1e-6),
        final_false_isolation_rate=final_false_isolation_rate,
        recovery_time_steps=float(recovery_time if recovery_time >= 0 else config.steps),
        recovery_success=float(recovery_time >= 0),
        pre_shock_memory_mass=pre_shock_memory_mass,
        late_memory_mass=late_memory_mass,
        memory_recovery_ratio=late_memory_mass / max(pre_shock_memory_mass, 0.01),
        pre_shock_core_memory_mass=pre_shock_core_memory_mass,
        late_core_memory_mass=late_core_memory_mass,
        core_memory_recovery_ratio=late_core_memory_mass / max(pre_shock_core_memory_mass, 0.01),
        topological_memory_index=memory_supported_samples / max(1, config.steps // config.persistence_delta),
        recall_event_rate=recall_events / max(1, config.steps // config.persistence_delta),
        guardian_event_rate=guardian_events / max(1, config.steps // config.persistence_delta),
        absorber_event_rate=absorber_events / max(1, config.steps // config.persistence_delta),
        scaffold_event_rate=scaffold_events / max(1, config.steps // config.persistence_delta),
        modulation_event_rate=modulation_events / max(1, config.steps // config.persistence_delta),
        modulation_target_fraction=(
            modulation_target_total / max(1, modulation_events * config.nodes)
        ),
        modulation_mean_ki=modulation_ki_total / max(1, modulation_target_total),
        scaffold_edge_mass=float(np.mean(scaffold_mass_history[late_slice])),
        controller_absorber_rate=controller_absorber_steps / controller_total_steps,
        controller_scaffold_rate=controller_scaffold_steps / controller_total_steps,
        controller_observe_rate=controller_observe_steps / controller_total_steps,
        controller_switch_rate=controller_switches / controller_total_steps,
        controller_reward_mean=float(reward_controller["reward_total"] / max(1, reward_controller["reward_updates"])),
        learned_coherence_critical=float(controller_policy["coherence_critical"]),
        learned_cluster_priority=float(controller_policy["cluster_priority"]),
        learned_disturbance_priority=float(controller_policy["disturbance_priority"]),
        learned_action_value_observe=float(reward_controller["values"]["observe"]),
        learned_action_value_absorber=float(reward_controller["values"]["absorber"]),
        learned_action_value_scaffold=float(reward_controller["values"]["scaffold"]),
        resonance_energy_late=float(np.mean(energy_history[late_slice])),
        late_trust_mean=float(np.mean(trust_mean_history[late_slice])),
        inverse_triad_event_rate=inverse_triad_events / max(1, config.steps // config.persistence_delta),
        inverse_triad_mean_anchor_trust=inverse_triad_anchor_trust_total / max(1, inverse_triad_events),
        inverse_triad_mean_bridge_strength=inverse_triad_bridge_total / max(1, inverse_triad_events),
        inverse_triad_mean_shell_score=inverse_triad_shell_total / max(1, inverse_triad_events),
        inverse_triad_propagation_fraction=inverse_triad_propagation_fraction,
        inverse_triad_boundary_recovery_steps=inverse_triad_boundary_recovery_steps,
        inverse_triad_core_stability_delta=inverse_triad_core_stability_delta,
        inverse_triad_core_contamination_rate=inverse_triad_core_contamination_rate,
        phase_manipulation_event_rate=phase_manipulation_events / max(
            1, config.steps // config.persistence_delta
        ),
        phase_manipulation_mean_target_trust=phase_manipulation_target_trust_total
        / max(1, phase_manipulation_events),
        phase_manipulation_mean_bidirectional_balance=phase_manipulation_balance_total
        / max(1, phase_manipulation_events),
        phase_manipulation_upward_spread_fraction=phase_manipulation_upward_spread_fraction,
        phase_manipulation_downward_spread_fraction=phase_manipulation_downward_spread_fraction,
        phase_manipulation_recovery_steps=phase_manipulation_recovery_steps,
        phase_manipulation_core_stability_delta=phase_manipulation_core_stability_delta,
        phase_manipulation_core_contamination_rate=phase_manipulation_core_contamination_rate,
        quantized_scalar_anchor_gap_late=quantized_scalar_anchor_gap_late,
        quantized_scalar_lock_fraction_late=quantized_scalar_lock_fraction_late,
        quantized_repulsive_fraction_late=quantized_repulsive_fraction_late,
        quantized_vector_drive_mean_late=quantized_vector_drive_mean_late,
    )
    return (
        metrics,
        shock_rows,
        guardian_rows,
        absorber_rows,
        scaffold_rows,
        modulation_rows,
        inverse_triad_rows,
        phase_manipulation_rows,
        coherence_history,
    )


def run_phase6_experiments(
    config: PhaseVIConfig,
    output_dir: Path,
    runs: int,
    seed_base: int,
    scenarios: list[PhaseVIScenario] | None = None,
    model_labels: dict[str, str] | None = None,
    experiment_tag: str | None = None,
    baseline_summary_path: Path | None = None,
    comparison_filename: str = "phase5_phase6_delta.csv",
    comparison_json_key: str = "phase5_phase6_delta",
) -> dict[str, Any]:
    scenarios = scenarios or build_phase6_scenarios()
    model_labels = model_labels or PHASE6_MODEL_LABELS
    experiment_tag = experiment_tag or getattr(config, "experiment_tag", "Phase VI")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_metrics: list[PhaseVIRunMetrics] = []
    shock_rows: list[dict[str, Any]] = []
    guardian_rows: list[dict[str, Any]] = []
    absorber_rows: list[dict[str, Any]] = []
    scaffold_rows: list[dict[str, Any]] = []
    modulation_rows: list[dict[str, Any]] = []
    inverse_triad_rows: list[dict[str, Any]] = []
    phase_manipulation_rows: list[dict[str, Any]] = []
    timeline_sums: dict[str, np.ndarray] = {
        scenario.name: np.zeros(config.steps, dtype=float) for scenario in scenarios
    }

    for scenario_index, scenario in enumerate(scenarios):
        for run_index in range(runs):
            seed = seed_base + scenario_index * 100 + run_index
            (
                metrics,
                current_shock_rows,
                current_guardian_rows,
                current_absorber_rows,
                current_scaffold_rows,
                current_modulation_rows,
                current_inverse_triad_rows,
                current_phase_manipulation_rows,
                coherence_history,
            ) = run_single_phase6(
                config=config,
                scenario=scenario,
                run_index=run_index,
                seed=seed,
            )
            run_metrics.append(metrics)
            shock_rows.extend(current_shock_rows)
            guardian_rows.extend(current_guardian_rows)
            absorber_rows.extend(current_absorber_rows)
            scaffold_rows.extend(current_scaffold_rows)
            modulation_rows.extend(current_modulation_rows)
            inverse_triad_rows.extend(current_inverse_triad_rows)
            phase_manipulation_rows.extend(current_phase_manipulation_rows)
            timeline_sums[scenario.name] += coherence_history

        checkpoint_phase6_progress(
            output_dir=output_dir,
            config=config,
            scenarios=scenarios,
            runs=runs,
            experiment_tag=experiment_tag,
            baseline_summary_path=baseline_summary_path,
            comparison_filename=comparison_filename,
            comparison_json_key=comparison_json_key,
            run_metrics=run_metrics,
            shock_rows=shock_rows,
            guardian_rows=guardian_rows,
            absorber_rows=absorber_rows,
            scaffold_rows=scaffold_rows,
            modulation_rows=modulation_rows,
            inverse_triad_rows=inverse_triad_rows,
            phase_manipulation_rows=phase_manipulation_rows,
            completed_count=scenario_index + 1,
        )

    timeline_means = {
        scenario.name: timeline_sums[scenario.name] / runs for scenario in scenarios
    }
    summary_rows, comparison_rows, run_rows = checkpoint_phase6_progress(
        output_dir=output_dir,
        config=config,
        scenarios=scenarios,
        runs=runs,
        experiment_tag=experiment_tag,
        baseline_summary_path=baseline_summary_path,
        comparison_filename=comparison_filename,
        comparison_json_key=comparison_json_key,
        run_metrics=run_metrics,
        shock_rows=shock_rows,
        guardian_rows=guardian_rows,
        absorber_rows=absorber_rows,
        scaffold_rows=scaffold_rows,
        modulation_rows=modulation_rows,
        inverse_triad_rows=inverse_triad_rows,
        phase_manipulation_rows=phase_manipulation_rows,
        completed_count=len(scenarios),
    )
    save_csv(
        output_dir / "shock_events.csv",
        shock_rows,
        fieldnames=[
            "scenario",
            "model_kind",
            "bootstrap_mode",
            "false_mode",
            "run_index",
            "seed",
            "shock_step",
            "phase_nodes",
            "geometry_nodes",
            "memory_retention",
            "core_memory_retention",
            "scaffold_retention",
        ],
    )
    save_csv(
        output_dir / "guardian_events.csv",
        guardian_rows,
        fieldnames=[
            "scenario",
            "model_kind",
            "bootstrap_mode",
            "false_mode",
            "run_index",
            "seed",
            "step",
            "guardian_size",
            "coherence",
            "cluster_fraction",
        ],
    )
    save_csv(
        output_dir / "absorber_events.csv",
        absorber_rows,
        fieldnames=[
            "scenario",
            "model_kind",
            "bootstrap_mode",
            "false_mode",
            "run_index",
            "seed",
            "step",
            "emitter_size",
            "distressed_size",
            "distressed_ratio",
            "mean_disturbance",
        ],
    )
    save_csv(
        output_dir / "scaffold_events.csv",
        scaffold_rows,
        fieldnames=[
            "scenario",
            "model_kind",
            "bootstrap_mode",
            "false_mode",
            "run_index",
            "seed",
            "step",
            "anchor_size",
            "recruit_size",
            "cluster_fraction",
            "mean_affinity",
        ],
    )
    save_csv(
        output_dir / "modulation_events.csv",
        modulation_rows,
        fieldnames=[
            "scenario",
            "model_kind",
            "bootstrap_mode",
            "false_mode",
            "run_index",
            "seed",
            "step",
            "anchor_size",
            "target_size",
            "beam_count",
            "mean_ki",
            "mean_beam_response",
        ],
    )
    save_csv(
        output_dir / "inverse_triad_events.csv",
        inverse_triad_rows,
        fieldnames=[
            "scenario",
            "model_kind",
            "bootstrap_mode",
            "false_mode",
            "run_index",
            "seed",
            "step",
            "zone",
            "triad_indices",
            "core_size",
            "triad_edge_count",
            "mean_trust",
            "mean_memory_support",
            "mean_bridge_strength",
            "mean_disturbance",
            "shell_score",
        ],
    )
    save_csv(
        output_dir / "phase_manipulation_events.csv",
        phase_manipulation_rows,
        fieldnames=[
            "scenario",
            "model_kind",
            "bootstrap_mode",
            "false_mode",
            "run_index",
            "seed",
            "step",
            "node_index",
            "target_trust",
            "balance",
            "higher_size",
            "lower_size",
            "higher_strength",
            "lower_strength",
            "score",
        ],
    )
    save_json(
        output_dir / "scenario_summary.json",
        {
            "config": asdict(config),
            "runs_per_scenario": runs,
            "scenarios": [asdict(item) for item in scenarios],
            "run_metrics": run_rows,
            "shock_events": shock_rows,
            "guardian_events": guardian_rows,
            "absorber_events": absorber_rows,
            "scaffold_events": scaffold_rows,
            "modulation_events": modulation_rows,
            "inverse_triad_events": inverse_triad_rows,
            "phase_manipulation_events": phase_manipulation_rows,
            "summary": summary_rows,
            comparison_json_key: comparison_rows,
            "coherence_timelines_mean": {
                key: value.tolist() for key, value in timeline_means.items()
            },
        },
    )

    render_grouped_metric(
        summary_rows,
        output_dir / "coherence_recovery_comparison.png",
        "coherence_recovery_ratio",
        f"{experiment_tag} Coherence Recovery Ratio",
        model_labels,
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "recovery_time_comparison.png",
        "recovery_time_steps",
        f"{experiment_tag} Recovery Time After Shock",
        model_labels,
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "cluster_recovery_comparison.png",
        "cluster_recovery_ratio",
        f"{experiment_tag} Cluster Recovery Ratio",
        model_labels,
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "memory_recovery_comparison.png",
        "memory_recovery_ratio",
        f"{experiment_tag} Dynamic Memory Recovery Ratio",
        model_labels,
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "core_memory_recovery_comparison.png",
        "core_memory_recovery_ratio",
        f"{experiment_tag} Core Memory Recovery Ratio",
        model_labels,
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "guardian_event_rate_comparison.png",
        "guardian_event_rate",
        f"{experiment_tag} Guardian Intervention Rate",
        model_labels,
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "absorber_event_rate_comparison.png",
        "absorber_event_rate",
        f"{experiment_tag} Absorber Intervention Rate",
        model_labels,
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "scaffold_event_rate_comparison.png",
        "scaffold_event_rate",
        f"{experiment_tag} Scaffold Intervention Rate",
        model_labels,
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "modulation_event_rate_comparison.png",
        "modulation_event_rate",
        f"{experiment_tag} Multifocal Modulation Rate",
        model_labels,
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "inverse_triad_event_rate_comparison.png",
        "inverse_triad_event_rate",
        f"{experiment_tag} Inverse Triad Event Rate",
        model_labels,
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "scaffold_edge_mass_comparison.png",
        "scaffold_edge_mass",
        f"{experiment_tag} Scaffold Edge Mass",
        model_labels,
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "inverse_triad_core_contamination_comparison.png",
        "inverse_triad_core_contamination_rate",
        f"{experiment_tag} Inverse Triad Core Contamination",
        model_labels,
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "phase_manipulation_event_rate_comparison.png",
        "phase_manipulation_event_rate",
        f"{experiment_tag} Phase Manipulation Event Rate",
        model_labels,
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "phase_manipulation_upward_spread_comparison.png",
        "phase_manipulation_upward_spread_fraction",
        f"{experiment_tag} Phase Manipulation Upward Spread",
        model_labels,
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "phase_manipulation_downward_spread_comparison.png",
        "phase_manipulation_downward_spread_fraction",
        f"{experiment_tag} Phase Manipulation Downward Spread",
        model_labels,
    )
    render_grouped_metric(
        summary_rows,
        output_dir / "controller_reward_comparison.png",
        "controller_reward_mean",
        f"{experiment_tag} Controller Mean Reward",
        model_labels,
    )
    render_recovery_timeline(
        scenarios,
        timeline_means,
        config,
        output_dir / "shock_response_timeline.png",
        model_labels,
        experiment_tag,
    )

    print(f"\n{experiment_tag} summary (recovery / cluster / memory / core / success):")
    for row in summary_rows:
        print(
            f"- {row['scenario']}: "
            f"coh_recovery={row['coherence_recovery_ratio_mean']:.3f}, "
            f"cluster_recovery={row['cluster_recovery_ratio_mean']:.3f}, "
            f"memory_recovery={row['memory_recovery_ratio_mean']:.3f}, "
            f"core_recovery={row['core_memory_recovery_ratio_mean']:.3f}, "
            f"success={row['recovery_success_mean']:.3f}"
        )

    return {
        "output_dir": output_dir,
        "summary_rows": summary_rows,
        "run_rows": run_rows,
        "shock_rows": shock_rows,
        "guardian_rows": guardian_rows,
        "absorber_rows": absorber_rows,
        "scaffold_rows": scaffold_rows,
        "modulation_rows": modulation_rows,
        "inverse_triad_rows": inverse_triad_rows,
        "phase_manipulation_rows": phase_manipulation_rows,
        "comparison_rows": comparison_rows,
    }
