from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

from resonance_field.phase3_config import PhaseIIIConfig
from resonance_field.topologies import (
    bootstrap_kick,
    build_false_frequencies,
    capped_velocity,
    clipped_scalar_velocity,
    helical_distance,
    reflect_bounds,
    toroidal_delta,
)


@dataclass
class LayeredStepResult:
    validated_adjacency: np.ndarray
    validated_weights: np.ndarray
    pair_distance: np.ndarray
    phase_drive: np.ndarray
    resonance_energy: float
    triangle_density: float


def weighted_resonance_energy(weights: np.ndarray, theta: np.ndarray) -> float:
    phase_diff = theta[None, :] - theta[:, None]
    return float(0.5 * np.sum(np.cos(phase_diff) * weights))


def pair_resonance_mask(
    pair_distance: np.ndarray,
    freq: np.ndarray,
    config: PhaseIIIConfig,
) -> np.ndarray:
    freq_gap = np.abs(freq[:, None] - freq[None, :])
    mask = (pair_distance < config.radius) & (freq_gap < config.epsilon)
    np.fill_diagonal(mask, False)
    return mask


def triangle_validate(
    base_adjacency: np.ndarray,
    config: PhaseIIIConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    base_float = base_adjacency.astype(float)
    common_neighbors = base_float @ base_float
    supported_edges = base_adjacency & (common_neighbors > 0.0)
    weight_scale = np.maximum(np.max(common_neighbors), 1.0)

    weights = np.where(base_adjacency, config.triangle_weaken, 0.0)
    weights[supported_edges] = 1.0 + (
        config.triangle_reinforce * common_neighbors[supported_edges] / weight_scale
    )
    validated = weights > config.activation_threshold

    triangle_count = float(np.trace(common_neighbors @ base_float) / 6.0)
    total_triangles = comb(base_adjacency.shape[0], 3)
    triangle_density = triangle_count / total_triangles if total_triangles else 0.0
    return validated, weights, triangle_density


class PhaseIIIModel:
    kind = "base"

    def initialize_state(
        self,
        config: PhaseIIIConfig,
        rng: np.random.Generator,
        false_mode: str,
    ) -> dict[str, np.ndarray]:
        state = self.initialize_geometry(config, rng)
        state["theta"] = rng.uniform(0.0, 2.0 * np.pi, size=config.nodes)
        state["freq"] = rng.uniform(0.95, 1.05, size=config.nodes)
        state["false_mask"] = np.zeros(config.nodes, dtype=bool)

        false_count = int(round(config.nodes * config.false_ratio)) if false_mode != "none" else 0
        if false_count:
            false_indices = rng.choice(config.nodes, size=false_count, replace=False)
            state["false_mask"][false_indices] = True
            state["freq"][state["false_mask"]] = build_false_frequencies(rng, false_count)
        return state

    def initialize_geometry(
        self,
        config: PhaseIIIConfig,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def step(
        self,
        state: dict[str, np.ndarray],
        config: PhaseIIIConfig,
    ) -> LayeredStepResult:
        raise NotImplementedError

    def apply_triangle_layer(
        self,
        base_adjacency: np.ndarray,
        config: PhaseIIIConfig,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return base_adjacency, base_adjacency.astype(float), 0.0


class PhaseIII2DModel(PhaseIIIModel):
    def initialize_geometry(
        self,
        config: PhaseIIIConfig,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        return {
            "pos": rng.random((config.nodes, 2)),
            "vel": np.zeros((config.nodes, 2), dtype=float),
        }

    def step(
        self,
        state: dict[str, np.ndarray],
        config: PhaseIIIConfig,
    ) -> LayeredStepResult:
        pos = state["pos"]
        vel = state["vel"]
        theta = state["theta"]
        freq = state["freq"]

        delta = toroidal_delta(pos)
        pair_distance = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(pair_distance, np.inf)
        base_adjacency = pair_resonance_mask(pair_distance, freq, config)
        validated_adjacency, validated_weights, triangle_density = self.apply_triangle_layer(
            base_adjacency,
            config,
        )

        phase_diff = theta[None, :] - theta[:, None]
        phase_drive = config.phase_coupling * np.sum(
            validated_weights * np.sin(phase_diff),
            axis=1,
        )

        unit_delta = delta / (pair_distance[..., None] + 1e-6)
        force = config.spatial_coupling * np.sum(
            validated_weights[:, :, None] * np.cos(phase_diff)[:, :, None] * unit_delta,
            axis=1,
        )

        state["vel"] = capped_velocity(config.drag * vel + force * config.dt, config.speed_cap)
        state["pos"] = (pos + state["vel"] * config.dt) % 1.0

        return LayeredStepResult(
            validated_adjacency=validated_adjacency,
            validated_weights=validated_weights,
            pair_distance=pair_distance,
            phase_drive=phase_drive,
            resonance_energy=weighted_resonance_energy(validated_weights, theta),
            triangle_density=triangle_density,
        )


class PhaseIIIHelixModel(PhaseIIIModel):
    def initialize_geometry(
        self,
        config: PhaseIIIConfig,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        max_t = config.helix_turns * 2.0 * np.pi
        return {
            "helix_t": rng.uniform(0.0, max_t, size=config.nodes),
            "helix_vel": np.zeros(config.nodes, dtype=float),
        }

    def step(
        self,
        state: dict[str, np.ndarray],
        config: PhaseIIIConfig,
    ) -> LayeredStepResult:
        helix_t = state["helix_t"]
        helix_vel = state["helix_vel"]
        theta = state["theta"]
        freq = state["freq"]

        pair_distance, raw_delta = helical_distance(helix_t, config)
        base_adjacency = pair_resonance_mask(pair_distance, freq, config)
        validated_adjacency, validated_weights, triangle_density = self.apply_triangle_layer(
            base_adjacency,
            config,
        )

        phase_diff = theta[None, :] - theta[:, None]
        phase_drive = config.phase_coupling * np.sum(
            validated_weights * np.sin(phase_diff),
            axis=1,
        )

        scalar_direction = raw_delta / (np.abs(raw_delta) + 1e-6)
        force = config.spatial_coupling * np.sum(
            validated_weights * np.cos(phase_diff) * scalar_direction,
            axis=1,
        )

        state["helix_vel"] = clipped_scalar_velocity(
            config.drag * helix_vel + force * config.dt,
            config.speed_cap,
        )
        state["helix_t"] = helix_t + state["helix_vel"] * config.dt
        max_t = config.helix_turns * 2.0 * np.pi
        state["helix_t"], state["helix_vel"] = reflect_bounds(
            state["helix_t"],
            state["helix_vel"],
            0.0,
            max_t,
        )

        return LayeredStepResult(
            validated_adjacency=validated_adjacency,
            validated_weights=validated_weights,
            pair_distance=pair_distance,
            phase_drive=phase_drive,
            resonance_energy=weighted_resonance_energy(validated_weights, theta),
            triangle_density=triangle_density,
        )


class BaselineLayeredModel(PhaseIII2DModel):
    kind = "baseline"


class TriangleLayeredModel(PhaseIII2DModel):
    kind = "triangle"

    def apply_triangle_layer(
        self,
        base_adjacency: np.ndarray,
        config: PhaseIIIConfig,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return triangle_validate(base_adjacency, config)


class HelixLayeredModel(PhaseIIIHelixModel):
    kind = "helix"


class HybridManifoldModel(PhaseIIIHelixModel):
    kind = "hybrid_manifold"

    def apply_triangle_layer(
        self,
        base_adjacency: np.ndarray,
        config: PhaseIIIConfig,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return triangle_validate(base_adjacency, config)


PHASE3_MODEL_REGISTRY = {
    "baseline": BaselineLayeredModel(),
    "triangle": TriangleLayeredModel(),
    "helix": HelixLayeredModel(),
    "hybrid_manifold": HybridManifoldModel(),
}
