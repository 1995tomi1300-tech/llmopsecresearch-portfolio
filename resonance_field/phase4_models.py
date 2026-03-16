from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np

from resonance_field.numeric_backend import (
    NumericBackend,
    pairwise_helical_geometry,
    pairwise_toroidal_geometry,
    weighted_force_2d,
    weighted_force_helix,
)
from resonance_field.phase4_config import PhaseIVConfig
from resonance_field.topologies import (
    build_false_frequencies,
    capped_velocity,
    clipped_scalar_velocity,
    helical_distance,
    reflect_bounds,
    toroidal_delta,
)


@dataclass
class BaseLayerResult:
    validated_adjacency: np.ndarray
    validated_weights: np.ndarray
    pair_distance: np.ndarray
    triangle_density: float
    geometry_data: np.ndarray


def pair_resonance_mask(
    pair_distance: np.ndarray,
    freq: np.ndarray,
    config: PhaseIVConfig,
) -> np.ndarray:
    freq_gap = np.abs(freq[:, None] - freq[None, :])
    mask = (pair_distance < config.radius) & (freq_gap < config.epsilon)
    np.fill_diagonal(mask, False)
    return mask


def triangle_validate(
    base_adjacency: np.ndarray,
    config: PhaseIVConfig,
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


class PhaseIVModel:
    kind = "base"

    def initialize_state(
        self,
        config: PhaseIVConfig,
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
        config: PhaseIVConfig,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def base_layer(
        self,
        state: dict[str, np.ndarray],
        config: PhaseIVConfig,
        backend: NumericBackend | None = None,
    ) -> BaseLayerResult:
        raise NotImplementedError

    def apply_motion(
        self,
        state: dict[str, np.ndarray],
        effective_weights: np.ndarray,
        phase_diff: np.ndarray,
        pair_distance: np.ndarray,
        geometry_data: np.ndarray,
        config: PhaseIVConfig,
        backend: NumericBackend | None = None,
    ) -> None:
        raise NotImplementedError

    def snapshot(self, state: dict[str, np.ndarray], config: PhaseIVConfig) -> dict[str, Any]:
        raise NotImplementedError


class PhaseIV2DModel(PhaseIVModel):
    def initialize_geometry(
        self,
        config: PhaseIVConfig,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        return {
            "pos": rng.random((config.nodes, 2)),
            "vel": np.zeros((config.nodes, 2), dtype=float),
        }

    def apply_triangle_layer(
        self,
        base_adjacency: np.ndarray,
        config: PhaseIVConfig,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return base_adjacency, base_adjacency.astype(float), 0.0

    def base_layer(
        self,
        state: dict[str, np.ndarray],
        config: PhaseIVConfig,
        backend: NumericBackend | None = None,
    ) -> BaseLayerResult:
        delta, pair_distance = pairwise_toroidal_geometry(
            state["pos"],
            backend or NumericBackend("cpu", "cpu", "numpy", "cpu", "float64"),
        )
        base_adjacency = pair_resonance_mask(pair_distance, state["freq"], config)
        validated_adjacency, validated_weights, triangle_density = self.apply_triangle_layer(
            base_adjacency,
            config,
        )
        return BaseLayerResult(
            validated_adjacency=validated_adjacency,
            validated_weights=validated_weights,
            pair_distance=pair_distance,
            triangle_density=triangle_density,
            geometry_data=delta,
        )

    def apply_motion(
        self,
        state: dict[str, np.ndarray],
        effective_weights: np.ndarray,
        phase_diff: np.ndarray,
        pair_distance: np.ndarray,
        geometry_data: np.ndarray,
        config: PhaseIVConfig,
        backend: NumericBackend | None = None,
    ) -> None:
        force = weighted_force_2d(
            effective_weights,
            np.cos(phase_diff),
            pair_distance,
            geometry_data,
            config.spatial_coupling,
            backend or NumericBackend("cpu", "cpu", "numpy", "cpu", "float64"),
        )
        state["vel"] = capped_velocity(
            config.drag * state["vel"] + force * config.dt,
            config.speed_cap,
        )
        state["pos"] = (state["pos"] + state["vel"] * config.dt) % 1.0

    def snapshot(self, state: dict[str, np.ndarray], config: PhaseIVConfig) -> dict[str, Any]:
        return {
            "projection": "2d",
            "x": state["pos"][:, 0].copy(),
            "y": state["pos"][:, 1].copy(),
        }


class PhaseIVHelixModel(PhaseIVModel):
    def initialize_geometry(
        self,
        config: PhaseIVConfig,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        max_t = config.helix_turns * 2.0 * np.pi
        return {
            "helix_t": rng.uniform(0.0, max_t, size=config.nodes),
            "helix_vel": np.zeros(config.nodes, dtype=float),
        }

    def apply_triangle_layer(
        self,
        base_adjacency: np.ndarray,
        config: PhaseIVConfig,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return base_adjacency, base_adjacency.astype(float), 0.0

    def base_layer(
        self,
        state: dict[str, np.ndarray],
        config: PhaseIVConfig,
        backend: NumericBackend | None = None,
    ) -> BaseLayerResult:
        raw_delta, pair_distance = pairwise_helical_geometry(
            state["helix_t"],
            config.helix_radius,
            config.helix_pitch,
            backend or NumericBackend("cpu", "cpu", "numpy", "cpu", "float64"),
        )
        base_adjacency = pair_resonance_mask(pair_distance, state["freq"], config)
        validated_adjacency, validated_weights, triangle_density = self.apply_triangle_layer(
            base_adjacency,
            config,
        )
        return BaseLayerResult(
            validated_adjacency=validated_adjacency,
            validated_weights=validated_weights,
            pair_distance=pair_distance,
            triangle_density=triangle_density,
            geometry_data=raw_delta,
        )

    def apply_motion(
        self,
        state: dict[str, np.ndarray],
        effective_weights: np.ndarray,
        phase_diff: np.ndarray,
        pair_distance: np.ndarray,
        geometry_data: np.ndarray,
        config: PhaseIVConfig,
        backend: NumericBackend | None = None,
    ) -> None:
        force = weighted_force_helix(
            effective_weights,
            np.cos(phase_diff),
            geometry_data,
            config.spatial_coupling,
            backend or NumericBackend("cpu", "cpu", "numpy", "cpu", "float64"),
        )
        state["helix_vel"] = clipped_scalar_velocity(
            config.drag * state["helix_vel"] + force * config.dt,
            config.speed_cap,
        )
        state["helix_t"] = state["helix_t"] + state["helix_vel"] * config.dt
        max_t = config.helix_turns * 2.0 * np.pi
        state["helix_t"], state["helix_vel"] = reflect_bounds(
            state["helix_t"],
            state["helix_vel"],
            0.0,
            max_t,
        )

    def snapshot(self, state: dict[str, np.ndarray], config: PhaseIVConfig) -> dict[str, Any]:
        helix_t = state["helix_t"]
        x = config.helix_radius * np.cos(helix_t)
        y = config.helix_radius * np.sin(helix_t)
        z = config.helix_pitch * helix_t
        return {
            "projection": "3d",
            "x": x.copy(),
            "y": y.copy(),
            "z": z.copy(),
        }


class BaselineMemoryModel(PhaseIV2DModel):
    kind = "baseline"


class TriangleMemoryModel(PhaseIV2DModel):
    kind = "triangle"

    def apply_triangle_layer(
        self,
        base_adjacency: np.ndarray,
        config: PhaseIVConfig,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return triangle_validate(base_adjacency, config)


class HelixMemoryModel(PhaseIVHelixModel):
    kind = "helix"


class HybridManifoldMemoryModel(PhaseIVHelixModel):
    kind = "hybrid_manifold"

    def apply_triangle_layer(
        self,
        base_adjacency: np.ndarray,
        config: PhaseIVConfig,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return triangle_validate(base_adjacency, config)


PHASE4_MODEL_REGISTRY = {
    "baseline": BaselineMemoryModel(),
    "triangle": TriangleMemoryModel(),
    "helix": HelixMemoryModel(),
    "hybrid_manifold": HybridManifoldMemoryModel(),
}
