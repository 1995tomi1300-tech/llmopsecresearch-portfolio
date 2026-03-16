from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np

from resonance_field.config import ExperimentConfig


@dataclass
class StepResult:
    support: np.ndarray
    pair_distance: np.ndarray
    phase_drive: np.ndarray
    resonance_energy: float
    triangle_density: float = 0.0
    spatial_clustering: float = 0.0


def build_false_frequencies(rng: np.random.Generator, count: int) -> np.ndarray:
    side = rng.random(count) < 0.5
    low_band = rng.uniform(0.62, 0.82, size=count)
    high_band = rng.uniform(1.18, 1.38, size=count)
    return np.where(side, low_band, high_band)


def bootstrap_kick(theta: np.ndarray, anchor: float, strength: float) -> np.ndarray:
    return theta + strength * np.sin(anchor - theta)


def toroidal_delta(pos: np.ndarray) -> np.ndarray:
    delta = pos[None, :, :] - pos[:, None, :]
    return (delta + 0.5) % 1.0 - 0.5


def capped_velocity(vel: np.ndarray, speed_cap: float) -> np.ndarray:
    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    scale = np.minimum(1.0, speed_cap / np.maximum(speed, 1e-9))
    return vel * scale


def clipped_scalar_velocity(vel: np.ndarray, speed_cap: float) -> np.ndarray:
    return np.clip(vel, -speed_cap, speed_cap)


def support_spatial_clustering(support: np.ndarray, pair_distance: np.ndarray, radius: float) -> float:
    upper = np.triu(support, k=1)
    if not np.any(upper):
        return 0.0
    values = pair_distance[upper]
    return float(np.clip(np.mean(1.0 - values / max(radius, 1e-9)), 0.0, 1.0))


def support_resonance_energy(support: np.ndarray, theta: np.ndarray) -> float:
    phase_diff = theta[None, :] - theta[:, None]
    return float(0.5 * np.sum(np.cos(phase_diff) * support))


def pair_mask(
    pair_distance: np.ndarray,
    freq: np.ndarray,
    radius: float,
    epsilon: float,
) -> np.ndarray:
    freq_gap = np.abs(freq[:, None] - freq[None, :])
    support = (pair_distance < radius) & (freq_gap < epsilon)
    np.fill_diagonal(support, False)
    return support


def triangle_support_terms(
    pair_support: np.ndarray,
    theta: np.ndarray,
    coupling: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    pair_float = pair_support.astype(float)
    common_neighbors = pair_float @ pair_float
    triangle_weights = np.where(pair_support, common_neighbors, 0.0)
    support = triangle_weights > 0.0

    z = np.exp(1j * theta)
    pair_complex = pair_float * (z[:, None] * z[None, :])
    triad_pair_sum = 0.5 * np.sum((pair_float @ pair_complex) * pair_float, axis=1)
    phase_drive = coupling * np.imag(np.conjugate(z) ** 2 * triad_pair_sum)

    triangle_count = float(np.trace(common_neighbors @ pair_float) / 6.0)
    total_triangles = comb(pair_support.shape[0], 3)
    density = triangle_count / total_triangles if total_triangles else 0.0
    return support, triangle_weights, phase_drive, density


def helix_coordinates(t: np.ndarray, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = config.helix_radius * np.cos(t)
    y = config.helix_radius * np.sin(t)
    z = config.helix_pitch * t
    return x, y, z


def helical_distance(t: np.ndarray, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    raw = t[None, :] - t[:, None]
    wrapped = (raw + np.pi) % (2.0 * np.pi) - np.pi
    turn_delta = (raw - wrapped) / (2.0 * np.pi)
    distance = np.sqrt(
        (config.helix_radius * wrapped) ** 2
        + (config.helix_pitch * 2.0 * np.pi * turn_delta) ** 2
    )
    np.fill_diagonal(distance, np.inf)
    return distance, raw


def reflect_bounds(
    values: np.ndarray,
    velocity: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    below = values < lower
    if np.any(below):
        values[below] = 2.0 * lower - values[below]
        velocity[below] *= -1.0

    above = values > upper
    if np.any(above):
        values[above] = 2.0 * upper - values[above]
        velocity[above] *= -1.0

    values = np.clip(values, lower, upper)
    return values, velocity


class ResonanceTopology:
    kind = "base"

    def initialize_state(
        self,
        config: ExperimentConfig,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def step(
        self,
        state: dict[str, np.ndarray],
        config: ExperimentConfig,
    ) -> StepResult:
        raise NotImplementedError

    def snapshot(self, state: dict[str, np.ndarray], config: ExperimentConfig) -> dict[str, Any]:
        raise NotImplementedError

    def manifold_coordinate(self, state: dict[str, np.ndarray]) -> np.ndarray | None:
        return None


class BaselineGraphTopology(ResonanceTopology):
    kind = "baseline"

    def initialize_state(
        self,
        config: ExperimentConfig,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        return {
            "pos": rng.random((config.nodes, 2)),
            "vel": np.zeros((config.nodes, 2), dtype=float),
        }

    def step(
        self,
        state: dict[str, np.ndarray],
        config: ExperimentConfig,
    ) -> StepResult:
        pos = state["pos"]
        vel = state["vel"]
        theta = state["theta"]
        freq = state["freq"]

        delta = toroidal_delta(pos)
        pair_distance = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(pair_distance, np.inf)
        support = pair_mask(pair_distance, freq, config.radius, config.epsilon)

        phase_diff = theta[None, :] - theta[:, None]
        phase_drive = config.phase_coupling * np.sum(np.sin(phase_diff) * support, axis=1)

        unit_delta = delta / (pair_distance[..., None] + 1e-6)
        force = config.spatial_coupling * np.sum(
            np.cos(phase_diff)[:, :, None] * support[:, :, None] * unit_delta,
            axis=1,
        )

        state["vel"] = capped_velocity(config.drag * vel + force * config.dt, config.speed_cap)
        state["pos"] = (pos + state["vel"] * config.dt) % 1.0

        return StepResult(
            support=support,
            pair_distance=pair_distance,
            phase_drive=phase_drive,
            resonance_energy=support_resonance_energy(support, theta),
            spatial_clustering=support_spatial_clustering(support, pair_distance, config.radius),
        )

    def snapshot(self, state: dict[str, np.ndarray], config: ExperimentConfig) -> dict[str, Any]:
        return {
            "projection": "2d",
            "x": state["pos"][:, 0].copy(),
            "y": state["pos"][:, 1].copy(),
        }


class TriangleSimplicialTopology(ResonanceTopology):
    kind = "triangle"

    def initialize_state(
        self,
        config: ExperimentConfig,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        return {
            "pos": rng.random((config.nodes, 2)),
            "vel": np.zeros((config.nodes, 2), dtype=float),
        }

    def step(
        self,
        state: dict[str, np.ndarray],
        config: ExperimentConfig,
    ) -> StepResult:
        pos = state["pos"]
        vel = state["vel"]
        theta = state["theta"]
        freq = state["freq"]

        delta = toroidal_delta(pos)
        pair_distance = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(pair_distance, np.inf)
        pair_support = pair_mask(pair_distance, freq, config.radius, config.epsilon)
        support, triangle_weights, phase_drive, triangle_density = triangle_support_terms(
            pair_support,
            theta,
            config.phase_coupling,
        )

        phase_diff = theta[None, :] - theta[:, None]
        weight_scale = np.maximum(np.max(triangle_weights), 1.0)
        weighted_support = triangle_weights / weight_scale
        unit_delta = delta / (pair_distance[..., None] + 1e-6)
        force = config.spatial_coupling * np.sum(
            np.cos(phase_diff)[:, :, None] * weighted_support[:, :, None] * unit_delta,
            axis=1,
        )

        state["vel"] = capped_velocity(config.drag * vel + force * config.dt, config.speed_cap)
        state["pos"] = (pos + state["vel"] * config.dt) % 1.0

        return StepResult(
            support=support,
            pair_distance=pair_distance,
            phase_drive=phase_drive,
            resonance_energy=support_resonance_energy(support, theta),
            triangle_density=triangle_density,
            spatial_clustering=support_spatial_clustering(support, pair_distance, config.radius),
        )

    def snapshot(self, state: dict[str, np.ndarray], config: ExperimentConfig) -> dict[str, Any]:
        return {
            "projection": "2d",
            "x": state["pos"][:, 0].copy(),
            "y": state["pos"][:, 1].copy(),
        }


class HelixFieldTopology(ResonanceTopology):
    kind = "helix"

    def initialize_state(
        self,
        config: ExperimentConfig,
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
        config: ExperimentConfig,
    ) -> StepResult:
        helix_t = state["helix_t"]
        helix_vel = state["helix_vel"]
        theta = state["theta"]
        freq = state["freq"]

        pair_distance, raw_delta = helical_distance(helix_t, config)
        support = pair_mask(pair_distance, freq, config.radius, config.epsilon)

        phase_diff = theta[None, :] - theta[:, None]
        phase_drive = config.phase_coupling * np.sum(np.sin(phase_diff) * support, axis=1)

        scalar_direction = raw_delta / (np.abs(raw_delta) + 1e-6)
        force = config.spatial_coupling * np.sum(
            np.cos(phase_diff) * support * scalar_direction,
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

        return StepResult(
            support=support,
            pair_distance=pair_distance,
            phase_drive=phase_drive,
            resonance_energy=support_resonance_energy(support, theta),
            spatial_clustering=support_spatial_clustering(support, pair_distance, config.radius),
        )

    def snapshot(self, state: dict[str, np.ndarray], config: ExperimentConfig) -> dict[str, Any]:
        x, y, z = helix_coordinates(state["helix_t"], config)
        return {
            "projection": "3d",
            "x": x.copy(),
            "y": y.copy(),
            "z": z.copy(),
        }

    def manifold_coordinate(self, state: dict[str, np.ndarray]) -> np.ndarray | None:
        return state["helix_t"].copy()


class HybridHelixTriangleTopology(ResonanceTopology):
    kind = "hybrid"

    def initialize_state(
        self,
        config: ExperimentConfig,
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
        config: ExperimentConfig,
    ) -> StepResult:
        helix_t = state["helix_t"]
        helix_vel = state["helix_vel"]
        theta = state["theta"]
        freq = state["freq"]

        pair_distance, raw_delta = helical_distance(helix_t, config)
        pair_support = pair_mask(pair_distance, freq, config.radius, config.epsilon)
        support, triangle_weights, phase_drive, triangle_density = triangle_support_terms(
            pair_support,
            theta,
            config.phase_coupling,
        )

        phase_diff = theta[None, :] - theta[:, None]
        weight_scale = np.maximum(np.max(triangle_weights), 1.0)
        weighted_support = triangle_weights / weight_scale
        scalar_direction = raw_delta / (np.abs(raw_delta) + 1e-6)
        force = config.spatial_coupling * np.sum(
            np.cos(phase_diff) * weighted_support * scalar_direction,
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

        return StepResult(
            support=support,
            pair_distance=pair_distance,
            phase_drive=phase_drive,
            resonance_energy=support_resonance_energy(support, theta),
            triangle_density=triangle_density,
            spatial_clustering=support_spatial_clustering(support, pair_distance, config.radius),
        )

    def snapshot(self, state: dict[str, np.ndarray], config: ExperimentConfig) -> dict[str, Any]:
        x, y, z = helix_coordinates(state["helix_t"], config)
        return {
            "projection": "3d",
            "x": x.copy(),
            "y": y.copy(),
            "z": z.copy(),
        }

    def manifold_coordinate(self, state: dict[str, np.ndarray]) -> np.ndarray | None:
        return state["helix_t"].copy()


TOPOLOGY_REGISTRY: dict[str, ResonanceTopology] = {
    "baseline": BaselineGraphTopology(),
    "triangle": TriangleSimplicialTopology(),
    "helix": HelixFieldTopology(),
    "hybrid": HybridHelixTriangleTopology(),
}
