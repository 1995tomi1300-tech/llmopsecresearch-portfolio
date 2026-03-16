from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np


_TORCH_MODULE = None
_TORCH_IMPORT_ERROR: Exception | None = None


@dataclass(frozen=True)
class NumericBackend:
    request: str
    mode: str
    library: str
    device: str
    precision: str
    note: str = ""


@dataclass(frozen=True)
class PhaseKernels:
    phase_diff: np.ndarray
    sin_phase: np.ndarray
    cos_phase: np.ndarray
    positive_alignment: np.ndarray


def _load_torch():
    global _TORCH_MODULE, _TORCH_IMPORT_ERROR
    if _TORCH_MODULE is not None:
        return _TORCH_MODULE
    if _TORCH_IMPORT_ERROR is not None:
        raise _TORCH_IMPORT_ERROR
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        _TORCH_IMPORT_ERROR = exc
        raise
    _TORCH_MODULE = torch
    return torch


def _torch_dtype(precision: str):
    torch = _load_torch()
    if precision == "float64":
        return torch.float64
    return torch.float32


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()


def resolve_numeric_backend(config: Any) -> NumericBackend:
    request = str(getattr(config, "compute_backend", "cpu")).lower()
    precision = str(getattr(config, "backend_precision", "float32")).lower()
    gpu_min_nodes = int(getattr(config, "gpu_min_nodes", 192))
    nodes = int(getattr(config, "nodes", 0))

    if request not in {"auto", "cpu", "gpu", "cuda"}:
        request = "auto"

    wants_gpu = request in {"gpu", "cuda"} or (request == "auto" and nodes >= gpu_min_nodes)
    if wants_gpu:
        try:
            torch = _load_torch()
            if torch.cuda.is_available():
                device_index = int(getattr(config, "backend_gpu_index", 0))
                device = f"cuda:{device_index}"
                return NumericBackend(
                    request=request,
                    mode="gpu",
                    library="torch",
                    device=device,
                    precision=precision,
                    note="gpu kernels enabled",
                )
            note = "torch present but CUDA unavailable"
        except Exception as exc:  # pragma: no cover - environment dependent
            note = f"gpu fallback: {exc}"
        return NumericBackend(
            request=request,
            mode="cpu",
            library="numpy",
            device="cpu",
            precision="float64",
            note=note,
        )

    return NumericBackend(
        request=request,
        mode="cpu",
        library="numpy",
        device="cpu",
        precision="float64",
        note="cpu path selected",
    )


def backend_info_dict(backend: NumericBackend) -> dict[str, str]:
    return {
        "request": backend.request,
        "mode": backend.mode,
        "library": backend.library,
        "device": backend.device,
        "precision": backend.precision,
        "note": backend.note,
    }


def phase_kernels(theta: np.ndarray, backend: NumericBackend) -> PhaseKernels:
    if backend.mode == "gpu" and backend.library == "torch":
        torch = _load_torch()
        theta_t = torch.as_tensor(theta, dtype=_torch_dtype(backend.precision), device=backend.device)
        phase_diff_t = theta_t[None, :] - theta_t[:, None]
        sin_t = torch.sin(phase_diff_t)
        cos_t = torch.cos(phase_diff_t)
        pos_t = torch.clamp(cos_t, min=0.0)
        return PhaseKernels(
            phase_diff=_as_numpy(phase_diff_t),
            sin_phase=_as_numpy(sin_t),
            cos_phase=_as_numpy(cos_t),
            positive_alignment=_as_numpy(pos_t),
        )

    phase_diff = theta[None, :] - theta[:, None]
    cos_phase = np.cos(phase_diff)
    return PhaseKernels(
        phase_diff=phase_diff,
        sin_phase=np.sin(phase_diff),
        cos_phase=cos_phase,
        positive_alignment=np.clip(cos_phase, 0.0, 1.0),
    )


def pairwise_toroidal_geometry(
    pos: np.ndarray,
    backend: NumericBackend,
) -> tuple[np.ndarray, np.ndarray]:
    if backend.mode == "gpu" and backend.library == "torch":
        torch = _load_torch()
        pos_t = torch.as_tensor(pos, dtype=_torch_dtype(backend.precision), device=backend.device)
        delta_t = pos_t[None, :, :] - pos_t[:, None, :]
        delta_t = torch.remainder(delta_t + 0.5, 1.0) - 0.5
        pair_distance_t = torch.linalg.vector_norm(delta_t, dim=2)
        idx = torch.arange(pos_t.shape[0], device=backend.device)
        pair_distance_t[idx, idx] = torch.inf
        return _as_numpy(delta_t), _as_numpy(pair_distance_t)

    delta = pos[None, :, :] - pos[:, None, :]
    delta = (delta + 0.5) % 1.0 - 0.5
    pair_distance = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(pair_distance, np.inf)
    return delta, pair_distance


def pairwise_helical_geometry(
    helix_t: np.ndarray,
    helix_radius: float,
    helix_pitch: float,
    backend: NumericBackend,
) -> tuple[np.ndarray, np.ndarray]:
    if backend.mode == "gpu" and backend.library == "torch":
        torch = _load_torch()
        t_t = torch.as_tensor(helix_t, dtype=_torch_dtype(backend.precision), device=backend.device)
        raw_t = t_t[None, :] - t_t[:, None]
        wrapped_t = torch.remainder(raw_t + np.pi, 2.0 * np.pi) - np.pi
        turn_delta_t = (raw_t - wrapped_t) / (2.0 * np.pi)
        pair_distance_t = torch.sqrt(
            (helix_radius * wrapped_t) ** 2
            + (helix_pitch * 2.0 * np.pi * turn_delta_t) ** 2
        )
        idx = torch.arange(t_t.shape[0], device=backend.device)
        pair_distance_t[idx, idx] = torch.inf
        return _as_numpy(raw_t), _as_numpy(pair_distance_t)

    raw = helix_t[None, :] - helix_t[:, None]
    wrapped = (raw + np.pi) % (2.0 * np.pi) - np.pi
    turn_delta = (raw - wrapped) / (2.0 * np.pi)
    pair_distance = np.sqrt(
        (helix_radius * wrapped) ** 2
        + (helix_pitch * 2.0 * np.pi * turn_delta) ** 2
    )
    np.fill_diagonal(pair_distance, np.inf)
    return raw, pair_distance


def weighted_force_2d(
    effective_weights: np.ndarray,
    cos_phase: np.ndarray,
    pair_distance: np.ndarray,
    geometry_data: np.ndarray,
    spatial_coupling: float,
    backend: NumericBackend,
) -> np.ndarray:
    if backend.mode == "gpu" and backend.library == "torch":
        torch = _load_torch()
        weights_t = torch.as_tensor(
            effective_weights, dtype=_torch_dtype(backend.precision), device=backend.device
        )
        cos_t = torch.as_tensor(cos_phase, dtype=_torch_dtype(backend.precision), device=backend.device)
        dist_t = torch.as_tensor(pair_distance, dtype=_torch_dtype(backend.precision), device=backend.device)
        geom_t = torch.as_tensor(geometry_data, dtype=_torch_dtype(backend.precision), device=backend.device)
        unit_delta_t = geom_t / (dist_t[..., None] + 1e-6)
        force_t = spatial_coupling * torch.sum(
            weights_t[:, :, None] * cos_t[:, :, None] * unit_delta_t,
            dim=1,
        )
        return _as_numpy(force_t)

    unit_delta = geometry_data / (pair_distance[..., None] + 1e-6)
    return spatial_coupling * np.sum(
        effective_weights[:, :, None] * cos_phase[:, :, None] * unit_delta,
        axis=1,
    )


def weighted_force_helix(
    effective_weights: np.ndarray,
    cos_phase: np.ndarray,
    raw_delta: np.ndarray,
    spatial_coupling: float,
    backend: NumericBackend,
) -> np.ndarray:
    if backend.mode == "gpu" and backend.library == "torch":
        torch = _load_torch()
        weights_t = torch.as_tensor(
            effective_weights, dtype=_torch_dtype(backend.precision), device=backend.device
        )
        cos_t = torch.as_tensor(cos_phase, dtype=_torch_dtype(backend.precision), device=backend.device)
        delta_t = torch.as_tensor(raw_delta, dtype=_torch_dtype(backend.precision), device=backend.device)
        scalar_t = delta_t / (torch.abs(delta_t) + 1e-6)
        force_t = spatial_coupling * torch.sum(weights_t * cos_t * scalar_t, dim=1)
        return _as_numpy(force_t)

    scalar_direction = raw_delta / (np.abs(raw_delta) + 1e-6)
    return spatial_coupling * np.sum(effective_weights * cos_phase * scalar_direction, axis=1)


def benchmark_pairwise_kernels(
    backend: NumericBackend,
    nodes: int,
    repeats: int,
    seed: int = 20260312,
) -> dict[str, float | str]:
    rng = np.random.default_rng(seed)
    pos = rng.random((nodes, 2))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=nodes)
    weights = rng.random((nodes, nodes))
    weights = 0.5 * (weights + weights.T)
    np.fill_diagonal(weights, 0.0)

    if backend.mode == "gpu" and backend.library == "torch":
        torch = _load_torch()
        torch.cuda.synchronize(device=backend.device)

    # Warm up kernels outside the measurement window so CUDA init does not dominate.
    delta, pair_distance = pairwise_toroidal_geometry(pos, backend)
    kernels = phase_kernels(theta, backend)
    weighted_force_2d(weights, kernels.cos_phase, pair_distance, delta, 0.012, backend)
    if backend.mode == "gpu" and backend.library == "torch":
        torch = _load_torch()
        torch.cuda.synchronize(device=backend.device)

    geom_time = 0.0
    phase_time = 0.0
    force_time = 0.0
    for _ in range(repeats):
        start = perf_counter()
        delta, pair_distance = pairwise_toroidal_geometry(pos, backend)
        if backend.mode == "gpu" and backend.library == "torch":
            _load_torch().cuda.synchronize(device=backend.device)
        geom_time += perf_counter() - start

        start = perf_counter()
        kernels = phase_kernels(theta, backend)
        if backend.mode == "gpu" and backend.library == "torch":
            _load_torch().cuda.synchronize(device=backend.device)
        phase_time += perf_counter() - start

        start = perf_counter()
        weighted_force_2d(weights, kernels.cos_phase, pair_distance, delta, 0.012, backend)
        if backend.mode == "gpu" and backend.library == "torch":
            _load_torch().cuda.synchronize(device=backend.device)
        force_time += perf_counter() - start

    return {
        "backend_mode": backend.mode,
        "backend_library": backend.library,
        "backend_device": backend.device,
        "nodes": nodes,
        "repeats": repeats,
        "geometry_ms": 1000.0 * geom_time / max(1, repeats),
        "phase_ms": 1000.0 * phase_time / max(1, repeats),
        "force_ms": 1000.0 * force_time / max(1, repeats),
        "total_ms": 1000.0 * (geom_time + phase_time + force_time) / max(1, repeats),
    }
