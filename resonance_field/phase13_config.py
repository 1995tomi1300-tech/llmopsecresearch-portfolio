from __future__ import annotations

from dataclasses import dataclass

from resonance_field.phase6_config import PhaseVIScenario
from resonance_field.phase12_config import PhaseXIIConfig


@dataclass(frozen=True)
class PhaseXIIIConfig(PhaseXIIConfig):
    experiment_tag: str = "Phase XIII"
    compute_backend: str = "auto"
    backend_precision: str = "float32"
    backend_gpu_index: int = 0
    gpu_min_nodes: int = 1024
    benchmark_nodes: int = 1024
    benchmark_repeats: int = 12
    controller_interval: int = 4
    cluster_refresh_interval: int = 4
    collect_event_rows: bool = False


PHASE13_MODEL_LABELS = {
    "baseline": "Baseline+GPU Backend",
    "triangle": "Triangle+GPU Backend",
    "helix": "Helix+GPU Backend",
    "hybrid_manifold": "Hybrid Manifold+GPU Backend",
}


def build_phase13_scenarios() -> list[PhaseVIScenario]:
    scenarios: list[PhaseVIScenario] = []
    for model_kind in ("baseline", "triangle", "helix", "hybrid_manifold"):
        for bootstrap_mode in ("single", "periodic"):
            for false_mode in ("none", "random", "noisy"):
                name = f"{model_kind}_{bootstrap_mode}_{false_mode}"
                label = (
                    f"{PHASE13_MODEL_LABELS[model_kind]} | "
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
