from __future__ import annotations

from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

from resonance_field.phase16_config import PhaseXVIConfig
from resonance_field.phase16_crossfire_singleton import build_crossfire_config
from resonance_field.phase16_quantized_pilot import apply_variant


TEACHING_PROFILE_NAME = "crossfire_loading_v1"
TEACHING_PROFILE_STRENGTH = 0.54
TEACHING_PROFILE_NOTES = (
    "Stable absorptive teaching regime extracted from the crossfire threshold map. "
    "Designed to raise coherence while preserving enough immune selectivity."
)


def interpolate_value(base: Any, target: Any, strength: float) -> Any:
    if isinstance(base, bool) and isinstance(target, bool):
        return target if strength > 0.0 else base
    if isinstance(base, int) and isinstance(target, int):
        return int(round(base + (target - base) * strength))
    if isinstance(base, float) and isinstance(target, float):
        return float(base + (target - base) * strength)
    return target if strength > 0.0 else base


def build_crossfire_profile(base: PhaseXVIConfig, strength: float) -> PhaseXVIConfig:
    if strength <= 0.0:
        return replace(base, quantized_scalar_vector_enabled=False)

    target = build_crossfire_config(base)
    overrides: dict[str, Any] = {}
    for field in fields(PhaseXVIConfig):
        name = field.name
        base_value = getattr(base, name)
        target_value = getattr(target, name)
        if base_value == target_value:
            continue
        overrides[name] = interpolate_value(base_value, target_value, strength)

    overrides["experiment_tag"] = f"Phase XVI Teaching Profile {strength:.2f}"
    overrides["quantized_scalar_vector_enabled"] = True
    overrides["modulation_training_enabled"] = True
    overrides["phase_manipulation_enabled"] = True
    return replace(base, **overrides)


def build_loading_teaching_profile(
    *,
    nodes: int = 120,
    steps: int = 220,
    backend: str = "auto",
    variant: str = "critical_push",
) -> PhaseXVIConfig:
    base = PhaseXVIConfig(
        nodes=nodes,
        steps=steps,
        shock_step=max(24, steps // 4),
        late_window=max(40, steps // 4),
        benchmark_repeats=2,
        compute_backend=backend,
        collect_event_rows=True,
    )
    base = apply_variant(base, variant)
    return build_crossfire_profile(base, TEACHING_PROFILE_STRENGTH)


def describe_loading_teaching_profile(config: PhaseXVIConfig) -> dict[str, Any]:
    return {
        "profile_name": TEACHING_PROFILE_NAME,
        "profile_strength": TEACHING_PROFILE_STRENGTH,
        "profile_notes": TEACHING_PROFILE_NOTES,
        "config": asdict(config),
    }


def write_loading_teaching_manifest(
    output_path: str | Path,
    *,
    nodes: int = 120,
    steps: int = 220,
    backend: str = "auto",
    variant: str = "critical_push",
) -> dict[str, Any]:
    import json

    config = build_loading_teaching_profile(
        nodes=nodes,
        steps=steps,
        backend=backend,
        variant=variant,
    )
    payload = describe_loading_teaching_profile(config)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
