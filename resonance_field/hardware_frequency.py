from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_cpu_mhz() -> dict[str, Any]:
    values: list[float] = []
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if "cpu MHz" not in line:
                continue
            try:
                values.append(float(line.split(":", 1)[1].strip()))
            except ValueError:
                continue
    if not values:
        return {
            "thread_count": 0,
            "mean_mhz": None,
            "min_mhz": None,
            "max_mhz": None,
            "samples_mhz": [],
        }
    return {
        "thread_count": len(values),
        "mean_mhz": sum(values) / len(values),
        "min_mhz": min(values),
        "max_mhz": max(values),
        "samples_mhz": values[:8],
    }


def read_gpu_clocks() -> list[dict[str, Any]]:
    query = ",".join(
        [
            "index",
            "name",
            "clocks.gr",
            "clocks.mem",
            "utilization.gpu",
            "temperature.gpu",
        ]
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []

    if completed.returncode != 0:
        return []

    rows: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 6:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "graphics_clock_mhz": float(parts[2]),
                    "memory_clock_mhz": float(parts[3]),
                    "utilization_gpu_pct": float(parts[4]),
                    "temperature_c": float(parts[5]),
                }
            )
        except ValueError:
            continue
    return rows


def sample_hardware_frequency() -> dict[str, Any]:
    return {
        "timestamp": _now(),
        "cpu": read_cpu_mhz(),
        "gpus": read_gpu_clocks(),
    }


def write_hardware_frequency(output_path: str | Path) -> dict[str, Any]:
    payload = sample_hardware_frequency()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
