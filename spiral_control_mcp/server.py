from __future__ import annotations

import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spiral_control_mcp.supervisor import (
    emergency_stop_all,
    get_job,
    list_jobs,
    run_probe_command,
    start_job,
    system_health,
    tail_job_log,
    stop_job,
)


SERVER_INFO = {
    "name": "spiral-control-mcp",
    "version": "0.1.0",
}
PROTOCOL_VERSION = "2024-11-05"


TOOLS = [
    {
        "name": "spiral_system_health",
        "description": "Return Docker, GPU, and tracked-job health.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "spiral_jobs_list",
        "description": "List tracked jobs managed by the control plane.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "spiral_job_status",
        "description": "Get one tracked job by id.",
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "spiral_job_tail",
        "description": "Read the tail of a tracked job log.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "lines": {"type": "integer", "default": 80},
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "spiral_job_stop",
        "description": "Stop one tracked job. Set force=true for hard kill.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "force": {"type": "boolean", "default": False},
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "spiral_emergency_stop_all",
        "description": "Emergency stop every tracked SPIRAL job.",
        "inputSchema": {
            "type": "object",
            "properties": {"force": {"type": "boolean", "default": True}},
        },
    },
    {
        "name": "spiral_start_native_dual_gpu_stack",
        "description": "Start the persistent native WSL dual-GPU MAX stack.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "gpu0_command": {"type": "string"},
                "gpu1_command": {"type": "string"},
                "keep_alive": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "spiral_stop_native_dual_gpu_stack",
        "description": "Stop the persistent native WSL dual-GPU MAX containers.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "spiral_native_dual_gpu_stack_status",
        "description": "Inspect the persistent native WSL dual-GPU MAX containers.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "spiral_start_dual_gpu_batch",
        "description": "Start the existing dual-GPU SPIRAL Mojo batch runner.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "inputs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "output_dir": {"type": "string"},
            },
            "required": ["inputs"],
        },
    },
    {
        "name": "spiral_start_nvda_lidar_batch",
        "description": "Start the NVDA/LiDAR SPIRAL batch pipeline.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "payload_path": {"type": "string"},
                "output_dir": {"type": "string"},
            },
        },
    },
    {
        "name": "spiral_start_batch_benchmark",
        "description": "Benchmark single-GPU and dual-GPU SPIRAL batch execution.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "inputs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "output_dir": {"type": "string"},
                "repeat_inputs": {"type": "integer", "default": 4},
            },
            "required": ["inputs"],
        },
    },
    {
        "name": "spiral_hardware_frequency_snapshot",
        "description": "Read current CPU/GPU hardware frequencies.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "spiral_start_population_baseline",
        "description": "Run the no-teaching population baseline on the reserved CPU lane with telemetry.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "inputs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "output_path": {"type": "string"},
                "logical_total": {"type": "number"},
            },
            "required": ["inputs"],
        },
    },
    {
        "name": "spiral_start_phase14_lane",
        "description": "Run Phase XIV multifocal training on the reserved CPU lane with telemetry.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "output_dir": {"type": "string"},
                "runs": {"type": "integer"},
                "nodes": {"type": "integer"},
                "steps": {"type": "integer"},
                "backend": {"type": "string"},
                "gpu_index": {"type": "integer"},
            },
        },
    },
]


def _text_result(payload: Any, *, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, indent=2),
            }
        ],
        "isError": is_error,
    }


def _require(arguments: dict[str, Any], key: str) -> Any:
    if key not in arguments:
        raise KeyError(f"Missing required argument: {key}")
    return arguments[key]


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "spiral_system_health":
        return _text_result(system_health())
    if name == "spiral_jobs_list":
        return _text_result({"jobs": list_jobs()})
    if name == "spiral_job_status":
        return _text_result(get_job(str(_require(arguments, "job_id"))))
    if name == "spiral_job_tail":
        return _text_result(
            tail_job_log(
                str(_require(arguments, "job_id")),
                lines=int(arguments.get("lines", 80)),
            )
        )
    if name == "spiral_job_stop":
        return _text_result(
            stop_job(
                str(_require(arguments, "job_id")),
                force=bool(arguments.get("force", False)),
            )
        )
    if name == "spiral_emergency_stop_all":
        return _text_result(
            {
                "jobs": emergency_stop_all(force=bool(arguments.get("force", True))),
                "native_stack": run_probe_command(
                    ["bash", "/mnt/d/stop_spiral_wsl_native_dual_gpu.sh"]
                ),
            }
        )
    if name == "spiral_start_native_dual_gpu_stack":
        keep_alive = "1" if bool(arguments.get("keep_alive", True)) else "0"
        parts = [f"KEEP_ALIVE={keep_alive}"]
        gpu0_command = arguments.get("gpu0_command")
        gpu1_command = arguments.get("gpu1_command")
        if gpu0_command:
            parts.append(f"GPU0_COMMAND={shlex.quote(str(gpu0_command))}")
        if gpu1_command:
            parts.append(f"GPU1_COMMAND={shlex.quote(str(gpu1_command))}")
        command = [
            "bash",
            "-lc",
            " ".join(parts) + " /mnt/d/run_spiral_wsl_native_dual_gpu.sh",
        ]
        job = start_job(
                label="native_dual_gpu_stack",
                command=command,
                meta={
                    "gpu0_command": gpu0_command,
                    "gpu1_command": gpu1_command,
                    "keep_alive": keep_alive,
                },
            )
        time.sleep(1.0)
        return _text_result(
            {
                "job": job,
                "stack_status": run_probe_command(
                    ["bash", "/mnt/d/status_spiral_wsl_native_dual_gpu.sh"]
                ),
            }
        )
    if name == "spiral_stop_native_dual_gpu_stack":
        return _text_result(
            run_probe_command(["bash", "/mnt/d/stop_spiral_wsl_native_dual_gpu.sh"])
        )
    if name == "spiral_native_dual_gpu_stack_status":
        return _text_result(
            run_probe_command(["bash", "/mnt/d/status_spiral_wsl_native_dual_gpu.sh"])
        )
    if name == "spiral_start_dual_gpu_batch":
        inputs = [str(item) for item in _require(arguments, "inputs")]
        command = ["python3", "/mnt/d/resonance_spiral_dual_gpu_batch.py", *inputs]
        output_dir = arguments.get("output_dir")
        if output_dir:
            command.extend(["--output-dir", str(output_dir)])
        return _text_result(
            start_job(
                label="dual_gpu_batch",
                command=command,
                meta={"inputs": inputs, "output_dir": output_dir},
            )
        )
    if name == "spiral_start_nvda_lidar_batch":
        payload_path = str(
            arguments.get(
                "payload_path",
                "/mnt/d/nvda_virtualis_ter_lidar/outputs/synthetic_lidar_bridge_payload.json",
            )
        )
        output_dir = str(
            arguments.get(
                "output_dir",
                "/mnt/d/nvda_virtualis_ter_lidar/outputs/dual_gpu_batch",
            )
        )
        return _text_result(
            start_job(
                label="nvda_lidar_batch",
                command=[
                    "bash",
                    "/mnt/d/run_nvda_lidar_spiral_batch.sh",
                    payload_path,
                    output_dir,
                ],
                meta={"payload_path": payload_path, "output_dir": output_dir},
            )
        )
    if name == "spiral_start_batch_benchmark":
        inputs = [str(item) for item in _require(arguments, "inputs")]
        output_dir = str(
            arguments.get(
                "output_dir",
                "/mnt/d/spiral_validation_outputs/batch_benchmark",
            )
        )
        repeat_inputs = int(arguments.get("repeat_inputs", 4))
        return _text_result(
            start_job(
                label="batch_benchmark",
                command=[
                    "python3",
                    "/mnt/d/resonance_spiral_batch_benchmark.py",
                    *inputs,
                    "--output-dir",
                    output_dir,
                    "--repeat-inputs",
                    str(repeat_inputs),
                ],
                meta={
                    "inputs": inputs,
                    "output_dir": output_dir,
                    "repeat_inputs": repeat_inputs,
                },
            )
        )
    if name == "spiral_hardware_frequency_snapshot":
        return _text_result(
            run_probe_command(["python3", "/mnt/d/spiral_hardware_frequency.py"])
        )
    if name == "spiral_start_population_baseline":
        inputs = [str(item) for item in _require(arguments, "inputs")]
        output_path = str(
            arguments.get(
                "output_path",
                "/mnt/d/spiral_validation_outputs/population_baseline.json",
            )
        )
        command = [
            "bash",
            "/mnt/d/run_spiral_population_baseline_lane.sh",
            output_path,
            *inputs,
        ]
        if "logical_total" in arguments and arguments["logical_total"] is not None:
            command.extend(["--logical-total", str(arguments["logical_total"])])
        return _text_result(
            start_job(
                label="population_baseline_lane",
                command=command,
                meta={
                    "inputs": inputs,
                    "output_path": output_path,
                    "logical_total": arguments.get("logical_total"),
                },
            )
        )
    if name == "spiral_start_phase14_lane":
        output_dir = str(
            arguments.get(
                "output_dir",
                "/mnt/d/resonance_phase14_outputs",
            )
        )
        command = ["bash", "/mnt/d/run_spiral_phase14_lane.sh", output_dir]
        if "runs" in arguments and arguments["runs"] is not None:
            command.extend(["--runs", str(arguments["runs"])])
        if "nodes" in arguments and arguments["nodes"] is not None:
            command.extend(["--nodes", str(arguments["nodes"])])
        if "steps" in arguments and arguments["steps"] is not None:
            command.extend(["--steps", str(arguments["steps"])])
        if "backend" in arguments and arguments["backend"] is not None:
            command.extend(["--backend", str(arguments["backend"])])
        if "gpu_index" in arguments and arguments["gpu_index"] is not None:
            command.extend(["--gpu-index", str(arguments["gpu_index"])])
        return _text_result(
            start_job(
                label="phase14_lane",
                command=command,
                meta={
                    "output_dir": output_dir,
                    "runs": arguments.get("runs"),
                    "nodes": arguments.get("nodes"),
                    "steps": arguments.get("steps"),
                    "backend": arguments.get("backend"),
                    "gpu_index": arguments.get("gpu_index"),
                },
            )
        )
    raise KeyError(f"Unknown tool: {name}")


def read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        header = line.decode("utf-8").strip()
        if ":" in header:
            key, value = header.split(":", 1)
            headers[key.strip().lower()] = value.strip()

    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode("utf-8"))


def write_message(payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def handle_request(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params", {})

    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {"listChanged": False},
                },
                "serverInfo": SERVER_INFO,
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        try:
            result = call_tool(
                str(params.get("name")),
                dict(params.get("arguments", {})),
            )
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": _text_result({"error": str(exc)}, is_error=True),
            }
    if method == "shutdown":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "exit":
        return None
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}",
        },
    }


def main() -> None:
    while True:
        message = read_message()
        if message is None:
            break
        response = handle_request(message)
        if response is not None:
            write_message(response)


if __name__ == "__main__":
    main()
