from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/d/spiral_control_mcp")
LOG_DIR = ROOT / "logs"
STATE_PATH = ROOT / "state.json"
DEFAULT_CWD = Path("/mnt/d")


def ensure_layout() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default_state() -> dict[str, Any]:
    return {
        "version": "0.1.0",
        "created_at": _now(),
        "updated_at": _now(),
        "jobs": {},
    }


def load_state() -> dict[str, Any]:
    ensure_layout()
    if not STATE_PATH.exists():
        state = _default_state()
        save_state(state)
        return state
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    ensure_layout()
    state["updated_at"] = _now()
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def refresh_job_state(job: dict[str, Any]) -> dict[str, Any]:
    pid = int(job.get("pid", 0) or 0)
    if pid <= 0:
        return job
    if job.get("status") in {"completed", "failed", "stopped", "killed"}:
        return job
    job["alive"] = _is_pid_alive(pid)
    if not job["alive"]:
        job["status"] = "exited"
    return job


def refresh_all_jobs() -> dict[str, Any]:
    state = load_state()
    for job_id, job in state["jobs"].items():
        state["jobs"][job_id] = refresh_job_state(job)
    save_state(state)
    return state


def list_jobs() -> list[dict[str, Any]]:
    state = refresh_all_jobs()
    jobs = []
    for job_id, job in state["jobs"].items():
        row = dict(job)
        row["job_id"] = job_id
        jobs.append(row)
    jobs.sort(key=lambda item: item.get("started_at", ""), reverse=True)
    return jobs


def get_job(job_id: str) -> dict[str, Any]:
    state = refresh_all_jobs()
    job = state["jobs"].get(job_id)
    if not job:
        raise KeyError(f"Unknown job_id: {job_id}")
    payload = dict(job)
    payload["job_id"] = job_id
    return payload


def start_job(
    *,
    label: str,
    command: list[str],
    cwd: str | Path = DEFAULT_CWD,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_layout()
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    log_path = LOG_DIR / f"{job_id}.log"
    log_handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    log_handle.write(f"[{_now()}] START {' '.join(command)}\n")
    log_handle.flush()
    log_handle.close()

    state = load_state()
    state["jobs"][job_id] = {
        "label": label,
        "command": command,
        "cwd": str(cwd),
        "pid": process.pid,
        "status": "running",
        "alive": True,
        "started_at": _now(),
        "log_path": str(log_path),
        "meta": meta or {},
    }
    save_state(state)
    return get_job(job_id)


def stop_job(job_id: str, force: bool = False) -> dict[str, Any]:
    state = load_state()
    job = state["jobs"].get(job_id)
    if not job:
        raise KeyError(f"Unknown job_id: {job_id}")

    pid = int(job.get("pid", 0) or 0)
    if pid > 0 and _is_pid_alive(pid):
        sig = signal.SIGKILL if force else signal.SIGTERM
        os.killpg(pid, sig)
        time.sleep(0.2)

    job["alive"] = pid > 0 and _is_pid_alive(pid)
    job["status"] = "killed" if force else "stopped"
    job["stopped_at"] = _now()
    state["jobs"][job_id] = job
    save_state(state)
    return get_job(job_id)


def emergency_stop_all(force: bool = True) -> dict[str, Any]:
    stopped: list[dict[str, Any]] = []
    for job in list_jobs():
        if job.get("status") == "running" or job.get("alive"):
            stopped.append(stop_job(job["job_id"], force=force))
    return {
        "stopped_count": len(stopped),
        "force": force,
        "jobs": stopped,
    }


def tail_job_log(job_id: str, lines: int = 80) -> dict[str, Any]:
    job = get_job(job_id)
    log_path = Path(job["log_path"])
    if not log_path.exists():
        return {"job_id": job_id, "lines": []}
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "job_id": job_id,
        "line_count": min(lines, len(content)),
        "lines": content[-lines:],
    }


def run_probe_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {
            "command": command,
            "error": str(exc),
        }


def system_health() -> dict[str, Any]:
    return {
        "timestamp": _now(),
        "docker_ps": run_probe_command(
            ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}\t{{.Image}}"]
        ),
        "nvidia_smi": run_probe_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "jobs": list_jobs(),
    }
