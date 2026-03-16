# SPIRAL Control MCP

Minimal local MCP control plane a SPIRAL futtatokhoz.

Fo cel:

- job inditas
- job allapot
- log tail
- leallitas
- emergency stop

Belso toolok:

- `spiral_system_health`
- `spiral_jobs_list`
- `spiral_job_status`
- `spiral_job_tail`
- `spiral_job_stop`
- `spiral_emergency_stop_all`
- `spiral_start_native_dual_gpu_stack`
- `spiral_stop_native_dual_gpu_stack`
- `spiral_native_dual_gpu_stack_status`
- `spiral_start_dual_gpu_batch`
- `spiral_start_nvda_lidar_batch`
- `spiral_start_batch_benchmark`

Futtatas:

```bash
bash /mnt/d/run_spiral_control_mcp.sh
```

Allapot es logok:

- state: `/mnt/d/spiral_control_mcp/state.json`
- logs: `/mnt/d/spiral_control_mcp/logs/`

Native stack status:

- `/mnt/d/status_spiral_wsl_native_dual_gpu.sh`

Megjegyzes:

Ez egy minimalis stdio MCP implementacio, hogy dependency nelkul is legyen
kontroll, kill-switch es biztonsagos operacios reteg a SPIRAL futtatok elott.
