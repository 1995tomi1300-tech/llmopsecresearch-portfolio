$CpuSet = if ($env:SPIRAL_CPU_SET) { $env:SPIRAL_CPU_SET } else { "29-31" }

if ($args.Count -eq 0) {
    Write-Error "Usage: run_spiral_cpu_lane.ps1 <command> [args...]"
    exit 1
}

$cmd = $args -join " "
wsl.exe -d Ubuntu -u root -- bash -lc "taskset -c $CpuSet $cmd"
