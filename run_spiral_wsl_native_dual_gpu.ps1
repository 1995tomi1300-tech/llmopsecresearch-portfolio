param(
    [string]$Distro = "Ubuntu",
    [string]$Image = "modular/max-nvidia-full:latest",
    [string]$Workspace = "/mnt/d",
    [string]$Command = "mojo --version && mojo /workspace/spiral_validation.mojo"
)

$ErrorActionPreference = "Stop"

$env:IMAGE = $Image
$env:WORKSPACE = $Workspace
$env:COMMAND = $Command

wsl.exe -d $Distro -- bash -lc "/mnt/d/run_spiral_wsl_native_dual_gpu.sh"
