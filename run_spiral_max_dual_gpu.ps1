param(
    [string]$Image = "modular/max-nvidia-full:latest",
    [string]$Workspace = "D:\",
    [string]$Command = "mojo --version && mojo /workspace/spiral_validation.mojo"
)

$ErrorActionPreference = "Stop"

function Start-SpiralContainer {
    param(
        [int]$GpuIndex,
        [string]$NameSuffix
    )

    $name = "spiral-max-gpu$NameSuffix"
    Write-Host "Launching $name on GPU $GpuIndex"
    docker run --rm -d `
        --name $name `
        --gpus "device=$GpuIndex" `
        --entrypoint /bin/bash `
        -v "${Workspace}:/workspace" `
        -w /workspace `
        $Image `
        -lc $Command | Out-Null
}

docker version | Out-Null

Start-SpiralContainer -GpuIndex 0 -NameSuffix "0"
Start-SpiralContainer -GpuIndex 1 -NameSuffix "1"

Write-Host ""
Write-Host "Containers started:"
Write-Host "  spiral-max-gpu0 -> GPU0"
Write-Host "  spiral-max-gpu1 -> GPU1"
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  docker logs -f spiral-max-gpu0"
Write-Host "  docker logs -f spiral-max-gpu1"
Write-Host "  docker exec -it spiral-max-gpu0 /bin/bash"
Write-Host "  docker exec -it spiral-max-gpu1 /bin/bash"
