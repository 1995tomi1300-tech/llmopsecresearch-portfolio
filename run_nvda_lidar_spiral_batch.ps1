param(
    [string]$Payload = "D:\nvda_virtualis_ter_lidar\outputs\synthetic_lidar_bridge_payload.json",
    [string]$OutputDir = "D:\nvda_virtualis_ter_lidar\outputs\dual_gpu_batch"
)

bash /mnt/d/run_nvda_lidar_spiral_batch.sh $Payload $OutputDir
