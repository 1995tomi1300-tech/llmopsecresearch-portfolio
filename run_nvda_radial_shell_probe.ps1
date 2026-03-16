param(
    [string]$OutputText = "D:\nvda_virtualis_ter_lidar\outputs\max_radial_shell_probe.txt",
    [string]$OutputJson = "D:\nvda_virtualis_ter_lidar\outputs\max_radial_shell_probe.json"
)

bash /mnt/d/run_nvda_radial_shell_probe.sh $OutputText $OutputJson
