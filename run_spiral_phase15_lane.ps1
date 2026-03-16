$OutputDir = if ($args.Length -gt 0) { $args[0] } else { "D:\\resonance_phase15_outputs" }
$Remaining = if ($args.Length -gt 1) { $args[1..($args.Length - 1)] } else { @() }

bash /mnt/d/run_spiral_phase15_lane.sh "/mnt/d/$([System.IO.Path]::GetFileName($OutputDir))" @Remaining
