param(
    [string]$Scenario = "",
    [string]$OutputDir = "D:\resonance_phase6_outputs\visual_confirmation"
)

if ([string]::IsNullOrWhiteSpace($Scenario)) {
    bash /mnt/d/run_resonance_visual_confirmation.sh "" $OutputDir
} else {
    bash /mnt/d/run_resonance_visual_confirmation.sh $Scenario $OutputDir
}
