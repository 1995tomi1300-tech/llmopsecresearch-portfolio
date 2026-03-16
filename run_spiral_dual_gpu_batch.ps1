param(
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
    [string[]]$Reports
)

python3 D:\resonance_spiral_dual_gpu_batch.py @Reports
