param(
    [string]$DatasetPath = "C:/Users/gunn/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3",
    [string]$OutputRoot = "artifacts/robustness_comparison",
    [ValidateSet("cpu", "cuda", "auto")][string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$pythonExecutable = ".venv/Scripts/python.exe"
$referenceCheckpoint = "artifacts/efficientnet_b0/apple_efficientnet_b0_checkpoint.pt"
$robustCheckpoint = "artifacts/efficientnet_b0_robust/apple_efficientnet_b0_robust_checkpoint.pt"
foreach ($checkpointPath in @($referenceCheckpoint, $robustCheckpoint)) {
    if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
        throw "Checkpoint missing: $checkpointPath. Run your training first."
    }
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Choose a new OutputRoot; comparison output must not already exist."
}
$referenceHash = (Get-FileHash -LiteralPath $referenceCheckpoint -Algorithm SHA256).Hash
$robustHash = (Get-FileHash -LiteralPath $robustCheckpoint -Algorithm SHA256).Hash
$models = @(
    @{ Name = "reference"; Checkpoint = $referenceCheckpoint },
    @{ Name = "robust"; Checkpoint = $robustCheckpoint }
)
foreach ($model in $models) {
    $modelOutput = Join-Path $OutputRoot $model.Name
    & $pythonExecutable -m appleleaf.cli.evaluate `
        --checkpoint $model.Checkpoint --dataset-path $DatasetPath `
        --output-dir (Join-Path $modelOutput "plantvillage") --device $Device
    if ($LASTEXITCODE -ne 0) { throw "PlantVillage evaluation failed." }

    & $pythonExecutable -m appleleaf.cli.background_audit `
        --checkpoint $model.Checkpoint --dataset-path $DatasetPath --split test --include-texture `
        --output (Join-Path $modelOutput "background_audit.csv") --device $Device
    if ($LASTEXITCODE -ne 0) { throw "Background audit failed." }

    & $pythonExecutable -m appleleaf.cli.evaluate_kashmir `
        --checkpoint $model.Checkpoint --subset scab-healthy `
        --output-dir (Join-Path $modelOutput "kashmir_scab_healthy") `
        --exclude-label-conflicts --exclude-missing-lfs --device $Device
    if ($LASTEXITCODE -ne 0) { throw "Kashmir evaluation failed." }
}
if ((Get-FileHash -LiteralPath $referenceCheckpoint -Algorithm SHA256).Hash -ne $referenceHash) {
    throw "Reference checkpoint changed unexpectedly."
}
if ((Get-FileHash -LiteralPath $robustCheckpoint -Algorithm SHA256).Hash -ne $robustHash) {
    throw "Robust checkpoint changed unexpectedly."
}
& $pythonExecutable scripts/compare_robustness_results.py --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) { throw "Comparison report failed." }
Write-Output "Comparison complete; both checkpoints unchanged. No training was performed."
