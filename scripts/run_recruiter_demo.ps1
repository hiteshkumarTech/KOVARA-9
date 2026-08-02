[CmdletBinding()]
param(
    [Parameter()]
    [string]$CheckpointPath,

    [Parameter()]
    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$DemoSeeds = @(4242, 4243)
$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$FinalResultsPath = Join-Path $RepositoryRoot "docs\day8-final-heldout-results.json"
$ConsumptionPath = Join-Path $RepositoryRoot "configs\evaluation\final_test_consumed.json"
$CandidatePath = Join-Path $RepositoryRoot "configs\training\mappo_final_candidate.yaml"
$FreezePath = Join-Path $RepositoryRoot "configs\training\mappo_final_candidate.freeze.json"

function Write-DemoHeading {
    param([Parameter(Mandatory)][string]$Text)
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor DarkGray
    Write-Host $Text -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor DarkGray
}

function Invoke-DemoCommand {
    param(
        [Parameter(Mandatory)][string]$Label,
        [Parameter(Mandatory)][scriptblock]$Command
    )
    Write-DemoHeading $Label
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Demo step failed with exit code ${LASTEXITCODE}: $Label"
    }
}

if (-not (Test-Path -LiteralPath $FinalResultsPath -PathType Leaf)) {
    throw "Final results JSON does not exist: $FinalResultsPath"
}
if (-not (Test-Path -LiteralPath $ConsumptionPath -PathType Leaf)) {
    throw "Final-test consumption record does not exist: $ConsumptionPath"
}

$FinalResults = Get-Content -LiteralPath $FinalResultsPath -Raw | ConvertFrom-Json
$Consumption = Get-Content -LiteralPath $ConsumptionPath -Raw | ConvertFrom-Json
if ($FinalResults.status -ne "complete") {
    throw "Final results are not marked complete."
}
if ($Consumption.status -ne "complete") {
    throw "Final-test consumption lock is not complete."
}

$TestStart = [int]$FinalResults.partitions.configured.test.start
$TestStop = [int]$FinalResults.partitions.configured.test.stop_exclusive
foreach ($DemoSeed in $DemoSeeds) {
    if ($DemoSeed -ge $TestStart -and $DemoSeed -lt $TestStop) {
        throw "Demo seed $DemoSeed overlaps the final-test partition."
    }
}

$ResolvedCheckpoint = $null
if ($CheckpointPath) {
    if (-not (Test-Path -LiteralPath $CheckpointPath -PathType Leaf)) {
        throw "Checkpoint path does not exist or is not a file: $CheckpointPath"
    }
    $ResolvedCheckpoint = (Resolve-Path -LiteralPath $CheckpointPath).Path
    if ([System.IO.Path]::GetExtension($ResolvedCheckpoint) -ne ".pt") {
        throw "Checkpoint path must identify a trusted .pt file: $ResolvedCheckpoint"
    }
}

Write-DemoHeading "KOVARA-9 v0.1 recruiter demo"
Write-Host "Research result: exploration transfer without task completion."
Write-Host "This is a simulator demonstration, not a benchmark or rescue deployment."
Write-Host "Demo seeds: $($DemoSeeds -join ', ') (outside final-test partition [$TestStart, $TestStop))."
Write-Host "Final-test consumption status: $($Consumption.status)."

if (-not $ResolvedCheckpoint) {
    Write-Host "No checkpoint supplied. This is expected: trained checkpoints are intentionally not committed."
}
else {
    Write-Host "Optional trusted checkpoint: $ResolvedCheckpoint"
}

if ($ValidateOnly) {
    Write-Host "DEMO VALIDATION COMPLETE: inputs, lock, seeds, and optional checkpoint are valid."
    exit 0
}

Set-Location -LiteralPath $RepositoryRoot
$env:UV_CACHE_DIR = Join-Path $env:TEMP "kovara9-recruiter-demo-uv-cache"
$DemoOutput = Join-Path $env:TEMP (
    "kovara9-recruiter-demo-{0}-{1}" -f [DateTime]::UtcNow.ToString("yyyyMMddHHmmss"), $PID
)
New-Item -ItemType Directory -Force $DemoOutput | Out-Null
$DemoPytestTemp = Join-Path $DemoOutput "pytest"
New-Item -ItemType Directory -Force $DemoPytestTemp | Out-Null
$Timer = [System.Diagnostics.Stopwatch]::StartNew()

Invoke-DemoCommand "1/7 - validate the demonstration environment and smoke training configuration" {
    uv run kovara9 config validate configs/environments/grid_rescue_easy.yaml
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    uv run kovara9 config validate configs/training/mappo_smoke.yaml
}

Invoke-DemoCommand "2/7 - verify the frozen candidate fingerprint and consumed-test status" {
    uv run kovara9 config verify-candidate `
        --candidate $CandidatePath `
        --freeze-record $FreezePath
}

Invoke-DemoCommand "3/7 - random-policy behavior, fixed seed, headless (not a benchmark)" {
    uv run kovara9 env run `
        --config configs/environments/grid_rescue_easy.yaml `
        --agent random `
        --seed $DemoSeeds[0] `
        --render none
}

Invoke-DemoCommand "4/7 - handcrafted frontier behavior, fixed seed, rendered (not learned)" {
    uv run kovara9 env run `
        --config configs/environments/grid_rescue_easy.yaml `
        --agent frontier `
        --seed $DemoSeeds[1] `
        --render ansi
}

Invoke-DemoCommand "5/7 - short untrained rollout smoke, no optimization or checkpoint" {
    uv run kovara9 rollout-smoke `
        --training-config configs/training/mappo_smoke.yaml `
        --steps 8
}

if ($ResolvedCheckpoint) {
    Invoke-DemoCommand "Optional - deterministic validation of a trusted local checkpoint" {
        uv run kovara9 evaluate-checkpoint `
            --checkpoint $ResolvedCheckpoint `
            --env-config configs/environments/grid_rescue_medium.yaml `
            --eval-config configs/evaluation/training_validation_smoke.yaml `
            --output (Join-Path $DemoOutput "checkpoint-validation") `
            --device cpu
    }
}

Invoke-DemoCommand "6/7 - regenerate six figures from committed JSON only" {
    uv run python scripts/generate_result_figures.py
}

Invoke-DemoCommand "7/7 - run focused Day 9 presentation-integrity tests" {
    uv run pytest tests/unit/test_day9_presentation.py `
        --basetemp $DemoPytestTemp `
        --no-cov `
        -q
}

$Timer.Stop()
Write-DemoHeading "Recorded final evidence and generated assets"
Write-Host ("Trained success: {0:P1}" -f [double]$FinalResults.training_seed_aggregates.trained.success_rate.mean)
Write-Host ("Random success:  {0:P1}" -f [double]$FinalResults.policy_results.random.pooled.success_rate)
Write-Host ("Frontier success:{0:P1}" -f [double]$FinalResults.policy_results.frontier.pooled.success_rate)
Write-Host "Recorded Day 8 test gate: $($FinalResults.quality_gates.pytest.passed) passed, $($FinalResults.quality_gates.pytest.skipped) skipped, $($FinalResults.quality_gates.pytest.coverage_percent)% coverage."
Write-Host "Figures:"
Get-ChildItem -LiteralPath (Join-Path $RepositoryRoot "docs\assets\results") -Filter "*.svg" |
    Sort-Object Name |
    ForEach-Object { Write-Host "  $($_.FullName)" }
Write-Host ("Demo runtime: {0:N1} seconds." -f $Timer.Elapsed.TotalSeconds)
Write-Host "No training or final-test evaluation was performed."
