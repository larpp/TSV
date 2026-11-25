$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $repo '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { $Python = Join-Path $repo '.venv/bin/python' }
if (-not (Test-Path -LiteralPath $Python)) { $Python = (Get-Command python -ErrorAction Stop).Source }
$columnArgs = @()
foreach ($role in $ProcessColumns.Keys) {
    $columnArgs += @('--column', "$role=$($ProcessColumns[$role])")
}
$commonArgs = @('--sensor-profile', $SensorProfile, '--step-profile', $StepProfile, '--output', $OutputPath) + $columnArgs
if ($null -ne $ContextLength) { $commonArgs += @('--context', $ContextLength) }
if ($null -ne $HorizonLength) { $commonArgs += @('--horizon', $HorizonLength) }
if ($null -ne $Stride) { $commonArgs += @('--stride', $Stride) }
if ($null -ne $StateThreshold) { $commonArgs += @('--state-threshold', $StateThreshold.ToString([cultureinfo]::InvariantCulture)) }
if ($null -ne $PastCovariates) { $commonArgs += @('--past-covariates') + $PastCovariates }
if ($null -ne $KnownCovariates) { $commonArgs += @('--known-covariates') + $KnownCovariates }
if ($null -ne $CategoricalPastCovariates) { $commonArgs += @('--categorical-past-covariates') + $CategoricalPastCovariates }
if ($null -ne $CategoricalKnownCovariates) { $commonArgs += @('--categorical-known-covariates') + $CategoricalKnownCovariates }
