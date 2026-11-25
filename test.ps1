param(
    [ValidateSet('auto','local')][string]$Pretrained = 'auto',
    [Nullable[int]]$ContextLength = $null,
    [Nullable[int]]$HorizonLength = 32,
    [Nullable[int]]$Stride = 32,
    [Nullable[double]]$StateThreshold = $null,
    [string]$TestDataPath = "$PSScriptRoot\data\test",
    [string]$CheckpointPath = '',
    [ValidateSet('auto','cuda','cpu')][string]$Device = 'auto',
    [string]$SensorProfile = "$PSScriptRoot\config\sensor_profile.json",
    [string]$StepProfile = "$PSScriptRoot\config\step_profile.json",
    # Null restores the ordered columns saved at training time.
    [string[]]$PastCovariates = $null,
    [string[]]$KnownCovariates = $null,
    [string[]]$CategoricalPastCovariates = $null,
    [string[]]$CategoricalKnownCovariates = $null,
    [hashtable]$ProcessColumns = @{
        main_recipe = 'PMStoredProcessRecipeName'
        main_step_seq = 'ProcessRecipeStepSeqNo'
        main_step_id = 'ProcessRecipeStepID'
        execution_recipe = 'PPExecname'
        substep_seq = 'PPExecStepSeqNo'
        substep_id = 'PPExecStepID'
        loop_setting = 'SubRecipeLoopSettingValue'
        loop_index = 'SubRecipeLoopMoniterValue'
    },
    [string]$OutputPath = "$PSScriptRoot\outputs"
)
. "$PSScriptRoot\scripts\entry.ps1"
if (-not $CheckpointPath) {
    $latest = Join-Path $OutputPath 'latest.json'
    $CheckpointPath = if (Test-Path -LiteralPath $latest) {
        Join-Path $OutputPath ((Get-Content -LiteralPath $latest -Raw | ConvertFrom-Json).checkpoint)
    } else { "$PSScriptRoot\checkpoints\TSV-28M\best.pt" }
}
& $Python -X utf8 -B -u "$PSScriptRoot\test.py" --pretrained $Pretrained @commonArgs --test-data $TestDataPath --checkpoint $CheckpointPath --device $Device
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed ($LASTEXITCODE)" }
