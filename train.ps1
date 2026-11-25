param(
    [ValidateSet('auto','local')][string]$Pretrained = 'auto',
    [int]$ContextLength = 128,
    [int]$HorizonLength = 32,
    [int]$Stride = 32,
    [double]$StateThreshold = 0.5,
    [string]$TrainDataPath = "$PSScriptRoot\data\train",
    [string]$ValidDataPath = "$PSScriptRoot\data\valid",
    [int]$Epochs = 100, [int]$HeadEpochs = 30, [int]$WarmupEpochs = 3,
    [int]$BatchSize = 8, [int]$Seed = 42, [int]$Patience = 8, [int]$HeadPatience = 5,
    [double]$LearningRate = 0.0001, [double]$BackboneLearningRate = 0.00001,
    [double]$AdapterLearningRate = 0.0001,
    [ValidateSet('auto','cuda','cpu')][string]$Device = 'auto',
    [string]$SensorProfile = "$PSScriptRoot\config\sensor_profile.json",
    [string]$StepProfile = "$PSScriptRoot\config\step_profile.json",
    [string[]]$PastCovariates = @('Temp_Set_U','Temp_Set_CU','Temp_Set_C','Temp_Set_CL','Temp_Set_L'),
    [string[]]$KnownCovariates = @(),
    [string[]]$CategoricalPastCovariates = @(),
    [string[]]$CategoricalKnownCovariates = @(),
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
& $Python -X utf8 -B -u "$PSScriptRoot\train.py" --pretrained $Pretrained @commonArgs --train-data $TrainDataPath --valid-data $ValidDataPath `
    --epochs $Epochs --p-epochs $HeadEpochs --warmup-epochs $WarmupEpochs --seed $Seed --batch-size $BatchSize `
    --patience $Patience --p-patience $HeadPatience --learning-rate $LearningRate --backbone-learning-rate $BackboneLearningRate `
    --adapter-learning-rate $AdapterLearningRate --device $Device
if ($LASTEXITCODE -ne 0) { throw "Training failed ($LASTEXITCODE)" }
