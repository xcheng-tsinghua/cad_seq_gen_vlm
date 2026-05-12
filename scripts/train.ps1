param(
  [string]$RawRoot = "/opt/data/private/data_set/cad_seq_img",
  [int]$ImageSize = 384,
  [int]$Epochs = 80,
  [int]$BatchSize = 8,
  [double]$Lr = 2e-4,
  [string]$SdModelId = "stabilityai/stable-diffusion-3.5-medium",
  [double]$WSdLatent = 0.2
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DatasetName = Split-Path -Leaf $RawRoot
$RunTag = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = Join-Path $ProjectRoot ("outputs/" + $DatasetName + "/train_" + $RunTag)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "RawRoot: $RawRoot"
Write-Host "OutputDir: $OutputDir"

python -m src.cad_seq_gen.train `
  --raw-root $RawRoot `
  --output-dir $OutputDir `
  --image-size $ImageSize `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --lr $Lr `
  --sd-model-id $SdModelId `
  --w-sd-latent $WSdLatent

$LatestCheckpoint = Join-Path $OutputDir "best.pt"
if (Test-Path $LatestCheckpoint) {
  $LatestFile = Join-Path (Join-Path $ProjectRoot ("outputs/" + $DatasetName)) "latest_best_checkpoint.txt"
  Set-Content -Path $LatestFile -Value $LatestCheckpoint -Encoding UTF8
  Write-Host "Latest checkpoint recorded: $LatestCheckpoint"
}

