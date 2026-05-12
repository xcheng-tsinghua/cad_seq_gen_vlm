param(
  [string]$RawRoot,
  [string]$Checkpoint = ""
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DatasetName = Split-Path -Leaf $RawRoot
$DatasetOutputRoot = Join-Path $ProjectRoot ("outputs/" + $DatasetName)
$RunTag = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = Join-Path $DatasetOutputRoot ("eval_" + $RunTag)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
  $LatestFile = Join-Path $DatasetOutputRoot "latest_best_checkpoint.txt"
  if (Test-Path $LatestFile) {
    $Checkpoint = (Get-Content -Path $LatestFile -Raw).Trim()
  } else {
    $BestList = Get-ChildItem -Path $DatasetOutputRoot -Recurse -Filter "best.pt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($BestList.Count -gt 0) {
      $Checkpoint = $BestList[0].FullName
    }
  }
}

if ([string]::IsNullOrWhiteSpace($Checkpoint) -or -not (Test-Path $Checkpoint)) {
  throw "Checkpoint not found. Please run scripts/train.ps1 first or pass -Checkpoint explicitly."
}

Write-Host "RawRoot: $RawRoot"
Write-Host "Checkpoint: $Checkpoint"
Write-Host "OutputDir: $OutputDir"

python -m src.cad_seq_gen.eval `
  --raw-root $RawRoot `
  --checkpoint $Checkpoint `
  --output-dir $OutputDir

