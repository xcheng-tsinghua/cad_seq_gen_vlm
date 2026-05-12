param(
  [string]$InputImage,
  [string]$ProcessedRoot,
  [string]$Checkpoint,
  [string]$OutputDir = "./outputs/infer_case_001_structured",
  [int]$NumSteps = 0
)

python -m src.cad_seq_gen.infer `
  --input-image $InputImage `
  --processed-root $ProcessedRoot `
  --checkpoint $Checkpoint `
  --output-dir $OutputDir `
  --num-steps $NumSteps

