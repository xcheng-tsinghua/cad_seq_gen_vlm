param(
  [string]$ProcessedRoot,
  [string]$Checkpoint,
  [string]$OutputDir = "./outputs/eval_structured"
)

python -m src.cad_seq_gen.eval `
  --processed-root $ProcessedRoot `
  --checkpoint $Checkpoint `
  --output-dir $OutputDir

