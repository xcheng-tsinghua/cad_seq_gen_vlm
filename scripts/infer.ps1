param(
  [string]$InputImage,
  [string]$ProcessedRoot,
  [string]$LoraDir,
  [string]$OutputDir = "./outputs/infer_case_001",
  [string]$PretrainedModel = "runwayml/stable-diffusion-v1-5",
  [string]$ControlNetModel = "lllyasviel/sd-controlnet-canny",
  [int]$NumSteps = 0
)

python -m src.cad_seq_gen.infer_sequence `
  --input-image $InputImage `
  --processed-root $ProcessedRoot `
  --pretrained-model $PretrainedModel `
  --controlnet-model $ControlNetModel `
  --lora-dir $LoraDir `
  --output-dir $OutputDir `
  --num-steps $NumSteps

