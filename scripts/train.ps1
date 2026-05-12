param(
  [string]$ProcessedRoot,
  [string]$PretrainedModel = "runwayml/stable-diffusion-v1-5",
  [string]$ControlNetModel = "lllyasviel/sd-controlnet-canny",
  [string]$OutputDir = "./outputs/cad_seq_lora",
  [int]$Epochs = 20,
  [int]$BatchSize = 2,
  [double]$Lr = 1e-4
)

python -m src.cad_seq_gen.train_controlnet_lora `
  --processed-root $ProcessedRoot `
  --pretrained-model $PretrainedModel `
  --controlnet-model $ControlNetModel `
  --output-dir $OutputDir `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --lr $Lr

