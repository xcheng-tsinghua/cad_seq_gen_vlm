param(
  [string]$ProcessedRoot,
  [string]$OutputDir = "./outputs/structured_v2",
  [int]$ImageSize = 384,
  [int]$Epochs = 80,
  [int]$BatchSize = 8,
  [double]$Lr = 2e-4,
  [string]$SdModelId = "stabilityai/stable-diffusion-3.5-medium",
  [double]$WSdLatent = 0.2
)

python -m src.cad_seq_gen.train `
  --processed-root $ProcessedRoot `
  --output-dir $OutputDir `
  --image-size $ImageSize `
  --epochs $Epochs `
  --batch-size $BatchSize `
  --lr $Lr `
  --sd-model-id $SdModelId `
  --w-sd-latent $WSdLatent

