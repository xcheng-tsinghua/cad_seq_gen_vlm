# vision_cad_emu35

MVP codebase for fine-tuning and deploying Emu3.5 as a multimodal autoregressive planner for vision-based CAD modeling step reverse generation.

The planner takes:

- `final_snapshot.png`: final rendered CAD part.
- `prev_depth_map.png`: the previous modeling state depth map.

It predicts:

- `Operation_Type: ...`
- `overlayed_all.png`: a CAD-style preview image for the current modeling step.

The downstream CAD executor/parser is intentionally outside this project. A small executor interface is included so the autoregressive loop can be connected to a real CAD backend later.

## Emu3.5 Adapter Boundary

All direct Emu3.5 calls live in:

```text
src/vision_cad_emu35/models/emu35_adapter.py
```

The adapter is designed around the public Emu3.5 repo utilities:

- `build_emu3p5`
- `build_image`
- `generate`
- `multimodal_decode`

If your installed Emu3.5 package exposes different names or output structures, update only this file. The data pipeline, trainer, evaluator, API, and web demo call the stable adapter methods:

```python
load_model()
build_training_sample(sample)
forward_loss(batch)
generate(final_snapshot, prev_depth_map, prompt, generation_config)
save_checkpoint(output_dir)
load_checkpoint(checkpoint_dir)
```

Official references:

- [BAAI Emu3.5 GitHub](https://github.com/baaivision/Emu3.5)
- [BAAI Emu3.5 on Hugging Face](https://huggingface.co/BAAI/Emu3.5)

## Setup

```bash
cd vision_cad_emu35
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,eval]"
```

Install the official Emu3.5 dependencies and set these config values:

```yaml
model:
  model_id_or_path: "/path/or/hf/id/to/Emu3.5"
  tokenizer_path: "/path/or/hf/id/to/Emu3.5"
  vision_tokenizer_path: "/path/or/hf/id/to/Emu3.5-VisionTokenizer"
  emu_repo_path: "/path/to/Emu3.5/repo"  # optional if already importable
```

## Dataset Layout

```text
root/
  CAD_PART_ID_VIEW_SUFFIX/
    final_snapshot.png
    roll_back_index_1/
      prev_depth_map.png
      current_depth_map.png
      operation_param.json
      overlayed_all.png
    roll_back_index_3/
      ...
```

Rollback indices do not need to be continuous. Splits are deterministic by `cad_part_id` by default to avoid leakage across views and steps.

## Prepare Manifests

```bash
python scripts/prepare_manifest.py \
  --dataset-root /path/to/dataset \
  --manifest-dir data/manifests \
  --add-stop-samples
```

Outputs:

- `manifest_all.jsonl`
- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `stats.json`
- `issues.jsonl`

## Train

```bash
python scripts/train.py \
  --config configs/train_80gb.yaml
```

The default profile assumes a single 80GB GPU and uses:

- bf16 mixed precision
- batch size 1
- gradient accumulation 8
- gradient checkpointing
- LoRA rank 64
- checkpoint save/resume
- TensorBoard logging

Resume:

```bash
python scripts/train.py \
  --config configs/train_80gb.yaml
```

Set `training.resume_from_checkpoint` in the YAML, or edit the file before launch.

## Evaluate

```bash
python scripts/evaluate.py \
  --config configs/train_80gb.yaml \
  --checkpoint outputs/emu35_finetune/best \
  --split test
```

Outputs include:

- `metrics.json`
- `operation_confusion_matrix.png`
- `qualitative_grid.png`
- `per_sample_results.jsonl`
- `failed_cases/`

Metrics include operation accuracy, confusion matrix, per-class precision/recall/F1, L1, MSE, PSNR, SSIM, optional LPIPS, and CAD color-mask IoU/F1 for yellow, cyan, red, blue, green, and magenta.

## Single-Step Inference

```bash
python scripts/infer_single.py \
  --config configs/infer.yaml \
  --checkpoint outputs/emu35_finetune/best \
  --final-snapshot examples/final_snapshot.png \
  --prev-depth-map examples/prev_depth_map.png \
  --output-dir outputs/infer_single
```

## Batch Inference

```bash
python scripts/infer_batch.py \
  --config configs/infer.yaml \
  --checkpoint outputs/emu35_finetune/best \
  --manifest data/manifests/test.jsonl \
  --output-dir outputs/infer_batch
```

## Autoregressive Inference

```bash
python scripts/infer_autoregressive.py \
  --config configs/infer.yaml \
  --checkpoint outputs/emu35_finetune/best \
  --final-snapshot examples/final_snapshot.png \
  --initial-depth-map examples/depth_0.png \
  --max-steps 20 \
  --output-dir outputs/autoregressive
```

For teacher-forced evaluation, provide known depth maps:

```bash
python scripts/infer_autoregressive.py \
  --config configs/infer.yaml \
  --checkpoint outputs/emu35_finetune/best \
  --final-snapshot examples/final_snapshot.png \
  --initial-depth-map examples/depth_0.png \
  --teacher-depth-sequence examples/depth_1.png examples/depth_2.png \
  --output-dir outputs/autoregressive_teacher
```

Without a teacher-forced sequence or real CAD executor, the loop raises `NotImplementedError` after a non-`<STOP>` operation, by design.

## API

```bash
python scripts/launch_api.py \
  --config configs/api.yaml \
  --checkpoint outputs/emu35_finetune/best
```

Endpoints:

- `GET /health`
- `POST /generate`
- `POST /generate_batch`
- `POST /autoregressive`

## Web Demo

```bash
python scripts/launch_web_demo.py \
  --config configs/api.yaml \
  --checkpoint outputs/emu35_finetune/best
```

Open the URL printed by the script. The demo lets you upload the final snapshot and previous depth map, then displays the predicted operation and generated overlay.

## Tests

```bash
pytest
```

The tests do not require Emu3.5. They cover operation type derivation, dataset scanning/splitting, and color-mask metrics.

## Known Limitations

- Exact Emu3.5 image-text training delimiters may vary by release. The current implementation uses native autoregressive target construction with official image tokenization and marks the delimiter check inside `Emu35Adapter.build_training_sample`.
- Real CAD execution is not included. Connect your CAD parser/rendering backend by implementing `Executor.run_step`.
- QLoRA depends on the official loader supporting quantized model loading. The adapter prepares k-bit training when PEFT is available, but you may need to extend `load_model` for your local Emu3.5 loader.

## MVP Roadmap

- Validate adapter delimiters against the installed Emu3.5 release.
- Add a real CAD executor integration.
- Add multi-view sample packing.
- Add distributed training profiles.
- Add richer operation-type constrained decoding.

