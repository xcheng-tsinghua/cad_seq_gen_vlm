# vision_cad_emu35

Frozen Emu3.5 RAG system for vision-based CAD modeling step reverse generation.

This project no longer fine-tunes Emu3.5. It keeps Emu3.5 frozen, builds a retrieval knowledge base from historical CAD modeling steps, retrieves similar examples for a query, and asks Emu3.5 to generate:

- `Operation_Type: <operation_type>`
- one CAD-style preview image, `overlayed_all.png`

The system supports an empty knowledge base. If no KB exists, the API and web demo still launch and run zero-shot using only the query images and drawing rules.

## Setup

```bash
cd vision_cad_emu35
python -m venv .venv
. .venv/bin/activate
pip install -U modelscope
pip install -e ".[dev]"
```

Use either conda or `.venv`, but not both at the same time. Before debugging runtime issues, confirm `which python` or `sys.executable` points at the environment where torch, transformers, and this project are installed.

## NVIDIA RTX PRO 6000 Blackwell Setup

Blackwell GPUs require a recent NVIDIA driver and a PyTorch build with CUDA 12.8 or newer. Do not let `pip install -e .` choose a generic PyTorch wheel; `pyproject.toml` intentionally does not depend on `torch`.

Recommended setup:

```bash
pip uninstall -y torch torchvision torchaudio xformers flash-attn bitsandbytes
pip install -r requirements-blackwell-cu128.txt
pip install -e . --no-deps
python scripts/check_gpu_env.py
```

`scripts/check_gpu_env.py` prints the PyTorch version, CUDA version, GPU name, compute capability, bf16 support, CUDA tensor allocation status, and optional acceleration import status.

Optional acceleration libraries are not required:

- `flash-attn`
- `xformers`
- `bitsandbytes`
- `vllm`

If they are absent, the project falls back to standard PyTorch inference with a warning.

For CPU-only model downloading or RAG KB building:

```bash
pip install -r requirements-cpu.txt
```

## Download Models from ModelScope Without GPU

Recommended for mainland China:

```bash
python scripts/download_models.py
```

The downloader does not import `torch`, does not load the model, and does not require a GPU. It writes:

```text
/root/autodl-tmp/data/BAAI/Emu3.5
/root/autodl-tmp/data/BAAI/Emu3.5-VisionTokenizer
```

ModelScope custom ids:

```bash
python scripts/download_models.py \
  --backend modelscope \
  --main-modelscope-id BAAI/Emu3.5 \
  --vision-tokenizer-modelscope-id BAAI/Emu3.5-VisionTokenizer
```

Hugging Face fallback:

```bash
python scripts/download_models.py \
  --backend huggingface \
  --hf-token $HF_TOKEN
```

Runtime loading always uses local paths by default. The adapter does not download anything.

## Install Official Emu3.5 Runtime Source Code

This project downloads Emu3.5 model weights, but it also needs the official Emu3.5 source code at runtime. [models/emu35_adapter.py](src/vision_cad_emu35/models/emu35_adapter.py) imports official utilities such as `build_emu3p5`, `build_image`, `generate`, and `multimodal_decode`.

Use this portability-oriented layout:

- Large model weights stay under `/root/autodl-tmp/data`.
- The RAG knowledge base stays under `/root/autodl-tmp/data/outputs/rag_kb`.
- Generated outputs and API artifacts should stay under `/root/autodl-tmp/data/outputs`.
- Official Emu3.5 runtime source code lives inside this project under `third_party/Emu3.5`.

Do not move model weights into `third_party`, and do not vendor downloaded model weights into this repository.

Recommended official source checkout path:

```text
third_party/Emu3.5
```

Clone the official runtime source:

```bash
mkdir -p third_party
git clone https://github.com/baaivision/Emu3.5.git third_party/Emu3.5
```

If GitHub access is slow, manually download/upload the repo or use an available GitHub mirror/proxy.

Then confirm `configs/rag.yaml` points to that checkout while keeping the local model-weight paths under `/root/autodl-tmp/data`:

```yaml
model:
  model_root: "/root/autodl-tmp/data"
  model_id_or_path: "/root/autodl-tmp/data/BAAI/Emu3.5"
  tokenizer_path: "/root/autodl-tmp/data/BAAI/Emu3.5"
  vision_tokenizer_path: "/root/autodl-tmp/data/BAAI/Emu3.5-VisionTokenizer"
  emu_repo_path: "third_party/Emu3.5"
  local_files_only: true
  attn_implementation: "eager"
  clear_transformers_remote_code_cache: true
  patch_tokenizer_source: true
```

If `model.emu_repo_path` is absolute, the runtime uses it directly. If it is relative, the runtime resolves it from the project root, so `third_party/Emu3.5` works no matter where you launch the scripts from.

Do not let the official Emu3.5 requirements reinstall or downgrade torch. The Blackwell environment should keep its CUDA-compatible torch build, such as `torch 2.12.0+cu130`.

If you install official Emu3.5 dependencies, filter out torch and optional acceleration packages first:

```bash
cp third_party/Emu3.5/requirements.txt /tmp/emu35_requirements_no_torch.txt
sed -i '/^torch/d;/^torchvision/d;/^torchaudio/d;/flash-attn/d;/flash_attn/d;/xformers/d;/bitsandbytes/d' /tmp/emu35_requirements_no_torch.txt
pip install -r /tmp/emu35_requirements_no_torch.txt
```

Check that the adapter can import the official runtime utilities:

```bash
python scripts/check_emu35_imports.py --config configs/rag.yaml
```

Check that the custom Emu3.5 tokenizer is compatible with the installed Transformers version:

```bash
python scripts/check_emu35_tokenizer.py --config configs/rag.yaml
```

Check that the project generation config contains the fields expected by the official Emu3.5 generation runtime:

```bash
python scripts/check_emu35_generation_cfg.py --config configs/rag.yaml
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

Rollback indices may be non-continuous. Operation types are derived exactly from `operation_param.json` by `get_exact_operation_type_from_param`.

## Normal RAG Workflow

1. Check GPU environment:

```bash
python scripts/check_gpu_env.py
```

2. Download model weights:

```bash
python scripts/download_models.py
```

3. Install or configure official Emu3.5 runtime source code:

```bash
mkdir -p third_party
git clone https://github.com/baaivision/Emu3.5.git third_party/Emu3.5
```

Make sure `configs/rag.yaml` contains:

```yaml
model:
  emu_repo_path: "third_party/Emu3.5"
```

4. Check official Emu3.5 imports:

```bash
python scripts/check_emu35_imports.py --config configs/rag.yaml
```

5. Check Emu3.5 tokenizer compatibility:

```bash
python scripts/check_emu35_tokenizer.py --config configs/rag.yaml
```

6. Check Emu3.5 generation config compatibility:

```bash
python scripts/check_emu35_generation_cfg.py --config configs/rag.yaml
```

7. Edit dataset path in `configs/rag.yaml`:

```yaml
data:
  dataset_root: "/path/to/your/dataset"
```

The default knowledge base path is also defined in `configs/rag.yaml`:

```yaml
rag:
  kb_dir: "/root/autodl-tmp/data/outputs/rag_kb"
```

8. Prepare manifest:

```bash
python scripts/prepare_manifest.py \
  --dataset-root /path/to/your/dataset \
  --manifest-dir data/manifests \
  --add-stop-samples
```

9. Build RAG knowledge base:

```bash
python scripts/build_kb.py \
  --config configs/rag.yaml \
  --dataset-root /path/to/your/dataset
```

10. Inspect KB:

```bash
python scripts/inspect_kb.py \
  --config configs/rag.yaml
```

11. Run single inference:

```bash
python scripts/infer_rag_single.py \
  --config configs/rag.yaml \
  --final-snapshot examples/final_snapshot.png \
  --prev-depth-map examples/prev_depth_map.png \
  --output-dir /root/autodl-tmp/data/outputs/rag_single
```

12. Launch web demo:

```bash
python scripts/launch_web_demo.py \
  --config configs/rag.yaml
```

Open:

```text
http://SERVER_IP:8000
```

The server binds to `0.0.0.0` by default so other computers on the network can access it.

## Changing the KB Path

Default KB path:

```text
/root/autodl-tmp/data/outputs/rag_kb
```

Option A: edit `configs/rag.yaml`:

```yaml
rag:
  kb_dir: "/new/kb/path"
```

Option B: override from CLI:

```bash
python scripts/build_kb.py \
  --config configs/rag.yaml \
  --dataset-root /path/to/dataset \
  --kb-dir /new/kb/path

python scripts/launch_web_demo.py \
  --config configs/rag.yaml \
  --kb-dir /new/kb/path
```

The same `--kb-dir` override is supported by `inspect_kb.py`, `infer_rag_single.py`, `infer_rag_batch.py`, `launch_api.py`, and `launch_web_demo.py`.

## Empty KB Mode

These cases are supported:

- `/root/autodl-tmp/data/outputs/rag_kb` does not exist.
- `kb_items.jsonl` is empty.
- `embeddings.npy` is missing.
- `embeddings.npy` has shape `(0, dim)`.

In all cases retrieval returns `[]`, the prompt builder creates a zero-shot prompt, and the web UI shows:

```text
Knowledge base is empty. The system is running in zero-shot mode.
```

Generation still requires a locally installed/loadable Emu3.5 model.

## API

```bash
python scripts/launch_api.py \
  --config configs/rag.yaml
```

Endpoints:

- `GET /health`
- `POST /retrieve`
- `POST /generate`
- `POST /reload_kb`

`GET /health` reports `kb_dir`, `kb_loaded`, `kb_empty`, and `kb_item_count`.

`POST /generate` accepts multipart fields:

- `final_snapshot`
- `prev_depth_map`
- `top_k`
- `prompt_extra`

## Troubleshooting

### Emu3ForCausalLM Does Not Support Flash Attention 2 Yet

Problem:

```text
ValueError: Emu3ForCausalLM does not support Flash Attention 2 yet.
```

Solution:

```yaml
model:
  attn_implementation: "eager"
```

This is the default in `configs/rag.yaml`. Supported values are `eager`, `sdpa`, `auto`, and `flash_attention_2`, but `auto` is resolved conservatively to `eager` for Emu3.5 so it does not accidentally select Flash Attention 2. If Flash Attention 2 is requested and Emu3.5 rejects it, the adapter retries once with eager attention.

Optional acceleration libraries are not required for the first working version:

- `flash-attn`
- `xformers`
- `bitsandbytes`
- `vllm`

### Invalid OMP_NUM_THREADS

If logs show:

```text
libgomp: Invalid value for environment variable OMP_NUM_THREADS
auto
```

the runtime scripts normalize invalid, missing, `0`, or `auto` values for `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to `8` before loading torch or Emu3.5.

The runtime also sets this memory-safer CUDA allocator default if it is not already configured:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Emu3Tokenizer special_tokens_set Error

Problem:

```text
AttributeError: Emu3Tokenizer has no attribute special_tokens_set
```

Cause:

The custom Emu3Tokenizer code is incompatible with the current Transformers initialization path, or stale Hugging Face remote-code cache is being used from:

```text
~/.cache/huggingface/modules/transformers_modules/
```

Solution:

```bash
python scripts/check_emu35_tokenizer.py --config configs/rag.yaml
```

The checker applies the local tokenizer compatibility patch, clears only the Emu3.5-specific Transformers remote-code cache when `model.clear_transformers_remote_code_cache: true`, loads the tokenizer with `local_files_only=True`, and runs a short encode/decode test.

Then retry:

```bash
python scripts/infer_rag_single.py \
  --config configs/rag.yaml \
  --final-snapshot examples/final_snapshot.png \
  --prev-depth-map examples/prev_depth_map.png \
  --output-dir outputs/rag_single
```

If stale cache is suspected, clear only the Emu3.5 remote-code cache under `~/.cache/huggingface/modules/transformers_modules/`. Do not delete model weights under `/root/autodl-tmp/data/BAAI/Emu3.5`.

### Emu3.5 Generation Config Missing Field

Problem:

```text
AttributeError: 'types.SimpleNamespace' object has no attribute 'unconditional_type'
```

Cause:

The official `third_party/Emu3.5/src/utils/generation_utils.py` expects an Emu3.5-shaped config object, including `unconditional_type`, `special_token_ids`, `classifier_free_guidance`, `sampling_params`, target image dimensions, and image CFG fields.

Solution:

```bash
python scripts/check_emu35_generation_cfg.py --config configs/rag.yaml
```

The adapter builds this object with `build_emu35_generation_cfg()` and fills safe defaults. `configs/rag.yaml` includes all configurable generation fields explicitly.

### Runtime Warnings

Warnings that should be fixed before treating inference as healthy:

- `Emu3ForCausalLM does not support Flash Attention 2 yet`: keep `model.attn_implementation: "eager"`.
- `Emu3Tokenizer has no attribute special_tokens_set`: run `python scripts/check_emu35_tokenizer.py --config configs/rag.yaml`.
- `SimpleNamespace has no attribute unconditional_type`: run `python scripts/check_emu35_generation_cfg.py --config configs/rag.yaml`.
- `do_sample=false` with `temperature`, `top_p`, or `top_k`: keep these generic generation fields as `null` when `do_sample: false`.
- `libgomp: Invalid value for environment variable OMP_NUM_THREADS`: let the runtime normalize env vars or export positive integers manually.

Warnings that are currently non-fatal compatibility noise:

- `seen_tokens` deprecation warnings from Transformers cache internals.
- `get_max_cache` deprecation warnings from Transformers cache internals.

Those cache warnings are from upstream API drift and should not block a working MVP unless they become errors in a later Transformers release.

## RAG Components

- `rag/image_embedding.py`: CLIP if available, simple CPU embedding by default.
- `rag/vector_store.py`: pure NumPy cosine-similarity vector store.
- `rag/retriever.py`: empty-KB-safe retrieval with optional operation type filtering.
- `rag/prompt_builder.py`: multimodal prompt packing with configurable retrieved example images.
- `models/emu35_adapter.py`: frozen Emu3.5 inference-only adapter.

## Not a Fine-Tuning Project

Fine-tuning scripts, LoRA/QLoRA training, optimizer loops, validation loss, and checkpoint training workflows have been removed. The current project is RAG-first and uses frozen Emu3.5 for inference only.

## Tests

```bash
pytest
```

Tests cover operation type extraction, dataset scanning, empty-KB behavior, prompt building, and vector retrieval.
