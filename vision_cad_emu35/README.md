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

Install the official Emu3.5 runtime code separately. If its utilities are not importable, set `model.emu_repo_path` in `configs/rag.yaml` to a local checkout of the official Emu3.5 repo.

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

1. Edit dataset path in `configs/rag.yaml`:

```yaml
data:
  dataset_root: "/path/to/your/dataset"
```

The default knowledge base path is also defined in `configs/rag.yaml`:

```yaml
rag:
  kb_dir: "/root/autodl-tmp/data/outputs/rag_kb"
```

2. Prepare manifest:

```bash
python scripts/prepare_manifest.py \
  --dataset-root /path/to/your/dataset \
  --manifest-dir data/manifests \
  --add-stop-samples
```

3. Build RAG knowledge base:

```bash
python scripts/build_kb.py \
  --config configs/rag.yaml \
  --dataset-root /path/to/your/dataset
```

4. Inspect KB:

```bash
python scripts/inspect_kb.py \
  --config configs/rag.yaml
```

5. Run single inference:

```bash
python scripts/infer_rag_single.py \
  --config configs/rag.yaml \
  --final-snapshot examples/final_snapshot.png \
  --prev-depth-map examples/prev_depth_map.png \
  --output-dir outputs/rag_single
```

6. Launch web demo:

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
