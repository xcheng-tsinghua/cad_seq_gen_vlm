# Vision2CAD

Frozen Emu3.5 system for vision-based CAD modeling step reverse generation and general multimodal inference.

This project no longer fine-tunes Emu3.5. It keeps Emu3.5 frozen, builds a retrieval knowledge base from historical CAD modeling steps, retrieves similar examples for a query, and asks Emu3.5 to generate:

- `Operation_Type: <operation_type>`
- one CAD-style preview image, `overlayed_all.png`

The system supports an empty knowledge base. If no KB exists, the API and web demo still launch and CAD-RAG runs zero-shot using only the query images and drawing rules. General Emu3.5 mode does not use the knowledge base at all.

## Modes

1. CAD-RAG Mode

Uses the CAD knowledge base and CAD-specific prompt to predict CAD modeling steps from `final_snapshot.png` and `prev_depth_map.png`. It returns `Operation_Type` plus the generated CAD preview image when Emu3.5 produces one.

2. General Emu3.5 Mode

Directly exposes frozen Emu3.5 for normal multimodal tasks. It accepts a user text prompt and zero or more images, skips RAG entirely, and returns raw text, generated image artifacts when present, and debug metadata.

## Project Layout

The repository now runs directly from the `cad_seq_gen_vlm` root, and the Python modules are flattened directly under `src/`:

```text
cad_seq_gen_vlm/
  configs/                 Runtime YAML configs.
  examples/                Small checked-in demo images.
  scripts/                 CLI entry points for download, checks, KB build, API, and demo.
  src/                    Python modules and top-level packages.
  tests/                   Unit tests.
  third_party/Emu3.5/      Local official Emu3.5 runtime checkout, ignored by Git.
```

Large local folders such as `data/`, `outputs/`, `checkpoints/`, `pretrained_lm/`, and `third_party/` are intentionally ignored. Keep model weights and generated artifacts outside Git.

## Filename Configuration

Project dataset, output artifact, manifest, and KB filenames are centralized in:

```text
src/filenames.py
```

Change names such as `final_snapshot.png`, `prev_depth_map.png`, `overlayed_all.png`, `operation_param.json`, `response.json`, or `kb_items.jsonl` there instead of editing scattered call sites.

## Environment Setup

The validated runtime uses conda, Python `3.12.13`, PyTorch `2.11.0+cu128`, and Transformers `4.48.2`.

```bash
cd /opt/data/private/networks/cad_seq_gen_vlm

conda create -n cad_vlm -c conda-forge python=3.12.13 -y
conda activate cad_vlm
python -m pip install --upgrade pip setuptools wheel

python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128

mkdir -p third_party
git clone https://github.com/baaivision/Emu3.5.git third_party/Emu3.5

python -m pip install -r requirements.txt
```

Install the download helpers if you need `scripts/download_models.py`:

```bash
python -m pip install modelscope huggingface_hub
```

For development-only tools:

```bash
python -m pip install pytest ruff
```

Quick verification:

```bash
python scripts/check_gpu_env.py
python scripts/check_emu35_imports.py --config configs/rag.yaml
python scripts/check_emu35_tokenizer.py --config configs/rag.yaml
python scripts/check_emu35_generation_cfg.py --config configs/rag.yaml
```

## CUDA 12.8 / Blackwell Notes
Optional acceleration libraries are not required:

- `flash-attn`
- `xformers`
- `bitsandbytes`
- `vllm`

If they are absent, the project falls back to standard PyTorch inference with a warning.

For CPU-only model downloading or RAG KB building:

## Download weights from ModelScope Without GPU

Recommended for mainland China:

```bash
python -m pip install modelscope huggingface_hub
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
python scripts/prepare_manifest.py --dataset-root /path/to/your/dataset --manifest-dir data/manifests --add-stop-samples
```

9. Build RAG knowledge base:

```bash
python scripts/build_kb.py --config configs/rag.yaml --dataset-root /path/to/your/dataset
```

10. Inspect KB:

```bash
python scripts/inspect_kb.py --config configs/rag.yaml
```

11. Run single inference:

```bash
python scripts/infer_rag_single.py --config configs/rag.yaml --final-snapshot examples/final_snapshot.png --prev-depth-map examples/prev_depth_map.png --output-dir /root/autodl-tmp/data/outputs/rag_single
```

12. Launch web demo:

```bash
python scripts/launch_web_demo.py --config configs/rag.yaml
```

Open:

```text
http://SERVER_IP:8000
```

The server binds to `0.0.0.0` by default so other computers on the network can access it.

## General Emu3.5 Mode

Text-only:

```bash
python scripts/infer_general.py \
  --config configs/general.yaml \
  --prompt "Explain what Emu3.5 can do." \
  --output-dir outputs/general_text
```

Image + text:

```bash
python scripts/infer_general.py \
  --config configs/general.yaml \
  --prompt "Describe this image." \
  --image examples/final_snapshot.png \
  --output-dir outputs/general_image
```

Multiple images:

```bash
python scripts/infer_general.py \
  --config configs/general.yaml \
  --prompt "Compare these two images." \
  --image examples/final_snapshot.png \
  --image examples/prev_depth_map.png \
  --output-dir outputs/general_multi_image
```

The script writes `response.json`, `raw_text.txt`, generated image files when present, and Emu3.5 debug events when enabled. Text-only responses are valid; no generated image is not treated as a failure.

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
- `POST /generate` (backward-compatible CAD-RAG endpoint)
- `POST /cad/generate`
- `POST /general/generate`
- `POST /reload_kb`

`GET /health` reports `model_loaded`, `modes_supported`, CAD-RAG KB status, and GPU info when available.

`POST /generate` and `POST /cad/generate` accept multipart fields:

- `final_snapshot`
- `prev_depth_map`
- `top_k`
- `prompt_extra`

`POST /general/generate` accepts multipart fields:

- `prompt`
- `images` repeated zero to five times

The web demo is launched the same way:

```bash
python scripts/launch_web_demo.py \
  --config configs/rag.yaml
```

Open:

```text
http://SERVER_IP:8000
```

The page includes both CAD-RAG and General Emu3.5 tabs.

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

- `src/rag/image_embedding.py`: CLIP if available, simple CPU embedding by default.
- `src/rag/vector_store.py`: pure NumPy cosine-similarity vector store.
- `src/rag/retriever.py`: empty-KB-safe retrieval with optional operation type filtering.
- `src/rag/prompt_builder.py`: multimodal prompt packing with configurable retrieved example images.
- `src/models/emu35_adapter.py`: frozen Emu3.5 inference-only adapter.

## Not a Fine-Tuning Project

Fine-tuning scripts, LoRA/QLoRA training, optimizer loops, validation loss, and checkpoint training workflows have been removed. The current project is RAG-first and uses frozen Emu3.5 for inference only.

## Tests

```bash
pytest
```

Tests cover operation type extraction, dataset scanning, empty-KB behavior, prompt building, and vector retrieval.

## An environment successfully running this project
```
GPU: RTX PRO 6000
(cad_vlm) root@autodl-container-eesuqcek0l-c2d01990:/opt/data/private/networks/cad_seq_gen_vlm# pip list
Package                  Version
------------------------ ------------
accelerate               1.13.0
aiofiles                 24.1.0
annotated-doc            0.0.4
annotated-types          0.7.0
antlr4-python3-runtime   4.9.3
anyio                    4.13.0
brotli                   1.2.0
certifi                  2026.5.20
charset-normalizer       3.4.7
click                    8.4.1
cuda-bindings            12.9.4
cuda-pathfinder          1.2.2
cuda-toolkit             12.8.1
einops                   0.8.2
fastapi                  0.136.3
ffmpy                    1.0.0
filelock                 3.29.0
fsspec                   2026.4.0
gradio                   5.49.1
gradio_client            1.13.3
groovy                   0.1.2
h11                      0.16.0
hf-xet                   1.5.0
httpcore                 1.0.9
httpx                    0.28.1
huggingface_hub          0.36.2
idna                     3.16
imageio                  2.37.0
imageio-ffmpeg           0.6.0
Jinja2                   3.1.6
markdown-it-py           4.2.0
MarkupSafe               3.0.3
mdurl                    0.1.2
mpmath                   1.3.0
networkx                 3.6.1
numpy                    2.4.4
nvidia-cublas-cu12       12.8.4.1
nvidia-cuda-cupti-cu12   12.8.90
nvidia-cuda-nvrtc-cu12   12.8.93
nvidia-cuda-runtime-cu12 12.8.90
nvidia-cudnn-cu12        9.19.0.56
nvidia-cufft-cu12        11.3.3.83
nvidia-cufile-cu12       1.13.1.3
nvidia-curand-cu12       10.3.9.90
nvidia-cusolver-cu12     11.7.3.90
nvidia-cusparse-cu12     12.5.8.93
nvidia-cusparselt-cu12   0.7.1
nvidia-nccl-cu12         2.28.9
nvidia-nvjitlink-cu12    12.8.93
nvidia-nvshmem-cu12      3.4.5
nvidia-nvtx-cu12         12.8.90
omegaconf                2.3.0
orjson                   3.11.9
packaging                26.0
pandas                   2.3.3
pillow                   11.3.0
pip                      26.0.1
protobuf                 7.35.0
psutil                   7.2.2
pydantic                 2.11.10
pydantic_core            2.33.2
pydub                    0.25.1
Pygments                 2.20.0
python-dateutil          2.9.0.post0
python-multipart         0.0.29
pytz                     2026.2
PyYAML                   6.0.3
regex                    2026.5.9
requests                 2.34.2
rich                     15.0.0
ruff                     0.15.14
safehttpx                0.1.7
safetensors              0.7.0
semantic-version         2.10.0
setuptools               70.2.0
shellingham              1.5.4
six                      1.17.0
starlette                0.52.1
sympy                    1.14.0
tiktoken                 0.13.0
tokenizers               0.21.4
tomlkit                  0.13.3
torch                    2.11.0+cu128
torchaudio               2.11.0+cu128
torchvision              0.26.0+cu128
tqdm                     4.67.3
transformers             4.48.2
triton                   3.6.0
typer                    0.25.1
typing_extensions        4.15.0
typing-inspection        0.4.2
tzdata                   2026.2
urllib3                  2.7.0
uvicorn                  0.48.0
websockets               15.0.1
wheel                    0.46.3
(cad_vlm) root@autodl-container-eesuqcek0l-c2d01990:/opt/data/private/networks/cad_seq_gen_vlm# conda list
# packages in environment at /root/miniconda3/envs/cad_vlm:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
_openmp_mutex             5.1                      52_gnu    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
accelerate                1.13.0                   pypi_0    pypi
aiofiles                  24.1.0                   pypi_0    pypi
annotated-doc             0.0.4                    pypi_0    pypi
annotated-types           0.7.0                    pypi_0    pypi
antlr4-python3-runtime    4.9.3                    pypi_0    pypi
anyio                     4.13.0                   pypi_0    pypi
brotli                    1.2.0                    pypi_0    pypi
bzip2                     1.0.8                h5eee18b_6    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
ca-certificates           2026.5.20            hbd8a1cb_0    conda-forge
certifi                   2026.5.20                pypi_0    pypi
charset-normalizer        3.4.7                    pypi_0    pypi
click                     8.4.1                    pypi_0    pypi
cuda-bindings             12.9.4                   pypi_0    pypi
cuda-pathfinder           1.2.2                    pypi_0    pypi
cuda-toolkit              12.8.1                   pypi_0    pypi
einops                    0.8.2                    pypi_0    pypi
fastapi                   0.136.3                  pypi_0    pypi
ffmpy                     1.0.0                    pypi_0    pypi
filelock                  3.29.0                   pypi_0    pypi
fsspec                    2026.4.0                 pypi_0    pypi
gradio                    5.49.1                   pypi_0    pypi
gradio-client             1.13.3                   pypi_0    pypi
groovy                    0.1.2                    pypi_0    pypi
h11                       0.16.0                   pypi_0    pypi
hf-xet                    1.5.0                    pypi_0    pypi
httpcore                  1.0.9                    pypi_0    pypi
httpx                     0.28.1                   pypi_0    pypi
huggingface-hub           0.36.2                   pypi_0    pypi
icu                       78.3                 h33c6efd_0    conda-forge
idna                      3.16                     pypi_0    pypi
imageio                   2.37.0                   pypi_0    pypi
imageio-ffmpeg            0.6.0                    pypi_0    pypi
jinja2                    3.1.6                    pypi_0    pypi
ld_impl_linux-64          2.44                 h9e0c5a2_3    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libexpat                  2.8.0                h7354ed3_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libffi                    3.4.8                hc5d346e_2    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libgcc                    15.2.0               h69a1729_8    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libgcc-ng                 15.2.0               h166f726_8    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libstdcxx                 15.2.0               h39759b7_8    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libuuid                   1.41.5               h5eee18b_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libuv                     1.52.1               h280c20c_0    conda-forge
libxcb                    1.17.0               h9b100fa_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
libzlib                   1.3.2                h25fd6f3_2    conda-forge
markdown-it-py            4.2.0                    pypi_0    pypi
markupsafe                3.0.3                    pypi_0    pypi
mdurl                     0.1.2                    pypi_0    pypi
mpmath                    1.3.0                    pypi_0    pypi
ncurses                   6.5                  h7934f7d_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
networkx                  3.6.1                    pypi_0    pypi
nodejs                    22.22.3              h273caaf_0    conda-forge
numpy                     2.4.4                    pypi_0    pypi
nvidia-cublas-cu12        12.8.4.1                 pypi_0    pypi
nvidia-cuda-cupti-cu12    12.8.90                  pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.8.93                  pypi_0    pypi
nvidia-cuda-runtime-cu12  12.8.90                  pypi_0    pypi
nvidia-cudnn-cu12         9.19.0.56                pypi_0    pypi
nvidia-cufft-cu12         11.3.3.83                pypi_0    pypi
nvidia-cufile-cu12        1.13.1.3                 pypi_0    pypi
nvidia-curand-cu12        10.3.9.90                pypi_0    pypi
nvidia-cusolver-cu12      11.7.3.90                pypi_0    pypi
nvidia-cusparse-cu12      12.5.8.93                pypi_0    pypi
nvidia-cusparselt-cu12    0.7.1                    pypi_0    pypi
nvidia-nccl-cu12          2.28.9                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.8.93                  pypi_0    pypi
nvidia-nvshmem-cu12       3.4.5                    pypi_0    pypi
nvidia-nvtx-cu12          12.8.90                  pypi_0    pypi
omegaconf                 2.3.0                    pypi_0    pypi
openssl                   3.6.2                h35e630c_0    conda-forge
orjson                    3.11.9                   pypi_0    pypi
packaging                 26.0            py312h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pandas                    2.3.3                    pypi_0    pypi
pillow                    11.3.0                   pypi_0    pypi
pip                       26.0.1             pyhc872135_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
protobuf                  7.35.0                   pypi_0    pypi
psutil                    7.2.2                    pypi_0    pypi
pthread-stubs             0.3                  h0ce48e5_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
pydantic                  2.11.10                  pypi_0    pypi
pydantic-core             2.33.2                   pypi_0    pypi
pydub                     0.25.1                   pypi_0    pypi
pygments                  2.20.0                   pypi_0    pypi
python                    3.12.13              h4d16e0c_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
python-dateutil           2.9.0.post0              pypi_0    pypi
python-multipart          0.0.29                   pypi_0    pypi
pytz                      2026.2                   pypi_0    pypi
pyyaml                    6.0.3                    pypi_0    pypi
readline                  8.3                  hc2a1206_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
regex                     2026.5.9                 pypi_0    pypi
requests                  2.34.2                   pypi_0    pypi
rich                      15.0.0                   pypi_0    pypi
ruff                      0.15.14                  pypi_0    pypi
safehttpx                 0.1.7                    pypi_0    pypi
safetensors               0.7.0                    pypi_0    pypi
semantic-version          2.10.0                   pypi_0    pypi
setuptools                70.2.0                   pypi_0    pypi
shellingham               1.5.4                    pypi_0    pypi
six                       1.17.0                   pypi_0    pypi
sqlite                    3.51.2               h3e8d24a_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
starlette                 0.52.1                   pypi_0    pypi
sympy                     1.14.0                   pypi_0    pypi
tiktoken                  0.13.0                   pypi_0    pypi
tk                        8.6.15               h54e0aa7_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
tokenizers                0.21.4                   pypi_0    pypi
tomlkit                   0.13.3                   pypi_0    pypi
torch                     2.11.0+cu128             pypi_0    pypi
torchaudio                2.11.0+cu128             pypi_0    pypi
torchvision               0.26.0+cu128             pypi_0    pypi
tqdm                      4.67.3                   pypi_0    pypi
transformers              4.48.2                   pypi_0    pypi
triton                    3.6.0                    pypi_0    pypi
typer                     0.25.1                   pypi_0    pypi
typing-extensions         4.15.0                   pypi_0    pypi
typing-inspection         0.4.2                    pypi_0    pypi
tzdata                    2026.2                   pypi_0    pypi
urllib3                   2.7.0                    pypi_0    pypi
uvicorn                   0.48.0                   pypi_0    pypi
websockets                15.0.1                   pypi_0    pypi
wheel                     0.46.3          py312h06a4308_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
xorg-libx11               1.8.12               h9b100fa_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
xorg-libxau               1.0.12               h9b100fa_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
xorg-libxdmcp             1.1.5                h9b100fa_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
xorg-xorgproto            2024.1               h5eee18b_1    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
xz                        5.8.2                h448239c_0    https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
zlib                      1.3.2                h25fd6f3_2    conda-forge
```



