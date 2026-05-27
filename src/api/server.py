import io
import json
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from PIL import Image

from config import AppConfig
from model_paths import ensure_default_local_model_paths, validate_local_model_paths
from models.emu35_adapter import Emu35Adapter
from rag.prompt_builder import RagPromptBuilder
from rag.retriever import RagRetriever
from utils.gpu import get_gpu_info
from utils.image_io import image_to_base64, save_image
from utils.runtime_env import normalize_thread_env

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
except ImportError as exc:
    FastAPI = File = Form = HTTPException = UploadFile = None  # type: ignore[assignment]
    CORSMiddleware = FileResponse = None  # type: ignore[assignment]
    _FASTAPI_IMPORT_ERROR: ImportError | None = exc
else:
    _FASTAPI_IMPORT_ERROR = None

try:
    APP_VERSION = version("vision-cad-emu35")
except PackageNotFoundError:
    APP_VERSION = "0.1.0"


def create_app(config: AppConfig, checkpoint: str | Path | None = None) -> Any:
    _ensure_fastapi()

    app = FastAPI(title="cad_seq_gen_vlm_rag", version=APP_VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.config = config
    app.state.adapter = None
    app.state.model_error = None
    app.state.retriever = RagRetriever(config.rag.kb_dir, config.rag)
    artifact_root = Path(config.api.artifacts_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)

    @app.on_event("startup")
    async def _startup() -> None:
        try:
            normalize_thread_env()
            ensure_default_local_model_paths(config.model)
            validate_local_model_paths(config.model)
            adapter = Emu35Adapter(config.model)
            adapter.load_model()
            app.state.adapter = adapter
        except Exception as exc:
            app.state.model_error = str(exc)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        retriever: RagRetriever = app.state.retriever
        return {
            "status": "ok",
            "model_loaded": app.state.adapter is not None,
            "model_error": app.state.model_error,
            **retriever.status(),
            "version": APP_VERSION,
            "gpu": get_gpu_info(),
        }

    @app.get("/")
    async def index() -> Any:
        index_path = Path(__file__).resolve().parents[1] / "web" / "index.html"
        return FileResponse(index_path)

    @app.post("/retrieve")
    async def retrieve(
        final_snapshot: UploadFile = File(...),
        prev_depth_map: UploadFile = File(...),
        top_k: int | None = Form(None),
    ) -> dict[str, Any]:
        final_image = await _upload_to_image(final_snapshot)
        prev_image = await _upload_to_image(prev_depth_map)
        examples = app.state.retriever.retrieve(final_image, prev_image, top_k=top_k or config.rag.top_k)
        return {
            "kb_dir": str(app.state.retriever.kb_dir),
            "retrieved_examples": examples,
            "zero_shot": len(examples) == 0,
            "message": "No retrieved examples available; running zero-shot mode." if not examples else None,
        }

    @app.post("/generate")
    async def generate(
        final_snapshot: UploadFile = File(...),
        prev_depth_map: UploadFile = File(...),
        top_k: int | None = Form(None),
        prompt_extra: str | None = Form(None),
    ) -> dict[str, Any]:
        adapter = _require_adapter(app)
        request_dir = artifact_root / uuid.uuid4().hex
        request_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        final_image = await _upload_to_image(final_snapshot)
        prev_image = await _upload_to_image(prev_depth_map)
        save_image(final_image, request_dir / "final_snapshot.png")
        save_image(prev_image, request_dir / "prev_depth_map.png")
        examples = app.state.retriever.retrieve(final_image, prev_image, top_k=top_k or config.rag.top_k)
        prompt = RagPromptBuilder(config.rag, image_size=config.model.image_size).build(
            final_image,
            prev_image,
            examples,
            prompt_extra=prompt_extra,
        )
        result = adapter.generate_multimodal(prompt.prompt_text, prompt.images, config.generation)
        generated_images = list(result.get("images") or [])
        if not generated_images and result.get("image") is not None:
            generated_images = [result["image"]]
        image_path: str | None = None
        generated_image_paths: list[str] = []
        if generated_images:
            overlay_path = request_dir / "overlayed_all.png"
            save_image(generated_images[0], overlay_path)
            image_path = str(overlay_path)
            generated_dir = request_dir / "generated_images"
            for index, image in enumerate(generated_images):
                target = generated_dir / f"image_{index:03d}.png"
                save_image(image, target)
                generated_image_paths.append(str(target))
        debug_events = result.get("emu35_events_debug") or result.get("metadata", {}).get("event_summaries") or []
        debug_events_path: str | None = None
        if getattr(config.generation, "save_debug_events", True) or not generated_images:
            debug_path = request_dir / "emu35_events_debug.json"
            debug_path.write_text(
                json.dumps(
                    {
                        "num_generation_events": result.get("metadata", {}).get("num_generation_events", len(debug_events)),
                        "num_generated_images": len(generated_images),
                        "raw_text_missing": result.get("raw_text_missing", not bool(result.get("raw_text", ""))),
                        "events": debug_events,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            debug_events_path = str(debug_path)
        response = {
            "operation_type": result.get("operation_type"),
            "raw_text": result.get("raw_text", ""),
            "raw_text_missing": result.get("raw_text_missing", not bool(result.get("raw_text", ""))),
            "image_base64": image_to_base64(generated_images[0]) if generated_images else None,
            "image_path": image_path,
            "num_generated_images": len(generated_images),
            "generated_image_paths": generated_image_paths,
            "debug_events_path": debug_events_path,
            "image_missing": not bool(generated_images),
            "retrieved_examples": prompt.retrieved_examples,
            "kb_dir": str(app.state.retriever.kb_dir),
            "zero_shot": len(examples) == 0,
            "metadata": {**result.get("metadata", {}), "image_roles": prompt.image_roles},
            "latency_seconds": time.perf_counter() - start,
            "warning": "No retrieved examples available; running zero-shot mode." if not examples else None,
        }
        if not generated_images:
            response.update(
                {
                    "image_missing_reason": "No PIL image was found in Emu3.5 generation events.",
                }
            )
        (request_dir / "prompt.txt").write_text(prompt.prompt_text, encoding="utf-8")
        (request_dir / "response.json").write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")
        return response

    @app.post("/reload_kb")
    async def reload_kb() -> dict[str, Any]:
        app.state.retriever = RagRetriever(config.rag.kb_dir, config.rag)
        retriever: RagRetriever = app.state.retriever
        return {
            **retriever.status(),
        }

    async def _upload_to_image(upload: UploadFile) -> Image.Image:
        content_type = upload.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Unsupported upload type for {upload.filename}: {content_type}")
        data = await upload.read()
        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image upload {upload.filename}: {exc}") from exc

    return app


def _require_adapter(app: Any) -> Any:
    if app.state.adapter is None:
        if HTTPException is None:
            raise RuntimeError("Model is not loaded.")
        detail = "Model is not loaded."
        if app.state.model_error:
            detail += f" {app.state.model_error}"
        raise HTTPException(status_code=503, detail=detail)
    return app.state.adapter


def _ensure_fastapi() -> None:
    if _FASTAPI_IMPORT_ERROR is not None:
        raise ImportError("fastapi and uvicorn are required for the API server.") from _FASTAPI_IMPORT_ERROR
