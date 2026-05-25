from __future__ import annotations

import io
import json
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from vision_cad_emu35 import __version__
from vision_cad_emu35.config import AppConfig
from vision_cad_emu35.model_paths import ensure_default_local_model_paths, validate_local_model_paths
from vision_cad_emu35.models.emu35_adapter import Emu35Adapter
from vision_cad_emu35.rag.prompt_builder import RagPromptBuilder
from vision_cad_emu35.rag.retriever import RagRetriever
from vision_cad_emu35.utils.gpu import get_gpu_info
from vision_cad_emu35.utils.image_io import image_to_base64, save_image


def create_app(config: AppConfig, checkpoint: str | Path | None = None) -> Any:
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise ImportError("fastapi and uvicorn are required for the API server.") from exc

    app = FastAPI(title="vision_cad_emu35_rag", version=__version__)
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
            "kb_loaded": True,
            "kb_empty": retriever.is_empty(),
            "kb_item_count": retriever.item_count(),
            "version": __version__,
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
        if result.get("image") is not None:
            save_image(result["image"], request_dir / "overlayed_all.png")
        response = {
            "operation_type": result.get("operation_type"),
            "image_base64": image_to_base64(result["image"]) if result.get("image") is not None else None,
            "retrieved_examples": prompt.retrieved_examples,
            "zero_shot": len(examples) == 0,
            "metadata": {**result.get("metadata", {}), "image_roles": prompt.image_roles},
            "latency_seconds": time.perf_counter() - start,
            "warning": "No retrieved examples available; running zero-shot mode." if not examples else None,
        }
        (request_dir / "prompt.txt").write_text(prompt.prompt_text, encoding="utf-8")
        (request_dir / "response.json").write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")
        return response

    @app.post("/reload_kb")
    async def reload_kb() -> dict[str, Any]:
        app.state.retriever = RagRetriever(config.rag.kb_dir, config.rag)
        retriever: RagRetriever = app.state.retriever
        return {
            "kb_loaded": True,
            "kb_empty": retriever.is_empty(),
            "kb_item_count": retriever.item_count(),
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
        try:
            from fastapi import HTTPException
        except ImportError:
            raise RuntimeError("Model is not loaded.")
        detail = "Model is not loaded."
        if app.state.model_error:
            detail += f" {app.state.model_error}"
        raise HTTPException(status_code=503, detail=detail)
    return app.state.adapter
