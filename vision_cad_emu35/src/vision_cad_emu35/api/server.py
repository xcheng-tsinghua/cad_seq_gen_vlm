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
from vision_cad_emu35.inference.autoregressive import AutoregressiveCADPlanner
from vision_cad_emu35.inference.single_step import load_adapter_for_inference
from vision_cad_emu35.utils.gpu import get_gpu_info
from vision_cad_emu35.utils.image_io import image_to_base64, save_image
from vision_cad_emu35.utils.jsonl import read_jsonl


def create_app(config: AppConfig, checkpoint: str | Path | None = None) -> Any:
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise ImportError("fastapi and uvicorn are required for the API server.") from exc

    app = FastAPI(title="vision_cad_emu35", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.config = config
    app.state.adapter = None
    app.state.checkpoint = str(checkpoint) if checkpoint else None
    artifact_root = Path(config.api.artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.adapter = load_adapter_for_inference(config, checkpoint)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok" if app.state.adapter is not None else "not_loaded",
            "model_loaded": app.state.adapter is not None,
            "version": __version__,
            "gpu": get_gpu_info(),
        }

    @app.get("/")
    async def index() -> Any:
        index_path = Path(__file__).resolve().parents[1] / "web" / "index.html"
        return FileResponse(index_path)

    @app.post("/generate")
    async def generate(
        final_snapshot: UploadFile = File(...),
        prev_depth_map: UploadFile = File(...),
        prompt: str | None = Form(None),
    ) -> dict[str, Any]:
        adapter = _require_adapter(app)
        request_dir = artifact_root / uuid.uuid4().hex
        request_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        try:
            final_image = await _upload_to_image(final_snapshot)
            prev_image = await _upload_to_image(prev_depth_map)
            save_image(final_image, request_dir / "final_snapshot.png")
            save_image(prev_image, request_dir / "prev_depth_map.png")
            result = adapter.generate(final_image, prev_image, prompt, config.generation)
            save_image(result["image"], request_dir / "overlayed_all.png")
            response = {
                "operation_type": result["operation_type"],
                "image_base64": image_to_base64(result["image"]),
                "metadata": result.get("metadata", {}),
                "latency_seconds": time.perf_counter() - start,
            }
            (request_dir / "response.json").write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")
            return response
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/generate_batch")
    async def generate_batch(
        manifest: UploadFile | None = File(None),
        final_snapshots: list[UploadFile] | None = File(None),
        prev_depth_maps: list[UploadFile] | None = File(None),
        prompt: str | None = Form(None),
    ) -> dict[str, Any]:
        adapter = _require_adapter(app)
        request_dir = artifact_root / uuid.uuid4().hex
        request_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        if manifest is not None:
            manifest_path = request_dir / "manifest.jsonl"
            manifest_path.write_bytes(await manifest.read())
            rows = list(read_jsonl(manifest_path))
            for idx, row in enumerate(rows):
                sample_dir = request_dir / f"sample_{idx:04d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                final_image = Image.open(row["final_snapshot_path"]).convert("RGB")
                prev_image = Image.open(row["prev_depth_map_path"]).convert("RGB")
                result = adapter.generate(final_image, prev_image, row.get("prompt"), config.generation)
                save_image(result["image"], sample_dir / "overlayed_all.png")
                results.append(_batch_result(row.get("sample_id", str(idx)), result, sample_dir))
        else:
            final_snapshots = final_snapshots or []
            prev_depth_maps = prev_depth_maps or []
            if not final_snapshots or len(final_snapshots) != len(prev_depth_maps):
                raise HTTPException(
                    status_code=400,
                    detail="Provide either a manifest file or equal-length final_snapshots and prev_depth_maps uploads.",
                )
            for idx, (final_upload, prev_upload) in enumerate(zip(final_snapshots, prev_depth_maps)):
                sample_dir = request_dir / f"sample_{idx:04d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                final_image = await _upload_to_image(final_upload)
                prev_image = await _upload_to_image(prev_upload)
                save_image(final_image, sample_dir / "final_snapshot.png")
                save_image(prev_image, sample_dir / "prev_depth_map.png")
                result = adapter.generate(final_image, prev_image, prompt, config.generation)
                save_image(result["image"], sample_dir / "overlayed_all.png")
                results.append(_batch_result(str(idx), result, sample_dir))
        (request_dir / "batch_response.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        return {"results": results}

    @app.post("/autoregressive")
    async def autoregressive(
        final_snapshot: UploadFile = File(...),
        initial_depth_map: UploadFile = File(...),
        max_steps: int = Form(20),
        prompt: str | None = Form(None),
    ) -> dict[str, Any]:
        adapter = _require_adapter(app)
        request_dir = artifact_root / uuid.uuid4().hex
        request_dir.mkdir(parents=True, exist_ok=True)
        final_image = await _upload_to_image(final_snapshot)
        depth_image = await _upload_to_image(initial_depth_map)
        final_path = request_dir / "final_snapshot.png"
        depth_path = request_dir / "initial_depth_map.png"
        save_image(final_image, final_path)
        save_image(depth_image, depth_path)
        planner = AutoregressiveCADPlanner(adapter, generation_config=config.generation)
        try:
            steps = planner.run(final_path, depth_path, request_dir / "steps", max_steps=max_steps, prompt=prompt)
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        return {"steps": steps}

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


def _batch_result(sample_id: str, result: dict[str, Any], sample_dir: Path) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "operation_type": result["operation_type"],
        "output_dir": str(sample_dir),
        "metadata": result.get("metadata", {}),
    }


def _require_adapter(app: Any) -> Any:
    if app.state.adapter is None:
        try:
            from fastapi import HTTPException
        except ImportError:
            raise RuntimeError("Model is not loaded.")
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return app.state.adapter
