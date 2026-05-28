from __future__ import annotations

import pytest


def test_general_and_cad_routes_are_registered(tmp_path):
    pytest.importorskip("numpy")

    from config import AppConfig
    from api import server

    if server._FASTAPI_IMPORT_ERROR is not None:
        pytest.skip("fastapi is not installed")

    config = AppConfig()
    config.api.artifacts_dir = str(tmp_path / "artifacts")
    config.rag.kb_dir = str(tmp_path / "kb")

    try:
        app = server.create_app(config)
    except RuntimeError as exc:
        if "python-multipart" in str(exc):
            pytest.skip("python-multipart is not installed")
        raise

    route_paths = {route.path for route in app.routes}
    assert "/health" in route_paths
    assert "/generate" in route_paths
    assert "/cad/generate" in route_paths
    assert "/general/generate" in route_paths
    assert "/artifacts/{artifact_path:path}" in route_paths
