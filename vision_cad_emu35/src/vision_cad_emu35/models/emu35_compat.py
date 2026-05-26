from __future__ import annotations

import os
import re
import shutil
import warnings
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from vision_cad_emu35.config import ModelConfig, resolve_project_path
from vision_cad_emu35.model_paths import ensure_default_local_model_paths


UNSAFE_SPECIAL_TOKENS_SET_RE = re.compile(r"self\.special_tokens_set(?!\s*=)")


TOKENIZER_HELPER = '''
    def _get_emu3_special_tokens_set(self):
        special_tokens_set = getattr(self, "special_tokens_set", None)
        if special_tokens_set is not None:
            return set(special_tokens_set)

        tokens = set()

        try:
            tokens.update(self.all_special_tokens)
        except Exception:
            pass

        try:
            special_tokens_map = getattr(self, "special_tokens_map", {}) or {}
            for value in special_tokens_map.values():
                if isinstance(value, str):
                    tokens.add(value)
                elif isinstance(value, (list, tuple, set)):
                    tokens.update(x for x in value if isinstance(x, str))
        except Exception:
            pass

        try:
            special_tokens_map_extended = getattr(self, "special_tokens_map_extended", {}) or {}
            for value in special_tokens_map_extended.values():
                if isinstance(value, str):
                    tokens.add(value)
                elif isinstance(value, (list, tuple, set)):
                    tokens.update(x for x in value if isinstance(x, str))
        except Exception:
            pass

        return tokens

'''


@dataclass
class TokenizerCompatReport:
    tokenizer_source_paths: list[str] = field(default_factory=list)
    patched_source_paths: list[str] = field(default_factory=list)
    patch_needed: bool = False
    patch_applied: bool = False
    cache_detected: bool = False
    cache_cleared: bool = False
    cache_removed_paths: list[str] = field(default_factory=list)
    cache_skipped: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_emu3_tokenizer_compat(
    model_config: ModelConfig | dict[str, Any],
    clear_cache: bool | None = None,
    verbose: bool = False,
) -> TokenizerCompatReport:
    """Patch local Emu3 tokenizer source and clear stale Transformers remote-code cache."""

    ensure_default_local_model_paths(model_config)
    report = TokenizerCompatReport()
    source_paths = find_tokenization_emu3_sources(model_config)
    cache_tokenizer_sources = find_emu3_transformers_cache_tokenizer_files()
    if not source_paths and cache_tokenizer_sources:
        try:
            materialized_source = materialize_cached_tokenizer_source(model_config, cache_tokenizer_sources[0], verbose)
        except Exception as exc:
            message = f"failed to materialize cached tokenizer source: {type(exc).__name__}: {exc}"
            report.warnings.append(message)
            _emit(message, verbose)
        else:
            if materialized_source is not None:
                source_paths = [materialized_source]
    report.tokenizer_source_paths = [str(path) for path in source_paths]
    configured_patch_source = _get_attr_or_key(model_config, "patch_tokenizer_source")
    patch_source = True if configured_patch_source is None else bool(configured_patch_source)

    for source_path in source_paths:
        try:
            result = patch_emu3_tokenizer_file(source_path) if patch_source else inspect_emu3_tokenizer_file(source_path)
        except Exception as exc:
            message = f"failed to patch tokenizer source {source_path}: {type(exc).__name__}: {exc}"
            report.warnings.append(message)
            _emit(message, verbose)
            continue
        report.patch_needed = report.patch_needed or result["patch_needed"]
        report.patch_applied = report.patch_applied or result["patch_applied"]
        if result["patch_applied"]:
            report.patched_source_paths.append(str(source_path))
        if verbose:
            _emit(
                "tokenizer source path: "
                f"{source_path}; patch_needed={result['patch_needed']}; patch_applied={result['patch_applied']}",
                verbose,
            )
        if patch_source and result["patch_needed"] and not result["patch_applied"]:
            raise RuntimeError(
                "Emu3 tokenizer compatibility patch was required but was not applied. "
                f"Tokenizer source: {source_path}. "
                "Set model.patch_tokenizer_source: true or patch tokenization_emu3.py manually."
            )

    if clear_cache is None:
        configured_clear_cache = _get_attr_or_key(model_config, "clear_transformers_remote_code_cache")
        clear_cache = True if configured_clear_cache is None else bool(configured_clear_cache)
    cache_paths = find_emu3_transformers_cache_dirs()
    report.cache_detected = bool(cache_paths)
    if clear_cache:
        for cache_path in cache_paths:
            try:
                shutil.rmtree(cache_path)
            except FileNotFoundError:
                continue
            except Exception as exc:
                message = f"failed to remove Transformers remote-code cache {cache_path}: {type(exc).__name__}: {exc}"
                report.warnings.append(message)
                _emit(message, verbose)
                continue
            report.cache_cleared = True
            report.cache_removed_paths.append(str(cache_path))
            _emit(f"removed Transformers remote-code cache: {cache_path}", verbose)
    else:
        report.cache_skipped = True
        if cache_paths:
            _emit("Transformers remote-code cache detected but cache clearing is disabled.", verbose)
    if not cache_paths and verbose:
        _emit("No Emu3.5 Transformers remote-code cache directory was detected.", verbose)

    if not source_paths:
        message = (
            "No local tokenization_emu3.py source was found under tokenizer_path, model_id_or_path, "
            "or emu_repo_path. AutoTokenizer may keep using stale cached remote code."
        )
        report.warnings.append(message)
        _emit(message, verbose)
    elif report.patch_needed and not report.patch_applied:
        message = "Emu3 tokenizer compatibility patch was needed but was not applied."
        if patch_source:
            raise RuntimeError(message)
        report.warnings.append(message)
        _emit(message, verbose)
    return report


def find_tokenization_emu3_sources(model_config: ModelConfig | dict[str, Any]) -> list[Path]:
    roots: list[Path] = []
    for key in ("tokenizer_path", "model_id_or_path"):
        value = _get_attr_or_key(model_config, key)
        if value:
            roots.append(Path(str(value)).expanduser().resolve())
    emu_repo_path = _get_attr_or_key(model_config, "emu_repo_path")
    resolved_emu_repo_path = resolve_project_path(emu_repo_path) if emu_repo_path else None
    if resolved_emu_repo_path:
        roots.append(resolved_emu_repo_path)

    results: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        candidates = [root / "tokenization_emu3.py"]
        try:
            candidates.extend(root.rglob("tokenization_emu3.py"))
        except OSError:
            pass
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.exists() and resolved not in seen:
                seen.add(resolved)
                results.append(resolved)
    return results


def patch_emu3_tokenizer_file(path: str | Path) -> dict[str, bool]:
    source_path = Path(path)
    text = source_path.read_text(encoding="utf-8")
    patch_needed = bool(UNSAFE_SPECIAL_TOKENS_SET_RE.search(text))
    updated = UNSAFE_SPECIAL_TOKENS_SET_RE.sub(
        "self._get_emu3_special_tokens_set()",
        text,
    )
    if updated != text and "_get_emu3_special_tokens_set" not in text:
        updated = _insert_tokenizer_helper(updated)
    patch_applied = updated != text
    if patch_applied:
        backup_path = source_path.with_name(source_path.name + ".vision_cad_emu35_backup")
        if not backup_path.exists():
            shutil.copy2(source_path, backup_path)
        source_path.write_text(updated, encoding="utf-8")
    return {"patch_needed": patch_needed, "patch_applied": patch_applied}


def inspect_emu3_tokenizer_file(path: str | Path) -> dict[str, bool]:
    text = Path(path).read_text(encoding="utf-8")
    return {"patch_needed": bool(UNSAFE_SPECIAL_TOKENS_SET_RE.search(text)), "patch_applied": False}


def find_emu3_transformers_cache_dirs() -> list[Path]:
    base = _transformers_modules_cache_dir()
    if not base.exists():
        return []

    candidates: set[Path] = set()
    for relative in (
        Path("Emu3.5"),
        Path("BAAI") / "Emu3.5",
        Path("BAAI--Emu3.5"),
    ):
        path = base / relative
        if path.exists() and path.is_dir():
            candidates.add(path.resolve())

    try:
        for tokenization_file in base.rglob("tokenization_emu3.py"):
            parent = tokenization_file.parent.resolve()
            if _looks_like_emu3_cache(parent, tokenization_file):
                candidates.add(parent)
    except OSError:
        pass
    return sorted(candidates, key=lambda path: str(path))


def find_emu3_transformers_cache_tokenizer_files() -> list[Path]:
    base = _transformers_modules_cache_dir()
    if not base.exists():
        return []
    results: list[Path] = []
    try:
        for tokenization_file in base.rglob("tokenization_emu3.py"):
            if _looks_like_emu3_cache(tokenization_file.parent.resolve(), tokenization_file):
                results.append(tokenization_file.resolve())
    except OSError:
        pass
    return sorted(results, key=lambda path: str(path))


def materialize_cached_tokenizer_source(
    model_config: ModelConfig | dict[str, Any],
    cached_source: Path,
    verbose: bool,
) -> Path | None:
    target_root = _first_existing_path(
        _get_attr_or_key(model_config, "tokenizer_path"),
        _get_attr_or_key(model_config, "model_id_or_path"),
    )
    if target_root is None:
        _emit(
            f"cached Emu3 tokenizer source was found at {cached_source}, but tokenizer/model path does not exist.",
            verbose,
        )
        return None
    target = target_root / "tokenization_emu3.py"
    if not target.exists():
        shutil.copy2(cached_source, target)
        _emit(f"copied cached Emu3 tokenizer source to local path: {target}", verbose)
    return target.resolve()


def is_special_tokens_set_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "special_tokens_set" in text


def _insert_tokenizer_helper(text: str) -> str:
    match = re.search(r"^class\s+Emu3Tokenizer\b.*:\n", text, flags=re.MULTILINE)
    if not match:
        raise ValueError("Could not find `class Emu3Tokenizer` in tokenization_emu3.py.")
    return text[: match.end()] + TOKENIZER_HELPER + text[match.end() :]


def _looks_like_emu3_cache(parent: Path, tokenization_file: Path) -> bool:
    path_text = str(parent).lower()
    if "emu3" in path_text:
        return True
    try:
        sample = tokenization_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "class Emu3Tokenizer" in sample


def _transformers_modules_cache_dir() -> Path:
    env_path = os.environ.get("HF_MODULES_CACHE")
    if env_path:
        path = Path(env_path).expanduser()
        if path.name == "transformers_modules":
            return path
        return path / "transformers_modules"
    return Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"


def _first_existing_path(*values: Any) -> Path | None:
    for value in values:
        if not value:
            continue
        path = Path(str(value)).expanduser().resolve()
        if path.exists() and path.is_dir():
            return path
    return None


def _emit(message: str, verbose: bool) -> None:
    if verbose:
        print(message)
    else:
        warnings.warn(message, RuntimeWarning)


def _get_attr_or_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    if is_dataclass(obj):
        return getattr(obj, key, None)
    return getattr(obj, key, None)
