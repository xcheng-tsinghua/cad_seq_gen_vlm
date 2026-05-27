from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import load_config, resolve_project_path


REQUIRED_IMPORTS = [
    ("src.utils.model_utils", "build_emu3p5"),
    ("src.utils.input_utils", "build_image"),
    ("src.utils.generation_utils", "generate"),
    ("src.utils.generation_utils", "multimodal_decode"),
]


def _read_emu_repo_path(config_path: Path) -> str | None:
    try:
        return load_config(config_path).model.emu_repo_path
    except ImportError as exc:
        if "PyYAML" not in str(exc):
            raise
        print("WARNING: PyYAML is not installed; using a minimal YAML reader for model.emu_repo_path only.")
        return _read_emu_repo_path_minimal(config_path)


def _read_emu_repo_path_minimal(config_path: Path) -> str | None:
    in_model = False
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if re.match(r"^\S", line):
            in_model = line.strip() == "model:"
            continue
        if not in_model:
            continue
        stripped = line.strip()
        if stripped.startswith("emu_repo_path:"):
            value = stripped.split(":", 1)[1].strip()
            if value in {"", "null", "None", "~"}:
                return None
            return value.strip("\"'")
    return None


def _list_repo_files(path: Path, limit: int = 80) -> list[str]:
    if not path.exists():
        return []
    items: list[str] = []
    try:
        for child in sorted(path.iterdir(), key=lambda item: item.name.lower()):
            suffix = "/" if child.is_dir() else ""
            items.append(f"{child.name}{suffix}")
            if len(items) >= limit:
                break
    except OSError as exc:
        return [f"<failed to list files: {exc}>"]
    return items


def _print_failure_help(repo_path: Path | None, configured_value: str | None) -> None:
    print("\nFAILED: official Emu3.5 runtime utilities are not importable.")
    print(f"configured emu_repo_path: {configured_value if configured_value else 'null'}")
    if repo_path and configured_value != str(repo_path):
        print(f"resolved emu_repo_path: {repo_path}")
    if repo_path:
        print(f"path exists: {repo_path.exists()}")
        print("files under emu_repo_path:")
        files = _list_repo_files(repo_path)
        if files:
            for item in files:
                print(f"  - {item}")
        else:
            print("  <none>")
    print("\nFix:")
    print("  1. Clone or upload the official Emu3.5 repo, for example:")
    print("     mkdir -p third_party")
    print("     git clone https://github.com/baaivision/Emu3.5.git third_party/Emu3.5")
    print("  2. Update configs/rag.yaml:")
    print('     model:')
    print('       emu_repo_path: "third_party/Emu3.5"')
    print("  3. Re-run:")
    print("     python scripts/check_emu35_imports.py --config configs/rag.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check official Emu3.5 runtime source imports.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "rag.yaml"))
    args = parser.parse_args()

    config_path = Path(args.config)
    emu_repo_path = _read_emu_repo_path(config_path)
    repo_path = resolve_project_path(emu_repo_path) if emu_repo_path else None

    print(f"config: {config_path.resolve()}")
    print(f"emu_repo_path: {emu_repo_path if emu_repo_path else 'null'}")
    if repo_path and emu_repo_path != str(repo_path):
        print(f"resolved emu_repo_path: {repo_path}")
    if repo_path:
        print(f"emu_repo_path exists: {repo_path.exists()}")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
            print("prepended emu_repo_path to sys.path: OK")
    else:
        print("prepended emu_repo_path to sys.path: SKIPPED, path is null")

    all_ok = True
    for module_name, attr_name in REQUIRED_IMPORTS:
        label = f"{module_name}.{attr_name}"
        try:
            module = importlib.import_module(module_name)
            getattr(module, attr_name)
        except Exception as exc:
            all_ok = False
            print(f"FAILED {label}: {type(exc).__name__}: {exc}")
        else:
            print(f"OK {label}")

    if all_ok:
        print("\nOK: official Emu3.5 runtime utilities are importable.")
        return 0

    _print_failure_help(repo_path, emu_repo_path)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
