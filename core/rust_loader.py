from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


def load_neural_trader_rust() -> tuple[ModuleType | None, str]:
    """Load the optional Rust hot-path extension from installed or local builds."""
    try:
        return importlib.import_module("neural_trader_rust"), "python_path"
    except Exception as first_exc:
        first_error = str(first_exc)

    repo_root = Path(__file__).resolve().parents[1]
    env_dir = os.getenv("NT_RUST_MODULE_DIR", "").strip()
    candidates: list[Path] = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            repo_root / "rust" / "target" / "release",
            repo_root / "rust" / "target" / "debug",
            repo_root / "rust" / "py-bindings" / "target" / "release",
            repo_root / "rust" / "py-bindings" / "target" / "debug",
        ]
    )

    errors: list[str] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if callable(add_dll_directory):
            try:
                add_dll_directory(path_str)
            except Exception:
                pass
        try:
            return importlib.import_module("neural_trader_rust"), path_str
        except Exception as exc:
            errors.append(f"{path_str}: {exc}")

    reason = "; ".join(errors[-3:]) if errors else first_error
    return None, reason or "module_not_found"
