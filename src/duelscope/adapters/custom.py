"""Load a user-written Detector subclass from a Python file."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

from duelscope.adapters.base import AdapterError, Detector, DetectorConfig


def load_custom(spec: str, cfg: DetectorConfig) -> Detector:
    """spec is 'path/to/file.py:ClassName' (class optional if the file has exactly one)."""
    path_str, _, cls_name = spec.rpartition(":")
    if not path_str or "/" in cls_name or "\\" in cls_name or cls_name.endswith(".py"):
        path_str, cls_name = spec, ""
    path = Path(path_str)
    if not path.is_file():
        raise AdapterError(
            f"Custom adapter file '{path}' not found. Use custom:path/file.py:Class."
        )
    mod_spec = importlib.util.spec_from_file_location(f"duelscope_custom_{path.stem}", path)
    module = importlib.util.module_from_spec(mod_spec)
    try:
        mod_spec.loader.exec_module(module)
    except Exception as exc:
        raise AdapterError(f"Importing '{path}' failed: {exc}") from exc
    classes = {
        n: c
        for n, c in inspect.getmembers(module, inspect.isclass)
        if issubclass(c, Detector) and c is not Detector and c.__module__ == module.__name__
    }
    if not cls_name:
        if len(classes) != 1:
            raise AdapterError(
                f"'{path}' defines {len(classes)} Detector subclasses; name one: "
                f"custom:{path}:ClassName (found: {', '.join(classes) or 'none'})."
            )
        cls_name = next(iter(classes))
    if cls_name not in classes:
        raise AdapterError(
            f"Class '{cls_name}' is not a Detector subclass in '{path}' "
            f"(found: {', '.join(classes) or 'none'})."
        )
    try:
        return classes[cls_name](cfg)
    except TypeError:
        return classes[cls_name]()
