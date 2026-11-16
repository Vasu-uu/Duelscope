"""Adapter registry, model-spec parsing and third-party plugin discovery."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points

from duelscope.adapters.base import AdapterError, Detector, DetectorConfig

Factory = Callable[[str, DetectorConfig], Detector]

_FACTORIES: dict[str, Factory] = {}
_EXTENSIONS: dict[str, str] = {}
_plugins_loaded = False


def register(scheme: str, factory: Factory, extensions: tuple[str, ...] = ()) -> None:
    """Register an adapter under `scheme:` (and optionally bare file extensions like '.onnx')."""
    _FACTORIES[scheme] = factory
    for ext in extensions:
        _EXTENSIONS[ext.lower()] = scheme


def _ultralytics(rest: str, cfg: DetectorConfig) -> Detector:
    from duelscope.adapters.ultralytics_ import UltralyticsDetector

    return UltralyticsDetector(rest, cfg)


def _onnx(rest: str, cfg: DetectorConfig) -> Detector:
    from duelscope.adapters.onnx_ import OnnxDetector

    return OnnxDetector(rest, cfg)


def _torchvision(rest: str, cfg: DetectorConfig) -> Detector:
    from duelscope.adapters.torchvision_ import TorchvisionDetector

    return TorchvisionDetector(rest, cfg)


def _custom(rest: str, cfg: DetectorConfig) -> Detector:
    from duelscope.adapters.custom import load_custom

    return load_custom(rest, cfg)


def _load_plugins() -> None:
    """Discover third-party adapters in the 'duelscope.adapters' entry-point group."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for ep in entry_points(group="duelscope.adapters"):
        try:
            obj = ep.load()
        except Exception as exc:
            print(f"[duelscope] WARNING: plugin adapter '{ep.name}' failed to load: {exc}")
            continue
        factory = obj if not (isinstance(obj, type) and issubclass(obj, Detector)) else None
        if factory is None:
            cls = obj
            factory = lambda rest, cfg, cls=cls: cls(rest, cfg)  # noqa: E731
        register(ep.name, factory)


register("ultralytics", _ultralytics, (".pt",))
register("onnx", _onnx, (".onnx",))
register("torchvision", _torchvision)
register("custom", _custom)
_BUILTIN = set(_FACTORIES)


def available_adapters() -> dict[str, str]:
    """Map of scheme -> origin ('built-in' or 'plugin') for list-adapters."""
    _load_plugins()
    return {k: ("built-in" if k in _BUILTIN else "plugin") for k in sorted(_FACTORIES)}


def parse_spec(spec: str) -> tuple[str, str]:
    """Split a model spec into (scheme, rest). Bare filenames are routed by extension."""
    _load_plugins()
    head, sep, rest = spec.partition(":")
    if sep and len(head) > 1 and head in _FACTORIES:  # len>1 keeps 'C:\\x.onnx' a path
        return head, rest
    for ext, scheme in _EXTENSIONS.items():
        if spec.lower().endswith(ext):
            return scheme, spec
    if sep and len(head) > 1 and head.isidentifier() and "/" not in head and "\\" not in head:
        raise AdapterError(
            f"Unknown adapter '{head}' in model spec '{spec}'. "
            f"Available: {', '.join(sorted(_FACTORIES))}. Run 'duelscope list-adapters'."
        )
    if "." not in spec.rsplit("/", 1)[-1]:  # e.g. 'yolov8n' -> let Ultralytics resolve it
        return "ultralytics", spec
    raise AdapterError(
        f"Cannot tell which adapter handles '{spec}'. Use a .pt/.onnx file or an explicit "
        "prefix like 'torchvision:fasterrcnn_resnet50_fpn' or 'custom:file.py:Class'."
    )


def create_detector(spec: str, cfg: DetectorConfig | None = None) -> Detector:
    """Build a (not yet loaded) detector from a model spec string."""
    scheme, rest = parse_spec(spec)
    return _FACTORIES[scheme](rest, cfg or DetectorConfig())
