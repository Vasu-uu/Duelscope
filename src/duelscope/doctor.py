"""Environment checks behind `duelscope doctor`."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

from duelscope.bench.hardware import _nvidia_smi

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass(frozen=True)
class Check:
    """One diagnostic line: status is ok/warn/fail; fix is a command or hint (may be empty)."""

    name: str
    status: str
    detail: str
    fix: str = ""


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def run_checks(cache_dir: Path = Path(".duelscope_cache")) -> list[Check]:
    """Inspect Python, CUDA, optional extras and the cache directory."""
    checks: list[Check] = []
    v = sys.version_info
    checks.append(
        Check(
            "Python",
            OK if v >= (3, 10) else FAIL,
            f"{v.major}.{v.minor}.{v.micro}",
            "" if v >= (3, 10) else "Duelscope needs Python 3.10 or newer.",
        )
    )
    gpu = _nvidia_smi()
    torch_ok = _has("torch")
    cuda = False
    if torch_ok:
        import torch

        cuda = bool(torch.cuda.is_available())
        checks.append(
            Check("PyTorch", OK, f"{torch.__version__} (CUDA build: {torch.version.cuda})")
        )
    else:
        checks.append(
            Check(
                "PyTorch",
                WARN,
                "not installed",
                "pip install 'duelscope[torchvision]'  (needed for Ultralytics/torchvision)",
            )
        )
    if gpu and cuda:
        checks.append(Check("GPU", OK, f"{gpu['name']} ({gpu['vram_mb']} MB), CUDA usable"))
    elif gpu and torch_ok:
        checks.append(
            Check(
                "GPU",
                WARN,
                f"{gpu['name']} found but PyTorch cannot use it (CPU-only build)",
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
                "  (pick the index for your CUDA version), then re-run 'duelscope doctor'",
            )
        )
    elif gpu:
        checks.append(Check("GPU", OK, f"{gpu['name']} ({gpu['vram_mb']} MB) detected"))
    else:
        checks.append(
            Check(
                "GPU",
                WARN,
                "none detected; benchmarks will run on CPU",
                "Use --device cpu to silence this, or install the NVIDIA driver.",
            )
        )
    for name, module, extra in (
        ("Ultralytics", "ultralytics", "ultralytics"),
        ("ONNX Runtime", "onnxruntime", "onnx"),
        ("torchvision", "torchvision", "torchvision"),
    ):
        ok = _has(module)
        checks.append(
            Check(
                name,
                OK if ok else WARN,
                "installed" if ok else "not installed",
                "" if ok else f"pip install 'duelscope[{extra}]'",
            )
        )
    if _has("onnxruntime") and gpu:
        import onnxruntime as ort

        has_cuda_ep = "CUDAExecutionProvider" in ort.get_available_providers()
        checks.append(
            Check(
                "ONNX CUDA provider",
                OK if has_cuda_ep else WARN,
                "available" if has_cuda_ep else "missing; ONNX models will run on CPU",
                "" if has_cuda_ep else "pip uninstall onnxruntime && pip install onnxruntime-gpu",
            )
        )
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        probe = cache_dir / ".write_test"
        probe.write_text("x")
        probe.unlink()
        checks.append(Check("Cache directory", OK, f"{cache_dir.resolve()} is writable"))
    except OSError as exc:
        checks.append(Check("Cache directory", FAIL, str(exc), "Choose another with --cache-dir."))
    return checks
