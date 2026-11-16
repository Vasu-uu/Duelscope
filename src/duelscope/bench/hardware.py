"""Collect hardware and software details so every report is reproducible."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata

import psutil

_LIBS = ("duelscope", "numpy", "torch", "torchvision", "ultralytics", "onnxruntime", "Pillow")


def _cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
            ).strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _nvidia_smi() -> dict | None:
    """GPU name, VRAM, driver and CUDA version from nvidia-smi, if present."""
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0]
        name, mem, driver = (x.strip() for x in q.split(","))
        full = subprocess.check_output(["nvidia-smi"], text=True, timeout=10)
        cuda = next(
            (
                ln.split("CUDA Version:")[1].split()[0]
                for ln in full.splitlines()
                if "CUDA Version" in ln
            ),
            None,
        )
        return {
            "name": name,
            "vram_mb": int(float(mem.split()[0])),
            "driver": driver,
            "cuda_driver_version": cuda,
        }
    except Exception:
        return None


def hardware_info() -> dict:
    """Return CPU, RAM, GPU, OS, Python and library versions. Never raises."""
    libs: dict[str, str] = {}
    for lib in _LIBS:
        try:
            libs[lib.lower()] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            continue
    gpu = _nvidia_smi()
    torch_cuda = False
    cuda_build = None
    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
        cuda_build = torch.version.cuda
        if torch_cuda and gpu is None:
            p = torch.cuda.get_device_properties(0)
            gpu = {"name": p.name, "vram_mb": p.total_memory // 2**20}
    except ImportError:
        pass
    return {
        "cpu": _cpu_name(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": gpu,
        "torch_cuda_available": torch_cuda,
        "torch_cuda_build": cuda_build,
        "os": platform.platform(),
        "python": platform.python_version(),
        "libraries": libs,
    }
