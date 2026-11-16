"""FP32 / FP16 / INT8 variants of an ONNX detector and the comparison table between them."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

from duelscope.adapters._utils import letterbox
from duelscope.adapters.base import AdapterError, load_rgb

VARIANTS = ("fp32", "fp16", "int8")


def _need_onnx():
    try:
        import onnx
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        raise AdapterError(
            "Quantisation needs onnx and onnxruntime. Run: pip install 'duelscope[onnx]'"
        ) from exc
    return onnx


def make_fp16(src: Path, dst: Path) -> Path:
    """Convert weights to FP16 (inputs/outputs stay FP32) with onnxruntime's converter."""
    onnx = _need_onnx()
    from onnxruntime.transformers.float16 import convert_float_to_float16

    onnx.save(convert_float_to_float16(onnx.load(str(src)), keep_io_types=True), str(dst))
    return dst


def head_nodes(onnx_path: Path) -> list[str]:
    """Names of nodes in the last top-level module (the detection head of Ultralytics exports).

    Quantising the head wrecks accuracy because boxes (pixels) and scores (0-1) share one
    tensor, so INT8 keeps it in floating point. Returns [] if the graph uses other naming.
    """
    onnx = _need_onnx()
    nodes = [n.name for n in onnx.load(str(onnx_path)).graph.node]
    idx = [int(m.group(1)) for n in nodes if (m := re.match(r"^/model\.(\d+)/", n))]
    if not idx:
        return []
    prefix = f"/model.{max(idx)}/"
    return [n for n in nodes if n.startswith(prefix)]


def make_int8(src: Path, dst: Path, calib_images: list[Path], imgsz: int = 640) -> Path:
    """Static INT8 (QDQ, per-channel) quantisation calibrated on the given images.

    The detection head stays in floating point (see `head_nodes`).
    """
    _need_onnx()
    if not calib_images:
        raise AdapterError("INT8 quantisation needs at least one calibration image.")
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    size = inp.shape[2] if isinstance(inp.shape[2], int) else imgsz
    del sess

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.it = iter(calib_images)

        def get_next(self):  # noqa: ANN201
            p = next(self.it, None)
            if p is None:
                return None
            lb, _, _ = letterbox(load_rgb(p), size)
            return {inp.name: lb.transpose(2, 0, 1)[None].astype(np.float32) / 255}

    quantize_static(
        str(src),
        str(dst),
        Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        nodes_to_exclude=head_nodes(src),
    )
    return dst


def make_variants(
    onnx_path: Path, out_dir: Path, calib_images: list[Path], imgsz: int = 640
) -> dict[str, Path]:
    """Write <stem>_fp32/_fp16/_int8.onnx into out_dir and return label -> path."""
    if not onnx_path.is_file():
        raise AdapterError(f"ONNX file '{onnx_path}' not found. Check the path.")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = onnx_path.stem
    fp32 = out_dir / f"{stem}_fp32.onnx"
    shutil.copyfile(onnx_path, fp32)
    return {
        "fp32": fp32,
        "fp16": make_fp16(onnx_path, out_dir / f"{stem}_fp16.onnx"),
        "int8": make_int8(onnx_path, out_dir / f"{stem}_int8.onnx", calib_images, imgsz),
    }


def quantisation_table(results: dict) -> list[dict]:
    """Rows comparing each model with the *_fp32 baseline (accuracy, speed, size)."""
    models = {m["name"]: m for m in results["models"]}
    base = next((m for n, m in models.items() if n.endswith("_fp32")), None)
    if base is None:
        raise ValueError("No model named '*_fp32' to use as the baseline.")
    rows = []
    for n, m in models.items():
        label = n.rsplit("_", 1)[-1]
        a, b = m["accuracy"], base["accuracy"]
        lat = m["speed"]["latency_ms"]["p50"] if m.get("speed") else None
        blat = base["speed"]["latency_ms"]["p50"] if base.get("speed") else None
        size, bsize = m["metadata"].get("weights_mb"), base["metadata"].get("weights_mb")
        rows.append(
            {
                "name": n,
                "precision": label,
                "map5095": a["map5095"],
                "map50": a["map50"],
                "d_map5095": (a["map5095"] - b["map5095"]) * 100,
                "d_map50": (a["map50"] - b["map50"]) * 100,
                "p50_ms": lat,
                "speedup": (blat / lat) if lat and blat else None,
                "size_mb": size,
                "size_ratio": (bsize / size) if size and bsize else None,
            }
        )
    return rows


def quant_insights(rows: list[dict]) -> list[str]:
    """Plain-language sentences about each non-baseline precision."""
    out = []
    for r in rows:
        if r["precision"] == "fp32":
            continue
        parts = [f"{r['precision'].upper()} changes mAP50-95 by {r['d_map5095']:+.1f}"]
        if r["size_ratio"]:
            parts.append(f"is {r['size_ratio']:.1f}x smaller")
        if r["speedup"]:
            parts.append(
                f"{r['speedup']:.2f}x the speed"
                if r["speedup"] >= 1
                else f"is {1 / r['speedup']:.2f}x slower"
            )
        out.append(" and ".join(parts) + " versus FP32 on this hardware.")
    return out
