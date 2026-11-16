from pathlib import Path

import numpy as np
import pytest

from duelscope.adapters.base import AdapterError
from duelscope.quant import head_nodes, make_variants, quant_insights, quantisation_table
from duelscope.synthetic import make_yolo_dataset
from duelscope.types import Detections
from duelscope.video import video_benchmark
from tests.helpers import MockDetector


def _video(path: Path, frames: int = 12) -> Path:
    cv2 = pytest.importorskip("cv2")
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for i in range(frames):
        img = np.full((120, 160, 3), 90, dtype=np.uint8)
        img[20:60, 10 + 5 * i : 50 + 5 * i] = (0, 200, 0)
        w.write(img)
    w.release()
    return path


def _mock(name: str, per_frame: list[int]):
    state = {"i": 0}

    def fn(_: Path) -> Detections:
        n = per_frame[state["i"] % len(per_frame)] if state["i"] >= 0 else 0
        state["i"] += 1
        return Detections.make([[5, 5, 30, 30]] * n, [0.9] * n, [0] * n)

    return MockDetector(fn, name, ["thing"])


def test_video_benchmark_metrics_and_output(tmp_path: Path) -> None:
    video = _video(tmp_path / "v.avi")
    steady, flicky = _mock("steady", [1]), _mock("flicky", [0, 2])
    out = tmp_path / "sbs.avi"
    res = video_benchmark([steady, flicky], video, out, conf=0.25, max_frames=8)
    assert res["frames_processed"] == 8 and out.is_file() and out.stat().st_size > 0
    by = {m["name"]: m for m in res["models"]}
    assert by["steady"]["flicker"] == 0.0
    assert by["flicky"]["flicker"] > 1.0  # alternates 0 and 2 detections
    assert by["steady"]["mean_detections"] == pytest.approx(1.0)
    assert by["steady"]["fps"] > 0 and by["steady"]["latency_ms"]["p50"] >= 0


def test_video_stride_and_errors(tmp_path: Path) -> None:
    video = _video(tmp_path / "v.avi")
    res = video_benchmark([_mock("m", [1])], video, None, stride=3)
    assert res["frames_processed"] == 4 and res["side_by_side"] is None
    with pytest.raises(AdapterError, match="not found"):
        video_benchmark([_mock("m", [1])], tmp_path / "none.avi")
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"not a video")
    with pytest.raises(AdapterError, match="cannot open|No frames"):
        video_benchmark([_mock("m", [1])], bad)


def _tiny_onnx(path: Path, prefix: str = "") -> Path:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    w = helper.make_tensor(
        "w",
        TensorProto.FLOAT,
        [4, 3, 3, 3],
        np.random.default_rng(0).normal(size=108).astype(np.float32).tolist(),
    )
    node = helper.make_node(
        "Conv", ["images", "w"], ["out"], pads=[1, 1, 1, 1], name=prefix + "conv"
    )
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 64, 64])],
        initializer=[w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def test_make_variants_produce_runnable_models(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    src = _tiny_onnx(tmp_path / "tiny.onnx")
    root = make_yolo_dataset(tmp_path / "d", n_images=4).parent
    calib = sorted((root / "images" / "val").glob("*.png"))
    out = make_variants(src, tmp_path / "v", calib, imgsz=64)
    assert set(out) == {"fp32", "fp16", "int8"} and all(p.is_file() for p in out.values())
    import onnxruntime as ort

    from duelscope.adapters._utils import letterbox
    from duelscope.adapters.base import load_rgb

    lb, _, _ = letterbox(load_rgb(calib[0]), 64)
    x = lb.transpose(2, 0, 1)[None].astype(np.float32) / 255
    ref = ort.InferenceSession(str(out["fp32"])).run(None, {"images": x})[0]
    for k, tol in (("fp16", 0.01), ("int8", 0.15)):
        got = ort.InferenceSession(str(out[k])).run(None, {"images": x})[0]
        assert got.shape == ref.shape
        assert np.abs(got - ref).mean() < tol * np.abs(ref).mean() + 1e-3
    with pytest.raises(AdapterError, match="not found"):
        make_variants(tmp_path / "missing.onnx", tmp_path / "x", calib)


def test_head_nodes_picks_last_top_level_module(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    assert head_nodes(_tiny_onnx(tmp_path / "a.onnx", "/model.3/")) == ["/model.3/conv"]
    assert head_nodes(_tiny_onnx(tmp_path / "b.onnx", "other_")) == []


def _model(name: str, ap: float, lat: float, size: float) -> dict:
    return {
        "name": name,
        "metadata": {"weights_mb": size},
        "accuracy": {"map5095": ap, "map50": ap + 0.2},
        "speed": {"latency_ms": {"p50": lat}},
    }


def test_quantisation_table_and_insights() -> None:
    res = {
        "models": [
            _model("m_fp32", 0.70, 100, 12),
            _model("m_fp16", 0.70, 50, 6),
            _model("m_int8", 0.61, 125, 3),
        ]
    }
    rows = {r["precision"]: r for r in quantisation_table(res)}
    assert rows["fp16"]["speedup"] == pytest.approx(2.0) and rows["fp16"]["size_ratio"] == 2.0
    assert rows["int8"]["d_map5095"] == pytest.approx(-9.0)
    text = " ".join(quant_insights(list(rows.values())))
    assert "INT8 changes mAP50-95 by -9.0 and is 4.0x smaller and is 1.25x slower" in text
    with pytest.raises(ValueError, match="_fp32"):
        quantisation_table({"models": [_model("x", 0.5, 1, 1)]})
