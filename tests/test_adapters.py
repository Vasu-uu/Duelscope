from pathlib import Path

import numpy as np
import pytest

from duelscope.adapters import (
    AdapterError,
    ClassMap,
    Detector,
    DetectorConfig,
    available_adapters,
    create_detector,
    parse_spec,
    register,
)
from duelscope.adapters._utils import batched_nms, box_iou, decode_yolo, letterbox, nms
from duelscope.adapters.custom import load_custom
from duelscope.adapters.onnx_ import OnnxDetector
from duelscope.types import Detections
from tests.helpers import MockDetector

ROOT = Path(__file__).resolve().parents[1]


def test_parse_spec_routes() -> None:
    assert parse_spec("yolov8n.pt") == ("ultralytics", "yolov8n.pt")
    assert parse_spec("models/m.ONNX") == ("onnx", "models/m.ONNX")
    assert parse_spec("torchvision:fasterrcnn_resnet50_fpn") == (
        "torchvision",
        "fasterrcnn_resnet50_fpn",
    )
    assert parse_spec("custom:a/b.py:Cls") == ("custom", "a/b.py:Cls")
    assert parse_spec(r"C:\models\m.onnx")[0] == "onnx"
    assert parse_spec("yolov8n")[0] == "ultralytics"


def test_parse_spec_errors() -> None:
    with pytest.raises(AdapterError, match="Unknown adapter 'nope'"):
        parse_spec("nope:thing")
    with pytest.raises(AdapterError, match="Cannot tell"):
        parse_spec("weights.xyz")


def test_registry_plugin_and_builtin_listing() -> None:
    register("mocktest", lambda rest, cfg: MockDetector(lambda p: Detections.empty(), rest))
    assert create_detector("mocktest:foo").name == "foo"
    ad = available_adapters()
    assert ad["ultralytics"] == "built-in" and ad["mocktest"] == "plugin"


def test_classmap_by_name_and_manual() -> None:
    cm = ClassMap.build(
        ["Person", "traffic_light", "car"],
        ["person", "Traffic Light", "vehicle"],
        manual={"car": "vehicle"},
    )
    assert cm.table.tolist() == [0, 1, 2]
    d = Detections.make([[0, 0, 5, 5]] * 2, [0.9, 0.8], [2, 7])
    out = cm.apply(d)
    assert out.classes.tolist() == [2] and len(out) == 1


def test_classmap_warns_and_fails_loudly() -> None:
    with pytest.warns(UserWarning, match="cannot predict"):
        cm = ClassMap.build(["a", "b", "zzz"], ["a", "b", "c"])
    assert cm.unmapped == ["zzz"] and cm.missing == ["c"]
    with pytest.raises(AdapterError, match="manual map"):
        ClassMap.build(["x"], ["y"])
    assert ClassMap.build(None, ["a", "b"]).table.tolist() == [0, 1]


def test_nms_and_iou() -> None:
    b = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    s = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms(b, s, 0.5).tolist() == [0, 2]
    assert batched_nms(b, s, np.array([0, 1, 0]), 0.5).tolist() == [0, 1, 2]
    assert box_iou(b[:1], b[:1])[0, 0] == pytest.approx(1.0)


def test_letterbox_geometry() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out, r, (px, py) = letterbox(img, 64)
    assert out.shape == (64, 64, 3) and r == pytest.approx(0.32) and (px, py) == (0.0, 16.0)


def test_decode_yolov8_and_end2end() -> None:
    nc, n = 3, 4
    out = np.zeros((1, 4 + nc, n), dtype=np.float32)
    out[0, :4, 0] = [50, 50, 20, 20]
    out[0, 4 + 1, 0] = 0.9
    out[0, :4, 1] = [51, 50, 20, 20]  # duplicate of the first, same class -> suppressed
    out[0, 4 + 1, 1] = 0.6
    out[0, :4, 2] = [200, 200, 10, 10]
    out[0, 4 + 2, 2] = 0.5
    b, s, c = decode_yolo(out, "yolov8", 0.25, 0.5, 100)
    assert c.tolist() == [1, 2] and s[0] == pytest.approx(0.9)
    np.testing.assert_allclose(b[0], [40, 40, 60, 60])
    e = np.array([[[1, 2, 3, 4, 0.9, 7], [1, 2, 3, 4, 0.1, 3]]], dtype=np.float32)
    b2, s2, c2 = decode_yolo(e, "end2end", 0.25, 0.5, 100)
    assert c2.tolist() == [7] and len(b2) == 1
    with pytest.raises(ValueError):
        decode_yolo(out, "bogus", 0.1, 0.5, 10)


def test_onnx_missing_file_message(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="not found"):
        OnnxDetector(str(tmp_path / "x.onnx")).load()


def test_custom_adapter_loading(tmp_path: Path) -> None:
    f = tmp_path / "mine.py"
    f.write_text(
        "from duelscope.adapters import Detector\n"
        "from duelscope.types import Detections\n"
        "class Mine(Detector):\n"
        "    def load(self): pass\n"
        "    def predict(self, images): return [Detections.empty() for _ in images]\n"
    )
    d = load_custom(f"{f}:Mine", DetectorConfig())
    assert isinstance(d, Detector) and len(d.predict([Path("x")])[0]) == 0
    assert load_custom(str(f), DetectorConfig()).name == "Mine"
    with pytest.raises(AdapterError, match="not a Detector subclass"):
        load_custom(f"{f}:Nope", DetectorConfig())
    with pytest.raises(AdapterError, match="not found"):
        load_custom("missing.py:X", DetectorConfig())


def test_default_warmup_and_fingerprint(tmp_path: Path) -> None:
    d = MockDetector(lambda p: Detections.empty(), "m")
    d.warmup(1)
    assert d.calls == 1 and d.metadata["framework"] == "mock"
    w = tmp_path / "w.bin"
    w.write_bytes(b"abc")
    d.weights_path = w
    import hashlib

    assert d.fingerprint() == hashlib.sha256(b"abc").hexdigest()[:16]
    assert d.metadata["weights_mb"] == 0.0


def test_torchvision_cpu_roundtrip() -> None:
    pytest.importorskip("torchvision")
    from duelscope.adapters.torchvision_ import TorchvisionDetector

    det = TorchvisionDetector(
        "fasterrcnn_mobilenet_v3_large_fpn", DetectorConfig(device="cpu"), weights=None
    )
    out = det.predict([np.full((96, 128, 3), 120, dtype=np.uint8)])
    assert len(out) == 1 and det.class_names[1] == "person" and det.num_params > 1_000_000
    assert det.last_timing["inference"] > 0
    with pytest.raises(AdapterError, match="Unknown torchvision"):
        TorchvisionDetector("resnet18")


@pytest.mark.ultralytics
def test_real_ultralytics_predicts() -> None:
    pytest.importorskip("ultralytics")
    w = ROOT / "weights" / "yolov8n.pt"
    if not w.is_file():
        pytest.skip("weights/yolov8n.pt not available (download it to run this test)")
    from duelscope.adapters.ultralytics_ import UltralyticsDetector

    det = UltralyticsDetector(str(w), DetectorConfig(device="cpu"))
    img = np.full((240, 320, 3), 128, dtype=np.uint8)
    out = det.predict([img])
    assert len(out) == 1 and det.class_names[0] == "person" and det.num_params > 1_000_000
    assert det.metadata["weights_mb"] > 5 and det.name == "yolov8n"
