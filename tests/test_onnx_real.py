from pathlib import Path

import pytest

from duelscope.adapters import DetectorConfig
from duelscope.adapters.onnx_ import OnnxDetector
from duelscope.data import load_dataset
from duelscope.synthetic import make_yolo_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.ultralytics
def test_real_onnx_export_matches_pt(tmp_path: Path) -> None:
    """Export yolov8n to ONNX, run it through OnnxDetector and compare with the .pt output."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("ultralytics")
    w = ROOT / "weights" / "yolov8n.pt"
    if not w.is_file():
        pytest.skip("weights/yolov8n.pt not available")
    import shutil

    from ultralytics import YOLO

    local = tmp_path / "yolov8n.pt"
    shutil.copy(w, local)
    onnx = Path(YOLO(str(local)).export(format="onnx", imgsz=640))
    img = load_dataset(make_yolo_dataset(tmp_path / "d", n_images=1)).records[0].path
    cfg = DetectorConfig(device="cpu", conf=0.25)
    det = OnnxDetector(str(onnx), cfg)
    out = det.predict([img])[0]
    ref = YOLO(str(local)).predict(str(img), conf=0.25, verbose=False)[0]
    assert det.class_names[0] == "person" and len(det.class_names) == 80
    assert abs(len(out) - len(ref.boxes)) <= 1
    assert det.last_timing["inference"] > 0
