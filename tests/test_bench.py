import time
from pathlib import Path

import numpy as np
import pytest

from duelscope.adapters import DetectorConfig
from duelscope.bench import benchmark, hardware_info
from duelscope.synthetic import make_yolo_dataset
from duelscope.types import Detections
from tests.helpers import MockDetector


def _images(tmp_path: Path, n: int = 3) -> list[Path]:
    root = make_yolo_dataset(tmp_path / "d", n_images=n).parent
    return sorted((root / "images" / "val").glob("*.png"))


def _sleepy(ms: float) -> MockDetector:
    def fn(_: Path) -> Detections:
        time.sleep(ms / 1000)
        return Detections.empty()

    return MockDetector(fn, "sleepy", cfg=DetectorConfig(device="cpu"))


def test_benchmark_structure_and_call_counts(tmp_path: Path) -> None:
    det = _sleepy(2.0)
    res = benchmark(det, _images(tmp_path), batch_sizes=(1, 4), warmup=2, iters=6)
    # 2 (detector warmup) + 2 (warmup) + 6 timed + bs1: 6 runs + bs4: max(6//4, 3)=3 runs of 4
    assert det.calls == 2 + 2 + 6 + 6 * 1 + 3 * 4
    lat = res.latency_ms
    assert (
        1.5 < lat["p50"] < 50 and lat["min"] <= lat["p50"] <= lat["p90"] <= lat["p99"] <= lat["max"]
    )
    assert lat["std"] >= 0 and res.fps_bs1 == pytest.approx(1e3 / lat["mean"])
    assert set(res.batch) == {"1", "4"}
    assert res.breakdown_ms == pytest.approx(
        {"preprocess": 0.1, "inference": 1.0, "postprocess": 0.1}
    )
    assert res.gpu_peak_mb is None and res.device == "cpu" and res.ram_mb > 0
    assert res.to_dict()["imgsz"] == 640


def test_benchmark_batch_throughput_scales_with_batch(tmp_path: Path) -> None:
    res = benchmark(_sleepy(1.0), _images(tmp_path), batch_sizes=(1, 4), warmup=1, iters=4)
    # the mock sleeps once per image, so throughput stays roughly constant (no free speed-up)
    assert res.batch["4"]["throughput_fps"] == pytest.approx(
        res.batch["1"]["throughput_fps"], rel=0.6
    )


def test_benchmark_needs_images() -> None:
    with pytest.raises(ValueError, match="at least one image"):
        benchmark(_sleepy(0), [])


def test_cuda_request_without_gpu_falls_back_to_cpu(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("GPU present")
    det = MockDetector(lambda p: Detections.empty(), cfg=DetectorConfig(device="cuda"))
    res = benchmark(det, _images(tmp_path, 1), warmup=1, iters=3)
    assert res.device == "cpu" and any("CUDA requested" in n for n in res.notes)


def test_hardware_info_never_raises_and_is_complete() -> None:
    info = hardware_info()
    for k in ("cpu", "cores_logical", "ram_gb", "gpu", "os", "python", "libraries"):
        assert k in info
    assert info["ram_gb"] > 0 and "duelscope" in info["libraries"]


@pytest.mark.gpu
def test_gpu_memory_is_reported(tmp_path: Path) -> None:
    from duelscope.adapters.torchvision_ import TorchvisionDetector

    det = TorchvisionDetector(
        "fasterrcnn_mobilenet_v3_large_fpn", DetectorConfig(device="cuda"), weights=None
    )
    res = benchmark(det, _images(tmp_path, 2), batch_sizes=(1,), warmup=1, iters=3)
    assert res.device == "cuda" and res.gpu_peak_mb and res.gpu_peak_mb > 0
    assert np.isfinite(res.latency_ms["p99"])
