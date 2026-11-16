import pytest


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked `gpu` when no CUDA device is present."""
    if _has_cuda():
        return
    skip = pytest.mark.skip(reason="no CUDA GPU available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
