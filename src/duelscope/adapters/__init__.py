"""Detector adapters and the registry that creates them."""

from duelscope.adapters.base import (
    AdapterError,
    Detector,
    DetectorConfig,
    ImageInput,
    resolve_device,
)
from duelscope.adapters.classmap import ClassMap
from duelscope.adapters.registry import available_adapters, create_detector, parse_spec, register

__all__ = [
    "AdapterError",
    "ClassMap",
    "Detector",
    "DetectorConfig",
    "ImageInput",
    "available_adapters",
    "create_detector",
    "parse_spec",
    "register",
    "resolve_device",
]
