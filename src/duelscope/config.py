"""Run configuration (validated with pydantic, loadable from YAML)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class SpeedConfig(BaseModel):
    """Speed benchmark settings."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    images: int = Field(32, ge=1, description="Fixed subset size used for timing.")
    warmup: int = Field(5, ge=0)
    iters: int = Field(30, ge=3)
    batch_sizes: list[int] = [1, 8]


class GalleryConfig(BaseModel):
    """Failure gallery settings."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    k: int = Field(6, ge=1, description="Images per gallery view.")
    thumb_px: int = Field(480, ge=64, description="Longest thumbnail side in pixels.")
    quality: int = Field(70, ge=10, le=95, description="JPEG quality.")
    per_class_k: int = Field(2, ge=1)


class RunConfig(BaseModel):
    """Everything `duelscope run` needs."""

    model_config = ConfigDict(extra="forbid")
    data: Path
    models: list[str] = Field(min_length=1)
    split: str = "val"
    format: Literal["auto", "yolo", "coco"] = "auto"
    images_dir: Path | None = None
    out: Path = Path("report")
    imgsz: int = Field(640, ge=32)
    conf: float = Field(0.25, ge=0.0, le=1.0, description="Threshold for galleries/error analysis.")
    eval_conf: float = Field(0.001, ge=0.0, le=1.0, description="Threshold used for mAP.")
    iou: float = Field(0.6, gt=0.0, le=1.0, description="NMS IoU threshold.")
    batch: int = Field(8, ge=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    max_images: int | None = Field(None, ge=1)
    seed: int = 0
    cache: bool = True
    cache_dir: Path = Path(".duelscope_cache")
    class_map: dict[str, str] = Field(
        default_factory=dict, description="Manual model-class-name -> dataset-class-name map."
    )
    speed: SpeedConfig = SpeedConfig()
    gallery: GalleryConfig = GalleryConfig()

    @field_validator("models")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("duplicate entries in 'models'; list each model once")
        return v


class ConfigError(Exception):
    """Raised for unreadable or invalid configuration; message says what to fix."""


def load_config(path: Path) -> dict:
    """Read a YAML config file into a plain dict (paths are resolved relative to the file)."""
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read config '{path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config '{path}' must be a YAML mapping of option names to values.")
    base = path.parent
    for key in ("data", "out", "images_dir", "cache_dir"):
        if isinstance(raw.get(key), str) and not Path(raw[key]).is_absolute():
            raw[key] = str((base / raw[key]).resolve())
    return raw


def build_config(values: dict) -> RunConfig:
    """Validate values into a RunConfig, turning pydantic errors into readable ConfigErrors."""
    try:
        return RunConfig(**values)
    except ValidationError as exc:
        lines = [
            f"  - {'.'.join(str(p) for p in e['loc']) or 'config'}: {e['msg']}"
            for e in exc.errors()
        ]
        raise ConfigError(
            "Invalid configuration:\n" + "\n".join(lines) + "\nSee 'duelscope init' for a sample."
        ) from exc
