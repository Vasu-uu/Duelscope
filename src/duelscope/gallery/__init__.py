"""Failure-case galleries."""

from duelscope.gallery.build import build_galleries
from duelscope.gallery.select import (
    a_ok_b_fails,
    all_models_fail,
    analyze_dataset,
    analyze_image,
    worst_images,
    worst_per_class,
)

__all__ = [
    "a_ok_b_fails",
    "all_models_fail",
    "analyze_dataset",
    "analyze_image",
    "build_galleries",
    "worst_images",
    "worst_per_class",
]
