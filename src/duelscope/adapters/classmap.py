"""Map model class ids to dataset class ids by name."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from duelscope.adapters.base import AdapterError
from duelscope.types import Detections


def _norm(name: str) -> str:
    return re.sub(r"[\s_\-]+", " ", name.strip().lower())


@dataclass
class ClassMap:
    """Lookup table from model class id to dataset class id (-1 = dropped)."""

    table: np.ndarray
    unmapped: list[str] = field(default_factory=list)  # model class names with no dataset match
    missing: list[str] = field(default_factory=list)  # dataset classes the model can never output

    @classmethod
    def build(
        cls,
        model_names: list[str] | None,
        dataset_names: list[str],
        manual: dict[str, str] | None = None,
        model_label: str = "model",
    ) -> ClassMap:
        """Match by normalised name; `manual` maps model-name -> dataset-name overrides."""
        if model_names is None:
            if len(dataset_names) == 0:
                raise AdapterError("Dataset has no classes.")
            return cls(np.arange(len(dataset_names), dtype=np.int64))
        ds_index = {_norm(n): i for i, n in enumerate(dataset_names)}
        man = {_norm(k): _norm(v) for k, v in (manual or {}).items()}
        table = np.full(len(model_names), -1, dtype=np.int64)
        unmapped: list[str] = []
        for i, n in enumerate(model_names):
            target = man.get(_norm(n), _norm(n))
            if target in ds_index:
                table[i] = ds_index[target]
            elif _norm(n) not in {"n a", "na", "background", "__background__"}:
                unmapped.append(n)
        covered = set(table[table >= 0].tolist())
        missing = [n for i, n in enumerate(dataset_names) if i not in covered]
        cm = cls(table, unmapped, missing)
        if len(covered) == 0:
            raise AdapterError(
                f"{model_label}: none of its classes match the dataset classes {dataset_names[:5]}"
                f"{'...' if len(dataset_names) > 5 else ''}. Provide a manual map under "
                "'class_map' in the config, e.g. {car: vehicle}."
            )
        if missing:
            warnings.warn(
                f"{model_label} cannot predict dataset classes {missing}; their AP will be 0.",
                stacklevel=2,
            )
        return cm

    def apply(self, det: Detections) -> Detections:
        """Remap class ids and drop detections of unmapped classes."""
        if len(det) == 0:
            return det
        safe = np.clip(det.classes, 0, len(self.table) - 1)
        mapped = np.where(det.classes < len(self.table), self.table[safe], -1)
        keep = mapped >= 0
        return Detections(det.boxes[keep], det.scores[keep], mapped[keep])
