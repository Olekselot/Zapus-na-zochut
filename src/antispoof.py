"""Захист від підробки: MobileNetV2 з Topologies, натренована на живе / друк / екран."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from src.config import (
    ANTISPOOF_META,
    ANTISPOOF_WEIGHTS,
    CLASS_NAMES,
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.images import face_tensor


@dataclass
class AntispoofSpec:
    width_multiplier: float
    image_size: int
    threshold: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    class_names: tuple[str, ...]


def build_model(num_classes: int = 3, width_multiplier: float = 0.5):
    import torch
    from topologies.models.mobilenet_v2 import MobileNetV2

    model = MobileNetV2(num_classes=num_classes, width_multiplier=width_multiplier, dropout=0.1)
    return model, torch


class AntispoofModel:
    def __init__(self, weights_path=ANTISPOOF_WEIGHTS, meta_path=ANTISPOOF_META) -> None:
        self.weights_path = weights_path
        self.meta_path = meta_path
        self._model = None
        self.spec: AntispoofSpec | None = None

    @property
    def available(self) -> bool:
        return self.weights_path.exists() and self.meta_path.exists()

    @property
    def threshold(self) -> float:
        self._ensure()
        assert self.spec is not None
        return self.spec.threshold

    @property
    def image_size(self) -> int:
        if not self.available:
            return DEFAULT_IMAGE_SIZE
        self._ensure()
        assert self.spec is not None
        return self.spec.image_size

    def describe(self) -> str:
        if not self.available:
            return (
                "Моделі захисту від підробки ще немає. "
                "Реєстрація, розпізнавання і журнал працюють, але оплату заблоковано, "
                "поки не з’явиться models/antispoof.pt."
            )
        self._ensure()
        assert self.spec is not None
        return (
            f"Захист від підробки завантажено. Поріг живості {self.spec.threshold:.3f}. "
            "Оплата проходить лише якщо обличчя живе і знайдене в базі."
        )

    def _ensure(self) -> None:
        if self._model is not None:
            return
        if not self.available:
            raise RuntimeError("немає ваг захисту від підробки")
        payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.spec = AntispoofSpec(
            width_multiplier=float(payload["width_multiplier"]),
            image_size=int(payload["image_size"]),
            threshold=float(payload["threshold"]),
            mean=tuple(payload.get("mean", IMAGENET_MEAN)),
            std=tuple(payload.get("std", IMAGENET_STD)),
            class_names=tuple(payload.get("class_names", CLASS_NAMES)),
        )
        model, torch = build_model(num_classes=len(self.spec.class_names), width_multiplier=self.spec.width_multiplier)
        try:
            state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(self.weights_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        self._model = model

    def predict_proba(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        self._ensure()
        assert self.spec is not None and self._model is not None
        import torch

        tensors = [
            face_tensor(image, self.spec.image_size, self.spec.mean, self.spec.std) for image in aligned_faces
        ]
        batch = torch.from_numpy(np.stack(tensors, axis=0))
        with torch.inference_mode():
            logits = self._model(batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return np.asarray(probs, dtype=np.float32)
