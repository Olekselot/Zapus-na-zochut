"""Захист від підробки: класифікатор живе / друк / екран поверх кропу з контекстом.

Основна модель — MobileNetV3 з вагами ImageNet, донавчена на наших даних.
MobileNetV2 з Topologies лишається для порівняння «з нуля проти донавчання».
"""

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

ARCHES = ("mnv3_small", "mnv3_large", "topologies_mnv2")


@dataclass
class AntispoofSpec:
    arch: str
    width_multiplier: float
    image_size: int
    threshold: float
    temperature: float
    crop_scale: float | None
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    class_names: tuple[str, ...]


def build_model(arch: str = "mnv3_small", num_classes: int = 3, width_multiplier: float = 0.5, pretrained: bool = False):
    """Повертає (модель, параметри голови). Голова — те, що вчиться першим, поки основа заморожена."""
    import torch.nn as nn

    if arch == "topologies_mnv2":
        from topologies.models.mobilenet_v2 import MobileNetV2

        model = MobileNetV2(num_classes=num_classes, width_multiplier=width_multiplier, dropout=0.1)
        return model, list(model.parameters())
    if arch not in ARCHES:
        raise ValueError(f"невідома архітектура {arch}, доступні: {', '.join(ARCHES)}")
    from torchvision import models

    if arch == "mnv3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    else:
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return model, list(model.classifier.parameters())


def spec_from_meta(payload: dict) -> AntispoofSpec:
    """Старі метадані без arch і crop_scale — це MobileNetV2 з Topologies на вузькому кропі."""
    crop = payload.get("crop_scale")
    return AntispoofSpec(
        arch=str(payload.get("arch", "topologies_mnv2")),
        width_multiplier=float(payload.get("width_multiplier", 0.5)),
        image_size=int(payload["image_size"]),
        threshold=float(payload["threshold"]),
        temperature=float(payload.get("temperature", 1.0)),
        crop_scale=None if crop is None else float(crop),
        mean=tuple(payload.get("mean", IMAGENET_MEAN)),
        std=tuple(payload.get("std", IMAGENET_STD)),
        class_names=tuple(payload.get("class_names", CLASS_NAMES)),
    )


class AntispoofModel:
    def __init__(self, weights_path=ANTISPOOF_WEIGHTS, meta_path=ANTISPOOF_META) -> None:
        self.weights_path = weights_path
        self.meta_path = meta_path
        self._model = None
        self.spec: AntispoofSpec | None = None

    @property
    def available(self) -> bool:
        return self.weights_path.exists() and self.meta_path.exists()

    def _read_spec(self) -> AntispoofSpec:
        if self.spec is None:
            self.spec = spec_from_meta(json.loads(self.meta_path.read_text(encoding="utf-8")))
        return self.spec

    @property
    def threshold(self) -> float:
        return self._read_spec().threshold

    @property
    def image_size(self) -> int:
        return self._read_spec().image_size if self.available else DEFAULT_IMAGE_SIZE

    @property
    def crop_scale(self) -> float | None:
        from src.config import DEFAULT_CROP_SCALE

        return self._read_spec().crop_scale if self.available else DEFAULT_CROP_SCALE

    def describe(self) -> str:
        if not self.available:
            return (
                "Моделі захисту від підробки ще немає. "
                "Реєстрація, розпізнавання і журнал працюють, але оплату заблоковано, "
                "поки не з’явиться models/antispoof.pt."
            )
        spec = self._read_spec()
        return (
            f"Захист від підробки завантажено ({spec.arch}). Поріг живості {spec.threshold:.3f}. "
            "Оплата проходить лише якщо обличчя живе і знайдене в базі."
        )

    def _ensure(self) -> None:
        if self._model is not None:
            return
        if not self.available:
            raise RuntimeError("немає ваг захисту від підробки")
        import torch

        spec = self._read_spec()
        model, _head = build_model(spec.arch, len(spec.class_names), spec.width_multiplier, pretrained=False)
        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        self._model = model

    def predict_proba(self, faces: list[np.ndarray]) -> np.ndarray:
        self._ensure()
        assert self.spec is not None and self._model is not None
        import torch

        tensors = [face_tensor(image, self.spec.image_size, self.spec.mean, self.spec.std) for image in faces]
        batch = torch.from_numpy(np.stack(tensors, axis=0))
        with torch.inference_mode():
            logits = self._model(batch) / self.spec.temperature
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return np.asarray(probs, dtype=np.float32)
