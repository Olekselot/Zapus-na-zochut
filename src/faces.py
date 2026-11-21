"""Пошук обличчя, вирівнювання по п'яти точках і вектор ArcFace на 512 чисел."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import DEFAULT_IMAGE_SIZE
from src.gallery import l2_normalize


@dataclass
class FaceSample:
    aligned_bgr: np.ndarray
    embedding: np.ndarray
    det_score: float


class FaceEngine:
    def __init__(self) -> None:
        self._app = None

    def _model(self):
        if self._app is not None:
            return self._app
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis
        except ImportError as error:
            raise RuntimeError(
                "Не встановлено insightface або onnxruntime. Виконайте pip install -r requirements.txt"
            ) from error
        providers = ["CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        context = 0 if providers[0].startswith("CUDA") else -1
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=context, det_size=(640, 640))
        self._app = app
        return app

    def analyze(self, image_bgr: np.ndarray, crop_size: int = DEFAULT_IMAGE_SIZE) -> FaceSample | None:
        from insightface.utils import face_align

        faces = self._model().get(image_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
        aligned = face_align.norm_crop(image_bgr, landmark=face.kps, image_size=crop_size)
        embedding = l2_normalize(np.asarray(face.embedding, dtype=np.float32))
        return FaceSample(aligned_bgr=aligned, embedding=embedding, det_score=float(face.det_score))
