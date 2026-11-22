"""Пошук обличчя, вирівнювання по п'яти точках, вектор ArcFace і кроп для антиспуфу."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import DEFAULT_CROP_SCALE, DEFAULT_IMAGE_SIZE
from src.gallery import l2_normalize


@dataclass
class FaceSample:
    aligned_bgr: np.ndarray
    embedding: np.ndarray
    det_score: float
    antispoof_bgr: np.ndarray | None = None
    bbox: np.ndarray | None = None
    kps: np.ndarray | None = None
    landmarks68: np.ndarray | None = None
    faces_found: int = 1
    second_face_ratio: float = 0.0

    @property
    def spoof_input(self) -> np.ndarray:
        return self.aligned_bgr if self.antispoof_bgr is None else self.antispoof_bgr


def _area(face) -> float:
    x1, y1, x2, y2 = [float(value) for value in face.bbox[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class FaceEngine:
    def __init__(self, det_size: int = 640, with_landmarks: bool = True, with_recognition: bool = True) -> None:
        """with_recognition=False — лише детектор, для нарізання кропів: так у кілька разів швидше."""
        self._app = None
        self.det_size = int(det_size)
        self.with_landmarks = with_landmarks
        self.with_recognition = with_recognition

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
        modules = ["detection"]
        modules += ["recognition"] if self.with_recognition else []
        modules += ["landmark_3d_68"] if self.with_landmarks else []
        app = FaceAnalysis(name="buffalo_l", providers=providers, allowed_modules=modules)
        app.prepare(ctx_id=context, det_size=(self.det_size, self.det_size))
        self._app = app
        return app

    def analyze(
        self,
        image_bgr: np.ndarray,
        crop_size: int = DEFAULT_IMAGE_SIZE,
        crop_scale: float | None = DEFAULT_CROP_SCALE,
        pick: str = "largest",
    ) -> FaceSample | None:
        """Найбільше обличчя на фото (pick="center" — найближче до центру, як у LFW).

        crop_scale=None дає старий вузький кроп для антиспуфу.
        """
        from insightface.utils import face_align

        faces = self._model().get(image_bgr)
        if not faces:
            return None
        ordered = sorted(faces, key=_area, reverse=True)
        face = ordered[0]
        if pick == "center":
            height, width = image_bgr.shape[:2]
            face = min(
                faces,
                key=lambda item: ((item.bbox[0] + item.bbox[2]) / 2 - width / 2) ** 2
                + ((item.bbox[1] + item.bbox[3]) / 2 - height / 2) ** 2,
            )
        others = [item for item in ordered if item is not face]
        second = _area(others[0]) / max(_area(face), 1e-6) if others else 0.0
        aligned = face_align.norm_crop(image_bgr, landmark=face.kps, image_size=112)
        if crop_scale is None:
            spoof = face_align.norm_crop(image_bgr, landmark=face.kps, image_size=crop_size)
        else:
            from src.images import context_crop

            spoof = context_crop(image_bgr, face.bbox, crop_scale, crop_size)
        landmarks = getattr(face, "landmark_3d_68", None)
        raw = getattr(face, "embedding", None)
        return FaceSample(
            aligned_bgr=aligned,
            embedding=None if raw is None else l2_normalize(np.asarray(raw, dtype=np.float32)),
            det_score=float(face.det_score),
            antispoof_bgr=spoof,
            bbox=np.asarray(face.bbox, dtype=np.float32),
            kps=np.asarray(face.kps, dtype=np.float32),
            landmarks68=None if landmarks is None else np.asarray(landmarks, dtype=np.float32),
            faces_found=len(faces),
            second_face_ratio=float(second),
        )
