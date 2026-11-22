"""Якість кадру: чи можна йому довіряти при оплаті й реєстрації."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.config import MAX_FRONTAL_YAW, MIN_BLUR, MIN_DET_SCORE, MIN_FACE_PX, MULTI_FACE_RATIO

PROBLEM_UA = {
    "multi_face": "у кадрі кілька облич",
    "low_score": "детектор не впевнений, що це обличчя",
    "small": "обличчя замале, підійдіть ближче",
    "blur": "кадр розмитий",
    "pose": "обличчя повернуте, дивіться прямо в камеру",
}


@dataclass(frozen=True)
class QualityRules:
    min_det_score: float = MIN_DET_SCORE
    min_face_px: float = MIN_FACE_PX
    min_blur: float = MIN_BLUR
    max_yaw: float = MAX_FRONTAL_YAW
    multi_face_ratio: float = MULTI_FACE_RATIO


def yaw_proxy(kps: np.ndarray) -> float:
    """Зсув носа від середини очей у міжочних відстанях.

    0 — анфас. Додатне значення: ніс ближче до правого краю кадру. Для камери
    без дзеркалення це означає, що людина повернула голову у свій лівий бік.
    """
    points = np.asarray(kps, dtype=np.float64).reshape(-1, 2)
    left_eye, right_eye, nose = points[0], points[1], points[2]
    distance = float(np.linalg.norm(right_eye - left_eye))
    if distance < 1e-6:
        return 0.0
    return float((nose[0] - (left_eye[0] + right_eye[0]) / 2) / distance)


def blur_score(image_bgr: np.ndarray) -> float:
    """Дисперсія лапласіана: що менша, то розмитіший кадр."""
    import cv2

    gray = cv2.cvtColor(np.ascontiguousarray(image_bgr), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def problems(sample, rules: QualityRules) -> list[str]:
    found: list[str] = []
    if sample.faces_found > 1 and sample.second_face_ratio >= rules.multi_face_ratio:
        found.append("multi_face")
    if sample.det_score < rules.min_det_score:
        found.append("low_score")
    if sample.bbox is not None:
        x1, y1, x2, y2 = [float(value) for value in sample.bbox[:4]]
        if min(x2 - x1, y2 - y1) < rules.min_face_px:
            found.append("small")
    if blur_score(sample.aligned_bgr) < rules.min_blur:
        found.append("blur")
    if sample.kps is not None and abs(yaw_proxy(sample.kps)) > rules.max_yaw:
        found.append("pose")
    return found


def split_by_quality(samples: list, rules: QualityRules) -> tuple[list, list[str]]:
    """Повертає придатні кадри і перелік проблем відкинутих."""
    kept, rejected = [], []
    for sample in samples:
        issues = problems(sample, rules)
        if issues:
            rejected.extend(issues)
        else:
            kept.append(sample)
    return kept, rejected


def main_problem(issues: list[str]) -> str:
    if not issues:
        return ""
    code = Counter(issues).most_common(1)[0][0]
    return PROBLEM_UA.get(code, code)
