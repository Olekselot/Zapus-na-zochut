"""Аугментації для антиспуфу.

Змінюємо світло, колір, масштаб і стиснення — те, що різниться між камерами.
Сильного розмиття немає: воно стирає муар екрана і текстуру паперу, на яких модель вчиться.
"""

from __future__ import annotations

import numpy as np


def augment(image_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    import cv2

    image = image_bgr
    height, width = image.shape[:2]
    if rng.random() < 0.8:
        area = rng.uniform(0.80, 1.0)
        aspect = rng.uniform(0.9, 1.1)
        crop_w = int(min(width, round(width * np.sqrt(area * aspect))))
        crop_h = int(min(height, round(height * np.sqrt(area / aspect))))
        left = int(rng.integers(0, width - crop_w + 1))
        top = int(rng.integers(0, height - crop_h + 1))
        image = cv2.resize(image[top : top + crop_h, left : left + crop_w], (width, height), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
    if rng.random() < 0.3:
        angle = rng.uniform(-10, 10)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
    values = image.astype(np.float32)
    if rng.random() < 0.8:
        values = values * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25)
    if rng.random() < 0.3:
        gamma = rng.uniform(0.8, 1.25)
        values = 255.0 * np.power(np.clip(values, 0, 255) / 255.0, gamma)
    if rng.random() < 0.3:
        gray = values.mean(axis=2, keepdims=True)
        values = gray + (values - gray) * rng.uniform(0.7, 1.3)
    if rng.random() < 0.2:
        values = values * rng.uniform(0.92, 1.08, size=(1, 1, 3))
    image = np.clip(values, 0, 255).astype(np.uint8)
    if rng.random() < 0.1:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    if rng.random() < 0.3:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
        if ok:
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(image)
