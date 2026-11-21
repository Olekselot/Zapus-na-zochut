"""Читання фото і перетворення вирізаного обличчя на тензор для MobileNetV2."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def imread(path: Path | str) -> np.ndarray:
    import cv2

    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"не вдалося прочитати фото: {path}")
    return image


def as_bgr(image: np.ndarray) -> np.ndarray:
    """Gradio віддає RGB. Детектор і OpenCV працюють з BGR."""
    import cv2

    array = np.asarray(image)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    elif array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    else:
        array = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2BGR)
    if array.dtype != np.uint8:
        values = array.astype(np.float32)
        if values.max() <= 1.0:
            values *= 255.0
        array = np.clip(values, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def bgr_to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("очікувалось кольорове фото")
    return np.ascontiguousarray(array[:, :, :3][:, :, ::-1])


def face_tensor(
    aligned_bgr: np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    import cv2

    image = aligned_bgr
    if image.shape[0] != image_size or image.shape[1] != image_size:
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    rgb = (rgb - mean_array) / std_array
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32)
