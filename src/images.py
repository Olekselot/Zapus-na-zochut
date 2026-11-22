"""Читання фото, кроп з контекстом для антиспуфу і тензор для мережі."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import DEFAULT_CROP_SCALE, DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def imread(path: Path | str) -> np.ndarray:
    import cv2

    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"не вдалося прочитати фото: {path}")
    return image


def imwrite(path: Path | str, image: np.ndarray, quality: int = 92) -> None:
    import cv2

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError(f"не вдалося закодувати фото: {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(encoded.tobytes())


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


def shrink(image: np.ndarray, max_side: int) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    side = max(height, width)
    if side <= max_side:
        return image
    ratio = max_side / side
    return cv2.resize(image, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA)


def context_crop(
    image_bgr: np.ndarray,
    bbox: np.ndarray,
    scale: float = DEFAULT_CROP_SCALE,
    size: int = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Квадрат у scale разів більший за обличчя, щоб у кадр потрапили рамка екрана чи край аркуша.

    Якщо такий квадрат не влазить у фото, він зменшується до меншої сторони фото
    і зсувається всередину. Чорних полів немає: вони були б штучною ознакою.
    """
    import cv2

    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in np.asarray(bbox, dtype=np.float32)[:4]]
    side = max(x2 - x1, y2 - y1) * float(scale)
    side = max(1.0, min(side, float(width), float(height)))
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(round(min(max(center_x - side / 2, 0.0), width - side)))
    top = int(round(min(max(center_y - side / 2, 0.0), height - side)))
    edge = int(round(side))
    crop = image_bgr[top : top + edge, left : left + edge]
    interpolation = cv2.INTER_AREA if edge > size else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interpolation)


def face_tensor(
    face_bgr: np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    import cv2

    image = face_bgr
    if image.shape[0] != image_size or image.shape[1] != image_size:
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    rgb = (rgb - mean_array) / std_array
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32)
