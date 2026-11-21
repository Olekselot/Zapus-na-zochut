"""П'ять кадрів з локальної вебкамери. Це камера комп'ютера, на якому запущено демо."""

from __future__ import annotations

import time

import numpy as np

from src.config import FRAME_COUNT


def capture_frames(count: int = FRAME_COUNT, camera_index: int = 0, gap_s: float = 0.2) -> list[np.ndarray]:
    import cv2

    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(
            "Не вдалося відкрити камеру. Закрийте програми, які її займають, "
            "або задайте інший індекс змінною CAMERA_INDEX."
        )
    frames: list[np.ndarray] = []
    try:
        for _ in range(3):
            camera.read()
        for index in range(count):
            ok, frame = camera.read()
            if ok and frame is not None:
                frames.append(frame)
            if index != count - 1:
                time.sleep(gap_s)
    finally:
        camera.release()
    if len(frames) < min(3, count):
        raise RuntimeError(f"Камера дала лише {len(frames)} кадрів із {count}.")
    return frames
