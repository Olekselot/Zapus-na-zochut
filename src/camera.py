"""Кадри з локальної вебкамери. Це камера комп'ютера, на якому запущено демо."""

from __future__ import annotations

import time

import numpy as np

from src.config import CHALLENGE_FPS, CHALLENGE_SECONDS, FRAME_COUNT


def open_camera(camera_index: int = 0):
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
    for _ in range(3):
        camera.read()
    return camera


def capture_frames(count: int = FRAME_COUNT, camera_index: int = 0, gap_s: float = 0.3) -> list[np.ndarray]:
    camera = open_camera(camera_index)
    frames: list[np.ndarray] = []
    try:
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


def capture_sequence(
    seconds: float = CHALLENGE_SECONDS,
    fps: float = CHALLENGE_FPS,
    camera_index: int = 0,
) -> list[np.ndarray]:
    """Безперервний запис для активної перевірки: кадр не частіше ніж fps разів на секунду."""
    camera = open_camera(camera_index)
    frames: list[np.ndarray] = []
    step = 1.0 / max(fps, 1.0)
    started = time.monotonic()
    last = -step
    try:
        while time.monotonic() - started < seconds:
            ok, frame = camera.read()
            now = time.monotonic() - started
            if ok and frame is not None and now - last >= step:
                frames.append(frame)
                last = now
    finally:
        camera.release()
    if len(frames) < 5:
        raise RuntimeError(f"Камера дала лише {len(frames)} кадрів за {seconds:.0f} с.")
    return frames
