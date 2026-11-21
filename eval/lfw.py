"""Спільне читання пар LFW для метрик розпізнавання."""

from __future__ import annotations

import numpy as np


def load_test_pairs(max_pairs: int | None = None, resize: float | None = 1.0):
    """Тестові пари sklearn LFW. Мітка True означає одну й ту саму людину."""
    from sklearn.datasets import fetch_lfw_pairs

    bunch = fetch_lfw_pairs(
        subset="test",
        color=True,
        resize=resize,
        download_if_missing=True,
        funneled=True,
    )
    pairs = np.asarray(bunch.pairs)
    genuine = np.asarray(bunch.target).astype(int) == 1
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
        genuine = genuine[:max_pairs]
    return pairs, genuine


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        values = array.astype(np.float32)
        if np.nanmax(values) <= 1.0:
            values *= 255.0
        array = np.clip(values, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    return np.ascontiguousarray(array[:, :, :3])


def to_bgr(image: np.ndarray) -> np.ndarray:
    import cv2

    rgb = to_uint8_rgb(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
