"""Зведення п'яти кадрів: медіана живості й середній вектор обличчя."""

from __future__ import annotations

import numpy as np

from src.gallery import l2_normalize


def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("немає векторів для усереднення")
    stacked = np.stack([l2_normalize(vector) for vector in embeddings], axis=0)
    return l2_normalize(stacked.mean(axis=0))
