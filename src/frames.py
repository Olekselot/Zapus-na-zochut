"""Зведення кадрів: середній вектор обличчя."""

from __future__ import annotations

import numpy as np

from src.gallery import template


def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("немає векторів для усереднення")
    return template(embeddings)
