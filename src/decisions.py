"""Рішення «живе чи підробка» для одного кадру і для пачки кадрів."""

from __future__ import annotations

import numpy as np

from src.config import CLASS_NAMES


def summarize_liveness(probs: np.ndarray, threshold: float) -> tuple[bool, str, float]:
    """Повертає (чи живе, назва класу, медіана P(live)).

    Для кількох кадрів живість — це медіана P(live) і більшість кадрів,
    у яких live є найімовірнішим класом. Одне фото є окремим випадком пачки з одного кадру.
    """
    matrix = np.asarray(probs, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 3 or len(matrix) == 0:
        raise ValueError("очікувались імовірності форми [N, 3]")
    median_live = float(np.median(matrix[:, 0]))
    majority_live = bool(np.mean(matrix.argmax(axis=1) == 0) >= 0.5)
    is_live = bool(median_live >= threshold and majority_live)
    if is_live:
        return True, CLASS_NAMES[0], median_live
    mean_probs = matrix.mean(axis=0)
    attack_index = 1 + int(np.argmax(mean_probs[1:]))
    return False, CLASS_NAMES[attack_index], median_live
