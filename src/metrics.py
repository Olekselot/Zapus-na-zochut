"""Банківські метрики розпізнавання і захисту від підробки.

Розпізнавання: FAR — частка чужих пар, яких система прийняла; FRR — частка своїх,
яких відхилила; TAR — частка своїх, яких прийняла; EER — поріг, де FAR ≈ FRR.
Захист: APCER — частка атак, прийнятих як жива людина; BPCER — частка живих,
відхилених як підробка; ACER — їхнє середнє.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RecognitionPoint:
    threshold: float
    far: float
    tar: float
    frr: float
    accuracy: float


@dataclass(frozen=True)
class SpoofRates:
    apcer: float
    bpcer: float
    acer: float
    attacks: int
    bona_fide: int


def _as_bool(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=bool)


def presentation_rates(is_live: np.ndarray, accepted_as_live: np.ndarray) -> SpoofRates:
    truth = _as_bool(is_live)
    accepted = _as_bool(accepted_as_live)
    if truth.shape != accepted.shape:
        raise ValueError("мітки і рішення мають бути однакової довжини")
    attacks = ~truth
    bona = truth
    apcer = float(accepted[attacks].mean()) if attacks.any() else float("nan")
    bpcer = float((~accepted[bona]).mean()) if bona.any() else float("nan")
    if np.isnan(apcer) or np.isnan(bpcer):
        acer = float("nan")
    else:
        acer = (apcer + bpcer) / 2
    return SpoofRates(apcer, bpcer, acer, int(attacks.sum()), int(bona.sum()))


def live_mask_from_probs(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Живе обличчя: медіана тут не потрібна, це рішення для одного кадру.

    Кадр живий, якщо клас live має найбільшу ймовірність і вона не нижча за поріг.
    """
    matrix = np.asarray(probs, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("очікувалась матриця ймовірностей [N, 3]")
    live_wins = matrix.argmax(axis=1) == 0
    return live_wins & (matrix[:, 0] >= threshold)


def choose_live_threshold(probs: np.ndarray, is_live: np.ndarray, apcer_target: float = 0.05) -> float:
    """Найнижчий поріг P(live), за якого APCER ще не вищий за ціль.

    Вищий поріг рідше приймає підробку і частіше відхиляє живу людину.
    Серед порогів, що тримають APCER у цілі, беремо найнижчий: так BPCER менший.
    Якщо жоден поріг ціль не виконує, беремо поріг з найменшим ACER.
    """
    matrix = np.asarray(probs, dtype=np.float64)
    truth = _as_bool(is_live)
    if len(matrix) != len(truth):
        raise ValueError("ймовірності і мітки різної довжини")
    if not (~truth).any() or not truth.any():
        raise ValueError("для порогу потрібні і живі кадри, і атаки")

    candidates = np.unique(np.concatenate([matrix[:, 0], np.array([0.0, 1.0])]))
    feasible: list[tuple[float, float]] = []
    best_acer = float("inf")
    best_threshold = 1.0
    for threshold in candidates:
        rates = presentation_rates(truth, live_mask_from_probs(matrix, float(threshold)))
        if rates.acer < best_acer or (rates.acer == best_acer and float(threshold) < best_threshold):
            best_acer = rates.acer
            best_threshold = float(threshold)
        if rates.apcer <= apcer_target:
            feasible.append((float(threshold), rates.bpcer))
    if not feasible:
        return best_threshold
    feasible.sort(key=lambda item: (item[0], item[1]))
    return feasible[0][0]


def recognition_curve(scores: np.ndarray, genuine: np.ndarray) -> list[RecognitionPoint]:
    """Пари з оцінкою схожості. genuine=True означає, що це одна й та сама людина."""
    values = np.asarray(scores, dtype=np.float64)
    labels = _as_bool(genuine)
    if len(values) != len(labels):
        raise ValueError("оцінки і мітки пар різної довжини")
    if not labels.any() or not (~labels).any():
        raise ValueError("потрібні і свої пари, і чужі")

    thresholds = np.unique(np.concatenate([values, values + 1e-6, np.array([values.max() + 1.0])]))
    points: list[RecognitionPoint] = []
    for threshold in thresholds:
        accepted = values >= threshold
        far = float(accepted[~labels].mean())
        tar = float(accepted[labels].mean())
        accuracy = float((accepted == labels).mean())
        points.append(
            RecognitionPoint(
                threshold=float(threshold),
                far=far,
                tar=tar,
                frr=1.0 - tar,
                accuracy=accuracy,
            )
        )
    return points


def operating_point(points: list[RecognitionPoint], far_target: float) -> RecognitionPoint | None:
    """Найщедріший поріг, за якого FAR ще не вищий за ціль. Так TAR найбільший."""
    allowed = [point for point in points if point.far <= far_target]
    if not allowed:
        return None
    return min(allowed, key=lambda point: (point.threshold, -point.tar))


def equal_error_point(points: list[RecognitionPoint]) -> RecognitionPoint:
    return min(points, key=lambda point: (abs(point.far - point.frr), point.threshold))
