"""Банківські метрики розпізнавання і захисту від підробки.

Розпізнавання. FMR/FNMR — помилки порівняння лише для пар, де обидва обличчя знайдено.
FAR/FRR — помилки системи: пара, де детектор не знайшов обличчя (FTA), рахується відмовою.
TAR = 1 - FRR. EER — поріг, де FAR ≈ FRR.

Захист (ISO/IEC 30107-3). APCER рахується окремо для кожного типу атаки (друк, екран),
підсумковий APCER — найгірший з них. BPCER — частка живих, відхилених як підробка.
ACER = (APCER + BPCER) / 2 лишаємо для порівняння зі статтями, стандарт його не радить.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import LABEL_NAMES


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
    apcer_by_type: dict


def _as_bool(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=bool)


def _labels(values: np.ndarray) -> np.ndarray:
    """Булеві «живе / ні» перетворюються на 0 / 1, цілі мітки класів лишаються."""
    array = np.asarray(values)
    if array.dtype == bool:
        return np.where(array, 0, 1)
    return array.astype(int)


def presentation_rates(labels: np.ndarray, accepted_as_live: np.ndarray) -> SpoofRates:
    """labels — 0 живе, 1 друк, 2 екран, 3 підробка невідомого типу (або булеве is_live)."""
    classes = _labels(labels)
    accepted = _as_bool(accepted_as_live)
    if classes.shape != accepted.shape:
        raise ValueError("мітки і рішення мають бути однакової довжини")
    bona = classes == 0
    by_type: dict[str, float] = {}
    for index in sorted(set(classes[~bona].tolist())):
        name = LABEL_NAMES[index] if index < len(LABEL_NAMES) else str(index)
        by_type[name] = float(accepted[classes == index].mean())
    apcer = max(by_type.values()) if by_type else float("nan")
    bpcer = float((~accepted[bona]).mean()) if bona.any() else float("nan")
    acer = float("nan") if np.isnan(apcer) or np.isnan(bpcer) else (apcer + bpcer) / 2
    return SpoofRates(apcer, bpcer, acer, int((~bona).sum()), int(bona.sum()), by_type)


def live_mask_from_probs(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Кадр живий, якщо клас live має найбільшу ймовірність і вона не нижча за поріг."""
    matrix = np.asarray(probs, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("очікувалась матриця ймовірностей [N, 3]")
    live_wins = matrix.argmax(axis=1) == 0
    return live_wins & (matrix[:, 0] >= threshold)


def _candidates(probs: np.ndarray) -> np.ndarray:
    return np.unique(np.concatenate([np.asarray(probs, dtype=np.float64)[:, 0], np.array([0.0, 1.0 + 1e-9])]))


def choose_live_threshold(probs: np.ndarray, labels: np.ndarray, apcer_target: float = 0.05) -> float:
    """Найнижчий поріг P(live), за якого найгірший APCER ще не вищий за ціль.

    Серед порогів, що тримають APCER у цілі, найнижчий дає найменший BPCER.
    Якщо жоден поріг ціль не виконує, беремо поріг з найменшим ACER.
    """
    matrix = np.asarray(probs, dtype=np.float64)
    classes = _labels(labels)
    if len(matrix) != len(classes):
        raise ValueError("ймовірності і мітки різної довжини")
    if not (classes != 0).any() or not (classes == 0).any():
        raise ValueError("для порогу потрібні і живі кадри, і атаки")
    best_acer, best_threshold = float("inf"), 1.0
    for threshold in _candidates(matrix):
        rates = presentation_rates(classes, live_mask_from_probs(matrix, float(threshold)))
        if rates.apcer <= apcer_target:
            return float(threshold)
        if rates.acer < best_acer:
            best_acer, best_threshold = rates.acer, float(threshold)
    return best_threshold


def threshold_for_bpcer(probs: np.ndarray, labels: np.ndarray, bpcer_target: float = 0.05) -> float:
    """Найвищий поріг P(live), за якого BPCER ще не вищий за ціль.

    Обмежуємо відмови живим людям і за цієї умови пропускаємо якомога менше атак.
    Корисно, коли ціль за APCER дає неприйнятний BPCER.
    """
    matrix = np.asarray(probs, dtype=np.float64)
    classes = _labels(labels)
    if len(matrix) != len(classes):
        raise ValueError("ймовірності і мітки різної довжини")
    if not (classes == 0).any():
        raise ValueError("для порогу потрібні живі кадри")
    best = 0.0
    for threshold in _candidates(matrix):
        if presentation_rates(classes, live_mask_from_probs(matrix, float(threshold))).bpcer <= bpcer_target:
            best = max(best, float(threshold))
    return best


def det_curve(probs: np.ndarray, labels: np.ndarray) -> list[tuple[float, float, float]]:
    """(поріг, найгірший APCER, BPCER) для кожного порогу — точки DET-кривої."""
    matrix = np.asarray(probs, dtype=np.float64)
    classes = _labels(labels)
    points = []
    for threshold in _candidates(matrix):
        rates = presentation_rates(classes, live_mask_from_probs(matrix, float(threshold)))
        points.append((float(threshold), rates.apcer, rates.bpcer))
    return points


def bpcer_at_apcer(probs: np.ndarray, labels: np.ndarray, targets=(0.01, 0.05, 0.10)) -> dict[str, float | None]:
    """BPCER при найнижчому порозі, що тримає найгірший APCER не вище за ціль."""
    points = det_curve(probs, labels)
    result: dict[str, float | None] = {}
    for target in targets:
        allowed = [point for point in points if point[1] <= target]
        result[f"{target:g}"] = None if not allowed else min(allowed, key=lambda point: point[0])[2]
    return result


def recognition_curve(scores: np.ndarray, genuine: np.ndarray) -> list[RecognitionPoint]:
    """Пари з оцінкою схожості. NaN — пара, де обличчя не знайдено, вона завжди відхиляється."""
    values = np.asarray(scores, dtype=np.float64)
    labels = _as_bool(genuine)
    if len(values) != len(labels):
        raise ValueError("оцінки і мітки пар різної довжини")
    if not labels.any() or not (~labels).any():
        raise ValueError("потрібні і свої пари, і чужі")
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("немає жодної пари з обличчями")
    thresholds = np.unique(np.concatenate([finite, finite + 1e-6, np.array([finite.max() + 1.0])]))
    filled = np.where(np.isfinite(values), values, -np.inf)
    points: list[RecognitionPoint] = []
    for threshold in thresholds:
        accepted = filled >= threshold
        far = float(accepted[~labels].mean())
        tar = float(accepted[labels].mean())
        points.append(
            RecognitionPoint(
                threshold=float(threshold),
                far=far,
                tar=tar,
                frr=1.0 - tar,
                accuracy=float((accepted == labels).mean()),
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


def min_resolvable_far(impostor_pairs: int) -> float:
    """Правило трьох: з 0 помилок на n чужих парах можна стверджувати лише FAR < 3/n (95%)."""
    return 3.0 / max(int(impostor_pairs), 1)


def rates_at(scores: np.ndarray, genuine: np.ndarray, threshold: float) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    labels = _as_bool(genuine)
    found = np.isfinite(values)
    accepted = np.where(found, values, -np.inf) >= threshold
    return {
        "far": float(accepted[~labels].mean()),
        "frr": float((~accepted[labels]).mean()),
        "fmr": float(accepted[~labels & found].mean()) if (~labels & found).any() else float("nan"),
        "fnmr": float((~accepted[labels & found]).mean()) if (labels & found).any() else float("nan"),
        "fta": float((~found).mean()),
    }


def cross_validate(
    scores: np.ndarray,
    genuine: np.ndarray,
    folds: np.ndarray,
    far_target: float,
) -> dict[str, object]:
    """Поріг підбирається на інших фолдах і перевіряється на відкладеному — без підгонки під тест."""
    values = np.asarray(scores, dtype=np.float64)
    labels = _as_bool(genuine)
    fold_ids = np.asarray(folds)
    rows = []
    for fold in sorted(set(fold_ids.tolist())):
        test = fold_ids == fold
        train_points = recognition_curve(values[~test], labels[~test])
        point = operating_point(train_points, far_target)
        best = max(train_points, key=lambda item: (item.accuracy, -item.threshold))
        threshold = float("inf") if point is None else point.threshold
        rates = rates_at(values[test], labels[test], threshold)
        accepted = np.where(np.isfinite(values[test]), values[test], -np.inf) >= best.threshold
        rows.append(
            {
                "fold": int(fold),
                "threshold": threshold,
                "accuracy_threshold": best.threshold,
                "accuracy": float((accepted == labels[test]).mean()),
                **rates,
                "tar": 1.0 - rates["frr"],
            }
        )

    def stat(key: str) -> dict[str, float]:
        column = np.asarray([row[key] for row in rows], dtype=np.float64)
        return {"mean": float(np.nanmean(column)), "std": float(np.nanstd(column))}

    return {
        "far_target": far_target,
        "folds": rows,
        "accuracy": stat("accuracy"),
        "tar": stat("tar"),
        "far": stat("far"),
        "frr": stat("frr"),
        "fmr": stat("fmr"),
        "fnmr": stat("fnmr"),
        "threshold": stat("threshold"),
        "resolvable_far": min_resolvable_far(min(int((~labels & (fold_ids != fold)).sum()) for fold in set(fold_ids.tolist()))),
    }
