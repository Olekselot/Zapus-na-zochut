"""Пошук клієнта за косинусною схожістю векторів обличчя."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import EMBEDDING_DIM


@dataclass(frozen=True)
class Match:
    client_id: int | None
    code: str | None
    similarity: float | None
    accepted: bool


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.shape != (EMBEDDING_DIM,):
        raise ValueError(f"очікувався вектор з {EMBEDDING_DIM} чисел")
    if not np.isfinite(values).all():
        raise ValueError("вектор обличчя містить нескінченні значення")
    norm = float(np.linalg.norm(values))
    if norm < 1e-12:
        raise ValueError("порожній вектор обличчя")
    return values / np.float32(norm)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(l2_normalize(left), l2_normalize(right)))


def best_match(query: np.ndarray, entries: list[tuple[int, str, np.ndarray]], threshold: float) -> Match:
    """Оцінка клієнта — максимальний косинус серед усіх його збережених векторів.

    Приймаємо збіг, якщо ця оцінка не нижча за поріг. Поріг підбирається так,
    щоб чужих пропускало мало (орієнтир FAR 1% на LFW).
    """
    if not entries:
        return Match(None, None, None, False)

    query_vector = l2_normalize(query)
    grouped: dict[tuple[int, str], list[np.ndarray]] = {}
    for client_id, code, vector in entries:
        grouped.setdefault((client_id, code), []).append(vector)

    best_id: int | None = None
    best_code: str | None = None
    best_score = -2.0
    for (client_id, code), vectors in grouped.items():
        score = max(float(np.dot(query_vector, l2_normalize(vector))) for vector in vectors)
        if score > best_score or (score == best_score and (best_id is None or client_id < best_id)):
            best_score = score
            best_id = client_id
            best_code = code

    accepted = best_score >= threshold
    if not accepted:
        return Match(None, None, best_score, False)
    return Match(best_id, best_code, best_score, True)
