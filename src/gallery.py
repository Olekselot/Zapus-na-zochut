"""Пошук клієнта: косинус до шаблону клієнта і відрив від другого кандидата."""

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
    second_similarity: float | None = None
    reason: str | None = None


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


def template(vectors: list[np.ndarray]) -> np.ndarray:
    """Шаблон клієнта — нормований середній вектор усіх його фото."""
    if not vectors:
        raise ValueError("немає векторів для шаблону")
    stacked = np.stack([l2_normalize(vector) for vector in vectors], axis=0)
    return l2_normalize(stacked.mean(axis=0))


def client_templates(entries: list[tuple[int, str, np.ndarray]]) -> list[tuple[int, str, np.ndarray]]:
    grouped: dict[tuple[int, str], list[np.ndarray]] = {}
    for client_id, code, vector in entries:
        grouped.setdefault((client_id, code), []).append(vector)
    return [(client_id, code, template(vectors)) for (client_id, code), vectors in grouped.items()]


def best_match(
    query: np.ndarray,
    entries: list[tuple[int, str, np.ndarray]],
    threshold: float,
    margin: float = 0.0,
) -> Match:
    """Оцінка клієнта — косинус до його шаблону, а не максимум по всіх фото.

    Максимум по багатьох фото дає чужому більше шансів випадково пройти поріг,
    підібраний на парах «фото проти фото». Шаблон цього не робить.
    Збіг приймаємо, якщо оцінка не нижча за поріг і найкращий клієнт
    відривається від другого хоча б на margin.
    """
    if not entries:
        return Match(None, None, None, False, reason="empty")
    query_vector = l2_normalize(query)
    scored = sorted(
        ((float(np.dot(query_vector, vector)), client_id, code) for client_id, code, vector in client_templates(entries)),
        key=lambda item: (-item[0], item[1]),
    )
    best_score, best_id, best_code = scored[0]
    second = scored[1][0] if len(scored) > 1 else None
    if best_score < threshold:
        return Match(None, None, best_score, False, second, "below_threshold")
    if second is not None and best_score - second < margin:
        return Match(None, None, best_score, False, second, "ambiguous")
    return Match(best_id, best_code, best_score, True, second, None)


def consistent(vectors: list[np.ndarray], threshold: float) -> tuple[list[int], np.ndarray]:
    """Індекси векторів, що схожі на медоїд групи не менше за threshold, і шаблон із них.

    Медоїд — вектор, найближчий у середньому до всіх інших. Так одне чуже фото
    не тягне шаблон у свій бік.
    """
    normalized = np.stack([l2_normalize(vector) for vector in vectors], axis=0)
    similarity = normalized @ normalized.T
    medoid = int(np.argmax(similarity.sum(axis=1)))
    keep = [index for index in range(len(vectors)) if similarity[medoid, index] >= threshold]
    return keep, template([normalized[index] for index in keep])
