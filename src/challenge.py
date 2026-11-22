"""Активна перевірка живості: поворот голови в названий бік або кліпання.

Фото чи роздруківка такого не зроблять, а заздалегідь записане відео
не знає, який бік система попросить цього разу.
"""

from __future__ import annotations

import secrets

import numpy as np

from src.quality import yaw_proxy

CHALLENGES = {
    "left": "Поверніть голову ліворуч і поверніться назад",
    "right": "Поверніть голову праворуч і поверніться назад",
    "blink": "Дивіться в камеру і кліпніть два-три рази",
}
TURN_PEAK = 0.25
FRONTAL = 0.12
BLINK_DROP = 0.70
BLINK_RECOVER = 0.90


def random_challenge(with_blink: bool = True) -> str:
    options = list(CHALLENGES) if with_blink else ["left", "right"]
    return secrets.choice(options)


def check_turn(proxies: list[float], direction: str) -> tuple[bool, str]:
    """Спершу анфас, потім пік у потрібний бік, потім знову майже анфас."""
    values = np.asarray(proxies, dtype=np.float64)
    if len(values) < 4:
        return False, "Замало кадрів з обличчям."
    sign = 1.0 if direction == "left" else -1.0
    signed = values * sign
    if np.min(signed) <= -TURN_PEAK:
        return False, "Голову повернуто не в той бік."
    peak = int(np.argmax(signed))
    if signed[peak] < TURN_PEAK:
        return False, "Поворот замалий або його не було."
    if not np.any(np.abs(values[:peak]) <= FRONTAL):
        return False, "Перед поворотом треба дивитись прямо в камеру."
    if not np.any(np.abs(values[peak + 1 :]) <= FRONTAL * 1.5):
        return False, "Після повороту треба повернутись анфас."
    return True, "Поворот зараховано."


def eye_aspect_ratio(landmarks68: np.ndarray) -> float:
    """EAR за 68 точками iBUG: очі — точки 36–41 і 42–47."""
    points = np.asarray(landmarks68, dtype=np.float64)[:, :2]

    def one(eye: np.ndarray) -> float:
        vertical = np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])
        horizontal = np.linalg.norm(eye[0] - eye[3])
        return float(vertical / max(2.0 * horizontal, 1e-6))

    return (one(points[36:42]) + one(points[42:48])) / 2


def check_blink(ears: list[float]) -> tuple[bool, str]:
    """Хоча б одне відносне падіння EAR з поверненням до відкритих очей."""
    values = np.asarray(ears, dtype=np.float64)
    if len(values) < 5:
        return False, "Замало кадрів з обличчям."
    baseline = float(np.median(values))
    if baseline <= 0:
        return False, "Очі не знайдено."
    closed = np.where(values <= baseline * BLINK_DROP)[0]
    for index in closed:
        before = values[:index]
        after = values[index + 1 :]
        if np.any(before >= baseline * BLINK_RECOVER) and np.any(after >= baseline * BLINK_RECOVER):
            return True, "Кліпання зараховано."
    return False, "Кліпання не помітно."


def verify(samples: list, kind: str) -> tuple[bool, str]:
    if kind in {"left", "right"}:
        proxies = [yaw_proxy(sample.kps) for sample in samples if sample.kps is not None]
        return check_turn(proxies, kind)
    if kind == "blink":
        ears = [eye_aspect_ratio(sample.landmarks68) for sample in samples if sample.landmarks68 is not None]
        return check_blink(ears)
    return False, "Невідоме завдання."
