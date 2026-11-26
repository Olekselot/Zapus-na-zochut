"""Виміряти холодний старт і гарячу інференцію захисту від підробки."""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.antispoof import AntispoofModel


def percentile(values: list[float], percent: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percent))


def main() -> None:
    parser = argparse.ArgumentParser(description="p50/p95 затримки моделі антиспуфу")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()
    if args.rounds < 1 or args.batch_size < 1:
        raise SystemExit("--rounds і --batch-size мають бути більшими за нуль.")

    cold_started = time.perf_counter()
    model = AntispoofModel()
    if not model.warm_up():
        raise SystemExit("Немає models/antispoof.pt і/або models/antispoof_meta.json.")
    cold_seconds = time.perf_counter() - cold_started

    rng = np.random.default_rng(42)
    faces = [rng.integers(0, 255, (model.image_size, model.image_size, 3), dtype=np.uint8) for _ in range(args.batch_size)]
    warmups = min(5, args.rounds)
    for _ in range(warmups):
        model.predict_proba(faces)

    hot: list[float] = []
    for _ in range(args.rounds):
        started = time.perf_counter()
        model.predict_proba(faces)
        hot.append(time.perf_counter() - started)

    print(f"Холодний старт: {cold_seconds:.3f} с")
    print(f"Гаряча інференція, batch {args.batch_size}: p50 {1000 * percentile(hot, 50):.1f} мс, p95 {1000 * percentile(hot, 95):.1f} мс")


if __name__ == "__main__":
    main()
