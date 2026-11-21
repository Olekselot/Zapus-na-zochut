"""Порівняння Facenet512, VGG-Face і ArcFace з deepface на тих самих парах LFW."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.lfw import load_test_pairs, to_uint8_rgb  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.metrics import equal_error_point, operating_point, recognition_curve  # noqa: E402

MODELS = ("Facenet512", "VGG-Face", "ArcFace")


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    left = left / max(float(np.linalg.norm(left)), 1e-12)
    right = right / max(float(np.linalg.norm(right)), 1e-12)
    return float(np.dot(left, right))


def represent(image: np.ndarray, model_name: str) -> np.ndarray | None:
    from deepface import DeepFace

    rgb = to_uint8_rgb(image)
    try:
        rows = DeepFace.represent(
            img_path=rgb,
            model_name=model_name,
            enforce_detection=False,
            detector_backend="skip",
            align=False,
        )
    except Exception:
        rows = DeepFace.represent(
            img_path=rgb,
            model_name=model_name,
            enforce_detection=False,
            detector_backend="opencv",
        )
    if not rows:
        return None
    return np.asarray(rows[0]["embedding"], dtype=np.float32)


def evaluate_model(model_name: str, pairs: np.ndarray, genuine: np.ndarray) -> dict:
    scores = []
    labels = []
    missed = 0
    for left, right, same in zip(pairs[:, 0], pairs[:, 1], genuine):
        first = represent(left, model_name)
        second = represent(right, model_name)
        if first is None or second is None:
            missed += 1
            continue
        scores.append(cosine(first, second))
        labels.append(bool(same))
    if not scores or not any(labels) or all(labels):
        return {"model": model_name, "error": "замало пар", "missed": missed}
    points = recognition_curve(np.asarray(scores), np.asarray(labels))
    eer = equal_error_point(points)
    far1 = operating_point(points, 0.01)
    return {
        "model": model_name,
        "pairs": len(scores),
        "missed": missed,
        "eer": eer.far,
        "eer_threshold": eer.threshold,
        "tar_at_far_1": None if far1 is None else far1.tar,
        "frr_at_far_1": None if far1 is None else far1.frr,
        "threshold_at_far_1": None if far1 is None else far1.threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Порівняти моделі deepface на LFW")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--resize", type=float, default=1.0)
    args = parser.parse_args()
    pairs, genuine = load_test_pairs(max_pairs=args.max_pairs, resize=args.resize)
    reports = []
    for model_name in MODELS:
        print(f"Модель {model_name}...", flush=True)
        report = evaluate_model(model_name, pairs, genuine)
        reports.append(report)
        print(report)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = RESULTS_DIR / "deepface_lfw.json"
    destination.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Звіт: {destination}")


if __name__ == "__main__":
    main()
