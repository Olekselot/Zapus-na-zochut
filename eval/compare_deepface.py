"""Порівняння ArcFace (insightface, як у демо) з Facenet512, VGG-Face і ArcFace з deepface.

Чесні умови: усі моделі бачать одне й те саме вирівняне обличчя від нашого детектора,
а різниться лише модель розпізнавання. Протокол той самий — 10 фолдів LFW з порогом
з інших фолдів. deepface приймає масив як BGR, тому кропи передаються саме так.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.lfw import embed_paths, lfw_home, load_pairs, pair_scores  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.metrics import cross_validate, equal_error_point, recognition_curve  # noqa: E402

MODELS = ("Facenet512", "VGG-Face", "ArcFace")


def aligned_crops(paths: list[Path], det_size: int) -> dict[str, np.ndarray | None]:
    from src.faces import FaceEngine
    from src.images import imread

    engine = FaceEngine(det_size=det_size, with_landmarks=False)
    crops = {}
    for index, path in enumerate(paths, start=1):
        found = engine.analyze(imread(path), pick="center")
        crops[str(path)] = None if found is None else found.aligned_bgr
        if index % 500 == 0:
            print(f"Кропів {index}/{len(paths)}", flush=True)
    return crops


def deepface_vectors(crops: dict[str, np.ndarray | None], model_name: str) -> dict[str, np.ndarray]:
    from deepface import DeepFace

    vectors = {}
    for key, crop in crops.items():
        if crop is None:
            vectors[key] = np.full(1, np.nan, dtype=np.float32)
            continue
        rows = DeepFace.represent(
            img_path=crop, model_name=model_name, detector_backend="skip", enforce_detection=False, align=False
        )
        vector = np.asarray(rows[0]["embedding"], dtype=np.float32)
        vectors[key] = vector / max(float(np.linalg.norm(vector)), 1e-12)
    return vectors


def summarize(name: str, scores: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> dict:
    points = recognition_curve(scores, labels)
    cv = cross_validate(scores, labels, folds, 0.01)
    return {
        "model": name,
        "pairs": int(len(scores)),
        "fta_pairs": int(np.isnan(scores).sum()),
        "accuracy": cv["accuracy"],
        "eer": equal_error_point(points).far,
        "tar_at_far_1": cv["tar"],
        "far_at_far_1": cv["far"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Порівняти моделі розпізнавання на LFW")
    parser.add_argument("--per-fold", type=int, default=None, help="Пар з кожного фолду для швидкого прогону")
    parser.add_argument("--det-size", type=int, default=640)
    args = parser.parse_args()
    home = lfw_home()
    pairs = load_pairs(home, args.per_fold)
    labels = np.asarray([same for _a, _b, same, _f in pairs])
    folds = np.asarray([fold for _a, _b, _s, fold in pairs])
    paths = sorted({path for first, second, _s, _f in pairs for path in (first, second)})

    ours = embed_paths(paths, det_size=args.det_size)
    reports = [summarize("insightface ArcFace (демо)", pair_scores(pairs, ours), labels, folds)]
    print(reports[-1], flush=True)
    crops = aligned_crops(paths, args.det_size)
    for model_name in MODELS:
        print(f"Модель {model_name}...", flush=True)
        vectors = deepface_vectors(crops, model_name)
        scores = []
        for first, second, _same, _fold in pairs:
            left, right = vectors[str(first)], vectors[str(second)]
            scores.append(float(np.dot(left, right)) if left.size > 1 and right.size > 1 else np.nan)
        reports.append(summarize(f"deepface {model_name}", np.asarray(scores), labels, folds))
        print(reports[-1], flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = RESULTS_DIR / "deepface_lfw.json"
    destination.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Звіт: {destination}")


if __name__ == "__main__":
    main()
