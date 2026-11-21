"""FAR, FRR, EER і TAR при заданому FAR на парах LFW тим самим ArcFace, що й у демо."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.lfw import load_test_pairs, to_bgr  # noqa: E402
from src.config import RECOGNITION_THRESHOLD_PATH, RESULTS_DIR  # noqa: E402
from src.faces import FaceEngine  # noqa: E402
from src.gallery import cosine  # noqa: E402
from src.metrics import equal_error_point, operating_point, recognition_curve  # noqa: E402


def embed_pairs(max_pairs: int | None, resize: float | None) -> tuple[list[float], list[bool], int]:
    pairs, genuine = load_test_pairs(max_pairs=max_pairs, resize=resize)
    engine = FaceEngine()
    scores: list[float] = []
    labels: list[bool] = []
    missed = 0
    for index, (left, right) in enumerate(pairs, start=1):
        first = engine.analyze(to_bgr(left))
        second = engine.analyze(to_bgr(right))
        if first is None or second is None:
            missed += 1
            continue
        scores.append(cosine(first.embedding, second.embedding))
        labels.append(bool(genuine[index - 1]))
        if index % 50 == 0 or index == len(pairs):
            print(f"Пар оброблено {index}/{len(pairs)}, без обличчя {missed}", flush=True)
    return scores, labels, missed


def main() -> None:
    parser = argparse.ArgumentParser(description="Метрики розпізнавання на LFW")
    parser.add_argument("--max-pairs", type=int, default=None, help="Скільки пар узяти з початку тестового списку")
    parser.add_argument("--resize", type=float, default=1.0)
    parser.add_argument(
        "--save-threshold",
        action="store_true",
        help="Перезаписати поріг демо. Без --max-pairs це робиться само, бо тоді прогін повний.",
    )
    args = parser.parse_args()
    scores, labels, missed = embed_pairs(args.max_pairs, args.resize)
    if not any(labels) or not any(not label for label in labels):
        raise SystemExit("Не вийшло зібрати і свої, і чужі пари. Спробуйте більший resize або більше пар.")
    import numpy as np

    points = recognition_curve(np.asarray(scores), np.asarray(labels))
    eer = equal_error_point(points)
    far1 = operating_point(points, 0.01)
    far01 = operating_point(points, 0.001)
    accuracy_point = max(points, key=lambda point: (point.accuracy, -point.threshold))
    print(f"Пар із двома обличчями: {len(scores)}. Пропущено: {missed}.")
    print(f"Найкраща accuracy {accuracy_point.accuracy:.3f} при порозі {accuracy_point.threshold:.3f}.")
    print(f"EER {eer.far:.3f}/{eer.frr:.3f} при порозі {eer.threshold:.3f}.")
    for title, point in (("FAR 1%", far1), ("FAR 0.1%", far01)):
        if point is None:
            print(f"{title}: на цих парах такого порогу немає.")
        else:
            print(
                f"{title}: поріг {point.threshold:.3f}, FAR {point.far:.4f}, "
                f"TAR {point.tar:.3f}, FRR {point.frr:.3f}."
            )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "pairs": len(scores),
        "missed": missed,
        "eer": {"threshold": eer.threshold, "far": eer.far, "frr": eer.frr},
        "far_1_percent": None if far1 is None else far1.__dict__,
        "far_0_1_percent": None if far01 is None else far01.__dict__,
        "best_accuracy": accuracy_point.__dict__,
    }
    destination = RESULTS_DIR / "recognition_lfw.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Звіт: {destination}")
    save = args.save_threshold or args.max_pairs is None
    if save and far1 is not None:
        RECOGNITION_THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
        RECOGNITION_THRESHOLD_PATH.write_text(
            json.dumps(
                {
                    "threshold": far1.threshold,
                    "far_target": 0.01,
                    "far": far1.far,
                    "tar": far1.tar,
                    "source": "lfw-test",
                    "pairs": len(scores),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Поріг демо оновлено: {RECOGNITION_THRESHOLD_PATH}")
    elif args.max_pairs is not None:
        print("Поріг демо не чіпав: прогін короткий. Повний запуск без --max-pairs запише поріг FAR 1%.")


if __name__ == "__main__":
    main()
