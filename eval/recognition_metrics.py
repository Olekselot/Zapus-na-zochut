"""Метрики розпізнавання на LFW тим самим ArcFace, що й у демо.

1. Стандартний протокол: 6000 пар, 10 фолдів. Поріг підбирається на 9 фолдах,
   перевіряється на десятому. Пари, де обличчя не знайдено, рахуються відмовою (FAR/FRR)
   і окремо виключаються (FMR/FNMR).
2. Розширений набір: усі пари фото між собою. Люди діляться на дві половини, поріг
   підбирається на одній і перевіряється на іншій. Тут мільйони чужих пар, тож видно FAR 0.1% і 0.01%,
   чого 3000 чужих пар стандартного протоколу не розрізняють.
Поріг демо береться з розширеного набору за --far-target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from eval.lfw import embed_paths, identity, lfw_home, load_pairs, pair_scores  # noqa: E402
from src.config import RECOGNITION_THRESHOLD_PATH, RESULTS_DIR  # noqa: E402
from src.metrics import (  # noqa: E402
    cross_validate,
    equal_error_point,
    min_resolvable_far,
    operating_point,
    rates_at,
    recognition_curve,
)


def all_pairs(keys: list[str], vectors: np.ndarray, people: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Оцінки всіх пар (i < j) і мітки «та сама людина»."""
    similarity = vectors @ vectors.T
    upper = np.triu_indices(len(keys), k=1)
    return similarity[upper], (people[:, None] == people[None, :])[upper]


def threshold_at(genuine_scores: np.ndarray, impostor_scores: np.ndarray, far_target: float) -> float:
    """Найнижчий поріг, за якого частка чужих пар з оцінкою ≥ порогу не більша за ціль."""
    ordered = np.sort(impostor_scores)[::-1]
    allowed = int(np.floor(far_target * len(ordered)))
    if allowed >= len(ordered):
        return float(ordered[-1])
    return float(np.nextafter(ordered[allowed], np.inf))


def extended(vectors: dict[str, np.ndarray], targets: list[float], seed: int) -> dict:
    keys = sorted(key for key, value in vectors.items() if np.isfinite(value).all())
    names = np.asarray([identity(Path(key)) for key in keys])
    matrix = np.stack([vectors[key] for key in keys]).astype(np.float32)
    people = np.unique(names)
    rng = np.random.default_rng(seed)
    half = set(rng.permutation(people)[: len(people) // 2].tolist())
    in_a = np.asarray([name in half for name in names])
    halves = []
    for mask in (in_a, ~in_a):
        scores, same = all_pairs([key for key, keep in zip(keys, mask) if keep], matrix[mask], names[mask])
        halves.append((scores[same], scores[~same]))
    report: dict = {"images": len(keys), "people": int(len(people)), "targets": {}}
    for target in targets:
        rows = []
        for (fit_gen, fit_imp), (test_gen, test_imp) in ((halves[0], halves[1]), (halves[1], halves[0])):
            threshold = threshold_at(fit_gen, fit_imp, target)
            rows.append(
                {
                    "threshold": threshold,
                    "far": float((test_imp >= threshold).mean()),
                    "tar": float((test_gen >= threshold).mean()),
                    "genuine_pairs": int(len(test_gen)),
                    "impostor_pairs": int(len(test_imp)),
                }
            )
        report["targets"][f"{target:g}"] = {
            "halves": rows,
            "far": float(np.mean([row["far"] for row in rows])),
            "tar": float(np.mean([row["tar"] for row in rows])),
            "resolvable_far": min_resolvable_far(min(row["impostor_pairs"] for row in rows)),
        }
    all_scores, all_same = all_pairs(keys, matrix, names)
    report["deploy_thresholds"] = {f"{target:g}": threshold_at(all_scores[all_same], all_scores[~all_same], target) for target in targets}
    report["score_hist"] = {
        "genuine": np.histogram(all_scores[all_same], bins=50, range=(-0.2, 1.0))[0].tolist(),
        "impostor": np.histogram(all_scores[~all_same], bins=50, range=(-0.2, 1.0))[0].tolist(),
    }
    return report


def save_plots(scores, labels, ext: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    points = recognition_curve(scores, labels)
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot([max(point.far, 1e-5) for point in points], [point.tar for point in points])
    left.set_xscale("log")
    left.set_xlabel("FAR")
    left.set_ylabel("TAR")
    left.set_title("ROC, LFW 6000 пар")
    left.grid(True, which="both", alpha=0.3)
    edges = np.linspace(-0.2, 1.0, 51)
    centers = (edges[:-1] + edges[1:]) / 2
    for key, label in (("genuine", "свої"), ("impostor", "чужі")):
        counts = np.asarray(ext["score_hist"][key], dtype=np.float64)
        right.plot(centers, counts / max(counts.sum(), 1), label=label)
    right.set_xlabel("косинус")
    right.set_title(f"Розподіл оцінок, усі пари ({ext['images']} фото)")
    right.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Метрики розпізнавання на LFW")
    parser.add_argument("--per-fold", type=int, default=None, help="Скільки пар брати з кожного фолду для швидкої перевірки")
    parser.add_argument("--far-target", type=float, default=0.001, help="FAR, за яким обирається поріг демо")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-threshold", action="store_true", help="Записати поріг навіть після короткого прогону")
    args = parser.parse_args()

    home = lfw_home()
    pairs = load_pairs(home, args.per_fold)
    paths = sorted({path for first, second, _s, _f in pairs for path in (first, second)})
    print(f"Пар: {len(pairs)}, унікальних фото: {len(paths)}.")
    vectors = embed_paths(paths, det_size=args.det_size)
    scores = pair_scores(pairs, vectors)
    labels = np.asarray([same for _a, _b, same, _f in pairs])
    folds = np.asarray([fold for _a, _b, _s, fold in pairs])
    fta = int(np.isnan(scores).sum())

    points = recognition_curve(scores, labels)
    eer = equal_error_point(points)
    cv = {f"{target:g}": cross_validate(scores, labels, folds, target) for target in (0.01, 0.001)}
    ext = extended({str(path): vectors[str(path)] for path in paths}, [0.01, 0.001, 0.0001], args.seed)

    print(f"Пар без обличчя (FTA): {fta} з {len(pairs)}.")
    print(f"Accuracy (10 фолдів): {cv['0.01']['accuracy']['mean']:.4f} ± {cv['0.01']['accuracy']['std']:.4f}.")
    print(f"EER {eer.far:.4f} при порозі {eer.threshold:.3f}.")
    for key, row in cv.items():
        warn = " (нижче роздільності!)" if float(key) < row["resolvable_far"] else ""
        print(
            f"FAR {float(key):.2%} крос-валідацією{warn}: TAR {row['tar']['mean']:.4f} ± {row['tar']['std']:.4f}, "
            f"фактичний FAR {row['far']['mean']:.4f}, поріг {row['threshold']['mean']:.3f} ± {row['threshold']['std']:.3f}, "
            f"FNMR без FTA {row['fnmr']['mean']:.4f}."
        )
    print(f"Розширений набір: {ext['images']} фото, {ext['people']} людей.")
    for key, row in ext["targets"].items():
        print(
            f"  FAR {float(key):.2%}: TAR {row['tar']:.4f}, фактичний FAR {row['far']:.5f} "
            f"(роздільність {row['resolvable_far']:.1e}), поріг {row['halves'][0]['threshold']:.3f}/{row['halves'][1]['threshold']:.3f}."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "pairs": len(pairs),
        "fta_pairs": fta,
        "eer": {"threshold": eer.threshold, "far": eer.far, "frr": eer.frr},
        "cross_validation": cv,
        "extended": ext,
    }
    destination = RESULTS_DIR / "recognition_lfw.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_plots(scores, labels, ext, RESULTS_DIR / "recognition_lfw.png")
    print(f"Звіт: {destination}")

    target_key = f"{args.far_target:g}"
    threshold = ext["deploy_thresholds"].get(target_key)
    if threshold is None:
        threshold = operating_point(points, args.far_target).threshold
    if args.per_fold is None or args.save_threshold:
        check = rates_at(scores, labels, threshold)
        RECOGNITION_THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
        RECOGNITION_THRESHOLD_PATH.write_text(
            json.dumps(
                {
                    "threshold": threshold,
                    "far_target": args.far_target,
                    "source": "lfw-all-pairs",
                    "lfw_6000_at_threshold": check,
                    "extended": ext["targets"].get(target_key),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Поріг демо {threshold:.3f} (FAR {args.far_target:.2%}) записано: {RECOGNITION_THRESHOLD_PATH}")
    else:
        print("Поріг демо не чіпав: прогін короткий. Повний запуск без --per-fold запише поріг.")


if __name__ == "__main__":
    main()
