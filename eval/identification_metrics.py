"""1:N — банк шукає обличчя серед усіх клієнтів, а не порівнює одну пару.

На векторах LFW з кешу (спершу запустіть recognition_metrics.py) імітуємо галерею з G клієнтів:
кожен зареєстрований за кількома фото, решта його фото — спроби оплати, а люди поза галереєю — чужі.
Порівнюємо три правила: максимум по фото (як було), шаблон клієнта, шаблон + відрив від другого.
FPIR — частка чужих, яких прийняли за когось із клієнтів; FNIR — частка своїх, яких не прийняли
або прийняли за іншого; wrong_id — своїх прийняли, але за іншого клієнта (гроші не з того рахунку).
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

from eval.lfw import identity  # noqa: E402
from src.config import MATCH_MARGIN, RESULTS_DIR, load_recognition_threshold  # noqa: E402

MODES = ("max", "template", "template+margin")


def decide(scores: np.ndarray, threshold: float, margin: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1)
    best = order[:, 0]
    top = scores[np.arange(len(scores)), best]
    second = scores[np.arange(len(scores)), order[:, 1]] if scores.shape[1] > 1 else np.full(len(scores), -np.inf)
    return best, (top >= threshold) & (top - second >= margin)


def trial(vectors, owners, by_person, gallery_people, outsiders, enroll, threshold, margin, rng):
    enrolled, owner_index, probes, probe_owner = [], [], [], []
    for index, person in enumerate(gallery_people):
        rows = rng.permutation(by_person[person])
        enrolled.extend(rows[:enroll])
        owner_index.extend([index] * enroll)
        probes.extend(rows[enroll:])
        probe_owner.extend([index] * (len(rows) - enroll))
    impostors = [rng.choice(by_person[person]) for person in outsiders]
    gallery = vectors[enrolled]
    owner_index = np.asarray(owner_index)
    templates = np.stack([gallery[owner_index == index].mean(axis=0) for index in range(len(gallery_people))])
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)
    result = {}
    for mode in MODES:
        def score(rows):
            query = vectors[rows]
            if mode == "max":
                every = query @ gallery.T
                return np.stack([every[:, owner_index == index].max(axis=1) for index in range(len(gallery_people))], axis=1)
            return query @ templates.T

        used_margin = margin if mode == "template+margin" else 0.0
        best, accepted = decide(score(np.asarray(probes)), threshold, used_margin)
        truth = np.asarray(probe_owner)
        _best_imp, accepted_imp = decide(score(np.asarray(impostors)), threshold, used_margin)
        result[mode] = {
            "fnir": float(np.mean(~(accepted & (best == truth)))),
            "wrong_id": float(np.mean(accepted & (best != truth))),
            "fpir": float(np.mean(accepted_imp)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ідентифікація 1:N на LFW")
    parser.add_argument("--cache", type=Path, default=RESULTS_DIR / "lfw_embeddings_d640.npz")
    parser.add_argument("--threshold", type=float, default=None, help="За замовчуванням — поріг демо")
    parser.add_argument("--margin", type=float, default=MATCH_MARGIN)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--enroll", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.cache.exists():
        raise SystemExit("Немає кешу векторів LFW. Спершу запустіть eval/recognition_metrics.py.")
    stored = np.load(args.cache, allow_pickle=False)
    keep = np.isfinite(stored["vectors"]).all(axis=1)
    vectors = stored["vectors"][keep].astype(np.float32)
    owners = np.asarray([identity(Path(key)) for key in stored["keys"][keep].tolist()])
    by_person: dict[str, np.ndarray] = {}
    for person in np.unique(owners):
        by_person[person] = np.where(owners == person)[0]
    eligible = sorted(person for person, rows in by_person.items() if len(rows) > args.enroll)
    threshold = load_recognition_threshold() if args.threshold is None else args.threshold
    rng = np.random.default_rng(args.seed)
    print(f"Фото {len(vectors)}, людей {len(by_person)}, придатних для галереї {len(eligible)}. Поріг {threshold:.3f}, відрив {args.margin}.")
    report = {"threshold": threshold, "margin": args.margin, "enroll": args.enroll, "sizes": {}}
    for size in args.sizes:
        if size > len(eligible):
            print(f"Галерея {size}: замало людей, пропускаю.")
            continue
        runs = []
        for _ in range(args.trials):
            gallery_people = list(rng.choice(eligible, size=size, replace=False))
            outsiders = [person for person in by_person if person not in set(gallery_people)]
            outsiders = list(rng.choice(outsiders, size=min(1000, len(outsiders)), replace=False))
            runs.append(trial(vectors, owners, by_person, gallery_people, outsiders, args.enroll, threshold, args.margin, rng))
        summary = {mode: {key: float(np.mean([run[mode][key] for run in runs])) for key in runs[0][mode]} for mode in MODES}
        report["sizes"][str(size)] = summary
        print(f"Галерея {size} клієнтів ({args.trials} повторів):")
        for mode, values in summary.items():
            print(f"  {mode:16s} FPIR {values['fpir']:.4f}  FNIR {values['fnir']:.4f}  wrong_id {values['wrong_id']:.4f}")
    destination = RESULTS_DIR / "identification_lfw.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Звіт: {destination}")


if __name__ == "__main__":
    main()
