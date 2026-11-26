"""APCER за типами атак, BPCER, BPCER при фіксованому APCER і DET-крива.

За замовчуванням рахує лише людей з test-частини, записаної в models/antispoof_meta.json
під час навчання. Для іншого набору (наприклад, навчали на CelebA-Spoof, перевіряємо
на своїй камері) — --split all: тоді беруться всі фото папки.
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

from src.antispoof import AntispoofModel  # noqa: E402
from src.config import ANTISPOOF_META, ANTISPOOF_WEIGHTS, CLASS_NAMES, LABEL_NAMES, RESULTS_DIR  # noqa: E402
from src.folders import collect_samples  # noqa: E402
from src.metrics import bpcer_at_apcer, det_curve, live_mask_from_probs, presentation_rates  # noqa: E402


def select(samples, split: str, data: Path, meta_path: Path = ANTISPOOF_META):
    if split == "all":
        return samples
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    people = set(meta.get("split", {}).get(split, []))
    if not people:
        raise SystemExit(f"У метаданих моделі немає частини {split}. Перенавчіть модель або візьміть --split all.")
    chosen = [sample for sample in samples if sample.person in people]
    if not chosen:
        raise SystemExit(
            f"У папці {data} немає людей з частини {split}. Якщо це інший набір, ніж для навчання, візьміть --split all."
        )
    return chosen


def save_det(points, path: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    apcer = np.asarray([point[1] for point in points])
    bpcer = np.asarray([point[2] for point in points])
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(np.clip(apcer, 1e-4, 1), np.clip(bpcer, 1e-4, 1))
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("APCER (найгірший тип атаки)")
    axis.set_ylabel("BPCER")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Метрики захисту від підробки")
    parser.add_argument("--data", type=Path, required=True, help="Папка live/print/phone/laptop/screen")
    parser.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--name", default=None, help="Ім'я файла звіту")
    parser.add_argument("--weights", type=Path, default=ANTISPOOF_WEIGHTS, help="Файл ваг моделі")
    parser.add_argument("--meta", type=Path, default=ANTISPOOF_META, help="Метадані цієї моделі")
    args = parser.parse_args()
    model = AntispoofModel(args.weights, args.meta)
    if not model.available:
        raise SystemExit(f"Немає ваг або метаданих моделі: {args.weights}, {args.meta}. Спочатку навчіть модель.")
    samples = select(collect_samples(args.data, tag=args.data.name), args.split, args.data, args.meta)
    if args.limit:
        samples = samples[: args.limit]

    from src.faces import FaceEngine
    from src.images import imread

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    engine = FaceEngine(det_size=int(meta.get("det_size", 640)), with_landmarks=False)
    faces, labels = [], []
    missed = np.zeros(len(LABEL_NAMES), dtype=int)
    for sample in samples:
        found = engine.analyze(imread(sample.path), crop_size=model.image_size, crop_scale=model.crop_scale)
        if found is None:
            missed[sample.label] += 1
            continue
        faces.append(found.spoof_input)
        labels.append(sample.label)
    if not faces:
        raise SystemExit("Детектор не знайшов жодного обличчя.")
    probabilities = np.concatenate([model.predict_proba(faces[start : start + 64]) for start in range(0, len(faces), 64)])
    label_array = np.asarray(labels)
    accepted = live_mask_from_probs(probabilities, model.threshold)
    rates = presentation_rates(label_array, accepted)
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((len(LABEL_NAMES), len(CLASS_NAMES)), dtype=int)
    for truth, guess in zip(label_array, predicted):
        confusion[truth, guess] += 1
    live_found = int((label_array == 0).sum())
    system_bpcer = (int((~accepted[label_array == 0]).sum()) + int(missed[0])) / max(live_found + int(missed[0]), 1)
    at_apcer = bpcer_at_apcer(probabilities, label_array)
    name = args.name or f"antispoof_{args.data.name}_{args.split}_{args.weights.stem}"
    print(f"Кадрів: {len(labels)} ({len({s.person for s in samples})} людей), без обличчя: {missed.tolist()}, поріг {model.threshold:.3f}.")
    print(f"APCER {rates.apcer:.3f} (найгірший) за типами: { {key: round(value, 3) for key, value in rates.apcer_by_type.items()} }.")
    print(f"BPCER {rates.bpcer:.3f}; з урахуванням кадрів без обличчя {system_bpcer:.3f}.")
    print(f"ACER {rates.acer:.3f} (лише для порівняння зі статтями).")
    print("BPCER при APCER ≤ 1/5/10 %:", at_apcer)
    print("Матриця помилок (рядки — правда live/print/screen/spoof, стовпці — прогноз live/print/screen):")
    print(confusion)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_det(det_curve(probabilities, label_array), RESULTS_DIR / f"{name}_det.png", f"DET: {args.data.name}, {args.split}")
    destination = RESULTS_DIR / f"{name}.json"
    destination.write_text(
        json.dumps(
            {
                "data": str(args.data),
                "weights": str(args.weights),
                "meta": str(args.meta),
                "model": {
                    "arch": meta.get("arch"),
                    "pretrained": meta.get("pretrained"),
                    "best_epoch": meta.get("best_epoch"),
                    "training_seconds": meta.get("training_seconds"),
                },
                "split": args.split,
                "people": len({sample.person for sample in samples}),
                "frames": len(labels),
                "missed_by_class": dict(zip(LABEL_NAMES, missed.tolist())),
                "threshold": model.threshold,
                "apcer": rates.apcer,
                "apcer_by_type": rates.apcer_by_type,
                "bpcer": rates.bpcer,
                "bpcer_with_fta": system_bpcer,
                "acer": rates.acer,
                "bpcer_at_apcer": at_apcer,
                "confusion": confusion.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Звіт: {destination}")


if __name__ == "__main__":
    main()
