"""APCER, BPCER і ACER на папці з живими обличчями і підробками."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.antispoof import AntispoofModel  # noqa: E402
from src.config import CLASS_NAMES, RESULTS_DIR  # noqa: E402
from src.folders import collect_samples  # noqa: E402
from src.metrics import live_mask_from_probs, presentation_rates  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Метрики захисту від підробки")
    parser.add_argument("--data", type=Path, required=True, help="Папка live/print/phone/laptop")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--name", default="antispoof", help="Ім'я файла звіту")
    args = parser.parse_args()
    model = AntispoofModel()
    if not model.available:
        raise SystemExit("Немає models/antispoof.pt. Спочатку навчіть модель.")
    samples = collect_samples(args.data)
    if args.limit:
        samples = samples[: args.limit]

    import numpy as np

    from src.faces import FaceEngine
    from src.images import imread

    engine = FaceEngine()
    faces = []
    labels = []
    missed = 0
    for sample in samples:
        found = engine.analyze(imread(sample.path), crop_size=model.image_size)
        if found is None:
            missed += 1
            continue
        faces.append(found.aligned_bgr)
        labels.append(sample.label)
    if not faces:
        raise SystemExit("Детектор не знайшов жодного обличчя.")
    probabilities = model.predict_proba(faces)
    label_array = np.asarray(labels)
    accepted = live_mask_from_probs(probabilities, model.threshold)
    rates = presentation_rates(label_array == 0, accepted)
    predicted = probabilities.argmax(axis=1)
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        mask = label_array == index
        per_class[name] = None if not mask.any() else float(np.mean(predicted[mask] == index))
    print(f"Кадрів: {len(labels)}, без обличчя: {missed}, поріг {model.threshold:.3f}.")
    print(f"APCER {rates.apcer:.3f} на {rates.attacks} атаках.")
    print(f"BPCER {rates.bpcer:.3f} на {rates.bona_fide} живих.")
    print(f"ACER {rates.acer:.3f}.")
    print("Частка вгаданого класу:", {key: None if value is None else round(value, 3) for key, value in per_class.items()})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = RESULTS_DIR / f"{args.name}.json"
    destination.write_text(
        json.dumps(
            {
                "data": str(args.data),
                "frames": len(labels),
                "missed": missed,
                "threshold": model.threshold,
                "apcer": rates.apcer,
                "bpcer": rates.bpcer,
                "acer": rates.acer,
                "attacks": rates.attacks,
                "bona_fide": rates.bona_fide,
                "class_recall": per_class,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Звіт: {destination}")


if __name__ == "__main__":
    main()
