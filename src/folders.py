"""Читання папок live/print/phone/laptop і розбиття за людиною, не за кадром."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from src.config import FOLDER_TO_LABEL, IMAGE_EXTENSIONS


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    person: str


def collect_samples(root: Path) -> list[Sample]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"немає папки з даними: {root}")
    samples: list[Sample] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label = FOLDER_TO_LABEL.get(class_dir.name.lower())
        if label is None:
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = image_path.relative_to(class_dir)
            person = relative.parts[0] if len(relative.parts) > 1 else f"file:{image_path.stem}"
            samples.append(Sample(image_path, label, person))
    if not samples:
        raise FileNotFoundError(
            "у папці немає фото. Очікується data/custom/live/<p01>/*.jpg і так само print, phone, laptop"
        )
    return samples


def split_by_person(
    samples: list[Sample],
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Усі фото однієї людини потрапляють лише в одну частину: train, validation або test."""
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("частки train і validation мають лишати місце для test")
    people = sorted({sample.person for sample in samples})
    order = people[:]
    random.Random(seed).shuffle(order)
    count = len(order)
    if count >= 3:
        val_count = max(1, int(round(count * val_ratio)))
        test_count = max(1, int(round(count * (1 - train_ratio - val_ratio))))
        if val_count + test_count >= count:
            val_count, test_count = 1, 1
        train_ids = set(order[: count - val_count - test_count])
        val_ids = set(order[count - val_count - test_count : count - test_count])
        test_ids = set(order[count - test_count :])
    elif count == 2:
        train_ids, val_ids, test_ids = {order[0]}, {order[1]}, set()
    else:
        train_ids, val_ids, test_ids = set(order), set(), set()

    def take(allowed: set[str]) -> list[Sample]:
        return [sample for sample in samples if sample.person in allowed]

    return take(train_ids), take(val_ids), take(test_ids)


def balance_classes(samples: list[Sample], seed: int = 42) -> list[Sample]:
    """Вирівнює кількість фото класів усередині вже відокремленої частини."""
    by_label: dict[int, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)
    if len(by_label) < 2:
        return samples
    cap = min(len(group) for group in by_label.values())
    rng = random.Random(seed)
    chosen: list[Sample] = []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        chosen.extend(group[:cap])
    return chosen
