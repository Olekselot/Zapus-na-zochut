"""Читання папок live/print/phone/laptop/screen і розбиття за людиною, не за кадром."""

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


def collect_samples(root: Path, tag: str | None = None) -> list[Sample]:
    """tag додається до коду людини, щоб p01 з різних наборів не злились в одну людину."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"немає папки з даними: {root}")
    prefix = "" if not tag else f"{tag}:"
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
            samples.append(Sample(image_path, label, prefix + person))
    if not samples:
        raise FileNotFoundError(
            "у папці немає фото. Очікується data/custom/live/<p01>/*.jpg і так само print, phone, laptop"
        )
    return samples


def split_people(
    people: list[str],
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[set[str], set[str], set[str]]:
    """Частка test = 1 - train - validation; нульова частка дає порожню частину."""
    test_ratio = 1 - train_ratio - val_ratio
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < -1e-9:
        raise ValueError("частки train і validation не можуть перевищувати 1")
    order = sorted(set(people))
    random.Random(seed).shuffle(order)
    count = len(order)
    if count >= 3:
        val_count = max(1, int(round(count * val_ratio))) if val_ratio > 1e-9 else 0
        test_count = max(1, int(round(count * test_ratio))) if test_ratio > 1e-9 else 0
        if val_count + test_count >= count:
            val_count, test_count = min(1, val_count), min(1, test_count)
        train_ids = set(order[: count - val_count - test_count])
        val_ids = set(order[count - val_count - test_count : count - test_count])
        test_ids = set(order[count - test_count :])
    elif count == 2:
        train_ids, val_ids, test_ids = {order[0]}, {order[1]}, set()
    else:
        train_ids, val_ids, test_ids = set(order), set(), set()
    return train_ids, val_ids, test_ids


def take(samples: list[Sample], allowed: set[str]) -> list[Sample]:
    return [sample for sample in samples if sample.person in allowed]


def split_by_person(
    samples: list[Sample],
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Усі фото однієї людини потрапляють лише в одну частину: train, validation або test."""
    train_ids, val_ids, test_ids = split_people([sample.person for sample in samples], seed, train_ratio, val_ratio)
    return take(samples, train_ids), take(samples, val_ids), take(samples, test_ids)


def person_folds(samples: list[Sample], folds: int, seed: int = 42) -> list[set[str]]:
    """Люди, розкладені на folds груп. Для folds = кількості людей це leave-one-person-out."""
    people = sorted({sample.person for sample in samples})
    if folds < 2 or folds > len(people):
        raise ValueError(f"фолдів має бути від 2 до кількості людей ({len(people)})")
    random.Random(seed).shuffle(people)
    return [set(people[index::folds]) for index in range(folds)]


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


def stratified_limit(samples: list[Sample], limit: int, seed: int = 42) -> list[Sample]:
    """Перші limit фото після перемішування, з усіма класами. Раніше брались перші за алфавітом папок."""
    if limit <= 0 or limit >= len(samples):
        return samples
    by_label: dict[int, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)
    rng = random.Random(seed)
    for group in by_label.values():
        rng.shuffle(group)
    chosen: list[Sample] = []
    while len(chosen) < limit:
        for label in sorted(by_label):
            if by_label[label] and len(chosen) < limit:
                chosen.append(by_label[label].pop())
    return chosen


def cap_per_person(samples: list[Sample], cap: int, seed: int = 42) -> list[Sample]:
    """Не більше cap фото на пару (людина, клас): краще більше різних людей, ніж сотні однакових кадрів."""
    if cap <= 0:
        return samples
    groups: dict[tuple[str, int], list[Sample]] = {}
    for sample in samples:
        groups.setdefault((sample.person, sample.label), []).append(sample)
    rng = random.Random(seed)
    chosen: list[Sample] = []
    for key in sorted(groups):
        group = groups[key][:]
        rng.shuffle(group)
        chosen.extend(group[:cap])
    return chosen
