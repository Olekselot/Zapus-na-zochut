"""Стан набору даних: скільки фото в кожному класі й у кожної людини, чого бракує, дублікати.

    python tools/dataset_report.py data/custom
    python tools/dataset_report.py data/celeba_spoof data/celeba_spoof_test --no-duplicates
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.config import LABEL_NAMES, RESULTS_DIR  # noqa: E402
from src.folders import collect_samples  # noqa: E402

MIN_PEOPLE = 8
MIN_PER_CELL = 15


def dhash(path: Path) -> int | None:
    import cv2

    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if bit else "0" for bit in bits), 2)


def report(root: Path, duplicates: bool) -> list[str]:
    samples = collect_samples(root)
    lines = [f"## {root}", ""]
    by_label = Counter(sample.label for sample in samples)
    lines.append("| клас | фото |")
    lines.append("|---|---|")
    for label in sorted(by_label):
        lines.append(f"| {LABEL_NAMES[label]} | {by_label[label]} |")
    lines.append("")
    loose = [sample for sample in samples if sample.person.startswith("file:")]
    people = sorted({sample.person for sample in samples if not sample.person.startswith("file:")})
    warnings = []
    if loose:
        warnings.append(f"{len(loose)} фото лежать без папки людини: розбиття за людьми для них неможливе.")
    if people:
        cells: dict[str, Counter] = defaultdict(Counter)
        for sample in samples:
            if not sample.person.startswith("file:"):
                cells[sample.person][sample.label] += 1
        present = sorted(by_label)
        lines.append("| людина | " + " | ".join(LABEL_NAMES[label] for label in present) + " |")
        lines.append("|---|" + "---|" * len(present))
        for person in people:
            lines.append(f"| {person} | " + " | ".join(str(cells[person][label]) for label in present) + " |")
            missing = [LABEL_NAMES[label] for label in present if cells[person][label] == 0]
            thin = [LABEL_NAMES[label] for label in present if 0 < cells[person][label] < MIN_PER_CELL]
            if missing:
                warnings.append(f"{person}: немає {', '.join(missing)}.")
            if thin:
                warnings.append(f"{person}: менше {MIN_PER_CELL} фото в {', '.join(thin)}.")
        lines.append("")
        if len(people) < MIN_PEOPLE:
            warnings.append(f"Людей {len(people)} — менше {MIN_PEOPLE}, метрики будуть шумні. Потрібно ще {MIN_PEOPLE - len(people)}.")
    counts = [count for count in by_label.values()]
    if len(counts) > 1 and max(counts) > 2 * min(counts):
        warnings.append("Класи незбалансовані більш ніж удвічі: навчайте з --balance.")
    if duplicates:
        groups: dict[tuple[str, int], list[tuple[int, Path]]] = defaultdict(list)
        unreadable = 0
        for sample in samples:
            value = dhash(sample.path)
            if value is None:
                unreadable += 1
                continue
            groups[(sample.person, sample.label)].append((value, sample.path))
        near = 0
        for items in groups.values():
            seen: list[int] = []
            for value, _path in items:
                if any(bin(value ^ other).count("1") <= 2 for other in seen):
                    near += 1
                else:
                    seen.append(value)
        if unreadable:
            warnings.append(f"Не читаються: {unreadable} фото.")
        if near:
            warnings.append(f"Майже однакових кадрів: {near} ({near / len(samples):.0%}). Краще різні ракурси й умови, ніж серії однакових.")
    lines.append("**Зауваги:** " + ("немає." if not warnings else ""))
    lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Звіт про набір даних")
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--no-duplicates", action="store_true", help="Не шукати майже однакові кадри (швидше)")
    args = parser.parse_args()
    lines = ["# Стан даних", ""]
    for root in args.roots:
        lines.extend(report(root, not args.no_duplicates))
    text = "\n".join(lines)
    print(text)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = RESULTS_DIR / "dataset_report.md"
    destination.write_text(text, encoding="utf-8")
    print(f"\nЗвіт: {destination}")


if __name__ == "__main__":
    main()
