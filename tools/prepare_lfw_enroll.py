"""Клієнти демо з LFW: кілька відомих людей з багатьма фото, розкладені на реєстрацію і оплату.

    python tools/prepare_lfw_enroll.py --people 6
Результат: data/enroll/<Ім'я_Прізвище>/enroll/*.jpg і data/enroll/<Ім'я_Прізвище>/pay/*.jpg.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.lfw import lfw_home  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Клієнти демо з LFW")
    parser.add_argument("--people", type=int, default=6)
    parser.add_argument("--enroll", type=int, default=8)
    parser.add_argument("--pay", type=int, default=4)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "enroll")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    folders = sorted((lfw_home() / "lfw_funneled").iterdir())
    rich = [folder for folder in folders if folder.is_dir() and len(list(folder.glob("*.jpg"))) >= args.enroll + args.pay]
    rng = random.Random(args.seed)
    chosen = rng.sample(rich, min(args.people, len(rich)))
    for folder in chosen:
        photos = sorted(folder.glob("*.jpg"))
        rng.shuffle(photos)
        for part, items in (("enroll", photos[: args.enroll]), ("pay", photos[args.enroll : args.enroll + args.pay])):
            destination = args.out / folder.name / part
            destination.mkdir(parents=True, exist_ok=True)
            for photo in items:
                shutil.copy2(photo, destination / photo.name)
        print(f"{folder.name}: {args.enroll} на реєстрацію, {args.pay} на оплату")
    print(f"Готово: {args.out}")


if __name__ == "__main__":
    main()
