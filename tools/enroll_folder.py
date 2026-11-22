"""Масова реєстрація клієнтів демо з data/enroll і перевірка оплати їхніми фото «на оплату».

    python tools/enroll_folder.py                       # зареєструвати всіх у data/bank.sqlite
    python tools/enroll_folder.py --check --db tmp.sqlite  # окрема база: реєстрація + пробні оплати

Ім'я береться з назви папки: George_W_Bush -> ім'я «George W», прізвище «Bush».
Фото відомих людей — це не атаки, але й не наша камера, тому перевірка живості тут вимкнена.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DB_PATH, IMAGE_EXTENSIONS  # noqa: E402


def photos(folder: Path) -> list[Path]:
    return sorted(path for path in folder.glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def split_name(folder_name: str) -> tuple[str, str]:
    parts = [part for part in folder_name.replace("-", " ").split("_") if part]
    if len(parts) == 1:
        return parts[0], parts[0]
    return " ".join(parts[:-1]), parts[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Зареєструвати клієнтів з папки")
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "enroll")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--check", action="store_true", help="Після реєстрації спробувати оплату фото з pay/")
    args = parser.parse_args()

    from src.antispoof import AntispoofModel
    from src.bank import Bank
    from src.faces import FaceEngine
    from src.images import imread
    from src.pipeline import Pipeline

    pipeline = Pipeline(Bank(args.db), FaceEngine(), AntispoofModel())
    pipeline.antispoof_enabled = False
    people = sorted(folder for folder in args.root.iterdir() if folder.is_dir())
    for folder in people:
        enroll_dir = folder / "enroll" if (folder / "enroll").exists() else folder
        first, last = split_name(folder.name)
        images = [imread(path) for path in photos(enroll_dir)]
        decision, pending = pipeline.start_registration(first, last, images, args.balance)
        if decision.status == "ask" and pending and pending.get("can_attach"):
            decision = pipeline.confirm_existing(pending)
        print(f"{folder.name}: {decision.title} — {decision.message}")
    if not args.check:
        return
    total = correct = 0
    for folder in people:
        first, last = split_name(folder.name)
        expected = f"{last} {first}"
        for path in photos(folder / "pay"):
            decision, _preview = pipeline.pay([imread(path)], 1.0)
            total += 1
            hit = decision.status == "paid" and decision.client_code == expected
            correct += hit
            print(f"  {path.parent.parent.name}/{path.name}: {decision.status} {decision.client_code or ''} "
                  f"схожість {decision.similarity if decision.similarity is not None else '-'}")
    if total:
        print(f"Правильних оплат: {correct}/{total} ({correct / total:.0%}).")


if __name__ == "__main__":
    main()
