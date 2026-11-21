"""Навчання MobileNetV2 (Topologies) відрізняти живе обличчя від друку й екрана."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_fcntl_shim() -> None:
    """topologies.data імпортує Unix-модуль fcntl навіть для звичайного навчання.

    На Windows його немає. Для рецепта vision-classify блокування файлів не потрібне,
    тож підставляємо порожній модуль до імпорту Trainer.
    """
    if sys.platform != "win32" or "fcntl" in sys.modules:
        return
    import types

    module = types.ModuleType("fcntl")
    module.LOCK_EX = 2
    module.LOCK_SH = 1
    module.LOCK_UN = 8
    module.LOCK_NB = 4
    module.flock = lambda *_args, **_kwargs: None
    sys.modules["fcntl"] = module


_install_fcntl_shim()

from src.config import (  # noqa: E402
    ANTISPOOF_META,
    ANTISPOOF_WEIGHTS,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODELS_DIR,
)
from src.folders import Sample, balance_classes, collect_samples, split_by_person  # noqa: E402
from src.metrics import choose_live_threshold, presentation_rates, live_mask_from_probs  # noqa: E402


def run_smoke() -> None:
    import torch
    from topologies import Trainer
    from topologies.models.mobilenet_v2 import MobileNetV2

    class TinyImages:
        def __len__(self) -> int:
            return 16

        def __getitem__(self, index: int) -> dict:
            return {
                "pixels": torch.rand(3, 64, 64),
                "labels": torch.tensor(index % 3, dtype=torch.long),
            }

    torch.manual_seed(0)
    model = MobileNetV2(num_classes=3, width_multiplier=0.25)
    trainer = Trainer(
        model,
        TinyImages(),
        recipe="vision-classify",
        max_epochs=1,
        batch_size=4,
        lr=1e-3,
        device="cpu",
        monitor="none",
    )
    trainer.fit()
    last = trainer.history[-1]
    print("Smoke-навчання пройшло. Останній крок:", {key: last[key] for key in sorted(last)})


def _align(samples: list[Sample], image_size: int, cache_dir: Path) -> tuple[list[tuple[Path, int]], int]:
    import cv2

    from src.faces import FaceEngine
    from src.images import imread

    cache_dir.mkdir(parents=True, exist_ok=True)
    engine = FaceEngine()
    ready: list[tuple[Path, int]] = []
    missed = 0
    for index, sample in enumerate(samples, start=1):
        stamp = sample.path.stat().st_mtime_ns
        digest = f"{sample.path.resolve()}|{stamp}|{image_size}"
        import hashlib

        name = hashlib.sha1(digest.encode("utf-8")).hexdigest()
        crop_path = cache_dir / f"{name}.jpg"
        fail_path = cache_dir / f"{name}.fail"
        if fail_path.exists():
            missed += 1
            continue
        if not crop_path.exists():
            found = engine.analyze(imread(sample.path), crop_size=image_size)
            if found is None:
                fail_path.write_text("no-face", encoding="utf-8")
                missed += 1
                continue
            ok, encoded = cv2.imencode(".jpg", found.aligned_bgr)
            if not ok:
                missed += 1
                continue
            crop_path.write_bytes(encoded.tobytes())
        ready.append((crop_path, sample.label))
        if index % 25 == 0 or index == len(samples):
            print(f"Вирівняно {index}/{len(samples)}, без обличчя {missed}", flush=True)
    return ready, missed


class CropDataset:
    def __init__(self, items: list[tuple[Path, int]], image_size: int, augment: bool) -> None:
        self.items = items
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        import torch

        from src.images import face_tensor, imread

        path, label = self.items[index]
        tensor = torch.from_numpy(face_tensor(imread(path), self.image_size))
        if self.augment and random.random() < 0.5:
            tensor = torch.flip(tensor, dims=[2])
        return {"pixels": tensor, "labels": torch.tensor(label, dtype=torch.long)}


def _probabilities(model, dataset, batch_size: int, device: str):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    chunks = []
    labels = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixels"].to(device)
            logits = model(pixels)
            chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
            labels.append(batch["labels"].numpy())
    return np.concatenate(chunks), np.concatenate(labels)


def _plain_history(history) -> list[dict]:
    rows = []
    for row in history:
        plain = {}
        for key, value in row.items():
            try:
                plain[key] = float(value)
            except (TypeError, ValueError):
                plain[key] = value
        rows.append(plain)
    return rows


def train(args: argparse.Namespace) -> None:
    import torch
    from topologies import Trainer
    from topologies.models.mobilenet_v2 import MobileNetV2

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples = collect_samples(Path(args.data))
    loose = [sample.person for sample in samples if sample.person.startswith("file:")]
    if loose:
        print(
            "Увага: частина фото лежить просто в папці класу, без папки людини. "
            "Розбиття тоді майже по кадрах і може зависити метрики. Краще live/p01/фото.jpg."
        )
    train_samples, val_samples, test_samples = split_by_person(samples, seed=args.seed)
    if args.balance:
        train_samples = balance_classes(train_samples, seed=args.seed)
    if args.limit:
        train_samples = train_samples[: args.limit]
        val_samples = val_samples[: max(1, args.limit // 5)]
    print(
        f"Фото: train {len(train_samples)}, validation {len(val_samples)}, test {len(test_samples)}. "
        "Test цей скрипт не чіпає: його рахує eval/antispoof_metrics.py."
    )
    cache = ROOT / "data" / "cache" / "antispoof"
    train_items, missed_train = _align(train_samples, args.image_size, cache)
    val_items, missed_val = _align(val_samples, args.image_size, cache)
    print(f"Після детектора: train {len(train_items)} (пропущено {missed_train}), validation {len(val_items)} (пропущено {missed_val}).")
    if len(train_items) < 6:
        raise SystemExit("Замало фото з обличчям для навчання. Перевірте камеру і структуру папок.")

    train_ds = CropDataset(train_items, args.image_size, augment=True)
    val_ds = CropDataset(val_items, args.image_size, augment=False) if val_items else None
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MobileNetV2(num_classes=3, width_multiplier=args.width, dropout=0.1)
    trainer_args = [
        model,
        train_ds,
    ]
    if val_ds is not None:
        trainer_args.append(val_ds)
    trainer = Trainer(
        *trainer_args,
        recipe="vision-classify",
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        monitor="plain",
        num_workers=0,
    )
    trainer.fit()

    threshold = 0.8
    threshold_split = "default"
    if val_ds is not None and len({label for _, label in val_items}) > 1 and any(label == 0 for _, label in val_items):
        probabilities, labels = _probabilities(model, val_ds, args.batch_size, device)
        is_live = labels == 0
        if (~is_live).any() and is_live.any():
            threshold = choose_live_threshold(probabilities, is_live, apcer_target=args.apcer)
            rates = presentation_rates(is_live, live_mask_from_probs(probabilities, threshold))
            threshold_split = "validation"
            print(
                f"Validation: APCER {rates.apcer:.3f}, BPCER {rates.bpcer:.3f}, "
                f"ACER {rates.acer:.3f}, поріг {threshold:.3f}."
            )
    else:
        print("Validation без обох класів. Лишаю консервативний поріг 0.80 і пишу це в метадані.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights) if args.weights else ANTISPOOF_WEIGHTS
    meta_path = Path(args.meta) if args.meta else ANTISPOOF_META
    torch.save(model.state_dict(), weights)
    meta = {
        "width_multiplier": args.width,
        "image_size": args.image_size,
        "threshold": threshold,
        "threshold_split": threshold_split,
        "apcer_target": args.apcer,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "class_names": list(CLASS_NAMES),
        "history": _plain_history(trainer.history),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Збережено ваги {weights}")
    print(f"Збережено опис {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Навчити захист від підробки обличчя")
    parser.add_argument("--data", type=Path, help="Папка live/print/phone/laptop")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--width", type=float, default=0.5, help="width_multiplier MobileNetV2, 0.5 = пресет small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apcer", type=float, default=0.05, help="Ціль APCER на validation")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=0, help="Обрізати train для короткої перевірки")
    parser.add_argument("--balance", action="store_true", help="Зрівняти кількість фото класів у train")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Одна епоха на випадкових тензорах, без датасету і без збереження")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.smoke:
        run_smoke()
    else:
        if arguments.data is None:
            raise SystemExit("Потрібен шлях --data або прапорець --smoke.")
        train(arguments)
