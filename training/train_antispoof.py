"""Навчання захисту від підробки: живе / друк / екран.

За замовчуванням — MobileNetV3-Small з вагами ImageNet: спершу вчиться лише голова,
потім уся мережа. Для порівняння можна взяти MobileNetV2 з Topologies з нуля (--arch topologies_mnv2).
Люди в train, validation і test не перетинаються. Поріг і температура підбираються на validation,
підсумкові метрики рахуються на test, якого навчання не бачило.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.antispoof import ARCHES  # noqa: E402
from src.config import (  # noqa: E402
    ANTISPOOF_META,
    ANTISPOOF_WEIGHTS,
    CACHE_DIR,
    CLASS_NAMES,
    DEFAULT_CROP_SCALE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_NAMES,
    MODELS_DIR,
    RESULTS_DIR,
    SPOOF_UNKNOWN,
)
from src.folders import (  # noqa: E402
    Sample,
    cap_per_person,
    collect_samples,
    person_folds,
    split_people,
    stratified_limit,
)
from src.metrics import bpcer_at_apcer, choose_live_threshold, live_mask_from_probs, presentation_rates  # noqa: E402


def load_samples(roots: list[Path]) -> list[Sample]:
    samples: list[Sample] = []
    for root in roots:
        samples.extend(collect_samples(root, tag=Path(root).name))
    return samples


def crop_samples(
    samples: list[Sample],
    image_size: int,
    crop_scale: float | None,
    det_size: int,
) -> tuple[list[tuple[Path, int, str]], int]:
    """Вирізає обличчя тим самим детектором, що й під час оплати, і кешує кропи на диску."""
    from src.images import imread, imwrite

    mode = "aligned" if crop_scale is None else f"ctx{crop_scale:g}"
    cache = CACHE_DIR / "antispoof" / f"{mode}_{image_size}_d{det_size}"
    cache.mkdir(parents=True, exist_ok=True)
    engine = None
    ready: list[tuple[Path, int, str]] = []
    missed = 0
    started = time.monotonic()
    for index, sample in enumerate(samples, start=1):
        key = f"{sample.path.resolve()}|{sample.path.stat().st_mtime_ns}"
        name = hashlib.sha1(key.encode("utf-8")).hexdigest()
        crop_path = cache / f"{name}.jpg"
        fail_path = cache / f"{name}.fail"
        if fail_path.exists():
            missed += 1
            continue
        if not crop_path.exists():
            if engine is None:
                from src.faces import FaceEngine

                engine = FaceEngine(det_size=det_size, with_landmarks=False, with_recognition=False)
            found = engine.analyze(imread(sample.path), crop_size=image_size, crop_scale=crop_scale)
            if found is None:
                fail_path.write_text("no-face", encoding="utf-8")
                missed += 1
                continue
            imwrite(crop_path, found.spoof_input, quality=95)
        ready.append((crop_path, sample.label, sample.person))
        if index % 500 == 0 or index == len(samples):
            rate = index / max(time.monotonic() - started, 1e-6)
            print(f"Кропи {index}/{len(samples)}, без обличчя {missed}, {rate:.1f} фото/с", flush=True)
    return ready, missed


class CropDataset:
    def __init__(self, items: list[tuple[Path, int, str]], image_size: int, augment: bool, seed: int = 0) -> None:
        self.items = items
        self.image_size = image_size
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        import torch

        from src.augment import augment
        from src.images import face_tensor, imread

        path, label, _person = self.items[index]
        image = imread(path)
        if self.augment:
            rng = np.random.default_rng((self.seed, self.epoch, index))
            image = augment(image, rng)
        tensor = torch.from_numpy(face_tensor(image, self.image_size, IMAGENET_MEAN, IMAGENET_STD))
        return tensor, label


def _loader(dataset: CropDataset, batch_size: int, shuffle: bool, balance: bool, workers: int, seed: int):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    sampler = None
    if balance:
        labels = np.asarray([label for _path, label, _person in dataset.items])
        counts = np.bincount(labels, minlength=len(LABEL_NAMES)).astype(np.float64)
        weights = 1.0 / np.maximum(counts[labels], 1.0)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(labels),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        generator=torch.Generator().manual_seed(seed),
    )


def predict_logits(model, dataset: CropDataset, batch_size: int, device: str, workers: int):
    import torch

    loader = _loader(dataset, batch_size, shuffle=False, balance=False, workers=workers, seed=0)
    chunks, labels = [], []
    model.eval()
    with torch.inference_mode():
        for pixels, target in loader:
            chunks.append(model(pixels.to(device)).float().cpu())
            labels.append(target)
    return torch.cat(chunks), torch.cat(labels)


def partial_cross_entropy(logits, target, smoothing: float = 0.0):
    """Мітка 3 — підробка невідомого типу (CelebA-Spoof з дзеркала): штраф -log(P(друк) + P(екран))."""
    import torch
    import torch.nn.functional as F

    known = target < SPOOF_UNKNOWN
    total = logits.new_zeros(())
    if known.any():
        total = total + F.cross_entropy(logits[known], target[known], label_smoothing=smoothing, reduction="sum")
    if (~known).any():
        log_probs = F.log_softmax(logits[~known], dim=1)
        total = total - torch.logsumexp(log_probs[:, 1:], dim=1).sum()
    return total / max(len(target), 1)


def hits(logits, target):
    """Влучання: точний клас, а для підробки невідомого типу — будь-який клас атаки."""
    import torch

    predicted = logits.argmax(1)
    return torch.where(target < SPOOF_UNKNOWN, predicted == target, predicted > 0)


def fit_temperature(logits, labels) -> float:
    """Температурне калібрування: одне число T, щоб ймовірності відповідали частоті помилок."""
    import torch

    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = partial_cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(min(max(log_t.exp().item(), 0.05), 20.0))


def train_one(args: argparse.Namespace, train_items, val_items, run_name: str) -> dict:
    import torch

    from src.antispoof import build_model

    training_started = time.monotonic()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained = args.pretrained and args.arch in {"mnv3_small", "mnv3_large"}
    model, head = build_model(args.arch, len(CLASS_NAMES), args.width, pretrained=pretrained)
    model.to(device)
    train_ds = CropDataset(train_items, args.image_size, augment=True, seed=args.seed)
    val_ds = CropDataset(val_items, args.image_size, augment=False)
    train_loader = _loader(train_ds, args.batch_size, True, args.balance, args.workers, args.seed)
    head_ids = {id(parameter) for parameter in head}
    freeze = args.freeze_epochs if pretrained else 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    warmup = min(len(train_loader), total_steps // 10)

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    best = {"val_loss": float("inf"), "epoch": -1}
    best_state = None
    history = []
    stale = 0
    for epoch in range(args.epochs):
        train_ds.epoch = epoch
        for parameter in model.parameters():
            parameter.requires_grad = epoch >= freeze or id(parameter) in head_ids
        model.train()
        started = time.monotonic()
        running, seen, correct = 0.0, 0, 0
        for pixels, target in train_loader:
            pixels, target = pixels.to(device), target.to(device)
            logits = model(pixels)
            loss = partial_cross_entropy(logits, target, args.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach()) * len(target)
            seen += len(target)
            correct += int(hits(logits, target).sum())
        row = {
            "epoch": epoch + 1,
            "train_loss": running / max(seen, 1),
            "train_acc": correct / max(seen, 1),
            "seconds": round(time.monotonic() - started, 1),
            "frozen_backbone": epoch < freeze,
        }
        if len(val_ds):
            logits, labels = predict_logits(model, val_ds, args.batch_size, device, args.workers)
            row["val_loss"] = float(partial_cross_entropy(logits, labels))
            row["val_acc"] = float(hits(logits, labels).float().mean())
            probs = torch.softmax(logits, 1).numpy()
            rates = presentation_rates(labels.numpy(), live_mask_from_probs(probs, 0.5))
            row["val_apcer@0.5"], row["val_bpcer@0.5"] = rates.apcer, rates.bpcer
            if row["val_loss"] < best["val_loss"] - 1e-4:
                best = {"val_loss": row["val_loss"], "epoch": epoch + 1}
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
        history.append(row)
        print(f"[{run_name}] " + ", ".join(f"{key} {value:.4f}" if isinstance(value, float) else f"{key} {value}" for key, value in row.items()), flush=True)
        if len(val_ds) and stale >= args.patience:
            print(f"[{run_name}] рання зупинка: validation не покращується {args.patience} епохи поспіль", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    return {
        "model": model,
        "history": history,
        "best_epoch": best["epoch"],
        "device": device,
        "training_seconds": round(time.monotonic() - training_started, 1),
    }


def calibrate(args, model, val_items) -> tuple[float, float, str]:
    """Температура і поріг P(live) на validation. Без обох класів — консервативний поріг 0.8."""
    import torch

    if not val_items or len({label for _p, label, _s in val_items}) < 2 or not any(label == 0 for _p, label, _s in val_items):
        return 1.0, 0.8, "default"
    logits, labels = predict_logits(model, CropDataset(val_items, args.image_size, False), args.batch_size, "cpu", args.workers)
    temperature = fit_temperature(logits, labels)
    probs = torch.softmax(logits / temperature, 1).numpy()
    return temperature, choose_live_threshold(probs, labels.numpy(), apcer_target=args.apcer), "validation"


def evaluate(args, model, items, temperature: float, threshold: float) -> dict | None:
    import torch

    if not items:
        return None
    logits, labels = predict_logits(model, CropDataset(items, args.image_size, False), args.batch_size, "cpu", args.workers)
    probs = torch.softmax(logits / temperature, 1).numpy()
    target = labels.numpy()
    rates = presentation_rates(target, live_mask_from_probs(probs, threshold))
    confusion = np.zeros((len(LABEL_NAMES), len(CLASS_NAMES)), dtype=int)
    for truth, guess in zip(target, probs.argmax(1)):
        confusion[truth, guess] += 1
    return {
        "frames": int(len(target)),
        "people": len({person for _p, _l, person in items}),
        "apcer": rates.apcer,
        "apcer_by_type": rates.apcer_by_type,
        "bpcer": rates.bpcer,
        "acer": rates.acer,
        "bpcer_at_apcer": bpcer_at_apcer(probs, target),
        "confusion": confusion.tolist(),
    }


def save_curves(history: list[dict], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [row["epoch"] for row in history]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(epochs, [row["train_loss"] for row in history], label="train")
    if "val_loss" in history[0]:
        left.plot(epochs, [row["val_loss"] for row in history], label="validation")
    left.set_title("Втрати")
    left.set_xlabel("епоха")
    left.legend()
    right.plot(epochs, [row["train_acc"] for row in history], label="train")
    if "val_acc" in history[0]:
        right.plot(epochs, [row["val_acc"] for row in history], label="validation")
    right.set_title("Точність")
    right.set_xlabel("епоха")
    right.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def prepare(args) -> tuple[list[Sample], list[tuple[Path, int, str]]]:
    samples = load_samples(args.data)
    if any(sample.person.split(":", 1)[-1].startswith("file:") for sample in samples):
        print("Увага: частина фото лежить просто в папці класу, без папки людини. Розбиття тоді майже по кадрах.")
    samples = cap_per_person(samples, args.max_per_person, seed=args.seed)
    if args.limit:
        samples = stratified_limit(samples, args.limit, seed=args.seed)
    counts = np.bincount([sample.label for sample in samples], minlength=len(LABEL_NAMES))
    print(f"Фото: {len(samples)} ({', '.join(f'{name} {count}' for name, count in zip(LABEL_NAMES, counts))}), людей {len({s.person for s in samples})}.")
    crop_scale = None if args.input == "aligned" else args.crop_scale
    items, missed = crop_samples(samples, args.image_size, crop_scale, args.det_size)
    print(f"З обличчям: {len(items)}, без обличчя: {missed}.")
    return samples, items


def run_folds(args, items) -> None:
    """Крос-валідація за людьми: кожна людина рівно раз потрапляє в test."""
    fake = [Sample(path, label, person) for path, label, person in items]
    groups = person_folds(fake, args.folds, seed=args.seed)
    reports = []
    for index, test_people in enumerate(groups):
        rest = sorted({person for _p, _l, person in items} - test_people)
        val_people = set(random.Random(args.seed + index).sample(rest, max(1, len(rest) // 6)))
        train_items = [item for item in items if item[2] not in test_people | val_people]
        val_items = [item for item in items if item[2] in val_people]
        test_items = [item for item in items if item[2] in test_people]
        _require_live_and_attack(train_items, "train")
        _require_live_and_attack(val_items, "validation")
        _require_live_and_attack(test_items, "test")
        result = train_one(args, train_items, val_items, f"fold {index + 1}/{args.folds}")
        temperature, threshold, _source = calibrate(args, result["model"], val_items)
        report = evaluate(args, result["model"], test_items, temperature, threshold)
        report.update(
            fold=index + 1,
            threshold=threshold,
            test_people=sorted(test_people),
            best_epoch=result["best_epoch"],
            training_seconds=result["training_seconds"],
        )
        reports.append(report)
        print(f"Fold {index + 1}: APCER {report['apcer']:.3f}, BPCER {report['bpcer']:.3f}", flush=True)

    def stat(key):
        values = np.asarray([report[key] for report in reports], dtype=np.float64)
        return {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}

    summary = {"folds": reports, "apcer": stat("apcer"), "bpcer": stat("bpcer"), "acer": stat("acer"), "args": _plain(args)}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = RESULTS_DIR / f"antispoof_cv_{args.run_name}.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Крос-валідація {args.folds} фолдів: APCER {summary['apcer']['mean']:.3f} ± {summary['apcer']['std']:.3f}, "
        f"BPCER {summary['bpcer']['mean']:.3f} ± {summary['bpcer']['std']:.3f}. Звіт: {destination}"
    )


def _plain(args) -> dict:
    return {key: str(value) if isinstance(value, Path) else ([str(item) for item in value] if isinstance(value, list) else value) for key, value in vars(args).items()}


def _require_live_and_attack(items, split_name: str) -> None:
    labels = {label for _path, label, _person in items}
    if 0 not in labels or not any(label != 0 for label in labels):
        raise SystemExit(f"Частина {split_name} має містити і live, і хоча б один клас атаки.")


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise SystemExit("--epochs, --batch-size і --patience мають бути більшими за нуль.")
    if args.workers < 0 or args.freeze_epochs < 0:
        raise SystemExit("--workers і --freeze-epochs не можуть бути від’ємними.")
    numeric = (args.width, args.crop_scale, args.lr, args.weight_decay, args.val_ratio, args.test_ratio, args.apcer, args.label_smoothing)
    if not all(math.isfinite(value) for value in numeric):
        raise SystemExit("Числові параметри навчання мають бути скінченними.")
    if args.image_size < 1 or args.det_size < 1 or args.width <= 0 or args.crop_scale <= 0:
        raise SystemExit("Розміри зображення, детектора, ширина моделі й crop-scale мають бути більшими за нуль.")
    if args.lr <= 0 or args.weight_decay < 0:
        raise SystemExit("--lr має бути більшим за нуль, --weight-decay — невід’ємним.")
    if not 0 <= args.val_ratio or not 0 <= args.test_ratio or args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("--val-ratio і --test-ratio мають бути невід’ємними, а їхня сума — меншою за 1.")
    if not 0 <= args.apcer <= 1 or not 0 <= args.label_smoothing < 1:
        raise SystemExit("--apcer має бути від 0 до 1, а --label-smoothing — від 0 включно до 1.")
    if args.folds < 0:
        raise SystemExit("--folds не може бути від’ємним.")


def train(args: argparse.Namespace) -> None:
    import torch

    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    _samples, items = prepare(args)
    if len(items) < 6:
        raise SystemExit("Замало фото з обличчям для навчання. Перевірте камеру і структуру папок.")
    if args.folds:
        run_folds(args, items)
        return
    people = [person for _p, _l, person in items]
    train_ids, val_ids, test_ids = split_people(people, args.seed, 1 - args.val_ratio - args.test_ratio, args.val_ratio)
    disjoint = not any(person.split(":", 1)[-1].startswith("file:") for person in people)
    if not disjoint and test_ids:
        print("Увага: фото без папок людей, тож test ділиться за кадрами і завищує метрики. Чесний test — окремий набір.")
    train_items = [item for item in items if item[2] in train_ids]
    val_items = [item for item in items if item[2] in val_ids]
    test_items = [item for item in items if item[2] in test_ids]
    _require_live_and_attack(train_items, "train")
    _require_live_and_attack(val_items, "validation")
    if test_items:
        _require_live_and_attack(test_items, "test")
    print(f"Людей: train {len(train_ids)}, validation {len(val_ids)}, test {len(test_ids)}. Кадрів: {len(train_items)}/{len(val_items)}/{len(test_items)}.")
    result = train_one(args, train_items, val_items, args.run_name)
    model = result["model"]
    temperature, threshold, source = calibrate(args, model, val_items)
    print(f"Температура {temperature:.3f}, поріг P(live) {threshold:.3f} ({source}).")
    validation = evaluate(args, model, val_items, temperature, threshold)
    test = evaluate(args, model, test_items, temperature, threshold)
    if test:
        print(
            f"Test ({test['people']} людей, {test['frames']} кадрів): APCER {test['apcer']:.3f} {test['apcer_by_type']}, "
            f"BPCER {test['bpcer']:.3f}, ACER {test['acer']:.3f}."
        )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights) if args.weights else ANTISPOOF_WEIGHTS
    meta_path = Path(args.meta) if args.meta else ANTISPOOF_META
    weights.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    meta = {
        "arch": args.arch,
        "pretrained": bool(args.pretrained and args.arch in {"mnv3_small", "mnv3_large"}),
        "width_multiplier": args.width,
        "image_size": args.image_size,
        "input": args.input,
        "crop_scale": None if args.input == "aligned" else args.crop_scale,
        "det_size": args.det_size,
        "threshold": threshold,
        "threshold_split": source,
        "temperature": temperature,
        "apcer_target": args.apcer,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "class_names": list(CLASS_NAMES),
        "data": [str(Path(root)) for root in args.data],
        "split": {"seed": args.seed, "train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)},
        "test_person_disjoint": disjoint,
        "best_epoch": result["best_epoch"],
        "training_seconds": result["training_seconds"],
        "validation": validation,
        "test": test,
        "history": result["history"],
        "args": _plain(args),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    save_curves(result["history"], RESULTS_DIR / f"antispoof_training_{args.run_name}.png")
    print(f"Збережено ваги {weights}")
    print(f"Збережено опис {meta_path}")


def run_smoke(args: argparse.Namespace | None = None) -> None:
    """Перевірка циклу без датасету: випадкові картинки, одна епоха, без збереження."""
    import tempfile

    from src.images import imwrite

    if args is None:
        args = parse_args(["--smoke", "--image-size", "64", "--epochs", "1", "--batch-size", "8", "--no-pretrained"])
    args.epochs = 1
    args.pretrained = False
    args.image_size = min(args.image_size, 64)
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as folder:
        items = []
        for index in range(24):
            path = Path(folder) / f"{index}.jpg"
            imwrite(path, rng.integers(0, 255, size=(args.image_size, args.image_size, 3), dtype=np.uint8))
            items.append((path, index % 3, f"p{index % 6}"))
        result = train_one(args, items[:18], items[18:], "smoke")
        temperature, threshold, source = calibrate(args, result["model"], items[18:])
        print(f"Smoke-навчання пройшло: {result['history'][-1]}, T {temperature:.2f}, поріг {threshold:.2f} ({source}).")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Навчити захист від підробки обличчя")
    parser.add_argument("--data", type=Path, nargs="+", help="Одна чи кілька папок live/print/phone/laptop/screen")
    parser.add_argument("--arch", default="mnv3_small", choices=ARCHES)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="Ваги ImageNet для MobileNetV3")
    parser.add_argument("--width", type=float, default=0.5, help="width_multiplier для topologies_mnv2")
    parser.add_argument("--input", default="context", choices=["context", "aligned"], help="Кроп з контекстом або вузький ArcFace-кроп")
    parser.add_argument("--crop-scale", type=float, default=DEFAULT_CROP_SCALE)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--det-size", type=int, default=640, help="Розмір входу детектора при нарізанні кропів")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--freeze-epochs", type=int, default=1, help="Скільки перших епох основа заморожена")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--balance", action="store_true", help="Зважений семплер: класи трапляються однаково часто")
    parser.add_argument("--max-per-person", type=int, default=0, help="Не більше N фото на людину в кожному класі")
    parser.add_argument("--limit", type=int, default=0, help="Перемішані N фото з усіх класів для короткої перевірки")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Частка людей у validation")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Частка людей у test; 0 — test окремим набором")
    parser.add_argument("--folds", type=int, default=0, help="Крос-валідація за людьми на K фолдах замість одного розбиття")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apcer", type=float, default=0.05, help="Ціль найгіршого APCER на validation")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--run-name", default="main")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Одна епоха на випадкових картинках, без датасету і без збереження")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.smoke:
        run_smoke(arguments)
    else:
        if not arguments.data:
            raise SystemExit("Потрібен шлях --data або прапорець --smoke.")
        train(arguments)
