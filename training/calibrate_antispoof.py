"""Перепідібрати поріг P(live) і температуру навченої моделі на людях, яких вона не бачила.

У дзеркалі CelebA-Spoof немає ID людей, тож validation ділить людей з train і завищує якість:
поріг, підібраний там, вийшов 0. Тут окремий набір (наприклад test-спліт) ділиться навпіл:
на першій половині — температура і поріг, на другій — звіт, який у виборі не брав участі.

    python training/calibrate_antispoof.py --data data/celeba_spoof_test --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.antispoof import AntispoofModel  # noqa: E402
from src.config import ANTISPOOF_META, ANTISPOOF_WEIGHTS, RESULTS_DIR  # noqa: E402
from src.folders import split_people, take  # noqa: E402
from src.metrics import (  # noqa: E402
    bpcer_at_apcer,
    choose_live_threshold,
    live_mask_from_probs,
    presentation_rates,
    threshold_for_bpcer,
)
from training.train_antispoof import crop_samples, fit_temperature, load_samples  # noqa: E402


def logits_for(model: AntispoofModel, items, batch_size: int):
    import torch

    from src.images import face_tensor, imread

    model.warm_up()
    spec, network = model.spec, model._model
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch = [face_tensor(imread(path), spec.image_size, spec.mean, spec.std) for path, _label, _person in items[start : start + batch_size]]
            chunks.append(network(torch.from_numpy(np.stack(batch))).float())
    labels = torch.as_tensor([label for _path, label, _person in items])
    return torch.cat(chunks), labels


def _relative(path: Path) -> str:
    """Шлях від кореня проєкту, щоб у звіті не було локальних абсолютних шляхів."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name


def rates_at(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    rates = presentation_rates(labels, live_mask_from_probs(probs, threshold))
    return {
        "threshold": round(float(threshold), 4),
        "apcer": rates.apcer,
        "bpcer": rates.bpcer,
        "acer": rates.acer,
        "live": int((labels == 0).sum()),
        "attacks": int((labels != 0).sum()),
    }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Поріг і температура захисту на окремому наборі людей")
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=Path, default=ANTISPOOF_WEIGHTS)
    parser.add_argument("--meta", type=Path, default=ANTISPOOF_META)
    parser.add_argument("--rule", choices=["bpcer", "apcer"], default="bpcer", help="Що обмежувати на половині для калібрування")
    parser.add_argument("--bpcer", type=float, default=0.05, help="Найбільша частка відхилених живих (правило bpcer)")
    parser.add_argument("--apcer", type=float, default=0.05, help="Найбільший APCER (правило apcer)")
    parser.add_argument("--holdout", type=float, default=0.5, help="Частка людей для звіту")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--write", action="store_true", help="Записати новий поріг і температуру в метадані")
    args = parser.parse_args()
    if not all(0 < value < 1 for value in (args.holdout, args.apcer, args.bpcer)):
        raise SystemExit("--holdout, --apcer і --bpcer мають бути між 0 і 1.")

    model = AntispoofModel(args.weights, args.meta)
    if not model.available:
        raise SystemExit(f"Немає ваг або метаданих: {args.weights}, {args.meta}.")
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    spec = model._read_spec()

    samples = load_samples(args.data)
    calib_ids, _none, report_ids = split_people([s.person for s in samples], args.seed, 1 - args.holdout, 0.0)
    items, missed = crop_samples(samples, spec.image_size, spec.crop_scale, int(meta.get("det_size", 640)))
    calib = [item for item in items if item[2] in calib_ids]
    report = [item for item in items if item[2] in report_ids]
    print(f"Кадрів з обличчям: калібрування {len(calib)}, звіт {len(report)}; без обличчя {missed}.")

    calib_logits, calib_labels = logits_for(model, calib, args.batch_size)
    report_logits, report_labels = logits_for(model, report, args.batch_size)
    temperature = fit_temperature(calib_logits, calib_labels)
    calib_probs = torch.softmax(calib_logits / temperature, 1).numpy()
    report_probs = torch.softmax(report_logits / temperature, 1).numpy()
    old_probs = torch.softmax(report_logits / spec.temperature, 1).numpy()
    if args.rule == "bpcer":
        threshold = threshold_for_bpcer(calib_probs, calib_labels.numpy(), bpcer_target=args.bpcer)
    else:
        threshold = choose_live_threshold(calib_probs, calib_labels.numpy(), apcer_target=args.apcer)

    labels = report_labels.numpy()
    result = {
        "data": [_relative(path) for path in args.data],
        "weights": _relative(args.weights),
        "seed": args.seed,
        "holdout": args.holdout,
        "rule": args.rule,
        "target": args.bpcer if args.rule == "bpcer" else args.apcer,
        "frames": {"calibration": len(calib), "report": len(report), "no_face": missed},
        "temperature": {"old": spec.temperature, "new": temperature},
        "report_old": rates_at(old_probs, labels, spec.threshold),
        "report_new": rates_at(report_probs, labels, threshold),
        "report_bpcer_at_apcer": bpcer_at_apcer(report_probs, labels),
        "calibration": rates_at(calib_probs, calib_labels.numpy(), threshold),
    }
    old, new = result["report_old"], result["report_new"]
    print(f"Звітна половина ({new['live']} живих, {new['attacks']} атак):")
    print(f"  старий поріг {old['threshold']:.3f}: APCER {old['apcer']:.4f}, BPCER {old['bpcer']:.4f}")
    print(f"  новий поріг  {new['threshold']:.3f}: APCER {new['apcer']:.4f}, BPCER {new['bpcer']:.4f}, T {temperature:.3f}")
    print(f"  BPCER при APCER 1/5/10 %: {result['report_bpcer_at_apcer']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = RESULTS_DIR / f"antispoof_calibration_{args.weights.stem}_{args.rule}.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Звіт: {destination}")

    if args.write:
        meta["validation_threshold"] = {"threshold": spec.threshold, "temperature": spec.temperature}
        meta["threshold"] = threshold
        meta["temperature"] = temperature
        meta["threshold_split"] = "calibration"
        meta["calibration"] = {
            key: result[key] for key in ("data", "seed", "holdout", "rule", "target", "frames", "report_old", "report_new", "report_bpcer_at_apcer")
        }
        meta["calibration"]["people"] = sorted(calib_ids)
        args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Поріг і температуру записано в {args.meta}. Фото калібрування eval/antispoof_metrics.py далі пропускає.")


if __name__ == "__main__":
    main()
