"""Активна перевірка, якість кадру, кропи, аугментації і навчальні функції."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.challenge import check_blink, check_turn, eye_aspect_ratio, random_challenge
from src.folders import Sample
from src.images import context_crop
from src.quality import blur_score, yaw_proxy

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _kps(shift: float) -> np.ndarray:
    return np.array([[70, 90], [130, 90], [100 + shift * 60, 120], [75, 150], [125, 150]], dtype=np.float32)


def _landmarks(openness: float) -> np.ndarray:
    points = np.zeros((68, 3), dtype=np.float32)
    for start, center in ((36, 70.0), (42, 130.0)):
        points[start + 0, :2] = (center - 15, 90)
        points[start + 3, :2] = (center + 15, 90)
        points[start + 1, :2] = (center - 5, 90 - openness)
        points[start + 2, :2] = (center + 5, 90 - openness)
        points[start + 4, :2] = (center + 5, 90 + openness)
        points[start + 5, :2] = (center - 5, 90 + openness)
    return points


class ChallengeTests(unittest.TestCase):
    def test_yaw_proxy_sign(self) -> None:
        self.assertAlmostEqual(yaw_proxy(_kps(0.0)), 0.0)
        self.assertGreater(yaw_proxy(_kps(0.4)), 0.3)
        self.assertLess(yaw_proxy(_kps(-0.4)), -0.3)

    def test_turn_left_and_back(self) -> None:
        sequence = [0.0, 0.05, 0.2, 0.4, 0.45, 0.3, 0.1, 0.02]
        self.assertTrue(check_turn(sequence, "left")[0])
        self.assertFalse(check_turn(sequence, "right")[0])
        self.assertFalse(check_turn([0.0, 0.2, 0.4, 0.45, 0.42], "left")[0])
        self.assertFalse(check_turn([0.0, 0.05, 0.1, 0.12, 0.05], "left")[0])
        self.assertFalse(check_turn([0.3, 0.4, 0.45, 0.1, 0.0], "left")[0])

    def test_blink(self) -> None:
        open_eye = eye_aspect_ratio(_landmarks(6))
        closed_eye = eye_aspect_ratio(_landmarks(1))
        self.assertLess(closed_eye, open_eye * 0.5)
        self.assertTrue(check_blink([open_eye] * 4 + [closed_eye] + [open_eye] * 4)[0])
        self.assertFalse(check_blink([open_eye] * 9)[0])
        self.assertFalse(check_blink([open_eye] * 4 + [closed_eye] * 5)[0])

    def test_random_challenge(self) -> None:
        self.assertIn(random_challenge(with_blink=False), {"left", "right"})


class ImageTests(unittest.TestCase):
    def test_context_crop_stays_inside(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :320] = 255
        crop = context_crop(image, np.array([280, 180, 360, 280]), scale=2.7, size=224)
        self.assertEqual(crop.shape, (224, 224, 3))
        big = context_crop(image, np.array([200, 50, 440, 470]), scale=2.7, size=224)
        self.assertEqual(big.shape, (224, 224, 3))
        corner = context_crop(image, np.array([0, 0, 50, 50]), scale=4.0, size=64)
        self.assertEqual(int(corner[0, 0, 0]), 255)

    def test_blur_score(self) -> None:
        rng = np.random.default_rng(0)
        sharp = rng.integers(0, 255, size=(112, 112, 3), dtype=np.uint8)
        self.assertGreater(blur_score(sharp), 100)
        self.assertEqual(blur_score(np.zeros((112, 112, 3), dtype=np.uint8)), 0.0)

    def test_augment_is_deterministic(self) -> None:
        from src.augment import augment

        rng = np.random.default_rng(1)
        image = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        first = augment(image, np.random.default_rng(5))
        second = augment(image, np.random.default_rng(5))
        self.assertEqual(first.shape, image.shape)
        self.assertEqual(first.dtype, np.uint8)
        self.assertTrue(np.array_equal(first, second))


class AntispoofMetricsPathTests(unittest.TestCase):
    def test_select_uses_the_requested_models_metadata(self) -> None:
        from eval.antispoof_metrics import select

        with tempfile.TemporaryDirectory() as folder:
            meta_path = Path(folder) / "scratch_meta.json"
            meta_path.write_text('{"split":{"test":["p02"]}}', encoding="utf-8")
            samples = [
                Sample(Path("p01.jpg"), 0, "p01"),
                Sample(Path("p02.jpg"), 1, "p02"),
            ]

            self.assertEqual(select(samples, "test", Path(folder), meta_path), [samples[1]])

    def test_select_all_skips_calibration_people(self) -> None:
        from eval.antispoof_metrics import select

        with tempfile.TemporaryDirectory() as folder:
            meta_path = Path(folder) / "meta.json"
            meta_path.write_text('{"calibration":{"people":["p01"]}}', encoding="utf-8")
            samples = [Sample(Path("p01.jpg"), 0, "p01"), Sample(Path("p02.jpg"), 3, "p02")]
            self.assertEqual(select(samples, "all", Path(folder), meta_path), [samples[1]])

    def test_model_warm_up_reports_missing_artifacts(self) -> None:
        from src.antispoof import AntispoofModel

        with tempfile.TemporaryDirectory() as folder:
            model = AntispoofModel(Path(folder) / "missing.pt", Path(folder) / "missing.json")
            self.assertFalse(model.warm_up())


@unittest.skipUnless(HAS_TORCH, "потрібен torch")
class TrainingTests(unittest.TestCase):
    def test_partial_loss_for_unknown_spoof(self) -> None:
        import torch

        from training.train_antispoof import hits, partial_cross_entropy

        logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 3.0, 3.0], [0.0, 5.0, -5.0]])
        target = torch.tensor([0, 3, 3])
        loss = partial_cross_entropy(logits, target)
        self.assertLess(float(loss), 0.05)
        self.assertEqual(hits(logits, target).tolist(), [True, True, True])
        wrong = partial_cross_entropy(torch.tensor([[5.0, 0.0, 0.0]]), torch.tensor([3]))
        self.assertGreater(float(wrong), 4.0)

    def test_temperature_softens_overconfident_logits(self) -> None:
        import torch

        from training.train_antispoof import fit_temperature

        torch.manual_seed(0)
        labels = torch.randint(0, 3, (300,))
        logits = torch.randn(300, 3) * 8
        self.assertGreater(fit_temperature(logits, labels), 1.5)

    def test_mobilenet_v2_scratch_builds_without_topologies(self) -> None:
        import torch

        from src.antispoof import build_model

        model, head = build_model("mnv2_scratch", num_classes=3, width_multiplier=0.25)
        model.eval()
        with torch.inference_mode():
            logits = model(torch.zeros((1, 3, 64, 64)))

        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertTrue(head)

    def test_worker_loader_uses_epoch_specific_augmentations(self) -> None:
        import torch
        from PIL import Image

        from training.train_antispoof import CropDataset, _loader

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "face.png"
            pixels = np.random.default_rng(42).integers(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(path)
            dataset = CropDataset([(path, 0, "p01")], image_size=64, augment=True, seed=7)
            loader = _loader(dataset, 1, shuffle=False, balance=False, workers=1, seed=7)

            dataset.epoch = 0
            first_iterator = iter(loader)
            first = next(first_iterator)[0]
            del first_iterator
            dataset.epoch = 1
            second_iterator = iter(loader)
            second = next(second_iterator)[0]
            del second_iterator

        self.assertFalse(torch.equal(first, second))

    def test_training_saves_weights_and_run_metadata(self) -> None:
        import json
        import torch
        from PIL import Image

        from training import train_antispoof

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(42)
            items = []
            for person in range(6):
                for label in (0, 1):
                    path = root / f"p{person}_{label}.png"
                    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(path)
                    items.append((path, label, f"p{person}"))

            weights = root / "models" / "scratch.pt"
            meta_path = root / "models" / "scratch_meta.json"
            results_dir = root / "results"
            args = train_antispoof.parse_args(
                [
                    "--data",
                    str(root),
                    "--arch",
                    "mnv2_scratch",
                    "--no-pretrained",
                    "--epochs",
                    "1",
                    "--image-size",
                    "64",
                    "--batch-size",
                    "8",
                    "--run-name",
                    "unit",
                    "--weights",
                    str(weights),
                    "--meta",
                    str(meta_path),
                ]
            )
            with (
                patch.object(train_antispoof, "prepare", return_value=([], items)),
                patch.object(train_antispoof, "MODELS_DIR", root / "models"),
                patch.object(train_antispoof, "RESULTS_DIR", results_dir),
                patch.object(torch, "set_num_threads"),
            ):
                train_antispoof.train(args)

            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertTrue(weights.is_file())
            self.assertEqual(metadata["arch"], "mnv2_scratch")
            self.assertEqual(metadata["best_epoch"], 1)
            self.assertIn("training_seconds", metadata)
            self.assertFalse(metadata["pretrained"])
            self.assertTrue((results_dir / "antispoof_training_unit.png").is_file())
            from src.antispoof import AntispoofModel

            model = AntispoofModel(weights, meta_path)
            probabilities = model.predict_proba([np.zeros((64, 64, 3), dtype=np.uint8)])
            self.assertEqual(tuple(probabilities.shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
