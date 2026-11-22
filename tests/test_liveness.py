"""Активна перевірка, якість кадру, кропи, аугментації і навчальні функції."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from src.challenge import check_blink, check_turn, eye_aspect_ratio, random_challenge
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


if __name__ == "__main__":
    unittest.main()
