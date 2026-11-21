"""Оплата на підставних обличчях: списання, підробка, невідомий і блокування без моделі."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.bank import Bank
from src.config import EMBEDDING_DIM
from src.decisions import summarize_liveness
from src.faces import FaceSample
from src.frames import average_embeddings
from src.gallery import l2_normalize
from src.pipeline import Pipeline


def _vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.normal(size=EMBEDDING_DIM).astype(np.float32))


class FakeFaces:
    def __init__(self, embedding: np.ndarray) -> None:
        self.embedding = embedding
        self.found = True

    def analyze(self, image, crop_size: int = 224):
        if not self.found:
            return None
        return FaceSample(
            aligned_bgr=np.zeros((crop_size, crop_size, 3), dtype=np.uint8),
            embedding=self.embedding,
            det_score=0.99,
        )


class FakeAnti:
    def __init__(self, rows: list[list[float]], threshold: float = 0.5) -> None:
        self.rows = np.asarray(rows, dtype=np.float32)
        self.available = True
        self.threshold = threshold
        self.image_size = 32
        self.calls = 0

    def predict_proba(self, images):
        self.calls += 1
        count = len(images)
        if len(self.rows) == 1:
            return np.repeat(self.rows, count, axis=0)
        return self.rows[:count]


class PaymentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.bank = Bank(Path(self._dir.name) / "bank.sqlite")
        self.client = _vector(10)
        self.bank.enroll("p01", [self.client], 100)
        self.faces = FakeFaces(self.client)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _pipeline(self, anti) -> Pipeline:
        return Pipeline(self.bank, self.faces, anti)

    def test_pays_live_client(self) -> None:
        anti = FakeAnti([[0.9, 0.05, 0.05]])
        decision, _preview = self._pipeline(anti).pay([np.zeros((4, 4, 3), dtype=np.uint8)], 25)
        self.assertEqual(decision.status, "paid")
        self.assertEqual(self.bank.balance_of(1), 75)
        self.assertEqual(decision.client_code, "p01")

    def test_blocks_spoof(self) -> None:
        anti = FakeAnti([[0.1, 0.2, 0.7]])
        decision, _preview = self._pipeline(anti).pay([np.zeros((4, 4, 3), dtype=np.uint8)], 25)
        self.assertEqual(decision.status, "spoof")
        self.assertEqual(self.bank.balance_of(1), 100)
        self.assertIn("екран", decision.message)

    def test_unknown_person(self) -> None:
        self.faces.embedding = _vector(99)
        anti = FakeAnti([[0.95, 0.03, 0.02]])
        decision, _preview = self._pipeline(anti).pay([np.zeros((4, 4, 3), dtype=np.uint8)], 10)
        self.assertEqual(decision.status, "unknown")
        self.assertEqual(self.bank.balance_of(1), 100)

    def test_insufficient_funds(self) -> None:
        anti = FakeAnti([[0.9, 0.05, 0.05]])
        decision, _preview = self._pipeline(anti).pay([np.zeros((4, 4, 3), dtype=np.uint8)], 500)
        self.assertEqual(decision.status, "insufficient")
        self.assertEqual(self.bank.balance_of(1), 100)

    def test_no_model_does_not_debit_or_invent_liveness(self) -> None:
        anti = FakeAnti([[0.1, 0.1, 0.8]])
        anti.available = False
        decision, _preview = self._pipeline(anti).pay([np.zeros((4, 4, 3), dtype=np.uint8)], 10)
        self.assertEqual(decision.status, "no_model")
        self.assertEqual(anti.calls, 0)
        self.assertEqual(self.bank.balance_of(1), 100)
        self.assertEqual(decision.client_code, "p01")

    def test_no_face(self) -> None:
        self.faces.found = False
        anti = FakeAnti([[0.9, 0.05, 0.05]])
        decision, _preview = self._pipeline(anti).pay([np.zeros((4, 4, 3), dtype=np.uint8)], 10, min_faces=1)
        self.assertEqual(decision.status, "no_face")
        self.assertEqual(anti.calls, 0)

    def test_five_frames_reject_when_majority_is_attack(self) -> None:
        rows = [
            [0.90, 0.05, 0.05],
            [0.90, 0.05, 0.05],
            [0.42, 0.50, 0.08],
            [0.42, 0.50, 0.08],
            [0.42, 0.50, 0.08],
        ]
        is_live, label, median = summarize_liveness(np.asarray(rows), threshold=0.4)
        self.assertFalse(is_live)
        self.assertEqual(label, "print")
        self.assertGreaterEqual(median, 0.4)
        anti = FakeAnti(rows, threshold=0.4)
        images = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in rows]
        decision, _preview = self._pipeline(anti).pay(images, 10, min_faces=3)
        self.assertEqual(decision.status, "spoof")
        self.assertEqual(decision.frames_used, 5)

    def test_disabled_guard_pays_spoof_and_skips_model(self) -> None:
        anti = FakeAnti([[0.1, 0.2, 0.7]])
        pipeline = self._pipeline(anti)
        pipeline.antispoof_enabled = False
        decision, _preview = pipeline.pay([np.zeros((4, 4, 3), dtype=np.uint8)], 25)
        self.assertEqual(decision.status, "paid")
        self.assertEqual(anti.calls, 0)
        self.assertEqual(self.bank.balance_of(1), 75)
        self.assertIn("вимкнено", decision.message)
        self.assertTrue(pipeline.toggle_antispoof())

    def test_disabled_guard_pays_without_trained_model(self) -> None:
        anti = FakeAnti([[0.1, 0.1, 0.8]])
        anti.available = False
        pipeline = self._pipeline(anti)
        pipeline.antispoof_enabled = False
        decision, _preview = pipeline.pay([np.zeros((4, 4, 3), dtype=np.uint8)], 10)
        self.assertEqual(decision.status, "paid")
        self.assertEqual(anti.calls, 0)
        self.assertEqual(self.bank.balance_of(1), 90)

    def test_average_embedding_matches_client(self) -> None:
        noisy = l2_normalize(self.client + 0.01)
        merged = average_embeddings([self.client, noisy])
        self.faces.embedding = merged
        # pay() averages the frame embeddings again; one frame keeps the vector.
        anti = FakeAnti([[0.9, 0.05, 0.05]])
        decision, _preview = self._pipeline(anti).pay([np.zeros((2, 2, 3), np.uint8)], 5)
        self.assertEqual(decision.status, "paid")


class RegistrationTests(unittest.TestCase):
    def test_one_match_asks_then_attaches_or_creates(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            bank = Bank(Path(folder) / "bank.sqlite")
            pipeline = Pipeline(bank, FakeFaces(_vector(1)), FakeAnti([[1, 0, 0]]))
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            decision, pending = pipeline.start_registration("Марко", "Коваль", [image], 40)
            self.assertEqual(decision.status, "enrolled")
            self.assertIsNone(pending)
            self.assertIn("Коваль Марко", decision.message)
            self.assertNotIn("c1", decision.message)

            decision, pending = pipeline.start_registration("марко", "коваль", [image], 90)
            self.assertEqual(decision.status, "ask")
            self.assertIsNotNone(pending)
            attached = pipeline.confirm_existing(pending)
            self.assertIn("Баланс без змін", attached.message)
            self.assertEqual(bank.balance_of(1), 40)
            self.assertEqual(bank.accounts()[0]["векторів"], 2)

            created = pipeline.confirm_new(pending)
            self.assertEqual(created.status, "enrolled")
            self.assertEqual(len(bank.find_by_name("Марко", "Коваль")), 2)
            blocked, again = pipeline.start_registration("Марко", "Коваль", [image], 10)
            self.assertEqual(blocked.status, "blocked")
            self.assertIsNone(again)
            self.assertIn("недоступна", blocked.message)
            self.assertEqual(len(bank.accounts()), 2)

            card = created.html()
            self.assertNotIn("c2", card)


if __name__ == "__main__":
    unittest.main()
