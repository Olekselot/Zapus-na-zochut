"""Оплата і реєстрація на підставних обличчях: безпека рахунків, ліміти, якість, кілька кадрів."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.bank import Bank
from src.config import EMBEDDING_DIM, HIGH_AMOUNT
from src.decisions import summarize_liveness
from src.faces import FaceSample
from src.gallery import l2_normalize
from src.pipeline import Pipeline

FRONTAL = np.array([[70, 90], [130, 90], [100, 120], [75, 150], [125, 150]], dtype=np.float32)
IMAGE = np.zeros((8, 8, 3), dtype=np.uint8)
PHOTOS = [IMAGE] * 5


def _vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.normal(size=EMBEDDING_DIM).astype(np.float32))


def _near(vector: np.ndarray, seed: int, noise: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(vector + noise * rng.normal(size=EMBEDDING_DIM).astype(np.float32) / np.sqrt(EMBEDDING_DIM))


class FakeFaces:
    def __init__(self, embedding: np.ndarray) -> None:
        self.embedding = embedding
        self.queue: list[np.ndarray] = []
        self.found = True
        self.sharp = True
        self.faces_found = 1
        self.second_ratio = 0.0

    def analyze(self, image, crop_size: int = 224, crop_scale=None, pick: str = "largest"):
        if not self.found:
            return None
        embedding = self.queue.pop(0) if self.queue else self.embedding
        rng = np.random.default_rng(0)
        aligned = rng.integers(0, 255, size=(112, 112, 3), dtype=np.uint8) if self.sharp else np.zeros((112, 112, 3), np.uint8)
        return FaceSample(
            aligned_bgr=aligned,
            embedding=embedding,
            det_score=0.99,
            antispoof_bgr=np.zeros((crop_size, crop_size, 3), dtype=np.uint8),
            bbox=np.array([0, 0, 200, 200], dtype=np.float32),
            kps=FRONTAL,
            faces_found=self.faces_found,
            second_face_ratio=self.second_ratio,
        )


class FakeAnti:
    def __init__(self, rows: list[list[float]], threshold: float = 0.5) -> None:
        self.rows = np.asarray(rows, dtype=np.float32)
        self.available = True
        self.threshold = threshold
        self.image_size = 32
        self.crop_scale = 2.7
        self.calls = 0

    def predict_proba(self, images):
        self.calls += 1
        if len(self.rows) == 1:
            return np.repeat(self.rows, len(images), axis=0)
        return self.rows[: len(images)]


LIVE = [[0.9, 0.05, 0.05]]
SCREEN = [[0.1, 0.2, 0.7]]


class PaymentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.bank = Bank(Path(self._dir.name) / "bank.sqlite")
        self.client = _vector(10)
        self.bank.enroll("p01", [self.client], 1000)
        self.faces = FakeFaces(self.client)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _pipeline(self, anti) -> Pipeline:
        return Pipeline(self.bank, self.faces, anti)

    def test_pays_live_client(self) -> None:
        decision, _preview = self._pipeline(FakeAnti(LIVE)).pay([IMAGE], 25)
        self.assertEqual(decision.status, "paid")
        self.assertEqual(self.bank.balance_of(1), 975)
        self.assertEqual(decision.client_code, "p01")

    def test_invalid_amounts_are_rejected_and_logged(self) -> None:
        anti = FakeAnti(LIVE)
        pipeline = self._pipeline(anti)

        for amount in (float("nan"), float("inf"), -1, 0):
            with self.subTest(amount=amount):
                decision, _preview = pipeline.pay([IMAGE], amount)
                self.assertEqual(decision.status, "error")

        self.assertEqual(self.bank.balance_of(1), 1000)
        self.assertEqual(anti.calls, 0)
        self.assertEqual(len(self.bank.transactions()), 4)
        self.assertTrue(all(row["статус"] == "error" for row in self.bank.transactions()))
        self.assertTrue(all(row["сума"] is None for row in self.bank.transactions()))

    def test_blocks_spoof(self) -> None:
        decision, _preview = self._pipeline(FakeAnti(SCREEN)).pay([IMAGE], 25)
        self.assertEqual(decision.status, "spoof")
        self.assertEqual(self.bank.balance_of(1), 1000)
        self.assertIn("екран", decision.message)

    def test_unknown_person(self) -> None:
        self.faces.embedding = _vector(99)
        decision, _preview = self._pipeline(FakeAnti(LIVE)).pay([IMAGE], 10)
        self.assertEqual(decision.status, "unknown")
        self.assertEqual(self.bank.balance_of(1), 1000)

    def test_ambiguous_between_two_clients(self) -> None:
        self.bank.enroll("p02", [_near(self.client, 3, 0.1)], 1000)
        decision, _preview = self._pipeline(FakeAnti(LIVE)).pay([IMAGE], 10)
        self.assertEqual(decision.status, "ambiguous")
        self.assertEqual(self.bank.balance_of(1), 1000)
        self.assertEqual(self.bank.balance_of(2), 1000)

    def test_insufficient_funds(self) -> None:
        decision, _preview = self._pipeline(FakeAnti(LIVE)).pay([IMAGE], 400)
        self.assertEqual(decision.status, "paid")
        decision, _preview = self._pipeline(FakeAnti(LIVE)).pay([IMAGE], 480)
        self.assertEqual(decision.status, "paid")
        decision, _preview = self._pipeline(FakeAnti(LIVE)).pay([IMAGE], 200)
        self.assertEqual(decision.status, "insufficient")
        self.assertEqual(self.bank.balance_of(1), 120)

    def test_no_model_does_not_debit_or_invent_liveness(self) -> None:
        anti = FakeAnti(SCREEN)
        anti.available = False
        decision, _preview = self._pipeline(anti).pay([IMAGE], 10)
        self.assertEqual(decision.status, "no_model")
        self.assertEqual(anti.calls, 0)
        self.assertEqual(self.bank.balance_of(1), 1000)
        self.assertEqual(decision.client_code, "p01")

    def test_no_face(self) -> None:
        self.faces.found = False
        anti = FakeAnti(LIVE)
        decision, _preview = self._pipeline(anti).pay([IMAGE], 10, min_faces=1)
        self.assertEqual(decision.status, "no_face")
        self.assertEqual(anti.calls, 0)

    def test_blurred_frame_is_rejected(self) -> None:
        self.faces.sharp = False
        decision, _preview = self._pipeline(FakeAnti(LIVE)).pay([IMAGE], 10)
        self.assertEqual(decision.status, "poor_quality")
        self.assertIn("розмитий", decision.message)

    def test_two_faces_in_frame(self) -> None:
        self.faces.faces_found, self.faces.second_ratio = 2, 0.8
        decision, _preview = self._pipeline(FakeAnti(LIVE)).pay([IMAGE], 10)
        self.assertEqual(decision.status, "multi_face")
        self.faces.second_ratio = 0.1
        decision, _preview = self._pipeline(FakeAnti(LIVE)).pay([IMAGE], 10)
        self.assertEqual(decision.status, "paid")

    def test_frames_from_different_people(self) -> None:
        self.faces.queue = [self.client, _vector(76), _vector(77), _vector(78), self.client]
        decision, _preview = self._pipeline(FakeAnti(LIVE)).pay([IMAGE] * 5, 10, min_faces=3)
        self.assertEqual(decision.status, "inconsistent")
        self.assertEqual(self.bank.balance_of(1), 1000)

    def test_single_outlier_frame_does_not_break_payment(self) -> None:
        self.faces.queue = [self.client, self.client, _vector(77), self.client, self.client]
        decision, _preview = self._pipeline(FakeAnti(LIVE)).pay([IMAGE] * 5, 10, min_faces=3)
        self.assertEqual(decision.status, "paid")
        self.assertEqual(decision.frames_used, 4)
        self.assertIn("1 відкинуто", self.bank.transactions()[0]["деталі"])

    def test_locked_pin_message_shows_time(self) -> None:
        client_id, _shown = self.bank.register_person("Марко", "Коваль", [_vector(40)], 5000, pin="2468")
        self.faces.embedding = _vector(40)
        pipeline = self._pipeline(FakeAnti(LIVE))
        big = HIGH_AMOUNT + 100
        for _ in range(5):
            decision = pipeline.pay([IMAGE], big, pin="0000")[0]
        self.assertEqual(decision.status, "pin_locked")
        self.assertIn("заблоковано до", decision.message)
        self.assertEqual(pipeline.pay([IMAGE], big, pin="2468")[0].status, "pin_locked")
        self.assertEqual(self.bank.balance_of(client_id), 5000)

    def test_five_frames_reject_when_majority_is_attack(self) -> None:
        rows = [[0.90, 0.05, 0.05]] * 2 + [[0.42, 0.50, 0.08]] * 3
        is_live, label, median = summarize_liveness(np.asarray(rows), threshold=0.4)
        self.assertFalse(is_live)
        self.assertEqual(label, "print")
        self.assertGreaterEqual(median, 0.4)
        decision, _preview = self._pipeline(FakeAnti(rows, threshold=0.4)).pay([IMAGE] * 5, 10, min_faces=3)
        self.assertEqual(decision.status, "spoof")
        self.assertEqual(decision.frames_used, 5)

    def test_disabled_guard_pays_spoof_and_skips_model(self) -> None:
        anti = FakeAnti(SCREEN)
        pipeline = self._pipeline(anti)
        pipeline.antispoof_enabled = False
        decision, _preview = pipeline.pay([IMAGE], 25)
        self.assertEqual(decision.status, "paid")
        self.assertEqual(anti.calls, 0)
        self.assertIn("вимкнено", decision.message)
        self.assertTrue(pipeline.toggle_antispoof())

    def test_high_amount_needs_pin(self) -> None:
        client_id, _shown = self.bank.register_person("Марко", "Коваль", [_vector(40)], 5000, pin="2468")
        self.faces.embedding = _vector(40)
        pipeline = self._pipeline(FakeAnti(LIVE))
        big = HIGH_AMOUNT + 100
        self.assertEqual(pipeline.pay([IMAGE], big)[0].status, "pin_required")
        self.assertEqual(pipeline.pay([IMAGE], big, pin="0000")[0].status, "pin_wrong")
        self.assertEqual(self.bank.balance_of(client_id), 5000)
        decision = pipeline.pay([IMAGE], big, pin="2468")[0]
        self.assertEqual(decision.status, "paid")
        self.assertEqual(self.bank.balance_of(client_id), 5000 - big)
        self.assertEqual(pipeline.pay([IMAGE], 50)[0].status, "paid")

    def test_high_amount_without_pin_on_account(self) -> None:
        decision, _preview = self._pipeline(FakeAnti(LIVE)).pay([IMAGE], HIGH_AMOUNT + 1, pin="1234")
        self.assertEqual(decision.status, "pin_required")
        self.assertEqual(self.bank.balance_of(1), 1000)


class RegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.bank = Bank(Path(self._dir.name) / "bank.sqlite")
        self.faces = FakeFaces(_vector(1))
        self.pipeline = Pipeline(self.bank, self.faces, FakeAnti([[1, 0, 0]]))

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_same_face_same_name_attaches(self) -> None:
        decision, pending = self.pipeline.start_registration("Марко", "Коваль", PHOTOS, 40)
        self.assertEqual(decision.status, "enrolled")
        self.assertIsNone(pending)
        self.assertIn("Коваль Марко", decision.message)
        self.assertNotIn("c1", decision.message)
        decision, pending = self.pipeline.start_registration("марко", "коваль", PHOTOS, 90)
        self.assertEqual(decision.status, "ask")
        self.assertTrue(pending["can_attach"])
        self.assertFalse(pending["can_create"])
        attached = self.pipeline.confirm_existing(pending)
        self.assertEqual(attached.status, "enrolled")
        self.assertEqual(self.bank.balance_of(1), 40)
        self.assertEqual(self.bank.accounts()[0]["векторів"], 10)
        self.assertEqual(self.pipeline.confirm_new(pending).status, "error")

    def test_invalid_balances_are_rejected(self) -> None:
        for balance in (float("nan"), float("inf"), -1, 0):
            with self.subTest(balance=balance):
                decision, pending = self.pipeline.start_registration("Марко", "Коваль", PHOTOS, balance)
                self.assertEqual(decision.status, "error")
                self.assertIsNone(pending)

        self.assertEqual(self.bank.accounts(), [])

    def test_stranger_cannot_take_over_account_by_name(self) -> None:
        self.pipeline.start_registration("Марко", "Коваль", PHOTOS, 40)
        self.faces.embedding = _vector(2)
        decision, pending = self.pipeline.start_registration("Марко", "Коваль", PHOTOS, 10)
        self.assertEqual(decision.status, "ask")
        self.assertFalse(pending["can_attach"])
        self.assertTrue(pending["can_create"])
        self.assertEqual(self.pipeline.confirm_existing(pending).status, "error")
        forged = dict(pending, can_attach=True, client_id=1, display="Коваль Марко")
        self.assertEqual(self.pipeline.confirm_existing(forged).status, "error")
        self.assertEqual(len(self.bank.client_vectors(1)), 5)
        created = self.pipeline.confirm_new(pending)
        self.assertEqual(created.status, "enrolled")
        self.assertNotIn("c2", created.html())
        self.faces.embedding = _vector(3)
        blocked, again = self.pipeline.start_registration("Марко", "Коваль", PHOTOS, 10)
        self.assertEqual(blocked.status, "blocked")
        self.assertIsNone(again)

    def test_same_face_other_name_is_duplicate(self) -> None:
        self.pipeline.start_registration("Марко", "Коваль", PHOTOS, 40)
        decision, pending = self.pipeline.start_registration("Олег", "Шевчук", PHOTOS, 40)
        self.assertEqual(decision.status, "duplicate")
        self.assertIsNone(pending)
        self.assertEqual(len(self.bank.accounts()), 1)

    def test_spoof_photos_are_not_enrolled(self) -> None:
        pipeline = Pipeline(self.bank, self.faces, FakeAnti(SCREEN))
        decision, _pending = pipeline.start_registration("Марко", "Коваль", PHOTOS, 40)
        self.assertEqual(decision.status, "spoof")
        self.assertEqual(len(self.bank.accounts()), 0)

    def test_mixed_people_rejected_and_pin_validated(self) -> None:
        self.faces.queue = [_vector(5), _vector(6), _vector(7)]
        decision, _pending = self.pipeline.start_registration("Марко", "Коваль", [IMAGE] * 3, 40)
        self.assertEqual(decision.status, "inconsistent")
        decision, _pending = self.pipeline.start_registration("Марко", "Коваль", PHOTOS, 40, pin="12")
        self.assertEqual(decision.status, "error")
        decision, _pending = self.pipeline.start_registration("Марко", "Коваль", PHOTOS, 40, pin="1357")
        self.assertEqual(decision.status, "enrolled")
        self.assertTrue(self.bank.has_pin(1))

    def test_too_few_usable_photos(self) -> None:
        decision, pending = self.pipeline.start_registration("Марко", "Коваль", [IMAGE] * 4, 40)
        self.assertEqual(decision.status, "poor_quality")
        self.assertIn("щонайменше 5", decision.message)
        self.assertIsNone(pending)
        self.faces.sharp = False
        decision, _pending = self.pipeline.start_registration("Марко", "Коваль", [IMAGE] * 8, 40)
        self.assertEqual(decision.status, "poor_quality")
        self.assertEqual(self.bank.accounts(), [])


if __name__ == "__main__":
    unittest.main()
