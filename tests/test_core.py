"""Перевірки банку, порогів і розбиття без камери й без нейромереж."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.bank import Bank
from src.config import EMBEDDING_DIM
from src.folders import Sample, balance_classes, collect_samples, split_by_person
from src.gallery import best_match, cosine, l2_normalize
from src.metrics import (
    choose_live_threshold,
    equal_error_point,
    live_mask_from_probs,
    operating_point,
    presentation_rates,
    recognition_curve,
)


def _vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.normal(size=EMBEDDING_DIM).astype(np.float32))


class GalleryTests(unittest.TestCase):
    def test_same_vector_is_one(self) -> None:
        vector = _vector(1)
        self.assertAlmostEqual(cosine(vector, vector), 1.0, places=5)

    def test_threshold_rejects_other_person(self) -> None:
        known = _vector(1)
        other = _vector(2)
        match = best_match(other, [(1, "p01", known)], threshold=0.99)
        self.assertFalse(match.accepted)
        self.assertIsNone(match.code)
        self.assertIsNotNone(match.similarity)

    def test_max_over_several_vectors(self) -> None:
        query = _vector(3)
        weak = _vector(4)
        match = best_match(query, [(7, "p07", weak), (7, "p07", query)], threshold=0.99)
        self.assertTrue(match.accepted)
        self.assertEqual(match.code, "p07")
        self.assertAlmostEqual(match.similarity or 0, 1.0, places=5)


class BankTests(unittest.TestCase):
    def test_enroll_debit_and_no_photos(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            bank = Bank(Path(folder) / "bank.sqlite")
            client_id = bank.enroll("p01", [_vector(1), _vector(1)], 100)
            self.assertEqual(bank.balance_of(client_id), 100)
            status, balance = bank.record(
                status="paid",
                amount=40,
                client_id=client_id,
                similarity=0.8,
                antispoof_score=0.9,
                debit=True,
            )
            self.assertEqual(status, "paid")
            self.assertEqual(balance, 60)
            self.assertEqual(bank.balance_of(client_id), 60)
            status, balance = bank.record(
                status="paid",
                amount=1000,
                client_id=client_id,
                debit=True,
            )
            self.assertEqual(status, "insufficient")
            self.assertEqual(bank.balance_of(client_id), 60)
            with self.assertRaises(ValueError):
                bank.enroll("p01", [_vector(1)], 10)
            import sqlite3

            connection = sqlite3.connect(bank.path)
            try:
                tables = {
                    name
                    for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                for table in tables:
                    columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
                    self.assertNotIn("photo", columns)
                    self.assertNotIn("image", columns)
                blob = connection.execute("SELECT vector FROM embeddings").fetchone()[0]
                self.assertEqual(len(blob), EMBEDDING_DIM * 4)
            finally:
                connection.close()

    def test_names_codes_and_duplicate_account(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bank.sqlite"
            import sqlite3

            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE clients (id INTEGER PRIMARY KEY, code TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL)"
            )
            connection.execute("INSERT INTO clients(code, created_at) VALUES ('p02', '2020-01-01')")
            connection.commit()
            connection.close()

            bank = Bank(path)
            connection = sqlite3.connect(path)
            try:
                connection.execute("INSERT INTO accounts(client_id, balance) VALUES (1, 1000)")
                connection.commit()
            finally:
                connection.close()
            self.assertEqual(bank.accounts()[0]["клієнт"], "p02")

            client_id, shown = bank.register_person("Іван", "Петренко", [_vector(1)], 80)
            self.assertEqual(shown, "Петренко Іван")
            self.assertEqual(bank.balance_of(client_id), 80)
            matches = bank.find_by_name("  іван ", "ПЕТРЕНКО")
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["id"], client_id)
            bank.add_embeddings(client_id, [_vector(2)])
            self.assertEqual(bank.balance_of(client_id), 80)
            labels = {label for _client, label, _vector in bank.gallery()}
            self.assertIn("Петренко Іван", labels)
            self.assertNotIn("c1", labels)

            _second, shown_again = bank.register_person("Іван", "Петренко", [_vector(3)], 10)
            self.assertEqual(shown_again, "Петренко Іван")
            self.assertEqual(len(bank.find_by_name("Іван", "Петренко")), 2)
            bank.record(status="paid", amount=5, client_id=client_id, similarity=0.9, debit=True)
            self.assertEqual(bank.transactions()[0]["клієнт"], "Петренко Іван")
            self.assertEqual(bank.balance_of(client_id), 75)


class MetricTests(unittest.TestCase):
    def test_separable_recognition(self) -> None:
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        genuine = np.array([True, True, False, False])
        points = recognition_curve(scores, genuine)
        point = operating_point(points, 0.01)
        self.assertIsNotNone(point)
        assert point is not None
        self.assertEqual(point.far, 0)
        self.assertEqual(point.tar, 1)
        eer = equal_error_point(points)
        self.assertLessEqual(eer.far, 0)
        self.assertLessEqual(eer.frr, 0)

    def test_spoof_threshold_blocks_attack(self) -> None:
        probs = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.8, 0.1, 0.1],
                [0.62, 0.3, 0.08],
                [0.2, 0.7, 0.1],
            ],
            dtype=np.float64,
        )
        is_live = np.array([True, True, False, True])
        threshold = choose_live_threshold(probs, is_live, apcer_target=0.0)
        accepted = live_mask_from_probs(probs, threshold)
        rates = presentation_rates(is_live, accepted)
        self.assertEqual(rates.apcer, 0)
        self.assertFalse(bool(accepted[2]))


class FolderTests(unittest.TestCase):
    def test_person_does_not_leak_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for person in ("p01", "p02", "p03", "p04"):
                for label in ("live", "print", "phone", "laptop"):
                    directory = root / label / person
                    directory.mkdir(parents=True)
                    (directory / "a.jpg").write_bytes(b"not-a-real-image")
            samples = collect_samples(root)
            self.assertTrue(all(sample.label == 2 for sample in samples if "phone" in str(sample.path) or "laptop" in str(sample.path)))
            train, val, test = split_by_person(samples, seed=1)
            groups = [train, val, test]
            seen: dict[str, int] = {}
            for index, group in enumerate(groups):
                for sample in group:
                    if sample.person in seen:
                        self.assertEqual(seen[sample.person], index)
                    seen[sample.person] = index
            self.assertEqual(set(seen), {"p01", "p02", "p03", "p04"})
            balanced = balance_classes(
                [
                    Sample(Path("a"), 0, "p01"),
                    Sample(Path("b"), 0, "p01"),
                    Sample(Path("c"), 1, "p02"),
                ],
                seed=0,
            )
            self.assertEqual(len(balanced), 2)


if __name__ == "__main__":
    unittest.main()
