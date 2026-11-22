"""Перевірки банку, порогів, метрик і розбиття без камери й без нейромереж."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.bank import Bank, check_pin, hash_pin, to_kop
from src.config import EMBEDDING_DIM, PIN_MAX_FAILURES
from src.folders import Sample, balance_classes, collect_samples, person_folds, split_by_person, stratified_limit
from src.gallery import best_match, consistent, cosine, l2_normalize, template
from src.metrics import (
    bpcer_at_apcer,
    choose_live_threshold,
    cross_validate,
    equal_error_point,
    live_mask_from_probs,
    min_resolvable_far,
    operating_point,
    presentation_rates,
    rates_at,
    recognition_curve,
)


def _vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.normal(size=EMBEDDING_DIM).astype(np.float32))


def _near(vector: np.ndarray, seed: int, noise: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(vector + noise * rng.normal(size=EMBEDDING_DIM).astype(np.float32) / np.sqrt(EMBEDDING_DIM))


class GalleryTests(unittest.TestCase):
    def test_same_vector_is_one(self) -> None:
        vector = _vector(1)
        self.assertAlmostEqual(cosine(vector, vector), 1.0, places=5)

    def test_threshold_rejects_other_person(self) -> None:
        match = best_match(_vector(2), [(1, "p01", _vector(1))], threshold=0.99)
        self.assertFalse(match.accepted)
        self.assertIsNone(match.code)
        self.assertEqual(match.reason, "below_threshold")

    def test_score_is_template_not_max(self) -> None:
        """Одне випадково схоже фото серед багатьох не повинно пропускати чужого."""
        query = _vector(3)
        entries = [(7, "p07", query)] + [(7, "p07", _vector(seed)) for seed in range(10, 19)]
        match = best_match(query, entries, threshold=0.5)
        self.assertFalse(match.accepted)
        expected = cosine(query, template([vector for _id, _code, vector in entries]))
        self.assertAlmostEqual(match.similarity or 0, expected, places=5)

    def test_ambiguous_when_two_clients_are_close(self) -> None:
        base = _vector(5)
        entries = [(1, "a", base), (2, "b", _near(base, 6, 0.1))]
        match = best_match(base, entries, threshold=0.4, margin=0.05)
        self.assertFalse(match.accepted)
        self.assertEqual(match.reason, "ambiguous")
        clear = best_match(base, [(1, "a", base), (2, "b", _vector(9))], threshold=0.4, margin=0.05)
        self.assertTrue(clear.accepted)
        self.assertEqual(clear.code, "a")

    def test_consistent_drops_outlier(self) -> None:
        person = _vector(20)
        vectors = [person, _near(person, 21, 0.3), _near(person, 22, 0.3), _vector(99)]
        keep, _template = consistent(vectors, threshold=0.5)
        self.assertEqual(keep, [0, 1, 2])


class BankTests(unittest.TestCase):
    def test_enroll_debit_and_no_photos(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            bank = Bank(Path(folder) / "bank.sqlite")
            client_id = bank.enroll("p01", [_vector(1), _vector(1)], 100)
            self.assertEqual(bank.balance_of(client_id), 100)
            status, balance = bank.record(status="paid", amount=40, client_id=client_id, similarity=0.8, debit=True)
            self.assertEqual((status, balance), ("paid", 60))
            status, _balance = bank.record(status="paid", amount=1000, client_id=client_id, debit=True)
            self.assertEqual(status, "insufficient")
            self.assertEqual(bank.balance_of(client_id), 60)
            with self.assertRaises(ValueError):
                bank.enroll("p01", [_vector(1)], 10)
            connection = sqlite3.connect(bank.path)
            try:
                tables = {name for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                for table in tables:
                    columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
                    self.assertNotIn("photo", columns)
                    self.assertNotIn("image", columns)
                blob = connection.execute("SELECT vector FROM embeddings").fetchone()[0]
                self.assertEqual(len(blob), EMBEDDING_DIM * 4)
            finally:
                connection.close()

    def test_money_in_kopiyky(self) -> None:
        self.assertEqual(to_kop(0.1) + to_kop(0.2), 30)
        self.assertEqual(to_kop("19.995"), 2000)
        with tempfile.TemporaryDirectory() as folder:
            bank = Bank(Path(folder) / "bank.sqlite")
            client_id = bank.enroll("p01", [_vector(1)], 0.3)
            for _ in range(3):
                bank.record(status="paid", amount=0.1, client_id=client_id, debit=True)
            self.assertEqual(bank.balance_of(client_id), 0.0)
            status, _ = bank.record(status="paid", amount=0.01, client_id=client_id, debit=True)
            self.assertEqual(status, "insufficient")

    def test_pin_hash_and_lockout(self) -> None:
        stored = hash_pin("1234")
        self.assertNotIn("1234", stored)
        self.assertTrue(check_pin("1234", stored))
        self.assertFalse(check_pin("4321", stored))
        with self.assertRaises(ValueError):
            hash_pin("12a4")
        with tempfile.TemporaryDirectory() as folder:
            bank = Bank(Path(folder) / "bank.sqlite")
            client_id, _shown = bank.register_person("Іван", "Петренко", [_vector(1)], 100, pin="1234")
            self.assertTrue(bank.has_pin(client_id))
            self.assertEqual(bank.verify_pin(client_id, "0000"), "wrong")
            self.assertEqual(bank.verify_pin(client_id, "1234"), "ok")
            states = [bank.verify_pin(client_id, "0000") for _ in range(PIN_MAX_FAILURES)]
            self.assertEqual(states[-1], "locked")
            self.assertEqual(bank.verify_pin(client_id, "1234"), "locked")
            other, _ = bank.register_person("Олена", "Іваненко", [_vector(2)], 100)
            self.assertEqual(bank.verify_pin(other, "1234"), "none")

    def test_migrates_old_real_balance(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bank.sqlite"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE clients (id INTEGER PRIMARY KEY, code TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL);
                CREATE TABLE embeddings (id INTEGER PRIMARY KEY, client_id INTEGER NOT NULL, vector BLOB NOT NULL);
                CREATE TABLE accounts (client_id INTEGER PRIMARY KEY, balance REAL NOT NULL);
                CREATE TABLE transactions (id INTEGER PRIMARY KEY, client_id INTEGER, amount REAL, similarity REAL,
                    antispoof_score REAL, status TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL);
                INSERT INTO clients(code, created_at) VALUES ('p02', '2020-01-01');
                INSERT INTO accounts(client_id, balance) VALUES (1, 99.99);
                INSERT INTO transactions(client_id, amount, status, created_at) VALUES (1, 12.5, 'paid', '2020-01-01');
                """
            )
            connection.commit()
            connection.close()
            bank = Bank(path)
            self.assertEqual(bank.balance_of(1), 99.99)
            self.assertEqual(bank.transactions()[0]["сума"], 12.5)
            self.assertEqual(bank.accounts()[0]["клієнт"], "p02")
            status, balance = bank.record(status="paid", amount=0.99, client_id=1, debit=True)
            self.assertEqual((status, balance), ("paid", 99.0))

    def test_names_codes_and_duplicate_account(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            bank = Bank(Path(folder) / "bank.sqlite")
            client_id, shown = bank.register_person("Іван", "Петренко", [_vector(1)], 80)
            self.assertEqual(shown, "Петренко Іван")
            self.assertEqual(bank.balance_of(client_id), 80)
            matches = bank.find_by_name("  іван ", "ПЕТРЕНКО")
            self.assertEqual([item["id"] for item in matches], [client_id])
            bank.add_embeddings(client_id, [_vector(2)])
            self.assertEqual(bank.balance_of(client_id), 80)
            self.assertEqual(len(bank.client_vectors(client_id)), 2)
            labels = {label for _client, label, _vector in bank.gallery()}
            self.assertEqual(labels, {"Петренко Іван"})
            bank.register_person("Іван", "Петренко", [_vector(3)], 10)
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
        assert point is not None
        self.assertEqual((point.far, point.tar), (0, 1))
        eer = equal_error_point(points)
        self.assertEqual((eer.far, eer.frr), (0, 0))

    def test_missing_face_counts_as_reject(self) -> None:
        scores = np.array([0.9, np.nan, 0.1, np.nan])
        genuine = np.array([True, True, False, False])
        rates = rates_at(scores, genuine, 0.5)
        self.assertEqual(rates["frr"], 0.5)
        self.assertEqual(rates["fnmr"], 0.0)
        self.assertEqual(rates["far"], 0.0)
        self.assertEqual(rates["fta"], 0.5)

    def test_cross_validation_does_not_see_test_fold(self) -> None:
        rng = np.random.default_rng(0)
        genuine = np.tile([True] * 50 + [False] * 50, 4)
        scores = np.where(genuine, rng.normal(0.7, 0.1, len(genuine)), rng.normal(0.1, 0.1, len(genuine)))
        folds = np.repeat(np.arange(4), 100)
        report = cross_validate(scores, genuine, folds, 0.01)
        self.assertEqual(len(report["folds"]), 4)
        self.assertGreater(report["tar"]["mean"], 0.9)
        self.assertAlmostEqual(report["resolvable_far"], 3 / 150)
        self.assertAlmostEqual(min_resolvable_far(3000), 0.001)

    def test_worst_attack_type_is_apcer(self) -> None:
        labels = np.array([0, 0, 1, 1, 2, 2])
        accepted = np.array([True, True, False, False, True, False])
        rates = presentation_rates(labels, accepted)
        self.assertEqual(rates.apcer_by_type, {"print": 0.0, "screen": 0.5})
        self.assertEqual(rates.apcer, 0.5)
        self.assertEqual(rates.bpcer, 0.0)

    def test_unknown_spoof_label_is_an_attack(self) -> None:
        rates = presentation_rates(np.array([0, 3, 3]), np.array([True, True, False]))
        self.assertEqual(rates.apcer_by_type, {"spoof": 0.5})
        self.assertEqual(rates.attacks, 2)

    def test_spoof_threshold_blocks_attack(self) -> None:
        probs = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.62, 0.3, 0.08], [0.2, 0.7, 0.1]])
        is_live = np.array([True, True, False, True])
        threshold = choose_live_threshold(probs, is_live, apcer_target=0.0)
        accepted = live_mask_from_probs(probs, threshold)
        self.assertEqual(presentation_rates(is_live, accepted).apcer, 0)
        self.assertFalse(bool(accepted[2]))
        at = bpcer_at_apcer(probs, np.array([0, 0, 1, 0]), targets=(0.0,))
        self.assertAlmostEqual(at["0"], 1 / 3)


class FolderTests(unittest.TestCase):
    def _tree(self, root: Path, people=("p01", "p02", "p03", "p04"), kinds=("live", "print", "phone", "laptop")) -> None:
        for person in people:
            for label in kinds:
                directory = root / label / person
                directory.mkdir(parents=True)
                for index in range(3):
                    (directory / f"{index}.jpg").write_bytes(b"not-a-real-image")

    def test_person_does_not_leak_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._tree(root)
            samples = collect_samples(root, tag="custom")
            self.assertTrue(all(sample.person.startswith("custom:") for sample in samples))
            self.assertTrue(all(sample.label == 2 for sample in samples if sample.path.parts[-3] in {"phone", "laptop"}))
            train, val, test = split_by_person(samples, seed=1)
            seen: dict[str, int] = {}
            for index, group in enumerate([train, val, test]):
                for sample in group:
                    self.assertEqual(seen.setdefault(sample.person, index), index)
            self.assertEqual(len(seen), 4)
            balanced = balance_classes([Sample(Path("a"), 0, "p01"), Sample(Path("b"), 0, "p01"), Sample(Path("c"), 1, "p02")])
            self.assertEqual(len(balanced), 2)

    def test_limit_keeps_every_class_and_folds_cover_people(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._tree(root)
            samples = collect_samples(root)
            limited = stratified_limit(samples, 6, seed=0)
            self.assertEqual(len(limited), 6)
            self.assertEqual({sample.label for sample in limited}, {0, 1, 2})
            folds = person_folds(samples, 4, seed=0)
            self.assertEqual(set().union(*folds), {"p01", "p02", "p03", "p04"})
            self.assertEqual(sum(len(fold) for fold in folds), 4)

    def test_unknown_spoof_folder(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for label in ("live", "spoof"):
                (root / label).mkdir()
                (root / label / "1.jpg").write_bytes(b"x")
            labels = {sample.label for sample in collect_samples(root)}
            self.assertEqual(labels, {0, 3})


if __name__ == "__main__":
    unittest.main()
