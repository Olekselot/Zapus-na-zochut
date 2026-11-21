"""Локальний банк. У SQLite лежать коди клієнтів, вектори, баланс і журнал, без фото."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import DB_PATH, EMBEDDING_DIM, ensure_runtime_dirs
from src.gallery import l2_normalize


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_name(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def display_name(first_name: str, last_name: str, code: str) -> str:
    """На екрані прізвище та ім’я. Старі записи без імені лишаються зі своїм кодом."""
    first = " ".join((first_name or "").split())
    last = " ".join((last_name or "").split())
    if last and first:
        return f"{last} {first}"
    return last or first or code


def encode_vector(vector: np.ndarray) -> bytes:
    return np.ascontiguousarray(l2_normalize(vector)).tobytes()


def decode_vector(blob: bytes) -> np.ndarray:
    values = np.frombuffer(blob, dtype=np.float32).copy()
    if values.shape != (EMBEDDING_DIM,):
        raise ValueError("у базі пошкоджений вектор обличчя")
    return values


class Bank:
    def __init__(self, path: Path | None = None) -> None:
        ensure_runtime_dirs()
        self.path = Path(path) if path is not None else DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS clients (
                    id INTEGER PRIMARY KEY,
                    code TEXT NOT NULL UNIQUE,
                    first_name TEXT NOT NULL DEFAULT '',
                    last_name TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY,
                    client_id INTEGER NOT NULL REFERENCES clients(id),
                    vector BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS accounts (
                    client_id INTEGER PRIMARY KEY REFERENCES clients(id),
                    balance REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY,
                    client_id INTEGER REFERENCES clients(id),
                    amount REAL,
                    similarity REAL,
                    antispoof_score REAL,
                    status TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                """
            )
            self._ensure_name_columns(connection)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _ensure_name_columns(connection: sqlite3.Connection) -> None:
        columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(clients)")}
        if "first_name" not in columns:
            connection.execute("ALTER TABLE clients ADD COLUMN first_name TEXT NOT NULL DEFAULT ''")
        if "last_name" not in columns:
            connection.execute("ALTER TABLE clients ADD COLUMN last_name TEXT NOT NULL DEFAULT ''")

    @staticmethod
    def _next_code(connection: sqlite3.Connection) -> str:
        used = {str(row["code"]) for row in connection.execute("SELECT code FROM clients")}
        number = 1
        while f"c{number}" in used:
            number += 1
        return f"c{number}"

    def enroll(
        self,
        code: str,
        embeddings: list[np.ndarray],
        balance: float,
        first_name: str = "",
        last_name: str = "",
    ) -> int:
        if not embeddings:
            raise ValueError("потрібен хоча б один вектор обличчя")
        stored = [encode_vector(vector) for vector in embeddings]
        with self._lock, self._connect() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO clients(code, first_name, last_name, created_at) VALUES (?, ?, ?, ?)",
                    (code, first_name.strip(), last_name.strip(), _now()),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(f"клієнт {code} уже є в базі") from error
            client_id = int(cursor.lastrowid)
            connection.executemany(
                "INSERT INTO embeddings(client_id, vector) VALUES (?, ?)",
                [(client_id, blob) for blob in stored],
            )
            connection.execute(
                "INSERT INTO accounts(client_id, balance) VALUES (?, ?)",
                (client_id, round(float(balance), 2)),
            )
            return client_id

    def register_person(
        self,
        first_name: str,
        last_name: str,
        embeddings: list[np.ndarray],
        balance: float,
    ) -> tuple[int, str]:
        """Новий клієнт отримує внутрішній код cN. Назовні повертається «Прізвище Ім’я»."""
        if not embeddings:
            raise ValueError("потрібен хоча б один вектор обличчя")
        stored = [encode_vector(vector) for vector in embeddings]
        with self._lock, self._connect() as connection:
            code = self._next_code(connection)
            cursor = connection.execute(
                "INSERT INTO clients(code, first_name, last_name, created_at) VALUES (?, ?, ?, ?)",
                (code, first_name.strip(), last_name.strip(), _now()),
            )
            client_id = int(cursor.lastrowid)
            connection.executemany(
                "INSERT INTO embeddings(client_id, vector) VALUES (?, ?)",
                [(client_id, blob) for blob in stored],
            )
            connection.execute(
                "INSERT INTO accounts(client_id, balance) VALUES (?, ?)",
                (client_id, round(float(balance), 2)),
            )
            return client_id, display_name(first_name, last_name, code)

    def add_embeddings(self, client_id: int, embeddings: list[np.ndarray]) -> int:
        """Додати вектори до наявного рахунку. Баланс не змінюється."""
        if not embeddings:
            raise ValueError("потрібен хоча б один вектор обличчя")
        stored = [encode_vector(vector) for vector in embeddings]
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT id FROM clients WHERE id = ?", (int(client_id),)).fetchone()
            if row is None:
                raise KeyError(client_id)
            connection.executemany(
                "INSERT INTO embeddings(client_id, vector) VALUES (?, ?)",
                [(int(client_id), blob) for blob in stored],
            )
            return len(stored)

    def find_by_name(self, first_name: str, last_name: str) -> list[dict[str, object]]:
        target_first = normalize_name(first_name)
        target_last = normalize_name(last_name)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id, code, first_name, last_name FROM clients ORDER BY id"
            ).fetchall()
        found: list[dict[str, object]] = []
        for row in rows:
            if normalize_name(str(row["first_name"])) != target_first:
                continue
            if normalize_name(str(row["last_name"])) != target_last:
                continue
            found.append(
                {
                    "id": int(row["id"]),
                    "name": display_name(str(row["first_name"]), str(row["last_name"]), str(row["code"])),
                }
            )
        return found

    def gallery(self) -> list[tuple[int, str, np.ndarray]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT e.client_id, c.code, c.first_name, c.last_name, e.vector
                FROM embeddings e
                JOIN clients c ON c.id = e.client_id
                ORDER BY e.client_id, e.id
                """
            ).fetchall()
        return [
            (
                int(row["client_id"]),
                display_name(str(row["first_name"]), str(row["last_name"]), str(row["code"])),
                decode_vector(row["vector"]),
            )
            for row in rows
        ]

    def balance_of(self, client_id: int) -> float:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT balance FROM accounts WHERE client_id = ?",
                (client_id,),
            ).fetchone()
        if row is None:
            raise KeyError(client_id)
        return float(row["balance"])

    def accounts(self) -> list[dict[str, object]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.code, c.first_name, c.last_name, a.balance, c.created_at,
                       (SELECT COUNT(*) FROM embeddings e WHERE e.client_id = c.id) AS vectors
                FROM clients c
                JOIN accounts a ON a.client_id = c.id
                ORDER BY c.last_name, c.first_name, c.id
                """
            ).fetchall()
        return [
            {
                "клієнт": display_name(str(row["first_name"]), str(row["last_name"]), str(row["code"])),
                "баланс": round(float(row["balance"]), 2),
                "векторів": int(row["vectors"]),
                "зареєстровано": row["created_at"],
            }
            for row in rows
        ]

    def record(
        self,
        status: str,
        amount: float | None,
        client_id: int | None = None,
        similarity: float | None = None,
        antispoof_score: float | None = None,
        detail: str = "",
        debit: bool = False,
    ) -> tuple[str, float | None]:
        """Записати операцію. Гроші змінюються лише коли debit=True і на рахунку вистачає суми.

        Повертає фактичний статус і баланс після спроби. Для відмов без клієнта баланс порожній.
        """
        with self._lock, self._connect() as connection:
            new_balance: float | None = None
            stored_status = status
            if debit:
                if client_id is None or amount is None:
                    raise ValueError("списання потребує клієнта і суми")
                row = connection.execute(
                    "SELECT balance FROM accounts WHERE client_id = ?",
                    (client_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(client_id)
                current = float(row["balance"])
                if current + 1e-9 < float(amount):
                    stored_status = "insufficient"
                    new_balance = current
                else:
                    new_balance = round(current - float(amount), 2)
                    connection.execute(
                        "UPDATE accounts SET balance = ? WHERE client_id = ?",
                        (new_balance, client_id),
                    )
                    stored_status = "paid"
            connection.execute(
                """
                INSERT INTO transactions(
                    client_id, amount, similarity, antispoof_score, status, detail, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    client_id,
                    None if amount is None else round(float(amount), 2),
                    similarity,
                    antispoof_score,
                    stored_status,
                    detail,
                    _now(),
                ),
            )
            return stored_status, new_balance

    def transactions(self, limit: int = 50) -> list[dict[str, object]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT t.created_at, t.status, c.code, c.first_name, c.last_name,
                       t.amount, t.similarity, t.antispoof_score, t.detail
                FROM transactions t
                LEFT JOIN clients c ON c.id = t.client_id
                ORDER BY t.id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [
            {
                "час": row["created_at"],
                "статус": row["status"],
                "клієнт": ""
                if row["code"] is None
                else display_name(str(row["first_name"] or ""), str(row["last_name"] or ""), str(row["code"])),
                "сума": None if row["amount"] is None else round(float(row["amount"]), 2),
                "схожість": None if row["similarity"] is None else round(float(row["similarity"]), 3),
                "живість": None if row["antispoof_score"] is None else round(float(row["antispoof_score"]), 3),
                "деталі": row["detail"],
            }
            for row in rows
        ]
