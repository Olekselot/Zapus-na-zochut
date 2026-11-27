"""Локальний банк. У SQLite лежать клієнти, вектори, баланс у копійках і журнал, без фото."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

from src.config import (
    DB_PATH,
    EMBEDDING_DIM,
    PIN_LOCK_MAX_SECONDS,
    PIN_LOCK_SECONDS,
    PIN_MAX_FAILURES,
    ensure_runtime_dirs,
)
from src.gallery import l2_normalize

PIN_PATTERN = re.compile(r"^\d{4,6}$")
PIN_ITERATIONS = 200_000

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS clients (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    first_name TEXT NOT NULL DEFAULT '',
    last_name TEXT NOT NULL DEFAULT '',
    pin_hash TEXT NOT NULL DEFAULT '',
    pin_failures INTEGER NOT NULL DEFAULT 0,
    pin_locked_until REAL NOT NULL DEFAULT 0,
    pin_lock_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY,
    client_id INTEGER NOT NULL REFERENCES clients(id),
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_client ON embeddings(client_id);
CREATE TABLE IF NOT EXISTS accounts (
    client_id INTEGER PRIMARY KEY REFERENCES clients(id),
    balance_kop INTEGER NOT NULL CHECK (balance_kop >= 0)
);
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY,
    client_id INTEGER REFERENCES clients(id),
    amount_kop INTEGER,
    similarity REAL,
    antispoof_score REAL,
    status TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def to_kop(value: float | int | str) -> int:
    """Гривні в цілі копійки без похибок float: 0.1 + 0.2 грн дає рівно 30 коп."""
    return int((Decimal(str(value)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def from_kop(value: int) -> float:
    return int(value) / 100


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


def valid_pin(pin: str) -> bool:
    return bool(PIN_PATTERN.match(pin or ""))


def hash_pin(pin: str) -> str:
    if not valid_pin(pin):
        raise ValueError("PIN має містити 4–6 цифр.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, PIN_ITERATIONS)
    return f"pbkdf2_sha256${PIN_ITERATIONS}${salt.hex()}${digest.hex()}"


def check_pin(pin: str, stored: str) -> bool:
    try:
        _scheme, iterations, salt, expected = stored.split("$")
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", (pin or "").encode("utf-8"), bytes.fromhex(salt), int(iterations))
    return hmac.compare_digest(digest.hex(), expected)


def _pin_state(pin_hash: str, locked_until: float, now: float) -> str:
    if not pin_hash:
        return "ні"
    if locked_until > now:
        return "заблоковано до " + datetime.fromtimestamp(locked_until).strftime("%H:%M")
    return "так"


class Bank:
    def __init__(self, path: Path | None = None, clock=time.time) -> None:
        ensure_runtime_dirs()
        self._clock = clock
        self.path = Path(path) if path is not None else DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._lock, self._connect() as connection:
            self._migrate_before(connection)
            connection.executescript(SCHEMA)
            self._migrate_after(connection)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _write(self):
        """Одна транзакція з блокуванням запису: перевірка балансу і списання не розірвуться."""
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
        return {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}

    def _migrate_before(self, connection: sqlite3.Connection) -> None:
        """Стара таблиця рахунків з балансом REAL перебудовується в копійки."""
        columns = self._columns(connection, "accounts")
        if columns and "balance_kop" not in columns and "balance" in columns:
            connection.executescript(
                """
                BEGIN;
                CREATE TABLE accounts_kop (
                    client_id INTEGER PRIMARY KEY REFERENCES clients(id),
                    balance_kop INTEGER NOT NULL CHECK (balance_kop >= 0)
                );
                INSERT INTO accounts_kop(client_id, balance_kop)
                    SELECT client_id, CAST(ROUND(balance * 100) AS INTEGER) FROM accounts;
                DROP TABLE accounts;
                ALTER TABLE accounts_kop RENAME TO accounts;
                COMMIT;
                """
            )

    def _migrate_after(self, connection: sqlite3.Connection) -> None:
        clients = self._columns(connection, "clients")
        for name, ddl in (
            ("first_name", "TEXT NOT NULL DEFAULT ''"),
            ("last_name", "TEXT NOT NULL DEFAULT ''"),
            ("pin_hash", "TEXT NOT NULL DEFAULT ''"),
            ("pin_failures", "INTEGER NOT NULL DEFAULT 0"),
            ("pin_locked_until", "REAL NOT NULL DEFAULT 0"),
            ("pin_lock_count", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if name not in clients:
                connection.execute(f"ALTER TABLE clients ADD COLUMN {name} {ddl}")
        transactions = self._columns(connection, "transactions")
        if "amount_kop" not in transactions:
            connection.execute("ALTER TABLE transactions ADD COLUMN amount_kop INTEGER")
            if "amount" in transactions:
                connection.execute(
                    "UPDATE transactions SET amount_kop = CAST(ROUND(amount * 100) AS INTEGER) WHERE amount IS NOT NULL"
                )

    @staticmethod
    def _next_code(connection: sqlite3.Connection) -> str:
        used = {str(row["code"]) for row in connection.execute("SELECT code FROM clients")}
        number = 1
        while f"c{number}" in used:
            number += 1
        return f"c{number}"

    @staticmethod
    def _insert_client(
        connection: sqlite3.Connection,
        code: str,
        first_name: str,
        last_name: str,
        stored: list[bytes],
        balance: float,
        pin_hash: str,
    ) -> int:
        balance_kop = to_kop(balance)
        if balance_kop < 0:
            raise ValueError("баланс не може бути від’ємним")
        cursor = connection.execute(
            "INSERT INTO clients(code, first_name, last_name, pin_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (code, first_name.strip(), last_name.strip(), pin_hash, _now()),
        )
        client_id = int(cursor.lastrowid)
        connection.executemany(
            "INSERT INTO embeddings(client_id, vector) VALUES (?, ?)",
            [(client_id, blob) for blob in stored],
        )
        connection.execute("INSERT INTO accounts(client_id, balance_kop) VALUES (?, ?)", (client_id, balance_kop))
        return client_id

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
        with self._write() as connection:
            try:
                return self._insert_client(connection, code, first_name, last_name, stored, balance, "")
            except sqlite3.IntegrityError as error:
                raise ValueError(f"клієнт {code} уже є в базі") from error

    def register_person(
        self,
        first_name: str,
        last_name: str,
        embeddings: list[np.ndarray],
        balance: float,
        pin: str | None = None,
        pin_hash: str | None = None,
    ) -> tuple[int, str]:
        """Новий клієнт отримує внутрішній код cN. Назовні повертається «Прізвище Ім’я»."""
        if not embeddings:
            raise ValueError("потрібен хоча б один вектор обличчя")
        stored = [encode_vector(vector) for vector in embeddings]
        if pin_hash is None:
            pin_hash = hash_pin(pin) if pin else ""
        with self._write() as connection:
            code = self._next_code(connection)
            client_id = self._insert_client(connection, code, first_name, last_name, stored, balance, pin_hash)
            return client_id, display_name(first_name, last_name, code)

    def add_embeddings(self, client_id: int, embeddings: list[np.ndarray]) -> int:
        """Додати вектори до наявного рахунку. Баланс не змінюється."""
        if not embeddings:
            raise ValueError("потрібен хоча б один вектор обличчя")
        stored = [encode_vector(vector) for vector in embeddings]
        with self._write() as connection:
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
            rows = connection.execute("SELECT id, code, first_name, last_name FROM clients ORDER BY id").fetchall()
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

    def client_vectors(self, client_id: int) -> list[np.ndarray]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT vector FROM embeddings WHERE client_id = ? ORDER BY id", (int(client_id),)
            ).fetchall()
        return [decode_vector(row["vector"]) for row in rows]

    def balance_of(self, client_id: int) -> float:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT balance_kop FROM accounts WHERE client_id = ?", (client_id,)).fetchone()
        if row is None:
            raise KeyError(client_id)
        return from_kop(row["balance_kop"])

    def has_pin(self, client_id: int) -> bool:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT pin_hash FROM clients WHERE id = ?", (int(client_id),)).fetchone()
        return bool(row and row["pin_hash"])

    def verify_pin(self, client_id: int, pin: str) -> str:
        """ok, wrong, locked або none.

        Після PIN_MAX_FAILURES помилок поспіль PIN блокується на PIN_LOCK_SECONDS,
        кожне наступне блокування вдвічі довше, але не довше за добу. Правильний PIN
        або адміністратор знімають блокування і скидають лічильники.
        """
        now = float(self._clock())
        with self._write() as connection:
            row = connection.execute(
                "SELECT pin_hash, pin_failures, pin_locked_until, pin_lock_count FROM clients WHERE id = ?",
                (int(client_id),),
            ).fetchone()
            if row is None or not row["pin_hash"]:
                return "none"
            if float(row["pin_locked_until"]) > now:
                return "locked"
            if check_pin(pin, str(row["pin_hash"])):
                connection.execute(
                    "UPDATE clients SET pin_failures = 0, pin_locked_until = 0, pin_lock_count = 0 WHERE id = ?",
                    (int(client_id),),
                )
                return "ok"
            failures = int(row["pin_failures"]) + 1
            if failures < PIN_MAX_FAILURES:
                connection.execute("UPDATE clients SET pin_failures = ? WHERE id = ?", (failures, int(client_id)))
                return "wrong"
            locks = int(row["pin_lock_count"]) + 1
            seconds = min(PIN_LOCK_SECONDS * 2 ** (locks - 1), PIN_LOCK_MAX_SECONDS)
            connection.execute(
                "UPDATE clients SET pin_failures = 0, pin_lock_count = ?, pin_locked_until = ? WHERE id = ?",
                (locks, now + seconds, int(client_id)),
            )
            return "locked"

    def pin_locked_until(self, client_id: int) -> float | None:
        """Момент (time.time), до якого PIN заблоковано, або None."""
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT pin_locked_until FROM clients WHERE id = ?", (int(client_id),)).fetchone()
        if row is None or float(row["pin_locked_until"]) <= float(self._clock()):
            return None
        return float(row["pin_locked_until"])

    def unlock_pin(self, code: str) -> bool:
        """Розблокування адміністратором за кодом клієнта (c1, c2...)."""
        with self._write() as connection:
            cursor = connection.execute(
                "UPDATE clients SET pin_failures = 0, pin_locked_until = 0, pin_lock_count = 0 WHERE code = ?",
                (str(code).strip(),),
            )
            return cursor.rowcount == 1

    def accounts(self) -> list[dict[str, object]]:
        now = float(self._clock())
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.code, c.first_name, c.last_name, c.pin_hash, c.pin_locked_until, a.balance_kop, c.created_at,
                       (SELECT COUNT(*) FROM embeddings e WHERE e.client_id = c.id) AS vectors
                FROM clients c
                JOIN accounts a ON a.client_id = c.id
                ORDER BY c.last_name, c.first_name, c.id
                """
            ).fetchall()
        return [
            {
                "код": str(row["code"]),
                "клієнт": display_name(str(row["first_name"]), str(row["last_name"]), str(row["code"])),
                "баланс": from_kop(row["balance_kop"]),
                "векторів": int(row["vectors"]),
                "PIN": _pin_state(str(row["pin_hash"]), float(row["pin_locked_until"]), now),
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

        Списання — один UPDATE з умовою balance_kop >= суми всередині BEGIN IMMEDIATE,
        тож дві одночасні оплати не зійдуть у мінус. Повертає фактичний статус і баланс.
        """
        amount_kop = None if amount is None else to_kop(amount)
        with self._write() as connection:
            new_balance: float | None = None
            stored_status = status
            if debit:
                if client_id is None or amount_kop is None:
                    raise ValueError("списання потребує клієнта і суми")
                if amount_kop <= 0:
                    raise ValueError("сума списання має бути більшою за нуль")
                cursor = connection.execute(
                    "UPDATE accounts SET balance_kop = balance_kop - ? WHERE client_id = ? AND balance_kop >= ?",
                    (amount_kop, client_id, amount_kop),
                )
                row = connection.execute(
                    "SELECT balance_kop FROM accounts WHERE client_id = ?", (client_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(client_id)
                stored_status = "paid" if cursor.rowcount == 1 else "insufficient"
                new_balance = from_kop(row["balance_kop"])
            connection.execute(
                """
                INSERT INTO transactions(
                    client_id, amount_kop, similarity, antispoof_score, status, detail, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (client_id, amount_kop, similarity, antispoof_score, stored_status, detail, _now()),
            )
            return stored_status, new_balance

    def transactions(self, limit: int = 50) -> list[dict[str, object]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT t.created_at, t.status, c.code, c.first_name, c.last_name,
                       t.amount_kop, t.similarity, t.antispoof_score, t.detail
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
                "сума": None if row["amount_kop"] is None else from_kop(row["amount_kop"]),
                "схожість": None if row["similarity"] is None else round(float(row["similarity"]), 3),
                "живість": None if row["antispoof_score"] is None else round(float(row["antispoof_score"]), 3),
                "деталі": row["detail"],
            }
            for row in rows
        ]
