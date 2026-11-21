"""Вісім кроків оплати: кадр, обличчя, вирівнювання, живість, вектор, база, поріг, списання."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import numpy as np

from src.bank import Bank
from src.config import CLASS_UA, DEFAULT_BALANCE, load_recognition_threshold
from src.decisions import summarize_liveness
from src.faces import FaceEngine, FaceSample
from src.frames import average_embeddings
from src.gallery import Match, best_match

NAME_PATTERN = re.compile(r"^[A-Za-zА-Яа-яІіЇїЄєҐґ'’\- ]{1,40}$")


def _preview(sample: FaceSample) -> np.ndarray:
    from src.images import bgr_to_rgb_uint8

    return bgr_to_rgb_uint8(sample.aligned_bgr)


STATUS_UA = {
    "paid": "проведено",
    "spoof": "підробка",
    "unknown": "невідомий",
    "no_face": "немає обличчя",
    "no_model": "немає моделі",
    "insufficient": "недостатньо коштів",
    "error": "помилка",
}


@dataclass
class Decision:
    status: str
    message: str
    client_code: str | None = None
    similarity: float | None = None
    live_score: float | None = None
    attack_label: str | None = None
    balance: float | None = None
    frames_used: int = 0
    amount: float | None = None
    title: str = ""

    def markdown(self) -> str:
        return self.message

    def html(self) -> str:
        if self.status in {"paid", "enrolled"}:
            background, color = "#e5f6ea", "#14532d"
        elif self.status == "ask":
            background, color = "#e8f1fb", "#1e3a5f"
        else:
            background, color = "#fdecea", "#7f1d1d"
        ink = f"color:{color} !important;opacity:1;"
        title = html.escape(self.title or STATUS_UA.get(self.status, "Відповідь"))
        chunks = [
            f'<div style="{ink}font-size:2rem;font-weight:700;line-height:1.2;">{title}</div>'
        ]
        if self.client_code:
            chunks.append(
                f'<div style="{ink}font-size:1.45rem;margin-top:10px;">{html.escape(self.client_code)}</div>'
            )
        facts: list[str] = []
        if self.amount is not None:
            facts.append(f"Сума {self.amount:.2f} грн")
        if self.balance is not None:
            facts.append(f"Баланс {self.balance:.2f} грн")
        if self.similarity is not None:
            facts.append(f"Схожість {self.similarity:.3f}")
        if self.live_score is not None:
            facts.append(f"Живість {self.live_score:.3f}")
        elif self.status == "paid":
            facts.append("Перевірку живості вимкнено")
        if facts and self.status == "paid":
            chunks.append(
                f'<div style="{ink}margin-top:16px;font-size:1.15rem;line-height:1.6;">'
                + "<br>".join(html.escape(item) for item in facts)
                + "</div>"
            )
        elif self.message:
            chunks.append(
                f'<p style="{ink}margin:16px 0 0;font-size:1.1rem;line-height:1.5;">{html.escape(self.message)}</p>'
            )
        return (
            f'<style>.pay-card, .pay-card * {{ color: {color} !important; }}</style>'
            f'<div class="pay-card" style="max-width:520px;margin:8px auto 20px;text-align:center;padding:28px 24px;'
            "border-radius:18px;box-shadow:0 10px 30px rgba(0,0,0,.08);"
            f'{ink}background:{background};">'
            + "".join(chunks)
            + "</div>"
        )


class Pipeline:
    def __init__(self, bank: Bank, faces: FaceEngine, antispoof) -> None:
        self.bank = bank
        self.faces = faces
        self.antispoof = antispoof
        self.antispoof_enabled = True

    def toggle_antispoof(self) -> bool:
        """Повертає новий стан: True, якщо перевірка живості знову увімкнена."""
        self.antispoof_enabled = not self.antispoof_enabled
        return self.antispoof_enabled

    def start_registration(
        self,
        first_name: str,
        last_name: str,
        images: list[np.ndarray],
        balance: float = DEFAULT_BALANCE,
    ) -> tuple[Decision, dict | None]:
        try:
            first = _clean_name(first_name, "Ім’я")
            last = _clean_name(last_name, "Прізвище")
        except ValueError as error:
            return Decision("error", str(error), title="Помилка"), None
        if balance <= 0:
            return Decision("error", "Початковий баланс має бути більшим за нуль.", title="Помилка"), None
        samples = self._samples(images)
        if not samples:
            return (
                Decision(
                    "no_face",
                    "На жодному фото не знайдено обличчя. Зніміть анфас при рівному світлі.",
                    title="Обличчя не знайдено",
                ),
                None,
            )
        embeddings = [sample.embedding.astype(np.float32).tolist() for sample in samples]
        note = _photo_note(len(samples), len(images))
        matches = self.bank.find_by_name(first, last)
        if len(matches) >= 2:
            return (
                Decision(
                    "blocked",
                    "На жаль, ця дія недоступна. Клієнтів із таким прізвищем та ім’ям уже кілька, "
                    "і незрозуміло, чий це рахунок.",
                    title="Дія недоступна",
                ),
                None,
            )
        if len(matches) == 1:
            shown = str(matches[0]["name"])
            return (
                Decision(
                    "ask",
                    f"У банку вже є {shown}. Це ваш рахунок?",
                    client_code=shown,
                    title="Це ваш рахунок?",
                ),
                {
                    "first": first,
                    "last": last,
                    "balance": float(balance),
                    "embeddings": embeddings,
                    "client_id": int(matches[0]["id"]),
                    "display": shown,
                    "note": note,
                    "photos": len(images),
                },
            )
        return self._create_person(first, last, embeddings, balance, len(images)), None

    def confirm_existing(self, pending: dict) -> Decision:
        vectors = [np.asarray(vector, dtype=np.float32) for vector in pending["embeddings"]]
        added = self.bank.add_embeddings(int(pending["client_id"]), vectors)
        balance = self.bank.balance_of(int(pending["client_id"]))
        return Decision(
            "enrolled",
            f"Фото додано до рахунку {pending['display']}. Нових векторів: {added}. "
            f"Баланс без змін: {balance:.2f} грн. Фото в базу не записувались.{pending['note']}",
            client_code=str(pending["display"]),
            balance=balance,
            frames_used=added,
            title="Фото додано",
        )

    def confirm_new(self, pending: dict) -> Decision:
        return self._create_person(
            str(pending["first"]),
            str(pending["last"]),
            list(pending["embeddings"]),
            float(pending["balance"]),
            int(pending["photos"]),
        )

    def _create_person(
        self,
        first: str,
        last: str,
        embeddings: list,
        balance: float,
        photo_count: int,
    ) -> Decision:
        vectors = [np.asarray(vector, dtype=np.float32) for vector in embeddings]
        try:
            _client_id, shown = self.bank.register_person(first, last, vectors, balance)
        except ValueError as error:
            return Decision("error", str(error), title="Помилка")
        note = _photo_note(len(vectors), photo_count)
        return Decision(
            "enrolled",
            f"Клієнта {shown} зареєстровано. Збережено векторів: {len(vectors)}. "
            f"Баланс {balance:.2f} грн. Фото в базу не записувались.{note}",
            client_code=shown,
            balance=float(balance),
            frames_used=len(vectors),
            title="Клієнта зареєстровано",
        )

    def pay(self, images: list[np.ndarray], amount: float, min_faces: int = 1) -> tuple[Decision, np.ndarray | None]:
        if amount <= 0:
            return Decision("error", "Сума має бути більшою за нуль.", title="Помилка"), None
        amount = round(float(amount), 2)
        samples = self._samples(images)
        preview = None if not samples else _preview(samples[0])
        if len(samples) < min_faces:
            decision = Decision(
                "no_face",
                "Обличчя не знайдено. Подивіться в камеру з відстані 40–70 см при рівному світлі.",
                frames_used=len(samples),
                amount=amount,
                title="Обличчя не знайдено",
            )
            self._log(decision, amount)
            return decision, preview

        live_score = None
        if self.antispoof_enabled and not self.antispoof.available:
            match = self._match(samples)
            who = ""
            if match.code is not None and match.similarity is not None:
                who = f" Найближчий клієнт {match.code}, схожість {match.similarity:.3f}, але гроші не списано."
            elif match.similarity is not None:
                who = f" Найближча схожість {match.similarity:.3f} нижче порогу, гроші не списано."
            decision = Decision(
                "no_model",
                "Захист від підробки ще не навчений, тому оплату заблоковано." + who,
                client_code=match.code,
                similarity=match.similarity,
                frames_used=len(samples),
                amount=amount,
                title="Оплату заблоковано",
            )
            self._log(decision, amount, client_id=match.client_id if match.accepted else None)
            return decision, preview

        if self.antispoof_enabled:
            probs = self.antispoof.predict_proba([sample.aligned_bgr for sample in samples])
            is_live, label, live_score = summarize_liveness(probs, self.antispoof.threshold)
            if not is_live:
                decision = Decision(
                    "spoof",
                    f"Відмова: спроба підробки ({CLASS_UA.get(label, label)}). Оплату не проведено. "
                    f"Оцінка живості {live_score:.3f}.",
                    live_score=live_score,
                    attack_label=label,
                    frames_used=len(samples),
                    amount=amount,
                    title="Спроба підробки",
                )
                self._log(decision, amount)
                return decision, preview

        match = self._match(samples)
        if live_score is None:
            detail = f"кадрів {len(samples)}, захист вимкнено"
            live_note = " Перевірку живості вимкнено."
        else:
            detail = f"кадрів {len(samples)}, живість {live_score:.3f}"
            live_note = f" Живість {live_score:.3f}."
        if not match.accepted:
            similarity = "немає бази" if match.similarity is None else f"{match.similarity:.3f}"
            decision = Decision(
                "unknown",
                "Відмова: клієнта не знайдено. "
                f"Найближча схожість {similarity}, поріг {load_recognition_threshold():.3f}.",
                similarity=match.similarity,
                live_score=live_score,
                frames_used=len(samples),
                amount=amount,
                title="Клієнта не знайдено",
            )
            self._log(decision, amount)
            return decision, preview

        status, balance = self.bank.record(
            status="paid",
            amount=amount,
            client_id=match.client_id,
            similarity=match.similarity,
            antispoof_score=live_score,
            detail=detail,
            debit=True,
        )
        if status == "insufficient":
            decision = Decision(
                "insufficient",
                f"Недостатньо коштів: на рахунку {balance:.2f} грн, потрібно {amount:.2f}.",
                client_code=match.code,
                similarity=match.similarity,
                live_score=live_score,
                balance=balance,
                frames_used=len(samples),
                amount=amount,
                title="Недостатньо коштів",
            )
            return decision, preview
        decision = Decision(
            "paid",
            f"Оплату проведено. Клієнт {match.code}, списано {amount:.2f} грн. "
            f"Баланс {balance:.2f} грн. Схожість {match.similarity:.3f}.{live_note}",
            client_code=match.code,
            similarity=match.similarity,
            live_score=live_score,
            balance=balance,
            frames_used=len(samples),
            amount=amount,
            title="Оплату проведено",
        )
        return decision, preview

    def _samples(self, images: list[np.ndarray]) -> list[FaceSample]:
        crop_size = self.antispoof.image_size if self.antispoof.available else 224
        found: list[FaceSample] = []
        for image in images:
            sample = self.faces.analyze(image, crop_size=crop_size)
            if sample is not None:
                found.append(sample)
        return found

    def _match(self, samples: list[FaceSample]) -> Match:
        embedding = average_embeddings([sample.embedding for sample in samples])
        return best_match(embedding, self.bank.gallery(), load_recognition_threshold())

    def _log(self, decision: Decision, amount: float, client_id: int | None = None) -> None:
        self.bank.record(
            status=decision.status,
            amount=amount,
            client_id=client_id,
            similarity=decision.similarity,
            antispoof_score=decision.live_score,
            detail=decision.attack_label or "",
            debit=False,
        )


def _clean_name(value: str, label: str) -> str:
    cleaned = " ".join(value.strip().split())
    if not cleaned or not NAME_PATTERN.match(cleaned) or not any(char.isalpha() for char in cleaned):
        raise ValueError(f"{label} має містити літери, без цифр, до 40 символів.")
    return cleaned


def _photo_note(kept: int, total: int) -> str:
    extra = ""
    if kept < total:
        extra = f" Пропущено фото без обличчя: {total - kept}."
    if kept < 3:
        extra += " Для стабільної оплати краще додати щонайменше 8 фото."
    return extra
