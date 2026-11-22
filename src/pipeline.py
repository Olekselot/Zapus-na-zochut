"""Оплата і реєстрація: кадр, обличчя, якість, живість, вектор, база, поріг, PIN, списання."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import numpy as np

from src.bank import Bank, hash_pin, valid_pin
from src.challenge import CHALLENGES, verify
from src.config import (
    CLASS_UA,
    CONSISTENCY_THRESHOLD,
    DEFAULT_BALANCE,
    HIGH_AMOUNT,
    HIGH_AMOUNT_EXTRA,
    MATCH_MARGIN,
    MAX_AMOUNT,
    load_recognition_threshold,
)
from src.decisions import summarize_liveness
from src.faces import FaceEngine, FaceSample
from src.gallery import Match, best_match, consistent, cosine, template
from src.metrics import live_mask_from_probs
from src.quality import QualityRules, main_problem, split_by_quality

NAME_PATTERN = re.compile(r"^[A-Za-zА-Яа-яІіЇїЄєҐґ'’\- ]{1,40}$")

STATUS_UA = {
    "paid": "проведено",
    "enrolled": "зареєстровано",
    "spoof": "підробка",
    "unknown": "невідомий",
    "ambiguous": "неоднозначно",
    "no_face": "немає обличчя",
    "poor_quality": "поганий кадр",
    "multi_face": "кілька облич",
    "inconsistent": "різні люди",
    "no_model": "немає моделі",
    "insufficient": "недостатньо коштів",
    "pin_required": "потрібен PIN",
    "pin_wrong": "невірний PIN",
    "pin_locked": "PIN заблоковано",
    "challenge_failed": "завдання не виконано",
    "duplicate": "обличчя вже в базі",
    "blocked": "недоступно",
    "error": "помилка",
}


def _preview(sample: FaceSample) -> np.ndarray:
    from src.images import bgr_to_rgb_uint8

    return bgr_to_rgb_uint8(sample.aligned_bgr)


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
        chunks = [f'<div style="{ink}font-size:2rem;font-weight:700;line-height:1.2;">{title}</div>']
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
            f"<style>.pay-card, .pay-card * {{ color: {color} !important; }}</style>"
            f'<div class="pay-card" style="max-width:520px;margin:8px auto 20px;text-align:center;padding:28px 24px;'
            "border-radius:18px;box-shadow:0 10px 30px rgba(0,0,0,.08);"
            f'{ink}background:{background};">' + "".join(chunks) + "</div>"
        )


class Pipeline:
    def __init__(self, bank: Bank, faces: FaceEngine, antispoof, rules: QualityRules | None = None) -> None:
        self.bank = bank
        self.faces = faces
        self.antispoof = antispoof
        self.rules = rules or QualityRules()
        self.antispoof_enabled = True

    def toggle_antispoof(self) -> bool:
        """Повертає новий стан: True, якщо перевірка живості знову увімкнена."""
        self.antispoof_enabled = not self.antispoof_enabled
        return self.antispoof_enabled

    @staticmethod
    def threshold_for(amount: float | None = None) -> float:
        """Для великих сум поріг схожості суворіший."""
        base = load_recognition_threshold()
        if amount is not None and amount > HIGH_AMOUNT:
            return base + HIGH_AMOUNT_EXTRA
        return base

    def start_registration(
        self,
        first_name: str,
        last_name: str,
        images: list[np.ndarray],
        balance: float = DEFAULT_BALANCE,
        pin: str = "",
    ) -> tuple[Decision, dict | None]:
        try:
            first = _clean_name(first_name, "Ім’я")
            last = _clean_name(last_name, "Прізвище")
        except ValueError as error:
            return Decision("error", str(error), title="Помилка"), None
        if balance <= 0 or balance > MAX_AMOUNT:
            return Decision("error", f"Початковий баланс має бути від 0.01 до {MAX_AMOUNT:.0f} грн.", title="Помилка"), None
        pin = (pin or "").strip()
        if pin and not valid_pin(pin):
            return Decision("error", "PIN має містити 4–6 цифр або лишіть поле порожнім.", title="Помилка"), None

        samples = self._samples(images)
        if not samples:
            return Decision(
                "no_face", "На жодному фото не знайдено обличчя. Зніміть анфас при рівному світлі.", title="Обличчя не знайдено"
            ), None
        kept, issues = split_by_quality(samples, self.rules)
        if not kept:
            return Decision(
                "poor_quality", f"Жодне фото не придатне: {main_problem(issues)}.", title="Фото не підходять"
            ), None
        keep, _template = consistent([sample.embedding for sample in kept], CONSISTENCY_THRESHOLD)
        if len(keep) * 2 < len(kept):
            return Decision(
                "inconsistent",
                "На фото, схоже, різні люди. Додайте фото лише однієї людини.",
                title="Різні люди на фото",
            ), None
        kept = [kept[index] for index in keep]

        spoofed = 0
        if self.antispoof_enabled and self.antispoof.available:
            probs = self.antispoof.predict_proba([sample.spoof_input for sample in kept])
            live = live_mask_from_probs(probs, self.antispoof.threshold)
            if not live.any():
                return Decision(
                    "spoof",
                    "Фото схожі на підробку: знімок екрана або роздруківки. Реєстрацію відхилено.",
                    title="Спроба підробки",
                ), None
            spoofed = int((~live).sum())
            kept = [sample for sample, ok in zip(kept, live) if ok]

        embeddings = [sample.embedding.astype(np.float32).tolist() for sample in kept]
        note = _photo_note(len(kept), len(images), spoofed)
        pending = {
            "first": first,
            "last": last,
            "balance": float(balance),
            "embeddings": embeddings,
            "pin_hash": hash_pin(pin) if pin else "",
            "note": note,
            "can_attach": False,
            "can_create": False,
        }
        threshold = self.threshold_for()
        same_face = best_match(template(embeddings), self.bank.gallery(), threshold)
        namesakes = self.bank.find_by_name(first, last)
        names = {int(item["id"]): str(item["name"]) for item in namesakes}

        if same_face.accepted:
            if same_face.client_id in names:
                shown = names[same_face.client_id]
                pending.update(client_id=int(same_face.client_id), display=shown, can_attach=True)
                return Decision(
                    "ask",
                    f"Обличчя збігається з рахунком {shown} (схожість {same_face.similarity:.3f}). Додати фото до нього?",
                    client_code=shown,
                    title="Це ваш рахунок?",
                ), pending
            return Decision(
                "duplicate",
                "Це обличчя вже зареєстроване на інший рахунок. Другий рахунок на те саме обличчя не створюється.",
                title="Обличчя вже в базі",
            ), None
        if len(namesakes) >= 2:
            return Decision(
                "blocked",
                "На жаль, ця дія недоступна. Клієнтів із таким прізвищем та ім’ям уже кілька, і незрозуміло, чий це рахунок.",
                title="Дія недоступна",
            ), None
        if len(namesakes) == 1:
            shown = str(namesakes[0]["name"])
            pending.update(display=shown, can_create=True)
            return Decision(
                "ask",
                f"У банку вже є {shown}, але обличчя з цим рахунком не збігається, тож додати до нього фото не можна. "
                "Створити окремий рахунок?",
                client_code=shown,
                title="Створити новий рахунок?",
            ), pending
        return self._create_person(pending), None

    def confirm_existing(self, pending: dict) -> Decision:
        """Фото додаються, лише якщо нове обличчя збігається з уже збереженим. Ім’я саме по собі нічого не доводить."""
        if not pending.get("can_attach"):
            return Decision("error", "Додати фото до цього рахунку не можна: обличчя не збігається.", title="Помилка")
        client_id = int(pending["client_id"])
        vectors = [np.asarray(vector, dtype=np.float32) for vector in pending["embeddings"]]
        stored = self.bank.client_vectors(client_id)
        similarity = cosine(template(vectors), template(stored)) if stored else -1.0
        if similarity < self.threshold_for():
            return Decision("error", "Обличчя не збігається з рахунком, фото не додано.", title="Помилка")
        added = self.bank.add_embeddings(client_id, vectors)
        balance = self.bank.balance_of(client_id)
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
        if not pending.get("can_create"):
            return Decision("error", "Новий рахунок для цього обличчя створити не можна.", title="Помилка")
        vectors = [np.asarray(vector, dtype=np.float32) for vector in pending["embeddings"]]
        if best_match(template(vectors), self.bank.gallery(), self.threshold_for()).accepted:
            return Decision("duplicate", "Це обличчя вже зареєстроване на інший рахунок.", title="Обличчя вже в базі")
        if len(self.bank.find_by_name(str(pending["first"]), str(pending["last"]))) >= 2:
            return Decision("blocked", "Клієнтів із таким прізвищем та ім’ям уже кілька.", title="Дія недоступна")
        return self._create_person(pending)

    def _create_person(self, pending: dict) -> Decision:
        vectors = [np.asarray(vector, dtype=np.float32) for vector in pending["embeddings"]]
        balance = float(pending["balance"])
        try:
            _client_id, shown = self.bank.register_person(
                str(pending["first"]), str(pending["last"]), vectors, balance, pin_hash=str(pending["pin_hash"])
            )
        except ValueError as error:
            return Decision("error", str(error), title="Помилка")
        pin_note = " PIN задано." if pending["pin_hash"] else f" Без PIN оплати понад {HIGH_AMOUNT:.0f} грн недоступні."
        return Decision(
            "enrolled",
            f"Клієнта {shown} зареєстровано. Збережено векторів: {len(vectors)}. "
            f"Баланс {balance:.2f} грн.{pin_note} Фото в базу не записувались.{pending['note']}",
            client_code=shown,
            balance=balance,
            frames_used=len(vectors),
            title="Клієнта зареєстровано",
        )

    def pay(
        self,
        images: list[np.ndarray],
        amount: float,
        min_faces: int = 1,
        pin: str | None = None,
        challenge: str | None = None,
    ) -> tuple[Decision, np.ndarray | None]:
        if amount <= 0 or amount > MAX_AMOUNT:
            return Decision("error", f"Сума має бути від 0.01 до {MAX_AMOUNT:.0f} грн.", title="Помилка"), None
        amount = round(float(amount), 2)
        samples = self._samples(images)
        preview = None if not samples else _preview(samples[0])

        def reject(status: str, message: str, title: str, client_id: int | None = None, **extra) -> tuple:
            decision = Decision(status, message, frames_used=len(samples), amount=amount, title=title, **extra)
            self._log(decision, amount, client_id=client_id)
            return decision, preview

        if len(samples) < min_faces:
            return reject("no_face", "Обличчя не знайдено. Подивіться в камеру з відстані 40–70 см при рівному світлі.", "Обличчя не знайдено")
        if challenge:
            done, note = verify(samples, challenge)
            if not done:
                return reject("challenge_failed", f"Завдання «{CHALLENGES.get(challenge, challenge)}» не виконано: {note}", "Завдання не виконано")
        if any(sample.faces_found > 1 and sample.second_face_ratio >= self.rules.multi_face_ratio for sample in samples):
            return reject("multi_face", "У кадрі кілька облич. Оплата проходить, коли перед камерою одна людина.", "Кілька облич")
        if len(samples) > 1:
            keep, _template = consistent([sample.embedding for sample in samples], CONSISTENCY_THRESHOLD)
            if len(keep) < len(samples):
                return reject("inconsistent", "У кадрах різні люди. Оплату зупинено.", "Різні люди в кадрах")
        kept, issues = split_by_quality(samples, self.rules)
        if not kept or (not challenge and len(kept) < min_faces):
            return reject("poor_quality", f"Кадр не підходить: {main_problem(issues)}.", "Поганий кадр")

        live_score = None
        threshold = self.threshold_for(amount)
        if self.antispoof_enabled and not self.antispoof.available:
            match = self._match(kept, threshold)
            who = ""
            if match.code is not None and match.similarity is not None:
                who = f" Найближчий клієнт {match.code}, схожість {match.similarity:.3f}, але гроші не списано."
            elif match.similarity is not None:
                who = f" Найближча схожість {match.similarity:.3f} нижче порогу, гроші не списано."
            return reject(
                "no_model",
                "Захист від підробки ще не навчений, тому оплату заблоковано." + who,
                "Оплату заблоковано",
                client_id=match.client_id if match.accepted else None,
                client_code=match.code,
                similarity=match.similarity,
            )
        if self.antispoof_enabled:
            probs = self.antispoof.predict_proba([sample.spoof_input for sample in kept])
            is_live, label, live_score = summarize_liveness(probs, self.antispoof.threshold)
            if not is_live:
                return reject(
                    "spoof",
                    f"Відмова: спроба підробки ({CLASS_UA.get(label, label)}). Оплату не проведено. Оцінка живості {live_score:.3f}.",
                    "Спроба підробки",
                    live_score=live_score,
                    attack_label=label,
                )

        match = self._match(kept, threshold)
        if not match.accepted:
            similarity = "немає бази" if match.similarity is None else f"{match.similarity:.3f}"
            if match.reason == "ambiguous":
                return reject(
                    "ambiguous",
                    f"Обличчя схоже одразу на двох клієнтів ({similarity} і {match.second_similarity:.3f}). "
                    "Оплату не проведено, щоб не списати гроші не з того рахунку.",
                    "Неоднозначний збіг",
                    similarity=match.similarity,
                    live_score=live_score,
                )
            return reject(
                "unknown",
                f"Відмова: клієнта не знайдено. Найближча схожість {similarity}, поріг {threshold:.3f}.",
                "Клієнта не знайдено",
                similarity=match.similarity,
                live_score=live_score,
            )

        pin_note = ""
        if amount > HIGH_AMOUNT:
            pin_state = "missing" if not pin else self.bank.verify_pin(int(match.client_id), pin)
            if not self.bank.has_pin(int(match.client_id)):
                pin_state = "none"
            texts = {
                "none": (f"Для сум понад {HIGH_AMOUNT:.0f} грн потрібен PIN, а клієнт його не задав.", "Потрібен PIN", "pin_required"),
                "missing": (f"Сума понад {HIGH_AMOUNT:.0f} грн: введіть PIN.", "Потрібен PIN", "pin_required"),
                "wrong": ("Невірний PIN. Оплату не проведено.", "Невірний PIN", "pin_wrong"),
                "locked": ("PIN заблоковано після кількох помилок. Оплати понад ліміт недоступні.", "PIN заблоковано", "pin_locked"),
            }
            if pin_state in texts:
                message, title, status = texts[pin_state]
                return reject(status, message, title, client_id=match.client_id, client_code=match.code, similarity=match.similarity)
            pin_note = ", PIN"

        live_part = "захист вимкнено" if live_score is None else f"живість {live_score:.3f}"
        challenge_part = f", завдання {challenge}" if challenge else ""
        detail = f"кадрів {len(kept)}, {live_part}{challenge_part}{pin_note}"
        status, balance = self.bank.record(
            status="paid",
            amount=amount,
            client_id=match.client_id,
            similarity=match.similarity,
            antispoof_score=live_score,
            detail=detail,
            debit=True,
        )
        common = dict(
            client_code=match.code,
            similarity=match.similarity,
            live_score=live_score,
            balance=balance,
            frames_used=len(kept),
            amount=amount,
        )
        if status == "insufficient":
            return Decision(
                "insufficient",
                f"Недостатньо коштів: на рахунку {balance:.2f} грн, потрібно {amount:.2f}.",
                title="Недостатньо коштів",
                **common,
            ), preview
        live_note = " Перевірку живості вимкнено." if live_score is None else f" Живість {live_score:.3f}."
        return Decision(
            "paid",
            f"Оплату проведено. Клієнт {match.code}, списано {amount:.2f} грн. "
            f"Баланс {balance:.2f} грн. Схожість {match.similarity:.3f}.{live_note}",
            title="Оплату проведено",
            **common,
        ), preview

    def _samples(self, images: list[np.ndarray]) -> list[FaceSample]:
        size = self.antispoof.image_size
        scale = self.antispoof.crop_scale
        found: list[FaceSample] = []
        for image in images:
            sample = self.faces.analyze(image, crop_size=size, crop_scale=scale)
            if sample is not None:
                found.append(sample)
        return found

    def _match(self, samples: list[FaceSample], threshold: float) -> Match:
        query = template([sample.embedding for sample in samples])
        return best_match(query, self.bank.gallery(), threshold, MATCH_MARGIN)

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


def _photo_note(kept: int, total: int, spoofed: int = 0) -> str:
    extra = ""
    if spoofed:
        extra += f" Відкинуто фото, схожих на підробку: {spoofed}."
    if kept + spoofed < total:
        extra += f" Пропущено непридатних фото: {total - kept - spoofed}."
    if kept < 3:
        extra += " Для стабільної оплати краще додати щонайменше 8 фото."
    return extra
