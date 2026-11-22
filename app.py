"""Інтерфейс демо: оплата, реєстрація і журнал."""

from __future__ import annotations

import hmac
import os

import gradio as gr
import numpy as np

from src.antispoof import AntispoofModel
from src.bank import Bank
from src.camera import capture_frames, capture_sequence
from src.challenge import CHALLENGES, random_challenge
from src.config import CHALLENGE_SECONDS, FRAME_COUNT, HIGH_AMOUNT, MIN_CAMERA_FACES
from src.faces import FaceEngine
from src.images import as_bgr, imread
from src.pipeline import STATUS_UA, Decision, Pipeline

PAGE_CSS = """
#guard-button {
    width: 100% !important;
    max-width: none !important;
    height: 100% !important;
    min-height: 0 !important;
    border-radius: 8px !important;
    margin: 0 !important;
    padding: 8px 16px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    color: white !important;
    border: none !important;
    box-shadow: none !important;
    white-space: nowrap !important;
}
#guard-button.guard-on {
    background: #1b7f3a !important;
}
#guard-button.guard-off {
    background: #c0392b !important;
}
"""


def _file_path(item) -> str | None:
    if item is None:
        return None
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("path") or item.get("name")
    path = getattr(item, "path", None) or getattr(item, "name", None)
    return str(path) if path else None


def _load_files(files) -> list[np.ndarray]:
    if not files:
        return []
    if not isinstance(files, list):
        files = [files]
    images = []
    for item in files:
        path = _file_path(item)
        if path:
            images.append(imread(path))
    return images


def _rows(records: list[dict[str, object]], headers: list[str]) -> list[list[object]]:
    return [[record.get(header, "") for header in headers] for record in records]


def _card(text: str, title: str = "Помилка") -> str:
    return Decision("error", text, title=title).html()


def build_demo() -> gr.Blocks:
    bank = Bank()
    antispoof = AntispoofModel()
    pipeline = Pipeline(bank, FaceEngine(), antispoof)
    camera_index = int(os.environ.get("CAMERA_INDEX", "0"))
    admin_pin = os.environ.get("ADMIN_PIN", "")

    account_headers = ["клієнт", "баланс", "векторів", "PIN", "зареєстровано"]
    journal_headers = ["час", "статус", "клієнт", "сума", "схожість", "живість", "деталі"]

    def tables():
        journal = bank.transactions()
        for row in journal:
            row["статус"] = STATUS_UA.get(str(row["статус"]), row["статус"])
        return _rows(bank.accounts(), account_headers), _rows(journal, journal_headers)

    def banner() -> str:
        if not pipeline.antispoof_enabled:
            text = "Захист від підробки вимкнено: оплату пройде навіть фото з телефона."
        elif not antispoof.available:
            text = "Модель захисту ще не навчена: оплату заблоковано, реєстрація працює."
        else:
            return ""
        return f'<div style="background:#fdecea;color:#7f1d1d;padding:10px 14px;border-radius:10px;font-weight:600;">{text}</div>'

    def guard_view():
        """Кнопка завжди показує справжній стан бекенду, навіть після перезавантаження сторінки."""
        if pipeline.antispoof_enabled:
            return gr.update(value="Захист увімкнено", elem_classes=["guard-on"])
        return gr.update(value="Захист вимкнено", elem_classes=["guard-off"])

    def on_load():
        accounts_rows, journal_rows = tables()
        return accounts_rows, journal_rows, guard_view(), banner()

    def toggle_guard(entered_pin):
        if admin_pin and not hmac.compare_digest(str(entered_pin or ""), admin_pin):
            return guard_view(), "Невірний PIN адміністратора, стан захисту не змінено.", banner()
        pipeline.toggle_antispoof()
        return guard_view(), "", banner()

    def pay_photo(image, amount, pin):
        if image is None:
            return _card("Додайте фото або один кадр із вебкамери.", "Потрібне фото"), None
        if amount is None:
            return _card("Вкажіть суму."), None
        try:
            decision, preview = pipeline.pay([as_bgr(image)], float(amount), min_faces=1, pin=pin or None)
        except Exception as error:
            return _card(f"Помилка: {error}"), None
        return decision.html(), preview

    def pay_camera(amount, pin):
        if amount is None:
            return _card("Вкажіть суму."), None
        try:
            frames = capture_frames(FRAME_COUNT, camera_index=camera_index)
            decision, preview = pipeline.pay(frames, float(amount), min_faces=MIN_CAMERA_FACES, pin=pin or None)
        except Exception as error:
            return _card(f"Помилка: {error}"), None
        return decision.html(), preview

    def new_challenge():
        kind = random_challenge(with_blink=True)
        text = (
            f"**Завдання:** {CHALLENGES[kind]}. Натисніть «Виконати завдання і оплатити» і починайте одразу: "
            f"камера пише {CHALLENGE_SECONDS:.0f} с."
        )
        return kind, text

    def pay_challenge(amount, pin, kind):
        done = (None, "Завдання використано. Для нової оплати отримайте нове.")
        if not kind:
            return _card("Спершу натисніть «Нове завдання».", "Немає завдання"), None, None, "Завдання ще не видано."
        if amount is None:
            return _card("Вкажіть суму."), None, kind, CHALLENGES[kind]
        try:
            frames = capture_sequence(camera_index=camera_index)
            decision, preview = pipeline.pay(frames, float(amount), min_faces=5, pin=pin or None, challenge=kind)
        except Exception as error:
            return _card(f"Помилка: {error}"), None, *done
        return decision.html(), preview, *done

    def ask_buttons(pending):
        attach = bool(pending and pending.get("can_attach"))
        create = bool(pending and pending.get("can_create"))
        return gr.update(visible=attach or create), gr.update(visible=attach), gr.update(visible=create)

    def register(first_name, last_name, balance, pin, files):
        hidden = (None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
        try:
            images = _load_files(files)
        except Exception as error:
            return _card(f"Не вдалося прочитати фото: {error}"), *hidden
        if not images:
            return _card("Додайте кілька фото клієнта.", "Потрібні фото"), *hidden
        try:
            decision, pending = pipeline.start_registration(
                str(first_name or ""), str(last_name or ""), images, float(balance or 0), str(pin or "")
            )
        except Exception as error:
            return _card(f"Помилка: {error}"), *hidden
        return decision.html(), pending, *ask_buttons(pending)

    def confirm(action, pending):
        hidden = (None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
        if not pending:
            return _card("Немає даних для підтвердження."), *hidden
        try:
            decision = action(pending)
        except Exception as error:
            return _card(f"Помилка: {error}"), *hidden
        return decision.html(), *hidden

    with gr.Blocks(title="Оплата за обличчям") as demo:
        with gr.Tabs():
            with gr.Tab("Оплата"):
                warning = gr.HTML()
                outcome = gr.HTML()
                with gr.Row():
                    amount = gr.Number(label="Сума, грн", value=25, minimum=0.01)
                    pin = gr.Textbox(label=f"PIN (для сум понад {HIGH_AMOUNT:.0f} грн)", type="password", max_lines=1)
                with gr.Row():
                    image = gr.Image(label="Одне фото або кадр з вебкамери", type="numpy", sources=["upload", "webcam"])
                    preview = gr.Image(label="Вирівняне обличчя", type="numpy")
                with gr.Row():
                    photo_button = gr.Button("Оплатити цим фото", variant="primary")
                    camera_button = gr.Button("Зняти 5 кадрів з камери і оплатити")
                with gr.Group():
                    challenge_text = gr.Markdown("Активна перевірка: система попросить повернути голову чи кліпнути.")
                    with gr.Row():
                        challenge_button = gr.Button("Нове завдання")
                        challenge_pay = gr.Button("Виконати завдання і оплатити", variant="primary")
                challenge_state = gr.State(None)
            with gr.Tab("Реєстрація"):
                registered = gr.HTML()
                with gr.Row(visible=False) as ask_row:
                    yes_button = gr.Button("Це мій рахунок", visible=False)
                    no_button = gr.Button("Створити новий", visible=False)
                first_name = gr.Textbox(label="Ім’я")
                last_name = gr.Textbox(label="Прізвище")
                with gr.Row():
                    start_balance = gr.Number(label="Початковий баланс, грн", value=1000, minimum=0.01)
                    new_pin = gr.Textbox(label="PIN, 4–6 цифр (необов’язково)", type="password", max_lines=1)
                files = gr.File(label="Кілька фото", file_count="multiple", file_types=["image"])
                register_button = gr.Button("Зареєструвати", variant="primary")
                pending = gr.State(None)
            with gr.Tab("Баланс і журнал"):
                with gr.Row():
                    guard_button = gr.Button("Захист увімкнено", elem_id="guard-button", elem_classes=["guard-on"], scale=1)
                    refresh = gr.Button("Оновити", scale=1, elem_id="refresh-button")
                admin_box = gr.Textbox(label="PIN адміністратора", type="password", max_lines=1, visible=bool(admin_pin))
                guard_note = gr.Markdown()
                accounts = gr.Dataframe(headers=account_headers, label="Рахунки", interactive=False, type="array")
                journal = gr.Dataframe(headers=journal_headers, label="Журнал операцій", interactive=False, type="array")

        ask_outputs = [registered, pending, ask_row, yes_button, no_button]
        photo_button.click(pay_photo, [image, amount, pin], [outcome, preview]).then(tables, None, [accounts, journal])
        camera_button.click(pay_camera, [amount, pin], [outcome, preview]).then(tables, None, [accounts, journal])
        challenge_button.click(new_challenge, None, [challenge_state, challenge_text])
        challenge_pay.click(
            pay_challenge, [amount, pin, challenge_state], [outcome, preview, challenge_state, challenge_text]
        ).then(tables, None, [accounts, journal])
        register_button.click(
            register, [first_name, last_name, start_balance, new_pin, files], ask_outputs
        ).then(tables, None, [accounts, journal])
        yes_button.click(lambda state: confirm(pipeline.confirm_existing, state), [pending], ask_outputs).then(
            tables, None, [accounts, journal]
        )
        no_button.click(lambda state: confirm(pipeline.confirm_new, state), [pending], ask_outputs).then(
            tables, None, [accounts, journal]
        )
        guard_button.click(toggle_guard, [admin_box], [guard_button, guard_note, warning])
        refresh.click(on_load, None, [accounts, journal, guard_button, warning])
        demo.load(on_load, None, [accounts, journal, guard_button, warning])
    return demo


if __name__ == "__main__":
    build_demo().launch(css=PAGE_CSS)
