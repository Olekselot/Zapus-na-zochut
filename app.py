"""Інтерфейс демо: оплата, реєстрація і журнал."""

from __future__ import annotations

import os

import gradio as gr
import numpy as np

from src.antispoof import AntispoofModel
from src.bank import Bank
from src.camera import capture_frames
from src.config import FRAME_COUNT, MIN_CAMERA_FACES
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
    table = []
    for record in records:
        table.append([record.get(header, "") for header in headers])
    return table


def _card(text: str, title: str = "Помилка") -> str:
    return Decision("error", text, title=title).html()


def build_demo() -> gr.Blocks:
    bank = Bank()
    pipeline = Pipeline(bank, FaceEngine(), AntispoofModel())
    camera_index = int(os.environ.get("CAMERA_INDEX", "0"))

    account_headers = ["клієнт", "баланс", "векторів", "зареєстровано"]
    journal_headers = ["час", "статус", "клієнт", "сума", "схожість", "живість", "деталі"]

    def tables():
        journal = bank.transactions()
        for row in journal:
            row["статус"] = STATUS_UA.get(str(row["статус"]), row["статус"])
        return _rows(bank.accounts(), account_headers), _rows(journal, journal_headers)

    def toggle_guard():
        enabled = pipeline.toggle_antispoof()
        if enabled:
            return gr.update(value="Захист увімкнено", elem_classes=["guard-on"])
        return gr.update(value="Захист вимкнено", elem_classes=["guard-off"])

    def pay_photo(image, amount):
        if image is None:
            return _card("Додайте фото або один кадр із вебкамери.", "Потрібне фото"), None
        if amount is None:
            return _card("Вкажіть суму."), None
        try:
            decision, preview = pipeline.pay([as_bgr(image)], float(amount), min_faces=1)
        except Exception as error:
            return _card(f"Помилка: {error}"), None
        return decision.html(), preview

    def pay_camera(amount):
        if amount is None:
            return _card("Вкажіть суму."), None
        try:
            frames = capture_frames(FRAME_COUNT, camera_index=camera_index)
            decision, preview = pipeline.pay(frames, float(amount), min_faces=MIN_CAMERA_FACES)
        except Exception as error:
            return _card(f"Помилка: {error}"), None
        return decision.html(), preview

    def register(first_name, last_name, balance, files):
        try:
            images = _load_files(files)
        except Exception as error:
            return _card(f"Не вдалося прочитати фото: {error}"), None, gr.update(visible=False)
        if not images:
            return _card("Додайте кілька фото клієнта.", "Потрібні фото"), None, gr.update(visible=False)
        try:
            decision, pending = pipeline.start_registration(
                str(first_name or ""),
                str(last_name or ""),
                images,
                float(balance),
            )
        except Exception as error:
            return _card(f"Помилка: {error}"), None, gr.update(visible=False)
        return decision.html(), pending, gr.update(visible=decision.status == "ask")

    def use_existing(pending):
        if not pending:
            return _card("Немає рахунку, який треба підтвердити."), None, gr.update(visible=False)
        try:
            decision = pipeline.confirm_existing(pending)
        except Exception as error:
            return _card(f"Помилка: {error}"), None, gr.update(visible=False)
        return decision.html(), None, gr.update(visible=False)

    def create_another(pending):
        if not pending:
            return _card("Немає даних для нового рахунку."), None, gr.update(visible=False)
        try:
            decision = pipeline.confirm_new(pending)
        except Exception as error:
            return _card(f"Помилка: {error}"), None, gr.update(visible=False)
        return decision.html(), None, gr.update(visible=False)

    with gr.Blocks(title="Оплата за обличчям") as demo:
        with gr.Tabs():
            with gr.Tab("Оплата"):
                outcome = gr.HTML()
                amount = gr.Number(label="Сума, грн", value=25, minimum=0.01)
                with gr.Row():
                    image = gr.Image(label="Одне фото або кадр з вебкамери", type="numpy", sources=["upload", "webcam"])
                    preview = gr.Image(label="Вирівняне обличчя", type="numpy")
                with gr.Row():
                    photo_button = gr.Button("Оплатити цим фото", variant="primary")
                    camera_button = gr.Button("Зняти 5 кадрів з камери і оплатити")
            with gr.Tab("Реєстрація"):
                registered = gr.HTML()
                with gr.Row(visible=False) as ask_row:
                    yes_button = gr.Button("Це мій рахунок")
                    no_button = gr.Button("Створити новий")
                first_name = gr.Textbox(label="Ім’я")
                last_name = gr.Textbox(label="Прізвище")
                start_balance = gr.Number(label="Початковий баланс, грн", value=1000, minimum=0.01)
                files = gr.File(label="Кілька фото", file_count="multiple", file_types=["image"])
                register_button = gr.Button("Зареєструвати", variant="primary")
                pending = gr.State(None)
            with gr.Tab("Баланс і журнал"):
                with gr.Row():
                    guard_button = gr.Button(
                        "Захист увімкнено",
                        elem_id="guard-button",
                        elem_classes=["guard-on"],
                        scale=1,
                    )
                    refresh = gr.Button("Оновити", scale=1, elem_id="refresh-button")
                accounts = gr.Dataframe(headers=account_headers, label="Рахунки", interactive=False, type="array")
                journal = gr.Dataframe(headers=journal_headers, label="Журнал операцій", interactive=False, type="array")

        photo_button.click(pay_photo, [image, amount], [outcome, preview]).then(tables, None, [accounts, journal])
        camera_button.click(pay_camera, [amount], [outcome, preview]).then(tables, None, [accounts, journal])
        register_button.click(
            register,
            [first_name, last_name, start_balance, files],
            [registered, pending, ask_row],
        ).then(tables, None, [accounts, journal])
        yes_button.click(use_existing, [pending], [registered, pending, ask_row]).then(
            tables, None, [accounts, journal]
        )
        no_button.click(create_another, [pending], [registered, pending, ask_row]).then(
            tables, None, [accounts, journal]
        )
        guard_button.click(toggle_guard, None, guard_button)
        refresh.click(tables, None, [accounts, journal])
        demo.load(tables, None, [accounts, journal])
    return demo


if __name__ == "__main__":
    build_demo().launch(css=PAGE_CSS)
