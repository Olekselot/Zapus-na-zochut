"""Збір власного набору: живі обличчя й атаки з тієї ж вебкамери, що й на захисті.

    python tools/collect.py --person p01 --kind live --condition рівне_світло
    python tools/collect.py --person p01 --kind print --condition a4_плоско
    python tools/collect.py --person p01 --kind phone
    python tools/collect.py --person p01 --kind enroll     # клієнти демо: data/enroll/p01/enroll
    python tools/collect.py --person p01 --kind pay        # фото «на оплату»: data/enroll/p01/pay

У вікні: ПРОБІЛ — почати / зупинити серію, S — один кадр, Q або Esc — вийти.
Кадр зберігається, лише якщо детектор бачить рівно одне велике обличчя і кадр не розмитий.
Прізвищ у назвах немає, лише код людини. Перед першим знімком людини питаємо згоду.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.camera import open_camera  # noqa: E402
from src.images import imwrite  # noqa: E402
from src.quality import blur_score  # noqa: E402

KINDS = {
    "live": "Жива людина: дивіться в камеру, трохи рухайте головою, міняйте вираз обличчя.",
    "print": "Роздруківка A4 перед камерою: плоско, потім трохи зігнуто, ближче і далі.",
    "phone": "Фото або відео обличчя на екрані телефона: різна яскравість, кути, відстань.",
    "laptop": "Фото або відео обличчя на екрані ноутбука чи монітора.",
    "enroll": "Реєстрація клієнта демо: анфас, рівне світло, 40–70 см.",
    "pay": "Фото «на оплату» для того ж клієнта, інший момент і ракурс.",
}
CONDITIONS = {
    "live": ["рівне_світло", "бокове_світло", "легкий_поворот", "окуляри", "темніше"],
    "print": ["a4_плоско", "a4_зігнуто", "ближче", "далі"],
    "phone": ["фото", "відео", "яскравість_макс", "під_кутом"],
    "laptop": ["фото", "відео", "під_кутом"],
    "enroll": ["анфас"],
    "pay": ["інший_ракурс"],
}


def target_dir(out: Path, person: str, kind: str) -> Path:
    if kind in {"enroll", "pay"}:
        return ROOT / "data" / "enroll" / person / kind
    return out / kind / person


def ensure_consent(out: Path, person: str) -> None:
    """Згода зберігається в consent.csv поруч із даними. Без «так» запис не почнеться."""
    path = out / "consent.csv"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            if any(row and row[0] == person and row[2] == "так" for row in csv.reader(handle)):
                return
    print(f"Людина {person} погоджується, що її фото використають лише для навчального проєкту,")
    print("локально, без публікації, і видалять після заліку (див. docs/DATA_COLLECTION.md)?")
    answer = input("Згода (так/ні): ").strip().lower()
    if answer not in {"так", "y", "yes"}:
        raise SystemExit("Без згоди знімати не можна.")
    fresh = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if fresh:
            writer.writerow(["person", "date", "consent"])
        writer.writerow([person, datetime.now().date().isoformat(), "так"])


def overlay(frame, lines: list[str], box, color) -> np.ndarray:
    import cv2

    view = frame.copy()
    if box is not None:
        x1, y1, x2, y2 = [int(value) for value in box]
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
    for index, line in enumerate(lines):
        cv2.putText(view, line, (12, 28 + 26 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, line, (12, 28 + 26 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def check(engine, frame, min_face: int, min_blur: float) -> tuple[bool, str, object]:
    faces = engine._model().get(frame)
    if not faces:
        return False, "no face", None
    faces = sorted(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse=True)
    face = faces[0]
    side = min(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1])
    if len(faces) > 1 and (faces[1].bbox[2] - faces[1].bbox[0]) > 0.5 * (face.bbox[2] - face.bbox[0]):
        return False, "2+ faces", face.bbox
    if side < min_face:
        return False, f"face {int(side)}px < {min_face}", face.bbox
    x1, y1, x2, y2 = [max(0, int(value)) for value in face.bbox]
    sharpness = blur_score(frame[y1:y2, x1:x2]) if y2 > y1 and x2 > x1 else 0.0
    if sharpness < min_blur:
        return False, f"blur {sharpness:.0f}", face.bbox
    return True, f"ok, blur {sharpness:.0f}", face.bbox


def main() -> None:
    parser = argparse.ArgumentParser(description="Зібрати фото для навчання і демо")
    parser.add_argument("--person", required=True, help="Код людини: p01, p02... Без прізвищ")
    parser.add_argument("--kind", required=True, choices=list(KINDS))
    parser.add_argument("--condition", default=None, help="Умова зйомки, потрапляє в назву файла і manifest")
    parser.add_argument("--count", type=int, default=25, help="Скільки кадрів за серію")
    parser.add_argument("--interval", type=float, default=0.4, help="Пауза між кадрами серії, с")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "custom")
    parser.add_argument("--min-face", type=int, default=80)
    parser.add_argument("--min-blur", type=float, default=30.0)
    args = parser.parse_args()
    if not args.person.replace("_", "").isalnum() or not args.person.isascii():
        raise SystemExit("Код людини — латиниця й цифри, наприклад p01.")
    condition = args.condition or CONDITIONS[args.kind][0]
    ensure_consent(args.out, args.person)

    import cv2

    from src.faces import FaceEngine

    engine = FaceEngine(det_size=320, with_landmarks=False, with_recognition=False)
    folder = target_dir(args.out, args.person, args.kind)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.csv"
    fresh = not manifest.exists()
    camera = open_camera(args.camera)
    window = f"zbir: {args.person} / {args.kind}"
    saved, running, last = len(list(folder.glob("*.jpg"))), False, 0.0
    start_count = saved
    try:
        with manifest.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if fresh:
                writer.writerow(["file", "person", "kind", "condition", "camera", "width", "height", "time", "blur"])
            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    raise SystemExit("Камера перестала віддавати кадри.")
                good, note, box = check(engine, frame, args.min_face, args.min_blur)
                now = time.monotonic()
                shot = False
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    running = not running
                if key == ord("s"):
                    shot = good
                if running and good and now - last >= args.interval:
                    shot = True
                if shot:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    name = f"{condition}_{stamp}.jpg"
                    imwrite(folder / name, frame, quality=95)
                    height, width = frame.shape[:2]
                    writer.writerow([str((folder / name).relative_to(ROOT)), args.person, args.kind, condition, args.camera, width, height, stamp, note])
                    handle.flush()
                    saved += 1
                    last = now
                    if saved - start_count >= args.count:
                        running = False
                        start_count = saved
                lines = [
                    f"{args.person} | {args.kind} | {condition}",
                    f"saved {saved} | series {'ON' if running else 'off'} | {note}",
                    "SPACE series  S shot  Q quit",
                ]
                try:
                    cv2.imshow(window, overlay(frame, lines, box, (0, 200, 0) if good else (0, 0, 255)))
                except cv2.error as error:
                    raise SystemExit(
                        "OpenCV без вікон (opencv-python-headless). Виконайте: pip install --force-reinstall opencv-python"
                    ) from error
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Збережено в {folder}: всього {saved} кадрів. Підказка для цього типу: {KINDS[args.kind]}")


if __name__ == "__main__":
    main()
