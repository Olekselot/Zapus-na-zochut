"""Шляхи, розміри векторів, пороги і правила безпеки за замовчуванням."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "eval" / "results"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "bank.sqlite"
ANTISPOOF_WEIGHTS = MODELS_DIR / "antispoof.pt"
ANTISPOOF_META = MODELS_DIR / "antispoof_meta.json"
RECOGNITION_THRESHOLD_PATH = MODELS_DIR / "recognition_threshold.json"

CLASS_NAMES = ("live", "print", "screen")
SPOOF_UNKNOWN = 3
LABEL_NAMES = CLASS_NAMES + ("spoof",)
CLASS_UA = {
    "live": "живе обличчя",
    "print": "роздруківка",
    "screen": "екран",
    "spoof": "підробка",
}
FOLDER_TO_LABEL = {
    "live": 0,
    "print": 1,
    "phone": 2,
    "laptop": 2,
    "screen": 2,
    "spoof": SPOOF_UNKNOWN,
}
EMBEDDING_DIM = 512
DEFAULT_RECOGNITION_THRESHOLD = 0.40
DEFAULT_IMAGE_SIZE = 224
DEFAULT_CROP_SCALE = 2.7
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_BALANCE = 1000.0
MAX_AMOUNT = 100_000.0
FRAME_COUNT = 5
MIN_CAMERA_FACES = 3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MATCH_MARGIN = 0.05
CONSISTENCY_THRESHOLD = 0.30
HIGH_AMOUNT = 500.0
HIGH_AMOUNT_EXTRA = 0.10
PIN_MAX_FAILURES = 5

MIN_DET_SCORE = 0.50
MIN_FACE_PX = 60
MIN_BLUR = 15.0
MAX_FRONTAL_YAW = 0.35
MULTI_FACE_RATIO = 0.50

CHALLENGE_SECONDS = 3.0
CHALLENGE_FPS = 10.0


def ensure_runtime_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_recognition_threshold() -> float:
    """Поріг косинуса. Після прогону LFW його переписує скрипт метрик."""
    if not RECOGNITION_THRESHOLD_PATH.exists():
        return DEFAULT_RECOGNITION_THRESHOLD
    payload = json.loads(RECOGNITION_THRESHOLD_PATH.read_text(encoding="utf-8"))
    return float(payload["threshold"])
