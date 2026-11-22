"""LFW: повні фото 250×250 з диска, 10 фолдів pairs.txt і кеш векторів.

sklearn тримає всі пари в пам'яті як float32 (для 6000 пар це ~9 ГБ), тому через нього
лише скачуємо архів, а фото читаємо самі по одному.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import RESULTS_DIR


def lfw_home() -> Path:
    from sklearn.datasets import get_data_home

    home = Path(get_data_home()) / "lfw_home"
    if not (home / "pairs.txt").exists() or not (home / "lfw_funneled").exists():
        try:
            from sklearn.datasets._lfw import _check_fetch_lfw

            _check_fetch_lfw(data_home=None, funneled=True, download_if_missing=True)
        except ImportError:
            from sklearn.datasets import fetch_lfw_pairs

            fetch_lfw_pairs(subset="10_folds", resize=0.05, funneled=True, download_if_missing=True)
    if not (home / "pairs.txt").exists():
        raise FileNotFoundError(f"не знайдено {home / 'pairs.txt'}")
    return home


def image_path(home: Path, name: str, number: int) -> Path:
    return home / "lfw_funneled" / name / f"{name}_{int(number):04d}.jpg"


def load_pairs(home: Path, per_fold: int | None = None) -> list[tuple[Path, Path, bool, int]]:
    """pairs.txt: 10 наборів, у кожному 300 своїх пар і 300 чужих. Повертає (фото1, фото2, своя, фолд)."""
    lines = (home / "pairs.txt").read_text(encoding="utf-8").split("\n")
    sets, size = (int(value) for value in lines[0].split())
    rows = [line.split("\t") for line in lines[1:] if line.strip()]
    pairs = []
    for fold in range(sets):
        chunk = rows[fold * 2 * size : (fold + 1) * 2 * size]
        matched, mismatched = chunk[:size], chunk[size:]
        if per_fold:
            matched, mismatched = matched[: per_fold // 2], mismatched[: per_fold - per_fold // 2]
        for name, first, second in matched:
            pairs.append((image_path(home, name, int(first)), image_path(home, name, int(second)), True, fold))
        for name_a, first, name_b, second in mismatched:
            pairs.append((image_path(home, name_a, int(first)), image_path(home, name_b, int(second)), False, fold))
    return pairs


def identity(path: Path) -> str:
    return path.parent.name


def embed_paths(paths: list[Path], cache_name: str = "lfw_embeddings", det_size: int = 640) -> dict[str, np.ndarray]:
    """Вектор для кожного фото, NaN — обличчя не знайдено. Кеш дописується, тож перерваний прогін продовжується."""
    from src.faces import FaceEngine
    from src.images import imread

    cache_path = RESULTS_DIR / f"{cache_name}_d{det_size}.npz"
    known: dict[str, np.ndarray] = {}
    if cache_path.exists():
        stored = np.load(cache_path, allow_pickle=False)
        known = dict(zip(stored["keys"].tolist(), stored["vectors"]))
    todo = sorted({str(path) for path in paths} - set(known))
    if todo:
        engine = FaceEngine(det_size=det_size, with_landmarks=False)
        for index, key in enumerate(todo, start=1):
            found = engine.analyze(imread(key), pick="center")
            known[key] = np.full(512, np.nan, dtype=np.float32) if found is None else found.embedding
            if index % 250 == 0 or index == len(todo):
                print(f"Векторів {index}/{len(todo)}", flush=True)
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                keys = np.asarray(sorted(known))
                np.savez(cache_path, keys=keys, vectors=np.stack([known[key] for key in keys.tolist()]))
    return known


def pair_scores(pairs, vectors: dict[str, np.ndarray]) -> np.ndarray:
    scores = []
    for first, second, _same, _fold in pairs:
        left, right = vectors[str(first)], vectors[str(second)]
        scores.append(float(np.dot(left, right)) if np.isfinite(left).all() and np.isfinite(right).all() else np.nan)
    return np.asarray(scores, dtype=np.float64)
