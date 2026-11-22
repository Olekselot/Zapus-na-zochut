"""Карта векторів облич (t-SNE): фото однієї людини мають злитися в окремий кластер.

Джерело — кеш векторів LFW після recognition_metrics.py або папка data/enroll/<людина>/...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from eval.lfw import identity  # noqa: E402
from src.config import IMAGE_EXTENSIONS, RESULTS_DIR  # noqa: E402


def from_cache(cache: Path, people: int) -> tuple[np.ndarray, np.ndarray]:
    stored = np.load(cache, allow_pickle=False)
    keep = np.isfinite(stored["vectors"]).all(axis=1)
    vectors = stored["vectors"][keep]
    names = np.asarray([identity(Path(key)) for key in stored["keys"][keep].tolist()])
    unique, counts = np.unique(names, return_counts=True)
    top = set(unique[np.argsort(-counts)[:people]].tolist())
    mask = np.asarray([name in top for name in names])
    return vectors[mask], names[mask]


def from_folder(root: Path) -> tuple[np.ndarray, np.ndarray]:
    from src.faces import FaceEngine
    from src.images import imread

    engine = FaceEngine(with_landmarks=False)
    vectors, names = [], []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        found = engine.analyze(imread(path))
        if found is not None:
            vectors.append(found.embedding)
            names.append(path.relative_to(root).parts[0])
    return np.asarray(vectors), np.asarray(names)


def main() -> None:
    parser = argparse.ArgumentParser(description="t-SNE векторів облич")
    parser.add_argument("--cache", type=Path, default=RESULTS_DIR / "lfw_embeddings_d640.npz")
    parser.add_argument("--folder", type=Path, default=None, help="Замість LFW: папка <людина>/**/*.jpg")
    parser.add_argument("--people", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    vectors, names = from_folder(args.folder) if args.folder else from_cache(args.cache, args.people)
    if len(vectors) < 5:
        raise SystemExit("Замало облич для карти.")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    points = TSNE(n_components=2, perplexity=min(30, len(vectors) - 1), metric="cosine", random_state=args.seed, init="pca").fit_transform(vectors)
    figure, axis = plt.subplots(figsize=(9, 7))
    for name in sorted(set(names.tolist())):
        mask = names == name
        axis.scatter(points[mask, 0], points[mask, 1], s=14, label=f"{name} ({mask.sum()})")
    axis.set_title("t-SNE векторів ArcFace: одна людина — один кластер")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(fontsize=7, loc="best", ncol=2)
    figure.tight_layout()
    destination = RESULTS_DIR / ("embedding_map_folder.png" if args.folder else "embedding_map_lfw.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=130)
    print(f"Карта: {destination} ({len(vectors)} облич, {len(set(names.tolist()))} людей)")


if __name__ == "__main__":
    main()
