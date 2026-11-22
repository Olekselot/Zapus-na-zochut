"""Збалансована підмножина CelebA-Spoof з дзеркала Hugging Face (Ar4ikov/celebA_spoof).

Дзеркало зберігає повний набір шардами parquet (72 ГБ), тож качаємо кілька шардів,
витягуємо JPEG як є, без перекодування, і одразу видаляємо шард.

У дзеркалі є лише мітка live / spoof, без типу атаки і без коду людини. Тому:
- підробки лягають у папку spoof (тип невідомий), навчання вміє з такими мітками працювати;
- фото лежать просто в папці класу, а розбиття train/validation всередині них іде за кадрами;
- чесний test береться з test-спліту дзеркала (офіційний intra-test, інші люди) в окрему папку.

Ліцензія CelebA-Spoof: лише некомерційне дослідження, без поширення. Папка data/ у git не потрапляє.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "Ar4ikov/celebA_spoof"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/data"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
HEADERS = {"User-Agent": "zapus-dataset-prep"}


def list_shards(split: str) -> list[str]:
    request = urllib.request.Request(API, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        entries = json.load(response)
    return sorted(entry["path"] for entry in entries if entry["path"].split("/")[-1].startswith(f"{split}-"))


def pick(shards: list[str], count: int) -> list[str]:
    """Рівномірно по всьому списку, а не перші count."""
    if count >= len(shards):
        return shards
    step = len(shards) / count
    return [shards[int(index * step)] for index in range(count)]


def download(remote: str, raw_dir: Path) -> Path:
    target = raw_dir / Path(remote).name
    if target.exists():
        return target
    partial = target.with_suffix(".part")
    request = urllib.request.Request(RESOLVE + remote, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    partial.rename(target)
    return target


def extract(shard: Path, out: Path, split: str, limits: dict[str, int], counts: dict[str, int], writer, rng) -> None:
    import pyarrow.parquet as pq

    table = pq.ParquetFile(shard)
    for group in range(table.metadata.num_row_groups):
        rows = table.read_row_group(group, columns=["Filepath", "Bbox", "Class"]).to_pylist()
        rng.shuffle(rows)
        for row in rows:
            label = str(row["Class"]).strip().lower()
            if label not in limits or counts[label] >= limits[label]:
                continue
            image = row["Filepath"]
            data = image.get("bytes")
            if not data:
                continue
            name = f"{split}_{Path(image.get('path') or f'{counts[label]}.jpg').stem}.jpg"
            destination = out / label / name
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            counts[label] += 1
            writer.writerow([f"{label}/{name}", label, json.dumps(row["Bbox"]), shard.name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Підготувати підмножину CelebA-Spoof")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--shards", type=int, default=4, help="Скільки шардів узяти (~4500 фото у train-шарді)")
    parser.add_argument("--out", type=Path, default=None, help="Куди класти live/ і spoof/")
    parser.add_argument("--max-live", type=int, default=8000)
    parser.add_argument("--max-spoof", type=int, default=8000)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "celeba_spoof")
    parser.add_argument("--keep", action="store_true", help="Не видаляти шард після розпакування")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = args.out or ROOT / "data" / ("celeba_spoof" if args.split != "test" else "celeba_spoof_test")
    out.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    limits = {"live": args.max_live, "spoof": args.max_spoof}
    counts = {label: len(list((out / label).glob("*.jpg"))) if (out / label).exists() else 0 for label in limits}
    shards = pick(list_shards(args.split), args.shards)
    rng = random.Random(args.seed)
    manifest = out / "manifest.csv"
    fresh = not manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if fresh:
            writer.writerow(["file", "class", "bbox", "shard"])
        for index, remote in enumerate(shards, start=1):
            if all(counts[label] >= limits[label] for label in limits):
                break
            print(f"[{index}/{len(shards)}] {remote}", flush=True)
            local = download(remote, args.raw_dir)
            extract(local, out, args.split, limits, counts, writer, rng)
            handle.flush()
            if not args.keep:
                local.unlink()
            print(f"  live {counts['live']}, spoof {counts['spoof']}", flush=True)
    print(f"Готово: {out} — live {counts['live']}, spoof {counts['spoof']}.")


if __name__ == "__main__":
    sys.exit(main())
