from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2

DATASET_DIR = Path("dataset_led")
META_FILE = DATASET_DIR / "metadata.jsonl"


def ensure_dataset_dirs():
    for label in ["on", "off", "ruido"]:
        (DATASET_DIR / label).mkdir(parents=True, exist_ok=True)


def save_sample(roi_bgr, label: str, meta: dict[str, Any] | None = None, sequence=None) -> str:
    ensure_dataset_dirs()

    ts = int(time.time() * 1000)
    base_name = f"{label}_{ts}"
    img_path = DATASET_DIR / label / f"{base_name}.jpg"

    cv2.imwrite(str(img_path), roi_bgr)

    seq_paths = []
    if sequence:
        seq_dir = DATASET_DIR / label / f"{base_name}_seq"
        seq_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(sequence):
            p = seq_dir / f"{i:03d}.jpg"
            cv2.imwrite(str(p), frame)
            seq_paths.append(str(p).replace("\\", "/"))

    row = {
        "file": str(img_path).replace("\\", "/"),
        "label": label,
        "timestamp": ts,
        "sequence_files": seq_paths,
    }
    if meta:
        row.update(meta)

    with open(META_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return str(img_path)