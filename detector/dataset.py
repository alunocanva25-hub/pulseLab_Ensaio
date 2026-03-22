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


def save_sample(roi_bgr, label: str, meta: dict[str, Any] | None = None) -> str:
    ensure_dataset_dirs()

    ts = int(time.time() * 1000)
    filename = f"{label}_{ts}.jpg"
    filepath = DATASET_DIR / label / filename

    cv2.imwrite(str(filepath), roi_bgr)

    row = {
        "file": str(filepath).replace("\\", "/"),
        "label": label,
        "timestamp": ts,
    }
    if meta:
        row.update(meta)

    with open(META_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return str(filepath)
