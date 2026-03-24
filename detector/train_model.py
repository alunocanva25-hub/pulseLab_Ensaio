from __future__ import annotations

import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from detector.temporal_features import frame_features, sequence_features

DATASET_DIR = Path("dataset_led")
META_FILE = DATASET_DIR / "metadata.jsonl"
MODEL_FILE = Path("detector_model.joblib")


def extract_features(img_bgr, sequence_imgs=None):
    base = frame_features(img_bgr)
    base_vec = [
        base["h_mean"], base["s_mean"], base["v_mean"],
        base["h_std"], base["s_std"], base["v_std"],
        base["gray_mean"], base["gray_std"],
        base["bright_ratio"], base["red_score"],
    ]

    seq_vec = sequence_features(sequence_imgs or [img_bgr])
    return base_vec + seq_vec


def load_sequence(paths):
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(img)
    return frames


def load_dataset():
    X = []
    y = []

    if not META_FILE.exists():
        raise FileNotFoundError("metadata.jsonl não encontrado. Colete amostras antes.")

    with open(META_FILE, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            img_path = Path(row["file"])
            label = row["label"]
            seq_paths = row.get("sequence_files", [])

            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            seq_imgs = load_sequence(seq_paths)
            feats = extract_features(img, seq_imgs)

            X.append(feats)
            y.append(label)

    if not X:
        raise RuntimeError("Nenhuma amostra válida encontrada.")

    return np.array(X, dtype=np.float32), np.array(y)


def train():
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_FILE)
    print(f"Modelo salvo em: {MODEL_FILE}")


if __name__ == "__main__":
    train()