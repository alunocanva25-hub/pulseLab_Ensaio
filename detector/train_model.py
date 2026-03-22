from __future__ import annotations

import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATASET_DIR = Path("dataset_led")
META_FILE = DATASET_DIR / "metadata.jsonl"
MODEL_FILE = Path("detector_model.joblib")


def extract_features(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))

    h_std = float(np.std(hsv[:, :, 0]))
    s_std = float(np.std(hsv[:, :, 1]))
    v_std = float(np.std(hsv[:, :, 2]))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_mean = float(np.mean(gray))
    gray_std = float(np.std(gray))

    # pixels bem brilhantes
    bright_ratio = float(np.mean(gray > 200))

    # vermelho dominante
    b, g, r = cv2.split(img_bgr)
    red_score = float(np.mean(r.astype(np.float32) - ((g.astype(np.float32) + b.astype(np.float32)) / 2.0)))

    return [
        h_mean, s_mean, v_mean,
        h_std, s_std, v_std,
        gray_mean, gray_std,
        bright_ratio,
        red_score,
    ]


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

            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            feats = extract_features(img)
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
        n_estimators=200,
        max_depth=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_FILE)
    print(f"Modelo salvo em: {MODEL_FILE}")


if __name__ == "__main__":
    train()
