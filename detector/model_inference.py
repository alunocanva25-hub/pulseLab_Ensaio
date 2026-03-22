from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from detector.train_model import extract_features

MODEL_FILE = Path("detector_model.joblib")


class DetectorModel:
    def __init__(self):
        self.model = None
        if MODEL_FILE.exists():
            self.model = joblib.load(MODEL_FILE)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, roi_bgr):
        if self.model is None:
            return {
                "loaded": False,
                "label": "desconhecido",
                "confidence": 0.0,
            }

        feats = np.array([extract_features(roi_bgr)], dtype=np.float32)
        probs = self.model.predict_proba(feats)[0]
        labels = self.model.classes_

        best_idx = int(np.argmax(probs))
        return {
            "loaded": True,
            "label": str(labels[best_idx]),
            "confidence": float(probs[best_idx]),
        }
