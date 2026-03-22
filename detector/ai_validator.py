from __future__ import annotations

from collections import deque
import numpy as np


class LEDAIVerifier:
    def __init__(self, history_size: int = 10):
        self.score_history = deque(maxlen=max(4, int(history_size)))

    def update(self, score: float):
        self.score_history.append(float(score))

    def validate(self, target: dict | None):
        if target is None:
            return {
                "is_valid_led": False,
                "confidence": 0.10,
                "reason": "Nenhum alvo encontrado",
            }

        score = float(target.get("score", 0.0))
        area = float(target.get("area", 0.0))
        brightness = float(target.get("brightness", 0.0))

        self.update(score)

        stability = 0.5
        if len(self.score_history) >= 4:
            arr = np.array(self.score_history, dtype=float)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            ratio = std_val / max(mean_val, 1e-6) if mean_val > 0 else 1.0

            if ratio < 0.20:
                stability = 0.9
            elif ratio < 0.40:
                stability = 0.6
            else:
                stability = 0.3

        confidence = 0.0
        if area >= 4:
            confidence += 0.25
        if brightness >= 70:
            confidence += 0.25
        confidence += 0.25 * stability
        if score >= 5:
            confidence += 0.15

        return {
            "is_valid_led": confidence >= 0.45,
            "confidence": round(min(confidence, 0.99), 2),
            "reason": f"score={round(score,2)}, area={round(area,2)}, brilho={round(brightness,2)}, estabilidade={round(stability,2)}",
        }