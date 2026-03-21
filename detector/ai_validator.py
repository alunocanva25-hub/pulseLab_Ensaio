from __future__ import annotations

from collections import deque
import numpy as np

class LEDAIVerifier:
    def __init__(self, history_size: int = 12):
        self.history_size = max(4, int(history_size))
        self.score_history = deque(maxlen=self.history_size)
        self.area_history = deque(maxlen=self.history_size)
        self.brightness_history = deque(maxlen=self.history_size)

    def update(self, score: float, area: float, brightness: float):
        self.score_history.append(float(score))
        self.area_history.append(float(area))
        self.brightness_history.append(float(brightness))

    def _stability(self) -> float:
        if len(self.score_history) < 4:
            return 0.5
        arr = np.array(self.score_history, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        if mean_val <= 0:
            return 0.0
        ratio = std_val / max(mean_val, 1e-6)
        if ratio <= 0.10: return 0.95
        if ratio <= 0.20: return 0.85
        if ratio <= 0.35: return 0.70
        if ratio <= 0.50: return 0.55
        return 0.35

    def validate(self, target: dict | None):
        if target is None:
            return {"is_valid_led": False, "confidence": 0.10, "reason": "Nenhum alvo encontrado"}
        area = float(target.get("area", 0.0))
        brightness = float(target.get("brightness", 0.0))
        saturation = float(target.get("saturation", 0.0))
        circularity = float(target.get("circularity", 0.0))
        score = float(target.get("score", 0.0))
        color = str(target.get("color", "-"))
        self.update(score, area, brightness)
        stability = self._stability()
        confidence = 0.0
        reasons = []
        if area >= 4:
            confidence += 0.20; reasons.append("área ok")
        else:
            reasons.append("área baixa")
        if brightness >= 70:
            confidence += 0.25; reasons.append("brilho ok")
        else:
            reasons.append("brilho baixo")
        if color in {"VERMELHO","AMARELO"}:
            if saturation >= 80:
                confidence += 0.20; reasons.append("saturação ok")
            else:
                reasons.append("saturação baixa")
        else:
            if brightness >= 180:
                confidence += 0.20; reasons.append("branco brilhante")
            else:
                reasons.append("branco fraco")
        if circularity >= 0.25:
            confidence += 0.10; reasons.append("forma aceitável")
        else:
            reasons.append("forma irregular")
        confidence += 0.25 * stability
        reasons.append(f"estabilidade {round(stability, 2)}")
        return {"is_valid_led": confidence >= 0.55, "confidence": round(min(confidence, 0.99), 2), "reason": ", ".join(reasons)}
