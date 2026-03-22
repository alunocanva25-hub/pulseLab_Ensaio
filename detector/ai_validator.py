from __future__ import annotations

from collections import deque
import numpy as np
import math


class LEDAIVerifier:
    def __init__(self, history_size: int = 20):
        self.score_history = deque(maxlen=history_size)
        self.color_history = deque(maxlen=8)
        self.area_history = deque(maxlen=8)
        self.brightness_history = deque(maxlen=8)
        self.cx_history = deque(maxlen=8)
        self.cy_history = deque(maxlen=8)

    def _color_match(self, detected_color: str, selected_color: str) -> bool:
        if selected_color == "AUTOMÁTICO":
            return True
        return detected_color == selected_color

    def _stability(self, values):
        if len(values) < 4:
            return 0.5

        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)

        if mean <= 0:
            return 0.2

        ratio = std / mean

        if ratio < 0.10:
            return 0.95
        if ratio < 0.20:
            return 0.85
        if ratio < 0.35:
            return 0.70
        if ratio < 0.50:
            return 0.50
        return 0.25

    def _position_stability(self):
        if len(self.cx_history) < 4:
            return 0.5

        xs = np.array(self.cx_history)
        ys = np.array(self.cy_history)

        std_pos = math.sqrt(np.std(xs) ** 2 + np.std(ys) ** 2)

        if std_pos < 2:
            return 0.95
        if std_pos < 4:
            return 0.80
        if std_pos < 7:
            return 0.60
        if std_pos < 12:
            return 0.40
        return 0.20

    def validate(self, target: dict | None, selected_color: str):
        if target is None:
            return {
                "is_valid_led": False,
                "confidence": 0.05,
                "reason": "Nenhum alvo"
            }

        score = float(target.get("score", 0))
        area = float(target.get("area", 0))
        brightness = float(target.get("brightness", 0))
        color = target.get("color", "N/A")
        cx = float(target.get("center_x", 0))
        cy = float(target.get("center_y", 0))

        self.score_history.append(score)
        self.color_history.append(color)
        self.area_history.append(area)
        self.brightness_history.append(brightness)
        self.cx_history.append(cx)
        self.cy_history.append(cy)

        color_ok = self._color_match(color, selected_color)

        color_consistency = self.color_history.count(color) / len(self.color_history)
        score_stability = self._stability(self.score_history)
        area_stability = self._stability(self.area_history)
        brightness_stability = self._stability(self.brightness_history)
        pos_stability = self._position_stability()

        confidence = 0.0

        if color_ok:
            confidence += 0.35
        if area >= 4:
            confidence += 0.15
        if brightness >= 70:
            confidence += 0.15
        if score >= 5:
            confidence += 0.10

        confidence += color_consistency * 0.10
        confidence += score_stability * 0.05
        confidence += area_stability * 0.04
        confidence += brightness_stability * 0.03
        confidence += pos_stability * 0.03

        return {
            "is_valid_led": confidence >= 0.55,
            "confidence": round(min(confidence, 0.99), 2),
            "reason": f"cor={color}, match={color_ok}, estabilidade={round(score_stability,2)}",
        }