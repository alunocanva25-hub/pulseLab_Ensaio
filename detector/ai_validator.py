from collections import deque
import numpy as np


class LEDAIVerifier:
    def __init__(self, history_size=10):
        self.score_history = deque(maxlen=history_size)
        self.color_history = deque(maxlen=5)

    def _color_match(self, target_color, selected_color):
        if selected_color == "AUTOMÁTICO":
            return True
        return target_color == selected_color

    def validate(self, target, selected_color):
        if target is None:
            return {
                "is_valid_led": False,
                "confidence": 0.0,
                "reason": "Nenhum alvo"
            }

        score = float(target.get("score", 0))
        area = float(target.get("area", 0))
        brilho = float(target.get("brightness", 0))
        cor = target.get("color", "N/A")

        self.score_history.append(score)
        self.color_history.append(cor)

        # =========================
        # 1. VALIDA COR SELECIONADA
        # =========================
        color_ok = self._color_match(cor, selected_color)

        # =========================
        # 2. CONSISTÊNCIA TEMPORAL
        # =========================
        consistency = self.color_history.count(cor) / len(self.color_history)

        # =========================
        # 3. ESTABILIDADE DO SCORE
        # =========================
        stability = 0.5
        if len(self.score_history) >= 4:
            arr = np.array(self.score_history)
            std = np.std(arr)
            mean = np.mean(arr)
            if mean > 0:
                ratio = std / mean
                if ratio < 0.2:
                    stability = 0.9
                elif ratio < 0.4:
                    stability = 0.6
                else:
                    stability = 0.3

        # =========================
        # SCORE FINAL
        # =========================
        confidence = 0.0

        if color_ok:
            confidence += 0.4

        if area > 5:
            confidence += 0.2

        if brilho > 80:
            confidence += 0.2

        confidence += consistency * 0.2
        confidence *= stability

        is_valid = confidence > 0.5

        return {
            "is_valid_led": is_valid,
            "confidence": round(confidence, 2),
            "reason": f"cor={cor}, match={color_ok}, consist={round(consistency,2)}"
        }