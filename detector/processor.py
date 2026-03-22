from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading

import av
import cv2

from detector.ai_validator import LEDAIVerifier
from detector.color_detector import analyze_best_target, build_color_masks, merge_masks
from detector.pulse_counter import ContadorPulso


@dataclass
class DetectorConfig:
    roi_size: float = 0.20
    show_overlay: bool = True
    smooth_window: int = 3
    detector_enabled: bool = True
    debounce_ms: int = 120
    limiar_on: float = 18.0
    limiar_off: float = 8.0
    led_color_mode: str = "VERMELHO"
    fast_pulse_mode: bool = True


class PulseDetectorProcessor:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.lock = threading.Lock()

        smooth_len = max(1, int(config.smooth_window))
        if config.fast_pulse_mode:
            smooth_len = min(smooth_len, 3)

        self.buffer = deque(maxlen=smooth_len)
        self.instant_buffer = deque(maxlen=2)

        debounce_ms = int(config.debounce_ms)
        if config.fast_pulse_mode:
            debounce_ms = min(debounce_ms, 120)

        self.contador = ContadorPulso(
            limiar_on=float(config.limiar_on),
            limiar_off=float(config.limiar_off),
            debounce_s=float(debounce_ms) / 1000.0,
        )

        self.ai = LEDAIVerifier(history_size=10)

        self.score = 0.0
        self.pulsos = 0
        self.status = "AGUARDANDO"
        self.last_color = "-"
        self.last_area = 0.0
        self.last_brilho = 0.0
        self.last_estado = "OFF"
        self.last_ai_confidence = 0.0
        self.last_ai_reason = "-"
        self.last_target_valid = False

    def _calc_full_score(self, merged_mask):
        if merged_mask is None or merged_mask.size == 0:
            return 0.0
        return float((merged_mask.sum() / (merged_mask.size * 255.0)) * 100.0)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        size = max(int(min(w, h) * float(self.config.roi_size)), 40)
        x = w // 2 - size // 2
        y = h // 2 - size // 2

        roi = img[y:y + size, x:x + size]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        color_masks = build_color_masks(hsv, self.config.led_color_mode)
        merged = merge_masks(color_masks)
        score_full = self._calc_full_score(merged)

        target = analyze_best_target(hsv, color_masks)
        ai_result = self.ai.validate(target, self.config.led_color_mode)

        with self.lock:
            self.last_target_valid = bool(ai_result["is_valid_led"])
            self.last_ai_confidence = float(ai_result["confidence"])
            self.last_ai_reason = str(ai_result["reason"])

            if target is not None:
                self.last_color = str(target["color"])
                self.last_area = float(target["area"])
                self.last_brilho = float(target["brightness"])
            else:
                self.last_color = "-"
                self.last_area = 0.0
                self.last_brilho = 0.0

            if target is not None and ai_result["is_valid_led"]:
                raw = score_full * 1.30
                self.status = f"{self.last_color}"
            elif target is not None:
                raw = score_full * 0.50
                self.status = f"{self.last_color} DUVIDOSO"
            else:
                raw = score_full * 0.15
                self.status = "SEM LED"

            area_pixels = float(merged.sum() / 255.0) if merged is not None else 0.0
            if area_pixels < 30:
                raw = 0.0

            self.instant_buffer.append(raw)
            instant = max(self.instant_buffer) if self.instant_buffer else raw

            self.buffer.append(raw)
            smooth = sum(self.buffer) / len(self.buffer)

            score = (instant * 0.70 + smooth * 0.30) if self.config.fast_pulse_mode else smooth
            self.score = score

            if self.config.detector_enabled:
                estado, pulsos, pulso = self.contador.atualizar(score)
                self.last_estado = estado
                self.pulsos = pulsos

                if pulso:
                    self.status = "PULSO DETECTADO"
                elif self.status == "SEM LED":
                    self.status = "LED ON" if estado == "ON" else "LED OFF"

        if self.config.show_overlay:
            cv2.rectangle(img, (x, y), (x + size, y + size), (0, 255, 255), 2)

            if target is not None and target.get("bbox"):
                tx, ty, tw, th = target["bbox"]
                cv2.rectangle(
                    img,
                    (x + tx, y + ty),
                    (x + tx + tw, y + ty + th),
                    (255, 200, 0),
                    2,
                )

            cv2.putText(
                img,
                f"{self.status} | P:{self.pulsos}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_snapshot(self):
        with self.lock:
            return {
                "status": self.status,
                "score": round(self.score, 2),
                "pulse_count": int(self.pulsos),
                "color": self.last_color,
                "area": round(self.last_area, 2),
                "brilho": round(self.last_brilho, 2),
                "estado": self.last_estado,
                "ai_confidence": round(self.last_ai_confidence, 2),
                "ai_reason": self.last_ai_reason,
                "target_valid": self.last_target_valid,
            }