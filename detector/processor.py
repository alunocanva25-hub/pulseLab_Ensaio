from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading

import av
import cv2

from detector.ai_validator import LEDAIVerifier
from detector.color_detector import analyze_best_target, build_color_masks, merge_masks
from detector.dataset import save_sample
from detector.model_inference import DetectorModel
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
    auto_calibrate: bool = True
    target_lock: bool = True


class PulseDetectorProcessor:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.lock = threading.Lock()

        smooth_len = max(1, int(config.smooth_window))
        if config.fast_pulse_mode:
            smooth_len = min(smooth_len, 3)

        self.buffer = deque(maxlen=smooth_len)
        self.instant_buffer = deque(maxlen=2)
        self.score_hist = deque(maxlen=20)

        debounce_ms = int(config.debounce_ms)
        if config.fast_pulse_mode:
            debounce_ms = min(debounce_ms, 120)

        self.contador = ContadorPulso(
            limiar_on=float(config.limiar_on),
            limiar_off=float(config.limiar_off),
            debounce_s=float(debounce_ms) / 1000.0,
        )

        self.ai = LEDAIVerifier(history_size=20)
        self.model = DetectorModel()

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
        self.last_hz = 0.0
        self.prev_center = None
        self.current_limiar_on = float(config.limiar_on)
        self.current_limiar_off = float(config.limiar_off)

        self.last_model_label = "desconhecido"
        self.last_model_conf = 0.0
        self.last_roi_bgr = None

    def _calc_full_score(self, merged_mask):
        if merged_mask is None or merged_mask.size == 0:
            return 0.0
        return float((merged_mask.sum() / (merged_mask.size * 255.0)) * 100.0)

    def _auto_thresholds(self):
        if len(self.score_hist) < 8:
            return self.current_limiar_on, self.current_limiar_off

        vals = list(self.score_hist)
        smin = min(vals)
        smax = max(vals)
        amp = max(smax - smin, 1.0)

        off = smin + amp * 0.20
        on = smin + amp * 0.55

        if on <= off:
            on = off + 1.0

        return round(on, 2), round(off, 2)

    def save_current_sample(self, label: str):
        if self.last_roi_bgr is None:
            return None

        meta = {
            "selected_color": self.config.led_color_mode,
            "score": round(self.score, 2),
            "status": self.status,
            "model_label": self.last_model_label,
            "model_conf": round(self.last_model_conf, 4),
            "ai_conf": round(self.last_ai_confidence, 4),
        }
        return save_sample(self.last_roi_bgr, label, meta)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        size = max(int(min(w, h) * float(self.config.roi_size)), 40)
        x = w // 2 - size // 2
        y = h // 2 - size // 2

        roi = img[y:y + size, x:x + size].copy()
        self.last_roi_bgr = roi.copy()

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        prev_center = self.prev_center if self.config.target_lock else None

        color_masks = build_color_masks(hsv, self.config.led_color_mode)
        merged = merge_masks(color_masks)
        score_full = self._calc_full_score(merged)

        target = analyze_best_target(
            hsv,
            color_masks,
            prev_center=prev_center,
            prefer_center_weight=1.0,
        )

        ai_result = self.ai.validate(target, self.config.led_color_mode)
        model_result = self.model.predict(roi)

        with self.lock:
            self.last_model_label = model_result["label"]
            self.last_model_conf = model_result["confidence"]

            self.last_target_valid = bool(ai_result["is_valid_led"])
            self.last_ai_confidence = float(ai_result["confidence"])
            self.last_ai_reason = str(ai_result["reason"])

            if target is not None:
                self.last_color = str(target["color"])
                self.last_area = float(target["area"])
                self.last_brilho = float(target["brightness"])
                self.prev_center = (target["center_x"], target["center_y"])
            else:
                self.last_color = "-"
                self.last_area = 0.0
                self.last_brilho = 0.0
                self.prev_center = None

            model_accept = model_result["label"] == "on" and model_result["confidence"] >= 0.60

            if target is not None and ai_result["is_valid_led"]:
                raw = score_full * 1.30
                self.status = f"{self.last_color}"
            elif target is not None:
                raw = score_full * 0.30
                self.status = f"{self.last_color} DUVIDOSO"
            else:
                raw = 0.0
                self.status = "SEM LED"

            area_pixels = float(merged.sum() / 255.0) if merged is not None else 0.0
            if area_pixels < 30:
                raw = 0.0

            self.instant_buffer.append(raw)
            instant = max(self.instant_buffer) if self.instant_buffer else raw

            self.buffer.append(raw)
            smooth = sum(self.buffer) / len(self.buffer)

            if self.config.fast_pulse_mode:
                score = (instant * 0.70) + (smooth * 0.30)
            else:
                score = smooth

            self.score = score
            self.score_hist.append(score)

            if self.config.auto_calibrate:
                self.current_limiar_on, self.current_limiar_off = self._auto_thresholds()
                self.contador.limiar_on = self.current_limiar_on
                self.contador.limiar_off = self.current_limiar_off

            allow_count = self.config.detector_enabled and self.last_target_valid
            if self.model.is_loaded:
                allow_count = allow_count and model_accept

            if allow_count:
                estado, pulsos, pulso, hz = self.contador.atualizar(score)
                self.last_estado = estado
                self.pulsos = pulsos
                self.last_hz = hz

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

            overlay = (
                f"{self.status} | P:{self.pulsos} | Hz:{self.last_hz} | "
                f"M:{self.last_model_label}:{round(self.last_model_conf,2)}"
            )
            cv2.putText(
                img,
                overlay,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
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
                "hz": round(self.last_hz, 2),
                "limiar_on": round(self.current_limiar_on, 2),
                "limiar_off": round(self.current_limiar_off, 2),
                "model_label": self.last_model_label,
                "model_confidence": round(self.last_model_conf, 2),
                "model_loaded": self.model.is_loaded,
            }
