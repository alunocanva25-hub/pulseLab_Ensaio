from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading

import av
import cv2
import numpy as np

from detector.color_detector import build_color_masks, merge_masks, analyze_best_target
from detector.dataset import save_sample


@dataclass
class CollectorConfig:
    roi_size: float = 0.20
    show_overlay: bool = True
    led_color_mode: str = "VERMELHO"
    sequence_size: int = 12


class PulseCollectorProcessor:
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.lock = threading.Lock()

        self.sequence = deque(maxlen=max(4, int(config.sequence_size)))
        self.last_roi_bgr = None

        self.status = "AGUARDANDO"
        self.detected_color = "-"
        self.area = 0.0
        self.brightness = 0.0
        self.score = 0.0
        self.reason = "Aguardando frames"

    def _calc_score(self, merged_mask):
        if merged_mask is None or merged_mask.size == 0:
            return 0.0
        return float((merged_mask.sum() / (merged_mask.size * 255.0)) * 100.0)

    def save_current_sample(self, label: str, meta: dict | None = None):
        if self.last_roi_bgr is None:
            return None

        extra = {
            "collector_status": self.status,
            "detected_color": self.detected_color,
            "area": round(self.area, 2),
            "brightness": round(self.brightness, 2),
            "score": round(self.score, 2),
        }
        if meta:
            extra.update(meta)

        return save_sample(
            self.last_roi_bgr,
            label,
            extra,
            sequence=list(self.sequence),
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        size = max(int(min(w, h) * float(self.config.roi_size)), 40)
        x = w // 2 - size // 2
        y = h // 2 - size // 2

        roi = img[y:y + size, x:x + size].copy()
        self.last_roi_bgr = roi.copy()
        self.sequence.append(roi.copy())

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        color_masks = build_color_masks(hsv, self.config.led_color_mode)
        merged = merge_masks(color_masks)
        score = self._calc_score(merged)

        target = analyze_best_target(hsv, color_masks, prev_center=None, prefer_center_weight=1.0)

        with self.lock:
            self.score = score

            if target is not None:
                self.detected_color = str(target.get("color", "-"))
                self.area = float(target.get("area", 0.0))
                self.brightness = float(target.get("brightness", 0.0))
                self.status = "ALVO ENCONTRADO"
                self.reason = "ROI pronta para coleta"
            else:
                self.detected_color = "-"
                self.area = 0.0
                self.brightness = 0.0
                self.status = "SEM ALVO"
                self.reason = "Nenhum alvo consistente no ROI"

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

            overlay = f"{self.status} | Cor:{self.detected_color} | Seq:{len(self.sequence)}"
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
                "detected_color": self.detected_color,
                "area": round(self.area, 2),
                "brightness": round(self.brightness, 2),
                "score": round(self.score, 2),
                "sequence_len": len(self.sequence),
                "roi_ready": self.last_roi_bgr is not None,
                "reason": self.reason,
            }