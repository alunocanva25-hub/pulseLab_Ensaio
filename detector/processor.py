from __future__ import annotations



from collections import deque

from dataclasses import dataclass



import av

import cv2

import numpy as np



from detector.ai_validator import LEDAIVerifier

from detector.color_detector import analyze_best_target, build_color_masks, merge_masks

from detector.pulse_counter import ContadorPulso





@dataclass

class DetectorConfig:

    roi_size: float = 0.20

    show_overlay: bool = True

    smooth_window: int = 5

    detector_enabled: bool = True

    debounce_ms: int = 250

    limiar_on: float = 30.0

    limiar_off: float = 20.0





class PulseDetectorProcessor:

    def __init__(self, config: DetectorConfig):

        self.config = config

        self.buffer = deque(maxlen=max(1, int(config.smooth_window)))



        self.contador = ContadorPulso(

            limiar_on=float(config.limiar_on),

            limiar_off=float(config.limiar_off),

            debounce_s=float(config.debounce_ms) / 1000.0,

        )



        self.ai_validator = LEDAIVerifier(history_size=12)



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

        return float((np.sum(merged_mask) / (merged_mask.size * 255.0)) * 100.0)



    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        h, w = img.shape[:2]



        size = int(min(w, h) * float(self.config.roi_size))

        size = max(size, 40)



        x = w // 2 - size // 2

        y = h // 2 - size // 2



        roi = img[y:y + size, x:x + size]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)



        color_masks = build_color_masks(hsv)

        merged_mask = merge_masks(color_masks)

        score_full = self._calc_full_score(merged_mask)



        target = analyze_best_target(hsv, color_masks)

        ai_result = self.ai_validator.validate(target)



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



        # lógica híbrida:

        # alvo bom -> score forte

        # alvo ruim mas visível -> score reduzido

        # sem alvo -> fallback baixo

        if target is not None and ai_result["is_valid_led"]:

            score = score_full * 1.25

            self.status = f"{self.last_color}"

        elif target is not None and not ai_result["is_valid_led"]:

            score = score_full * 0.45

            self.status = f"{self.last_color} DUVIDOSO"

        else:

            score = score_full * 0.20

            self.status = "SEM LED"



        self.buffer.append(score)

        score = float(np.mean(self.buffer))

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
