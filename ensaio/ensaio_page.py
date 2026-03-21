import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
from collections import deque

# =========================
# PROCESSADOR REAL
# =========================
class PulseProcessor(VideoProcessorBase):
    def __init__(self):
        self.buffer = deque(maxlen=3)
        self.pulsos = 0
        self.estado = "OFF"
        self.last_score = 0

    def detectar_cor(self, hsv, modo):
        if modo == "VERMELHO":
            m1 = cv2.inRange(hsv, (0, 80, 80), (10,255,255))
            m2 = cv2.inRange(hsv, (160,80,80), (180,255,255))
            return cv2.add(m1, m2)

        elif modo == "AMARELO":
            return cv2.inRange(hsv, (15,80,80), (35,255,255))

        elif modo == "BRANCO":
            return cv2.inRange(hsv, (0,0,200), (180,40,255))

        elif modo == "AZUL":
            return cv2.inRange(hsv, (90,80,80), (130,255,255))

        return None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        h, w = img.shape[:2]

        # ROI CENTRAL
        size = int(min(h, w) * 0.3)
        x = w//2 - size//2
        y = h//2 - size//2
        roi = img[y:y+size, x:x+size]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        modo = st.session_state.led_color_mode

        mask = self.detectar_cor(hsv, modo)

        score = (np.sum(mask) / (mask.size * 255)) * 100

        self.buffer.append(score)

        smooth = np.mean(self.buffer)

        # ===== PULSO RÁPIDO =====
        if st.session_state.fast_pulse_mode:
            valor = score * 0.7 + smooth * 0.3
        else:
            valor = smooth

        # ===== DETECÇÃO =====
        if self.estado == "OFF" and valor > 15:
            self.estado = "ON"

        elif self.estado == "ON" and valor < 5:
            self.estado = "OFF"
            self.pulsos += 1

        # ===== OVERLAY =====
        cv2.rectangle(img, (x,y), (x+size,y+size), (0,255,255), 2)

        cv2.putText(img, f"PULSOS: {self.pulsos}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        return img


# =========================
# UI PAGE
# =========================
def render_ensaio_page():
    st.subheader("🔴🟡⚪ Detector ao vivo")

    webrtc_streamer(
        key="pulse",
        video_processor_factory=PulseProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )