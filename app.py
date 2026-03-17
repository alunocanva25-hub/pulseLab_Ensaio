import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import time
from collections import deque

# =========================
# CONTADOR DE PULSO
# =========================
class ContadorPulso:
    def __init__(self):
        self.estado = "OFF"
        self.pulsos = 0
        self.ultimo_pulso = 0

        self.limiar_on = 15
        self.limiar_off = 5
        self.debounce = 0.2

    def atualizar(self, score):
        agora = time.time()

        if score > self.limiar_on:
            estado_atual = "ON"
        elif score < self.limiar_off:
            estado_atual = "OFF"
        else:
            estado_atual = self.estado

        pulso = False

        if (
            estado_atual == "ON"
            and self.estado == "OFF"
            and (agora - self.ultimo_pulso) > self.debounce
        ):
            self.pulsos += 1
            self.ultimo_pulso = agora
            pulso = True

        self.estado = estado_atual

        return estado_atual, self.pulsos, pulso


# =========================
# PROCESSADOR
# =========================
class PulseProcessor(VideoProcessorBase):
    def __init__(self):
        self.buffer = deque(maxlen=5)
        self.contador = ContadorPulso()
        self.pulsos = 0
        self.status = "AGUARDANDO"
        self.score = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # ROI CENTRAL
        size = int(min(w, h) * 0.25)
        x = w // 2 - size // 2
        y = h // 2 - size // 2

        roi = img[y:y+size, x:x+size]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # máscara vermelho
        m1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        m2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
        mask = cv2.add(m1, m2)

        # score total (fallback)
        score_full = (np.sum(mask) / (mask.size * 255)) * 100

        # contornos (LED)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        melhor = None
        melhor_score = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 3:
                continue

            mask_c = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(mask_c, [c], -1, 255, -1)

            brilho = cv2.mean(hsv[:, :, 2], mask=mask_c)[0]

            score = area * brilho

            if score > melhor_score:
                melhor_score = score
                melhor = (c, area, brilho)

        # =========================
        # NOVO BLOCO HÍBRIDO (IMPORTANTE)
        # =========================
        if melhor is not None:
            c, area, brilho = melhor

            if area >= 3 and brilho >= 70:
                score = score_full * 1.2
                self.status = "LED DETECTADO"
            else:
                score = score_full * 0.6
                self.status = "ALVO FRACO"

        else:
            score = score_full * 0.75
            self.status = "SEM ALVO"

        # suavização
        self.buffer.append(score)
        score_smooth = np.mean(self.buffer)

        self.score = score_smooth

        estado, pulsos, pulso = self.contador.atualizar(score_smooth)

        self.pulsos = pulsos

        if pulso:
            self.status = "PULSO DETECTADO"

        # overlay simples
        cv2.rectangle(img, (x, y), (x+size, y+size), (0, 255, 255), 2)
        cv2.putText(img, f"{estado} | P:{pulsos}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =========================
# UI
# =========================
st.set_page_config(layout="wide")
st.title("PulseLab v2 - Detector de Pulso")

ctx = webrtc_streamer(
    key="pulse",
    video_processor_factory=PulseProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_processor:
    st.metric("Pulsos", ctx.video_processor.pulsos)
    st.metric("Status", ctx.video_processor.status)
    st.metric("Score", round(ctx.video_processor.score, 2))