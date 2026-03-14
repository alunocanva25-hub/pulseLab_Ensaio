# pulselab_webrtc_live_prototype.py
# -----------------------------------------------------------------------------
# PulseLab - Protótipo ao vivo com streamlit-webrtc
#
# Objetivo:
# - usar câmera ao vivo na própria área de detecção
# - tentar priorizar câmera traseira
# - aplicar macro/zoom digital na própria imagem
# - desenhar ROI na própria imagem
# - mostrar OFF/ON em tempo real
# - contar pulsos automaticamente de forma experimental
#
# Dependências:
# pip install streamlit streamlit-webrtc av opencv-python-headless numpy pandas
# -----------------------------------------------------------------------------

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(page_title="PulseLab Live Prototype", page_icon="🔴", layout="wide")

# -----------------------------------------------------------------------------
# Estado
# -----------------------------------------------------------------------------
for key, value in {
    "calib_off": None,
    "calib_on": None,
    "saved_message": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------------------------------------------------------
# Configuração
# -----------------------------------------------------------------------------
st.title("🔴 PulseLab v6 - Protótipo ao vivo")
st.caption(
    "Câmera ao vivo + ROI + macro digital + OFF/ON em tempo real + contagem experimental de pulsos."
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    zoom_digital = st.slider("Zoom digital", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
with c2:
    roi_size = st.slider("Tamanho ROI", min_value=0.05, max_value=0.50, value=0.20, step=0.01)
with c3:
    threshold_margin = st.slider("Ajuste threshold", min_value=-50.0, max_value=50.0, value=0.0, step=0.5)
with c4:
    show_overlay = st.toggle("Mostrar overlay", value=True)

t1, t2, t3 = st.columns(3)
with t1:
    debounce_ms = st.number_input("Debounce (ms)", min_value=50, max_value=2000, value=250, step=10)
with t2:
    min_on_ms = st.number_input("Mín. ON (ms)", min_value=20, max_value=2000, value=80, step=10)
with t3:
    min_off_ms = st.number_input("Mín. OFF (ms)", min_value=20, max_value=2000, value=80, step=10)

detector_enabled = st.toggle("Detector ativo", value=True)

st.info(
    "Use a câmera traseira do celular, aproxime o LED e mantenha o LED dentro da ROI verde. "
    "Macro óptico real depende do aparelho/navegador; aqui usamos macro digital via recorte. "
    "O streamlit-webrtc é a base de vídeo em tempo real para Streamlit, enquanto st.camera_input é um widget de captura de foto. "
    "citeturn947611view0turn947611view1"
)


# -----------------------------------------------------------------------------
# Modelo de configuração
# -----------------------------------------------------------------------------
@dataclass
class DetectorConfig:
    zoom_digital: float
    roi_size: float
    calib_off: float | None
    calib_on: float | None
    threshold_margin: float
    debounce_ms: int
    min_on_ms: int
    min_off_ms: int
    detector_enabled: bool
    show_overlay: bool


class PulseDetectorProcessor:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.lock = threading.Lock()

        self.red_score = None
        self.status = "NÃO ANALISADO"
        self.confidence = None
        self.guidance = "Aguardando frames"
        self.pulse_count = 0
        self.detector_state = "IDLE"
        self.last_transition_ts = 0.0
        self.last_pulse_ts = 0.0
        self.threshold = None
        self.r_mean = None
        self.g_mean = None
        self.b_mean = None
        self.brightness = None

    def _get_threshold(self):
        if self.config.calib_off is None or self.config.calib_on is None:
            return None
        return ((float(self.config.calib_off) + float(self.config.calib_on)) / 2.0) + float(self.config.threshold_margin)

    def _process_detector(self, status: str, ts_now: float):
        if not self.config.detector_enabled:
            return

        current = self.detector_state
        debounce_s = float(self.config.debounce_ms) / 1000.0
        min_on_s = float(self.config.min_on_ms) / 1000.0
        min_off_s = float(self.config.min_off_ms) / 1000.0

        if current == "IDLE":
            if status == "LED DESLIGADO":
                self.detector_state = "OFF"
                self.last_transition_ts = ts_now
            elif status == "LED LIGADO":
                self.detector_state = "ON"
                self.last_transition_ts = ts_now
            return

        if current == "OFF" and status == "LED LIGADO":
            if (ts_now - self.last_transition_ts) >= min_off_s:
                self.detector_state = "ON"
                self.last_transition_ts = ts_now
            return

        if current == "ON" and status == "LED DESLIGADO":
            on_duration = ts_now - self.last_transition_ts
            since_last_pulse = ts_now - self.last_pulse_ts
            if on_duration >= min_on_s and since_last_pulse >= debounce_s:
                self.pulse_count += 1
                self.last_pulse_ts = ts_now
                self.detector_state = "OFF"
                self.last_transition_ts = ts_now
            return

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # Macro digital / recorte central
        zoom = max(1.0, float(self.config.zoom_digital))
        crop_w = max(20, int(w / zoom))
        crop_h = max(20, int(h / zoom))
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        crop = img[y1:y2, x1:x2].copy()

        ch, cw = crop.shape[:2]
        roi_w = max(8, int(cw * float(self.config.roi_size)))
        roi_h = max(8, int(ch * float(self.config.roi_size)))
        rx1 = max(0, cw // 2 - roi_w // 2)
        ry1 = max(0, ch // 2 - roi_h // 2)
        rx2 = min(cw, rx1 + roi_w)
        ry2 = min(ch, ry1 + roi_h)
        roi = crop[ry1:ry2, rx1:rx2]

        # Análise óptica
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        r = float(np.mean(roi_rgb[:, :, 0]))
        g = float(np.mean(roi_rgb[:, :, 1]))
        b = float(np.mean(roi_rgb[:, :, 2]))
        brightness = float(np.mean(roi_rgb))
        red_score = r - ((g + b) / 2.0)
        threshold = self._get_threshold()

        if threshold is not None:
            status = "LED LIGADO" if red_score >= threshold else "LED DESLIGADO"
            distance = abs(red_score - threshold)
            confidence = max(35.0, min(99.0, 45.0 + distance))
            if confidence < 60:
                guidance = "Baixa confiança - aproxime mais ou recalibre"
            elif confidence < 75:
                guidance = "Confiança média - estabilize a câmera"
            else:
                guidance = "Captura boa"
        else:
            status = "LED LIGADO" if red_score > 20 else "LED DESLIGADO"
            confidence = 78.0 if abs(red_score) > 20 else 55.0
            guidance = "Calibre OFF e ON"

        if brightness < 20:
            guidance = "Imagem muito escura"
        elif brightness > 240:
            guidance = "Imagem muito clara / estourada"

        ts_now = time.time()
        self._process_detector(status, ts_now)

        if self.config.show_overlay:
            cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
            cv2.putText(
                crop,
                f"{status} | Score: {red_score:.1f} | Pulsos: {self.pulse_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        with self.lock:
            self.red_score = red_score
            self.status = status
            self.confidence = confidence
            self.guidance = guidance
            self.threshold = threshold
            self.r_mean = r
            self.g_mean = g
            self.b_mean = b
            self.brightness = brightness

        return av.VideoFrame.from_ndarray(crop, format="bgr24")

    def get_snapshot(self):
        with self.lock:
            return {
                "red_score": self.red_score,
                "status": self.status,
                "confidence": self.confidence,
                "guidance": self.guidance,
                "pulse_count": self.pulse_count,
                "detector_state": self.detector_state,
                "threshold": self.threshold,
                "r_mean": self.r_mean,
                "g_mean": self.g_mean,
                "b_mean": self.b_mean,
                "brightness": self.brightness,
            }


config = DetectorConfig(
    zoom_digital=zoom_digital,
    roi_size=roi_size,
    calib_off=st.session_state.calib_off,
    calib_on=st.session_state.calib_on,
    threshold_margin=threshold_margin,
    debounce_ms=int(debounce_ms),
    min_on_ms=int(min_on_ms),
    min_off_ms=int(min_off_ms),
    detector_enabled=detector_enabled,
    show_overlay=show_overlay,
)

rtc_configuration = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
}

media_stream_constraints = {
    "video": {
        "facingMode": {"ideal": "environment"},
        "width": {"ideal": 1280},
        "height": {"ideal": 720},
    },
    "audio": False,
}

ctx = webrtc_streamer(
    key="pulselab-live-led",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    media_stream_constraints=media_stream_constraints,
    video_processor_factory=lambda: PulseDetectorProcessor(config),
    async_processing=True,
)

# -----------------------------------------------------------------------------
# Painel lateral / calibração
# -----------------------------------------------------------------------------
left, right = st.columns([1.6, 1])

with right:
    st.subheader("Leitura ao vivo")
    placeholder_metrics = st.empty()
    placeholder_info = st.empty()

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Salvar OFF", use_container_width=True):
            if ctx and ctx.video_processor:
                snap = ctx.video_processor.get_snapshot()
                if snap["red_score"] is not None:
                    st.session_state.calib_off = float(snap["red_score"])
                    st.session_state.saved_message = f"OFF salvo: {snap['red_score']:.2f}"
                else:
                    st.session_state.saved_message = "Sem leitura para salvar OFF."
    with b2:
        if st.button("Salvar ON", use_container_width=True):
            if ctx and ctx.video_processor:
                snap = ctx.video_processor.get_snapshot()
                if snap["red_score"] is not None:
                    st.session_state.calib_on = float(snap["red_score"])
                    st.session_state.saved_message = f"ON salvo: {snap['red_score']:.2f}"
                else:
                    st.session_state.saved_message = "Sem leitura para salvar ON."
    with b3:
        if st.button("Limpar calib.", use_container_width=True):
            st.session_state.calib_off = None
            st.session_state.calib_on = None
            st.session_state.saved_message = "Calibração limpa."

    if st.session_state.saved_message:
        st.success(st.session_state.saved_message)

    st.caption(
        "Se o navegador/aparelho permitir, a câmera traseira será priorizada com `facingMode=environment`. "
        "Macro óptico real continua dependendo do dispositivo. "
        "citeturn947611view0"
    )

# -----------------------------------------------------------------------------
# Atualização dos métricos enquanto o stream estiver ativo
# -----------------------------------------------------------------------------
if ctx and ctx.state.playing and ctx.video_processor:
    metrics_ph = right.empty()
    info_ph = right.empty()

    while ctx.state.playing:
        snap = ctx.video_processor.get_snapshot()
        if snap["red_score"] is not None:
            df = pd.DataFrame(
                [
                    {"Métrica": "Status", "Valor": snap["status"]},
                    {"Métrica": "Red Score", "Valor": round(float(snap["red_score"]), 2)},
                    {
                        "Métrica": "Confiança (%)",
                        "Valor": None if snap["confidence"] is None else round(float(snap["confidence"]), 1),
                    },
                    {"Métrica": "Pulsos", "Valor": snap["pulse_count"]},
                    {"Métrica": "Estado detector", "Valor": snap["detector_state"]},
                    {"Métrica": "Threshold", "Valor": None if snap["threshold"] is None else round(float(snap["threshold"]), 2)},
                    {"Métrica": "R médio", "Valor": None if snap["r_mean"] is None else round(float(snap["r_mean"]), 2)},
                    {"Métrica": "G médio", "Valor": None if snap["g_mean"] is None else round(float(snap["g_mean"]), 2)},
                    {"Métrica": "B médio", "Valor": None if snap["b_mean"] is None else round(float(snap["b_mean"]), 2)},
                    {"Métrica": "Brilho", "Valor": None if snap["brightness"] is None else round(float(snap["brightness"]), 2)},
                ]
            )
            metrics_ph.dataframe(df, use_container_width=True, hide_index=True)
            info_ph.info(snap["guidance"])
        time.sleep(0.2)
else:
    with left:
        st.info("Clique em START para abrir a câmera ao vivo na própria área de detecção.")
