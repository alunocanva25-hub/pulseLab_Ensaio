from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(page_title="Pulse Detector Vision", page_icon="🔴", layout="wide")

if "calib_off" not in st.session_state:
    st.session_state.calib_off = None
if "calib_on" not in st.session_state:
    st.session_state.calib_on = None
if "saved_message" not in st.session_state:
    st.session_state.saved_message = ""
if "last_announced_pulse" not in st.session_state:
    st.session_state.last_announced_pulse = 0
if "beep_enabled" not in st.session_state:
    st.session_state.beep_enabled = True
if "beep_ms" not in st.session_state:
    st.session_state.beep_ms = 120
if "beep_freq" not in st.session_state:
    st.session_state.beep_freq = 880


class DetectorPulso:
    def __init__(self, limiar_on: float = 30.0, limiar_off: float = 20.0, debounce: float = 0.2):
        self.estado_anterior = "OFF"
        self.pulsos = 0
        self.ultimo_pulso = 0.0
        self.limiar_on = float(limiar_on)
        self.limiar_off = float(limiar_off)
        self.debounce = float(debounce)

    def atualizar(self, red_score: float):
        if red_score > self.limiar_on:
            estado_atual = "ON"
        elif red_score < self.limiar_off:
            estado_atual = "OFF"
        else:
            estado_atual = self.estado_anterior

        agora = time.time()

        if (
            estado_atual == "ON"
            and self.estado_anterior == "OFF"
            and (agora - self.ultimo_pulso) > self.debounce
        ):
            self.pulsos += 1
            self.ultimo_pulso = agora

        self.estado_anterior = estado_atual
        return estado_atual, self.pulsos


st.title("🔴 Pulse Detector Vision")
st.caption("Detector focado só em LED: câmera, ROI, OFF/ON, histerese, debounce, pulso em tempo real e bip.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    zoom_digital = st.slider("Zoom digital", 1.0, 5.0, 2.2, 0.1)
with c2:
    roi_size = st.slider("Tamanho ROI", 0.05, 0.50, 0.18, 0.01)
with c3:
    detector_enabled = st.toggle("Detector ativo", value=True)
with c4:
    show_overlay = st.toggle("Mostrar overlay", value=True)

r1, r2, r3, r4 = st.columns(4)
with r1:
    smooth_window = st.slider("Média móvel (frames)", 1, 20, 6, 1)
with r2:
    debounce_ms = st.number_input("Debounce (ms)", min_value=50, max_value=3000, value=250, step=10)
with r3:
    min_on_ms = st.number_input("Mín. ON (ms)", min_value=20, max_value=3000, value=80, step=10)
with r4:
    min_off_ms = st.number_input("Mín. OFF (ms)", min_value=20, max_value=3000, value=80, step=10)

h1, h2, h3 = st.columns(3)
with h1:
    on_ratio = st.slider("Limiar ON (%)", 0.50, 0.95, 0.65, 0.01)
with h2:
    off_ratio = st.slider("Limiar OFF (%)", 0.05, 0.80, 0.35, 0.01)
with h3:
    threshold_margin = st.slider("Ajuste fino", -50.0, 50.0, 0.0, 0.5)

b1, b2, b3 = st.columns(3)
with b1:
    st.session_state.beep_enabled = st.toggle("Bip por pulso", value=st.session_state.beep_enabled)
with b2:
    st.session_state.beep_freq = st.number_input("Frequência bip (Hz)", min_value=300, max_value=2000, value=int(st.session_state.beep_freq), step=10)
with b3:
    st.session_state.beep_ms = st.number_input("Duração bip (ms)", min_value=50, max_value=1000, value=int(st.session_state.beep_ms), step=10)

st.info("Fluxo: START → aproximar LED → Salvar OFF → Salvar ON → observar OFF/ON, Pulsos e o bip.")

ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
twilio_cfg = st.secrets.get("twilio_turn", {})
if twilio_cfg:
    raw = twilio_cfg.get("ice_servers_json", "")
    if raw:
        try:
            ice_servers = json.loads(raw)
            st.success("TURN/Twilio carregado de Secrets.")
        except Exception as e:
            st.warning(f"Secrets TURN encontrados, mas JSON inválido: {e}")
else:
    st.warning("Sem TURN em Secrets. Em algumas redes o vídeo pode falhar.")

rtc_configuration = {"iceServers": ice_servers}
media_stream_constraints = {
    "video": {
        "facingMode": {"ideal": "environment"},
        "width": {"ideal": 1280},
        "height": {"ideal": 720},
    },
    "audio": False,
}


@dataclass
class DetectorConfig:
    zoom_digital: float
    roi_size: float
    calib_off: float | None
    calib_on: float | None
    threshold_margin: float
    on_ratio: float
    off_ratio: float
    smooth_window: int
    debounce_ms: int
    min_on_ms: int
    min_off_ms: int
    detector_enabled: bool
    show_overlay: bool


class PulseDetectorProcessor:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.lock = threading.Lock()
        self.red_buffer = deque(maxlen=max(1, int(config.smooth_window)))
        self.brightness_buffer = deque(maxlen=max(1, int(config.smooth_window)))
        self.red_score_raw = None
        self.red_score_smooth = None
        self.brightness_raw = None
        self.status = "NÃO ANALISADO"
        self.confidence = 0.0
        self.guidance = "Aguardando frames"
        self.pulse_count = 0
        self.threshold_on = None
        self.threshold_off = None
        self.last_pulse_quality = "-"
        self.frame_counter = 0
        self.pulse_events = deque(maxlen=100)
        self.detector_state = "OFF"

        self.detector_pulso = DetectorPulso(
            limiar_on=30.0,
            limiar_off=20.0,
            debounce=float(config.debounce_ms) / 1000.0,
        )

    def _compute_thresholds(self):
        if self.config.calib_off is None or self.config.calib_on is None:
            return None, None
        off = float(self.config.calib_off)
        on = float(self.config.calib_on)
        delta = on - off
        threshold_on = off + (delta * float(self.config.on_ratio)) + float(self.config.threshold_margin)
        threshold_off = off + (delta * float(self.config.off_ratio)) + float(self.config.threshold_margin)
        return threshold_on, threshold_off

    def _compute_confidence(self, signal_value: float, brightness: float):
        if self.threshold_on is None or self.threshold_off is None:
            base = min(90.0, max(45.0, 50.0 + abs(signal_value - 20.0)))
        else:
            ref = self.threshold_on if self.status == "LED LIGADO" else self.threshold_off
            distance = abs(signal_value - ref)
            base = min(99.0, max(35.0, 45.0 + distance))
        if brightness < 20 or brightness > 240:
            base -= 20
        if len(self.red_buffer) >= 3:
            std = float(np.std(np.array(self.red_buffer)))
            if std > 25:
                base -= 12
            elif std > 15:
                base -= 6
        return max(0.0, min(99.0, base))

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
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

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        r = float(np.mean(roi_rgb[:, :, 0]))
        g = float(np.mean(roi_rgb[:, :, 1]))
        b = float(np.mean(roi_rgb[:, :, 2]))
        brightness = float(np.mean(roi_rgb))
        red_score = r - ((g + b) / 2.0)

        self.red_buffer.append(red_score)
        self.brightness_buffer.append(brightness)

        red_score_smooth = float(np.mean(np.array(self.red_buffer)))
        brightness_smooth = float(np.mean(np.array(self.brightness_buffer)))

        self.threshold_on, self.threshold_off = self._compute_thresholds()

        if self.threshold_on is not None and self.threshold_off is not None:
            self.detector_pulso.limiar_on = self.threshold_on
            self.detector_pulso.limiar_off = self.threshold_off
        else:
            self.detector_pulso.limiar_on = 30.0 + float(self.config.threshold_margin)
            self.detector_pulso.limiar_off = 20.0 + float(self.config.threshold_margin)

        self.detector_pulso.debounce = float(self.config.debounce_ms) / 1000.0

        estado, pulsos = self.detector_pulso.atualizar(red_score_smooth)
        self.pulse_count = pulsos
        self.detector_state = estado
        self.status = "LED LIGADO" if estado == "ON" else "LED DESLIGADO"

        self.confidence = self._compute_confidence(red_score_smooth, brightness_smooth)

        if brightness_smooth < 20:
            self.guidance = "Imagem muito escura"
        elif brightness_smooth > 240:
            self.guidance = "Imagem muito clara / estourada"
        elif self.confidence < 60:
            self.guidance = "Baixa confiança - aproxime mais ou recalibre"
        elif self.confidence < 75:
            self.guidance = "Confiança média - estabilize a câmera"
        else:
            self.guidance = "Captura boa"

        self.red_score_raw = red_score
        self.red_score_smooth = red_score_smooth
        self.brightness_raw = brightness_smooth
        self.frame_counter += 1

        if len(self.pulse_events) == 0 or (self.pulse_events[0]["pulse"] != self.pulse_count and self.pulse_count > 0):
            quality = "ALTA" if self.confidence >= 80 else ("MÉDIA" if self.confidence >= 60 else "BAIXA")
            self.last_pulse_quality = quality
            self.pulse_events.appendleft({
                "t": round(time.time(), 3),
                "pulse": self.pulse_count,
                "quality": quality,
                "score": round(float(self.red_score_smooth or 0.0), 2),
            })

        if self.config.show_overlay:
            cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
            overlay = f"{self.status} | Score: {red_score_smooth:.1f} | Pulsos: {self.pulse_count} | Conf: {self.confidence:.0f}%"
            cv2.putText(crop, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(crop, format="bgr24")

    def get_snapshot(self):
        with self.lock:
            return {
                "red_score_raw": self.red_score_raw,
                "red_score_smooth": self.red_score_smooth,
                "status": self.status,
                "confidence": self.confidence,
                "guidance": self.guidance,
                "pulse_count": self.pulse_count,
                "detector_state": self.detector_state,
                "threshold_on": self.threshold_on,
                "threshold_off": self.threshold_off,
                "brightness": self.brightness_raw,
                "last_pulse_quality": self.last_pulse_quality,
                "pulse_events": list(self.pulse_events),
                "frame_counter": self.frame_counter,
            }

config = DetectorConfig(
    zoom_digital=zoom_digital,
    roi_size=roi_size,
    calib_off=st.session_state.calib_off,
    calib_on=st.session_state.calib_on,
    threshold_margin=threshold_margin,
    on_ratio=on_ratio,
    off_ratio=off_ratio,
    smooth_window=int(smooth_window),
    debounce_ms=int(debounce_ms),
    min_on_ms=int(min_on_ms),
    min_off_ms=int(min_off_ms),
    detector_enabled=detector_enabled,
    show_overlay=show_overlay,
)

ctx = webrtc_streamer(
    key="pulse-detector-vision",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    media_stream_constraints=media_stream_constraints,
    video_processor_factory=lambda: PulseDetectorProcessor(config),
    async_processing=True,
)

left, right = st.columns([1.6, 1])

def play_beep(freq: int, ms: int):
    components.html(
        f"""
        <script>
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "sine";
        osc.frequency.value = {int(freq)};
        osc.connect(gain);
        gain.connect(ctx.destination);
        gain.gain.value = 0.05;
        osc.start();
        setTimeout(() => {{
            osc.stop();
            ctx.close();
        }}, {int(ms)});
        </script>
        """,
        height=0,
    )

with right:
    st.subheader("Painel do detector")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Salvar OFF", use_container_width=True):
            if ctx and ctx.video_processor:
                snap = ctx.video_processor.get_snapshot()
                if snap["red_score_smooth"] is not None:
                    st.session_state.calib_off = float(snap["red_score_smooth"])
                    st.session_state.saved_message = f"OFF salvo: {snap['red_score_smooth']:.2f}"
    with b2:
        if st.button("Salvar ON", use_container_width=True):
            if ctx and ctx.video_processor:
                snap = ctx.video_processor.get_snapshot()
                if snap["red_score_smooth"] is not None:
                    st.session_state.calib_on = float(snap["red_score_smooth"])
                    st.session_state.saved_message = f"ON salvo: {snap['red_score_smooth']:.2f}"
    with b3:
        if st.button("Limpar calib.", use_container_width=True):
            st.session_state.calib_off = None
            st.session_state.calib_on = None
            st.session_state.saved_message = "Calibração limpa."

    if st.session_state.saved_message:
        st.success(st.session_state.saved_message)

if ctx and ctx.state.playing and ctx.video_processor:
    metrics_ph = right.empty()
    info_ph = right.empty()
    events_ph = st.empty()

    while ctx.state.playing:
        snap = ctx.video_processor.get_snapshot()
        if snap["red_score_smooth"] is not None:
            pulse_count = int(snap["pulse_count"])
            if st.session_state.beep_enabled and pulse_count > st.session_state.last_announced_pulse:
                st.session_state.last_announced_pulse = pulse_count
                play_beep(int(st.session_state.beep_freq), int(st.session_state.beep_ms))

            df = pd.DataFrame(
                [
                    {"Métrica": "Status", "Valor": snap["status"]},
                    {"Métrica": "Red Score bruto", "Valor": round(float(snap["red_score_raw"]), 2)},
                    {"Métrica": "Red Score médio", "Valor": round(float(snap["red_score_smooth"]), 2)},
                    {"Métrica": "Confiança (%)", "Valor": round(float(snap["confidence"]), 1)},
                    {"Métrica": "Pulsos", "Valor": snap["pulse_count"]},
                    {"Métrica": "Estado detector", "Valor": snap["detector_state"]},
                    {"Métrica": "Threshold ON", "Valor": None if snap["threshold_on"] is None else round(float(snap["threshold_on"]), 2)},
                    {"Métrica": "Threshold OFF", "Valor": None if snap["threshold_off"] is None else round(float(snap["threshold_off"]), 2)},
                    {"Métrica": "Brilho", "Valor": None if snap["brightness"] is None else round(float(snap["brightness"]), 2)},
                    {"Métrica": "Qualidade último pulso", "Valor": snap["last_pulse_quality"]},
                    {"Métrica": "Frames", "Valor": snap["frame_counter"]},
                ]
            )
            metrics_ph.dataframe(df, use_container_width=True, hide_index=True)
            info_ph.info(snap["guidance"])
            if snap["pulse_events"]:
                events_ph.dataframe(pd.DataFrame(snap["pulse_events"]), use_container_width=True, hide_index=True)
        time.sleep(0.2)
else:
    with left:
        st.info("Clique em START para abrir a câmera ao vivo.")

st.markdown("### Observação")
st.caption("Integração feita com a sua lógica base de DetectorPulso + bip por pulso. Próxima fase: plugar isso no v5.")
