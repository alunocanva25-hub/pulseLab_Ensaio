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
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(page_title="Pulse Counter Vision", page_icon="🔴", layout="wide")

for k, v in {"calib_off": None, "calib_on": None, "saved_message": ""}.items():
    if k not in st.session_state:
        st.session_state[k] = v

class ContadorPulso:
    def __init__(self, limiar_on: float, limiar_off: float, debounce_s: float = 0.25, min_on_s: float = 0.08, min_off_s: float = 0.08):
        self.limiar_on = float(limiar_on)
        self.limiar_off = float(limiar_off)
        self.debounce_s = float(debounce_s)
        self.min_on_s = float(min_on_s)
        self.min_off_s = float(min_off_s)
        self.estado = "OFF"
        self.pulsos = 0
        self.ultimo_pulso_ts = 0.0
        self.ultima_transicao_ts = time.time()
        self.logs = deque(maxlen=200)
        self.intervalo_medio = None

    def atualizar_parametros(self, limiar_on, limiar_off, debounce_s, min_on_s, min_off_s):
        self.limiar_on = float(limiar_on)
        self.limiar_off = float(limiar_off)
        self.debounce_s = float(debounce_s)
        self.min_on_s = float(min_on_s)
        self.min_off_s = float(min_off_s)

    def atualizar(self, score: float, confidence: float):
        agora = time.time()
        pulso_confirmado = False
        quality = "-"
        if self.estado == "OFF":
            tempo_off = agora - self.ultima_transicao_ts
            if score >= self.limiar_on and tempo_off >= self.min_off_s:
                self.estado = "ON"
                self.ultima_transicao_ts = agora
        elif self.estado == "ON":
            tempo_on = agora - self.ultima_transicao_ts
            if score <= self.limiar_off:
                if tempo_on >= self.min_on_s and (agora - self.ultimo_pulso_ts) >= self.debounce_s:
                    self.pulsos += 1
                    pulso_confirmado = True
                    quality = "ALTA" if confidence >= 80 else ("MÉDIA" if confidence >= 60 else "BAIXA")
                    intervalo = None if self.ultimo_pulso_ts <= 0 else agora - self.ultimo_pulso_ts
                    self.ultimo_pulso_ts = agora
                    self.logs.appendleft({
                        "pulso": self.pulsos,
                        "timestamp": round(agora, 3),
                        "score": round(float(score), 2),
                        "confidence": round(float(confidence), 1),
                        "quality": quality,
                        "intervalo_s": None if intervalo is None else round(intervalo, 3),
                    })
                    intervalos_validos = [x["intervalo_s"] for x in self.logs if x["intervalo_s"] is not None]
                    if intervalos_validos:
                        self.intervalo_medio = round(float(np.mean(intervalos_validos)), 3)
                self.estado = "OFF"
                self.ultima_transicao_ts = agora
        return {
            "estado": self.estado,
            "pulsos": self.pulsos,
            "pulso_confirmado": pulso_confirmado,
            "quality": quality,
            "intervalo_medio": self.intervalo_medio,
            "logs": list(self.logs),
        }

st.title("🔴 Pulse Counter Vision")
st.caption("Foco total na contagem de pulsos: ROI, OFF/ON, histerese, debounce e ciclo OFF→ON→OFF.")

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

st.info("Fluxo: START → aproximar LED → Salvar OFF → Salvar ON → observar pulsos confirmados.")

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
    "video": {"facingMode": {"ideal": "environment"}, "width": {"ideal": 1280}, "height": {"ideal": 720}},
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
        self.brightness = None
        self.status = "NÃO ANALISADO"
        self.confidence = 0.0
        self.guidance = "Aguardando frames"
        self.threshold_on = None
        self.threshold_off = None
        self.last_pulse_quality = "-"
        self.frame_counter = 0
        self.pulse_count = 0
        self.intervalo_medio = None
        self.contador = ContadorPulso(30.0, 20.0, float(config.debounce_ms)/1000.0, float(config.min_on_ms)/1000.0, float(config.min_off_ms)/1000.0)

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
        ref = 20.0
        if self.status == "LED LIGADO" and self.threshold_on is not None:
            ref = self.threshold_on
        elif self.status == "LED DESLIGADO" and self.threshold_off is not None:
            ref = self.threshold_off
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
        limiar_on = self.threshold_on if self.threshold_on is not None else 30.0 + float(self.config.threshold_margin)
        limiar_off = self.threshold_off if self.threshold_off is not None else 20.0 + float(self.config.threshold_margin)

        visual_estado = self.contador.estado
        if visual_estado == "OFF":
            self.status = "LED LIGADO" if red_score_smooth >= limiar_on else "LED DESLIGADO"
        else:
            self.status = "LED DESLIGADO" if red_score_smooth <= limiar_off else "LED LIGADO"

        self.confidence = self._compute_confidence(red_score_smooth, brightness_smooth)

        if self.config.detector_enabled:
            self.contador.atualizar_parametros(
                limiar_on=limiar_on,
                limiar_off=limiar_off,
                debounce_s=float(self.config.debounce_ms) / 1000.0,
                min_on_s=float(self.config.min_on_ms) / 1000.0,
                min_off_s=float(self.config.min_off_ms) / 1000.0,
            )
            result = self.contador.atualizar(red_score_smooth, self.confidence)
            self.pulse_count = result["pulsos"]
            self.intervalo_medio = result["intervalo_medio"]
            if result["pulso_confirmado"]:
                self.last_pulse_quality = result["quality"]

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
        self.brightness = brightness_smooth
        self.frame_counter += 1

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
                "detector_state": self.contador.estado,
                "threshold_on": self.threshold_on,
                "threshold_off": self.threshold_off,
                "brightness": self.brightness,
                "last_pulse_quality": self.last_pulse_quality,
                "pulse_events": list(self.contador.logs),
                "frame_counter": self.frame_counter,
                "intervalo_medio": self.intervalo_medio,
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
    key="pulse-counter-vision",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    media_stream_constraints=media_stream_constraints,
    video_processor_factory=lambda: PulseDetectorProcessor(config),
    async_processing=True,
)

left, right = st.columns([1.6, 1])
with right:
    st.subheader("Painel do contador")
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
            df = pd.DataFrame([
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
                {"Métrica": "Intervalo médio (s)", "Valor": snap["intervalo_medio"]},
                {"Métrica": "Frames", "Valor": snap["frame_counter"]},
            ])
            metrics_ph.dataframe(df, use_container_width=True, hide_index=True)
            info_ph.info(snap["guidance"])
            if snap["pulse_events"]:
                events_ph.dataframe(pd.DataFrame(snap["pulse_events"]), use_container_width=True, hide_index=True)
        time.sleep(0.2)
else:
    with left:
        st.info("Clique em START para abrir a câmera ao vivo.")

st.markdown("### Observação")
st.caption("Agora a prioridade é contagem de pulsos robusta. Depois ajustamos o bip e plugamos no v5.")
