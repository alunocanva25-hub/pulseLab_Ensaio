



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

st.set_page_config(page_title="PulseLab - Ensaio", page_icon="🔴", layout="wide")

# =============================================================================
# ESTADO
# =============================================================================
CONSTANTES_KDKH = [
    0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.625, 0.900, 0.960,
    1.000, 1.250, 1.500, 1.800, 2.000, 2.400, 2.800, 3.000, 3.125,
    3.600, 4.800, 5.400, 6.250, 7.200, 8.000, 9.600, 10.800, 21.600
]

DEFAULTS = {
    "modo_tela": "config_ensaio",   # config_ensaio | captura
    "tipo_deteccao": "Manual",      # Manual | LED | Tarja
    "calib_off": None,
    "calib_on": None,
    "saved_message": "",
    "tempo_ensaio_s": 10,
    "ensaio_inicio_ts": None,
    "meta_pulsos": 10,
    "classe_medidor": "B (1%)",
    "constante_kh": 3.600,
    "detector_verificado": False,
    "verificacao_led_seg": 3,
    "mostrar_painel_debug": False,
    "pulsos_manuais": 0,
    "ensaio_iniciado": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =============================================================================
# CONTADOR DE PULSO
# =============================================================================
class ContadorPulso:
    def __init__(
        self,
        limiar_on: float,
        limiar_off: float,
        debounce_s: float = 0.25,
        min_on_s: float = 0.08,
        min_off_s: float = 0.08,
    ):
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

                    if confidence >= 80:
                        quality = "ALTA"
                    elif confidence >= 60:
                        quality = "MÉDIA"
                    else:
                        quality = "BAIXA"

                    intervalo = None
                    if self.ultimo_pulso_ts > 0:
                        intervalo = agora - self.ultimo_pulso_ts

                    self.ultimo_pulso_ts = agora
                    self.logs.appendleft(
                        {
                            "pulso": self.pulsos,
                            "timestamp": round(agora, 3),
                            "score": round(float(score), 2),
                            "confidence": round(float(confidence), 1),
                            "quality": quality,
                            "intervalo_s": None if intervalo is None else round(intervalo, 3),
                        }
                    )

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


# =============================================================================
# TURN / WEBRTC
# =============================================================================
ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
twilio_cfg = st.secrets.get("twilio_turn", {})
if twilio_cfg:
    raw = twilio_cfg.get("ice_servers_json", "")
    if raw:
        try:
            ice_servers = json.loads(raw)
        except Exception:
            pass

rtc_configuration = {"iceServers": ice_servers}
media_stream_constraints = {
    "video": {
        "facingMode": {"ideal": "environment"},
        "width": {"ideal": 1280},
        "height": {"ideal": 720},
    },
    "audio": False,
}


# =============================================================================
# PROCESSADOR AO VIVO
# =============================================================================
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
        self.frame_counter = 0
        self.last_pulse_quality = "-"
        self.pulse_count = 0
        self.intervalo_medio = None

        self.contador = ContadorPulso(
            limiar_on=30.0,
            limiar_off=20.0,
            debounce_s=float(config.debounce_ms) / 1000.0,
            min_on_s=float(config.min_on_ms) / 1000.0,
            min_off_s=float(config.min_off_ms) / 1000.0,
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
            self.guidance = "Imagem escura"
        elif brightness_smooth > 240:
            self.guidance = "Imagem estourada"
        elif self.confidence < 60:
            self.guidance = "Baixa confiança"
        elif self.confidence < 75:
            self.guidance = "Confiança média"
        else:
            self.guidance = "Captura boa"

        self.red_score_raw = red_score
        self.red_score_smooth = red_score_smooth
        self.brightness = brightness_smooth
        self.frame_counter += 1

        if self.config.show_overlay:
            cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
            overlay = f"{self.status} | P:{self.pulse_count}"
            cv2.putText(
                crop,
                overlay,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

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


# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
.block-container {padding-top: 0.4rem; padding-bottom: 1rem;}
.top-strip {
    background: #111;
    color: #f7ea1c;
    border-radius: 14px;
    padding: 10px 12px;
    margin-bottom: 10px;
    font-weight: 800;
    font-size: 0.90rem;
}
.ensaio-toolbar {
    background: #f3f6fb;
    border: 1px solid #d9e2ef;
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 10px;
}
.compact-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 8px;
}
.action-row .stButton > button {
    width: 100%;
    min-height: 54px;
    font-weight: 800;
    border-radius: 14px;
}
.icon-btn .stButton > button {
    width: 100%;
    min-height: 42px;
    border-radius: 12px;
    font-size: 1.1rem;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR = CONFIG GERAL
# =============================================================================
with st.sidebar:
    st.header("Configuração geral")
    zoom_digital = st.slider("Zoom digital", 1.0, 5.0, 2.2, 0.1)
    roi_size = st.slider("Tamanho ROI", 0.05, 0.50, 0.18, 0.01)
    detector_enabled = st.toggle("Detector ativo", value=True)
    show_overlay = st.toggle("Mostrar overlay", value=True)
    smooth_window = st.slider("Média móvel (frames)", 1, 20, 6, 1)
    debounce_ms = st.number_input("Debounce (ms)", min_value=50, max_value=3000, value=250, step=10)
    min_on_ms = st.number_input("Mín. ON (ms)", min_value=20, max_value=3000, value=80, step=10)
    min_off_ms = st.number_input("Mín. OFF (ms)", min_value=20, max_value=3000, value=80, step=10)
    on_ratio = st.slider("Limiar ON (%)", 0.50, 0.95, 0.65, 0.01)
    off_ratio = st.slider("Limiar OFF (%)", 0.05, 0.80, 0.35, 0.01)
    threshold_margin = st.slider("Ajuste fino", -50.0, 50.0, 0.0, 0.5)

# =============================================================================
# TOPO CONFIG ENSAIO
# =============================================================================
st.title("PulseLab - Ensaio")

if not st.session_state.ensaio_iniciado:
    cfg1, cfg2, cfg3, cfg4 = st.columns(4)
    with cfg1:
        modos = ["Manual", "LED", "Tarja"]
        idx_modo = modos.index(st.session_state.tipo_deteccao) if st.session_state.tipo_deteccao in modos else 0
        st.session_state.tipo_deteccao = st.selectbox("Tipo de ensaio", modos, index=idx_modo)
    with cfg2:
        st.session_state.tempo_ensaio_s = st.number_input("Tempo do ensaio (s)", min_value=1, max_value=3600, value=int(st.session_state.tempo_ensaio_s), step=1)
    with cfg3:
        st.session_state.meta_pulsos = st.number_input("Meta de pulsos", min_value=1, max_value=9999, value=int(st.session_state.meta_pulsos), step=1)
    with cfg4:
        constantes_fmt = [f"{v:.3f}" for v in CONSTANTES_KDKH]
        valor_atual = f"{float(st.session_state.constante_kh):.3f}"
        idx_const = constantes_fmt.index(valor_atual) if valor_atual in constantes_fmt else constantes_fmt.index("3.600")
        selecionada = st.selectbox("Constante Kh/Kd", constantes_fmt, index=idx_const)
        st.session_state.constante_kh = float(selecionada)

    cfg5, cfg6, cfg7, cfg8 = st.columns(4)
    with cfg5:
        classes = ["A (2%)", "B (1%)", "C (0,5%)", "D (0,2%)"]
        idx_classe = classes.index(st.session_state.classe_medidor) if st.session_state.classe_medidor in classes else 1
        st.session_state.classe_medidor = st.selectbox("Classe do medidor", classes, index=idx_classe)
    with cfg6:
        st.session_state.verificacao_led_seg = st.number_input("Verificação detector (s)", min_value=1, max_value=30, value=int(st.session_state.verificacao_led_seg), step=1)
    with cfg7:
        if st.button("Abrir modo captura", use_container_width=True):
            st.session_state.modo_tela = "captura"
            st.rerun()
    with cfg8:
        st.empty()

    bar_text = (
        f"Tipo: {st.session_state.tipo_deteccao} | "
        f"Classe: {st.session_state.classe_medidor} | "
        f"Kh/Kd: {st.session_state.constante_kh:.3f} | "
        f"Tempo: {st.session_state.tempo_ensaio_s}s | "
        f"Meta: {st.session_state.meta_pulsos}"
    )
    st.markdown(f'<div class="ensaio-toolbar"><b>Configuração do ensaio:</b> {bar_text}</div>', unsafe_allow_html=True)

# =============================================================================
# WEBRTC
# =============================================================================
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

# =============================================================================
# TELA CAPTURA
# =============================================================================
if st.session_state.modo_tela == "captura":
    if st.session_state.ensaio_inicio_ts is None:
        st.session_state.ensaio_inicio_ts = time.time()

    tempo_decorrido = int(time.time() - st.session_state.ensaio_inicio_ts) if st.session_state.ensaio_iniciado else 0
    tempo_restante = max(int(st.session_state.tempo_ensaio_s) - tempo_decorrido, 0)

    snap = None
    if ctx and ctx.video_processor:
        snap = ctx.video_processor.get_snapshot()

    top_left, top_right = st.columns([10, 1])
    with top_right:
        if st.button("⚙️", help="Mostrar/ocultar calibração e diagnóstico"):
            st.session_state.mostrar_painel_debug = not st.session_state.mostrar_painel_debug
            st.rerun()

    if st.session_state.tipo_deteccao == "Manual":
        pulsos = int(st.session_state.pulsos_manuais)
        with top_left:
            st.markdown(
                f'<div class="top-strip">MANUAL | Pulsos: {pulsos} | Tempo restante: {tempo_restante}s</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <div class="compact-box" style="text-align:center;">
                <h2 style="margin:0;">Pulsos atuais</h2>
                <h1 style="margin:0;">{pulsos}</h1>
                <p style="margin:0;">Tempo restante: {tempo_restante}s</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info("Modo manual ativo. Use + e - para contagem.")

        st.markdown('<div class="action-row">', unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            if st.button("Iniciar", use_container_width=True):
                st.session_state.ensaio_iniciado = True
                if st.session_state.ensaio_inicio_ts is None:
                    st.session_state.ensaio_inicio_ts = time.time()
                st.rerun()
        with a2:
            if st.button("Finalizar", use_container_width=True):
                st.session_state.modo_tela = "config_ensaio"
                st.session_state.ensaio_inicio_ts = None
                st.session_state.ensaio_iniciado = False
                st.rerun()
        with a3:
            if st.button("+", use_container_width=True):
                st.session_state.pulsos_manuais += 1
                st.rerun()
        with a4:
            if st.button("-", use_container_width=True):
                if st.session_state.pulsos_manuais > 0:
                    st.session_state.pulsos_manuais -= 1
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        status = snap["status"] if snap and snap["status"] else "NÃO ANALISADO"
        score = f'{snap["red_score_smooth"]:.1f}' if snap and snap["red_score_smooth"] is not None else "-"
        pulsos = snap["pulse_count"] if snap else 0

        with top_left:
            st.markdown(
                f'<div class="top-strip">{st.session_state.tipo_deteccao} | {status} | Score: {score} | Pulsos: {pulsos} | Tempo restante: {tempo_restante}s</div>',
                unsafe_allow_html=True
            )

        st.caption("Área principal de captura")
        if not (ctx and ctx.state.playing):
            st.info("Clique em START para abrir a câmera ao vivo.")

        if st.session_state.mostrar_painel_debug:
            st.markdown('<div class="compact-box"><b>Calibrar / verificar detector</b></div>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("Salvar OFF", use_container_width=True):
                    if ctx and ctx.video_processor:
                        s = ctx.video_processor.get_snapshot()
                        if s["red_score_smooth"] is not None:
                            st.session_state.calib_off = float(s["red_score_smooth"])
                            st.session_state.saved_message = f'OFF salvo: {s["red_score_smooth"]:.2f}'
                            st.rerun()
            with c2:
                if st.button("Salvar ON", use_container_width=True):
                    if ctx and ctx.video_processor:
                        s = ctx.video_processor.get_snapshot()
                        if s["red_score_smooth"] is not None:
                            st.session_state.calib_on = float(s["red_score_smooth"])
                            st.session_state.saved_message = f'ON salvo: {s["red_score_smooth"]:.2f}'
                            st.rerun()
            with c3:
                if st.button("Verificar detector", use_container_width=True):
                    if ctx and ctx.video_processor:
                        inicio = time.time()
                        coletados = []
                        while (time.time() - inicio) < int(st.session_state.verificacao_led_seg):
                            s = ctx.video_processor.get_snapshot()
                            if s and s["red_score_smooth"] is not None:
                                coletados.append(float(s["red_score_smooth"]))
                            time.sleep(0.2)
                        if coletados:
                            amplitude = max(coletados) - min(coletados)
                            st.session_state.detector_verificado = amplitude > 5
                            if st.session_state.detector_verificado:
                                st.success(f"Detector respondeu. Amplitude: {amplitude:.2f}")
                            else:
                                st.warning(f"Sinal fraco. Amplitude: {amplitude:.2f}")
            with c4:
                if st.button("Limpar calibração", use_container_width=True):
                    st.session_state.calib_off = None
                    st.session_state.calib_on = None
                    st.session_state.saved_message = "Calibração limpa."
                    st.rerun()

            if st.session_state.saved_message:
                st.success(st.session_state.saved_message)

            if snap and snap["red_score_smooth"] is not None:
                st.markdown('<div class="compact-box"><b>Diagnóstico</b></div>', unsafe_allow_html=True)
                df = pd.DataFrame(
                    [
                        {"Métrica": "Status", "Valor": snap["status"]},
                        {"Métrica": "Score", "Valor": round(float(snap["red_score_smooth"]), 2)},
                        {"Métrica": "Confiança (%)", "Valor": round(float(snap["confidence"]), 1)},
                        {"Métrica": "Threshold ON", "Valor": None if snap["threshold_on"] is None else round(float(snap["threshold_on"]), 2)},
                        {"Métrica": "Threshold OFF", "Valor": None if snap["threshold_off"] is None else round(float(snap["threshold_off"]), 2)},
                        {"Métrica": "Qualidade último pulso", "Valor": snap["last_pulse_quality"]},
                        {"Métrica": "Intervalo médio (s)", "Valor": snap["intervalo_medio"]},
                    ]
                )
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown('<div class="action-row">', unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            if st.button("Iniciar", use_container_width=True):
                st.session_state.ensaio_iniciado = True
                if st.session_state.ensaio_inicio_ts is None:
                    st.session_state.ensaio_inicio_ts = time.time()
                st.rerun()
        with a2:
            if st.button("Finalizar", use_container_width=True):
                st.session_state.modo_tela = "config_ensaio"
                st.session_state.ensaio_inicio_ts = None
                st.session_state.ensaio_iniciado = False
                st.rerun()
        with a3:
            if st.button("+", use_container_width=True):
                st.info("Ajuste manual + reservado para integração.")
        with a4:
            if st.button("-", use_container_width=True):
                st.info("Ajuste manual - reservado para integração.")
        st.markdown('</div>', unsafe_allow_html=True)

        if ctx and ctx.state.playing and ctx.video_processor and st.session_state.mostrar_painel_debug:
            logs = ctx.video_processor.get_snapshot().get("pulse_events", [])
            if logs:
                st.markdown("### Eventos de pulso")
                st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)

# =============================================================================
# TELA CONFIG
# =============================================================================
else:
    st.info(
        "A configuração geral fica na barra lateral. "
        "A configuração do ensaio fica nesta tela. "
        "Ao abrir o modo captura, a câmera ocupa a maior parte do celular."
    )

    if ctx and ctx.state.playing and ctx.video_processor:
        s = ctx.video_processor.get_snapshot()
        if s and s["red_score_smooth"] is not None:
            st.markdown("### Prévia rápida do detector")
            df = pd.DataFrame(
                [
                    {"Métrica": "Status", "Valor": s["status"]},
                    {"Métrica": "Score", "Valor": round(float(s["red_score_smooth"]), 2)},
                    {"Métrica": "Pulsos", "Valor": s["pulse_count"]},
                    {"Métrica": "Confiança (%)", "Valor": round(float(s["confidence"]), 1)},
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)