# pulselab_detection_area_test.py
# -----------------------------------------------------------------------------
# PulseLab - Teste de análise na própria área de detecção
#
# Objetivo:
# - ao clicar em "Calibrar" ou "Iniciar detecção", abrir a câmera naquela área
# - analisar na própria área visual de trabalho
# - evitar criar uma segunda área separada para a análise
#
# IMPORTANTE:
# Em Streamlit puro, ainda usamos camera_input (captura do navegador).
# Então:
# - SIM, dá para abrir/mostrar a câmera ao clicar em um botão
# - NÃO é vídeo contínuo real frame a frame
# - a leitura continua sendo por capturas sucessivas
#
# Próxima etapa real para tempo real:
# - streamlit-webrtc + OpenCV
# -----------------------------------------------------------------------------

from __future__ import annotations

import io
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="PulseLab Detection Area Test", page_icon="🔴", layout="wide")

# =============================================================================
# STATE
# =============================================================================
def init_state():
    defaults = {
        "modo_interface": "Mobile",
        "camera_fallback_mobile": False,
        "macro_zoom": 2.0,
        "roi_size": 0.20,
        "calib_off": None,
        "calib_on": None,
        "threshold_margin": 0.0,
        "flow_mode": "idle",   # idle / calibrate / detect
        "last_capture_bytes": None,
        "last_status": "NÃO ANALISADO",
        "last_red_score": None,
        "last_confidence": None,
        "last_guidance": None,
        "pulse_count": 0,
        "detector_state": "IDLE",
        "detector_active": False,
        "last_transition_ts": 0.0,
        "last_pulse_ts": 0.0,
        "debounce_ms": 250,
        "min_on_ms": 80,
        "min_off_ms": 80,
        "timeline": [],
        "frame_index": 0,
        "auto_refresh": False,
        "refresh_seconds": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =============================================================================
# STYLE
# =============================================================================
st.markdown("""
<style>
.block-container {padding-top: 0.9rem; padding-bottom: 2rem;}
.detect-box {
    border: 2px solid #d8d8d8;
    border-radius: 18px;
    padding: 1rem;
    background: #ffffff;
    margin-bottom: 1rem;
}
.detect-title {
    font-size: 1.15rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}
.small-note {
    color: #666;
    font-size: 0.92rem;
}
div[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid #e6e6e6;
    border-radius: 14px;
    padding: 0.55rem 0.7rem;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPERS
# =============================================================================
def save_capture_bytes(uploaded_file):
    if uploaded_file is not None:
        st.session_state.last_capture_bytes = uploaded_file.getvalue()

def get_capture():
    """
    Mostra a câmera dentro da própria área de detecção.
    """
    if st.session_state.camera_fallback_mobile and st.session_state.modo_interface == "Mobile":
        up = st.file_uploader(
            "Capturar/selecionar imagem na área de detecção",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"uploader_{st.session_state.flow_mode}",
            help="Fallback para celular quando a câmera do navegador não abrir.",
        )
        save_capture_bytes(up)
    else:
        cam = st.camera_input(
            "Área de detecção",
            key=f"camera_{st.session_state.flow_mode}",
            help="Use a câmera traseira, aproxime o LED e mantenha o LED centralizado.",
        )
        save_capture_bytes(cam)

    if st.session_state.last_capture_bytes:
        return io.BytesIO(st.session_state.last_capture_bytes)
    return None

def get_threshold():
    if st.session_state.calib_off is None or st.session_state.calib_on is None:
        return None
    return ((float(st.session_state.calib_off) + float(st.session_state.calib_on)) / 2.0) + float(st.session_state.threshold_margin)

def append_timeline(ts_now: float, red_score: float, status: str, confidence: float):
    item = {
        "frame": st.session_state.frame_index,
        "ts": round(ts_now, 3),
        "red_score": round(red_score, 2),
        "status": status,
        "confidence": round(confidence, 1),
        "pulse_count": st.session_state.pulse_count,
        "detector_state": st.session_state.detector_state,
    }
    st.session_state.timeline.insert(0, item)
    st.session_state.timeline = st.session_state.timeline[:100]

def reset_detector():
    st.session_state.detector_active = False
    st.session_state.pulse_count = 0
    st.session_state.detector_state = "IDLE"
    st.session_state.last_transition_ts = 0.0
    st.session_state.last_pulse_ts = 0.0
    st.session_state.timeline = []
    st.session_state.frame_index = 0

def process_detector(status: str, ts_now: float):
    if not st.session_state.detector_active:
        return

    current = st.session_state.detector_state
    debounce_s = float(st.session_state.debounce_ms) / 1000.0
    min_on_s = float(st.session_state.min_on_ms) / 1000.0
    min_off_s = float(st.session_state.min_off_ms) / 1000.0

    if current == "IDLE":
        if status == "LED DESLIGADO":
            st.session_state.detector_state = "OFF"
            st.session_state.last_transition_ts = ts_now
        elif status == "LED LIGADO":
            st.session_state.detector_state = "ON"
            st.session_state.last_transition_ts = ts_now
        return

    if current == "OFF" and status == "LED LIGADO":
        if (ts_now - st.session_state.last_transition_ts) >= min_off_s:
            st.session_state.detector_state = "ON"
            st.session_state.last_transition_ts = ts_now
        return

    if current == "ON" and status == "LED DESLIGADO":
        on_duration = ts_now - st.session_state.last_transition_ts
        since_last_pulse = ts_now - st.session_state.last_pulse_ts
        if on_duration >= min_on_s and since_last_pulse >= debounce_s:
            st.session_state.pulse_count += 1
            st.session_state.last_pulse_ts = ts_now
            st.session_state.detector_state = "OFF"
            st.session_state.last_transition_ts = ts_now
        return

def analyze_capture(file_like):
    if file_like is None:
        return None

    img = Image.open(file_like).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Macro/zoom digital aplicado à própria área
    zoom = max(1.0, float(st.session_state.macro_zoom))
    crop_w = max(20, int(w / zoom))
    crop_h = max(20, int(h / zoom))

    cx = w // 2
    cy = h // 2

    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    crop = arr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    ch, cw = crop.shape[:2]
    roi_w = max(8, int(cw * float(st.session_state.roi_size)))
    roi_h = max(8, int(ch * float(st.session_state.roi_size)))

    rx1 = max(0, cw // 2 - roi_w // 2)
    ry1 = max(0, ch // 2 - roi_h // 2)
    rx2 = min(cw, rx1 + roi_w)
    ry2 = min(ch, ry1 + roi_h)

    roi = crop[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None

    r = float(np.mean(roi[:, :, 0]))
    g = float(np.mean(roi[:, :, 1]))
    b = float(np.mean(roi[:, :, 2]))
    brightness = float(np.mean(roi))
    red_score = r - ((g + b) / 2.0)

    threshold = get_threshold()
    status = "INDETERMINADO"
    confidence = 50.0
    guidance = "Calibre OFF e ON para melhorar a leitura"

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

    if brightness < 20:
        guidance = "Imagem muito escura"
    elif brightness > 240:
        guidance = "Imagem muito clara / estourada"

    st.session_state.last_status = status
    st.session_state.last_red_score = red_score
    st.session_state.last_confidence = confidence
    st.session_state.last_guidance = guidance

    img_main = Image.fromarray(crop)
    draw = ImageDraw.Draw(img_main)
    draw.rectangle((rx1, ry1, rx2, ry2), outline=(0, 255, 0), width=4)

    metrics = pd.DataFrame([
        {"Métrica": "R médio", "Valor": round(r, 2)},
        {"Métrica": "G médio", "Valor": round(g, 2)},
        {"Métrica": "B médio", "Valor": round(b, 2)},
        {"Métrica": "Brilho médio", "Valor": round(brightness, 2)},
        {"Métrica": "Red Score", "Valor": round(red_score, 2)},
        {"Métrica": "Threshold atual", "Valor": None if threshold is None else round(threshold, 2)},
        {"Métrica": "Status óptico", "Valor": status},
        {"Métrica": "Confiança (%)", "Valor": round(confidence, 1)},
        {"Métrica": "Zoom digital", "Valor": round(float(st.session_state.macro_zoom), 2)},
        {"Métrica": "ROI", "Valor": f"{int(st.session_state.roi_size * 100)}%"},
        {"Métrica": "Orientação", "Valor": guidance},
    ])

    return {
        "img_main": img_main,
        "metrics": metrics,
        "red_score": red_score,
        "status": status,
        "confidence": confidence,
    }

# =============================================================================
# UI
# =============================================================================
st.title("🔴 PulseLab - Análise na própria área")
st.caption(
    "Pergunta respondida: sim, dá para o botão abrir a câmera naquela área. "
    "Mas em Streamlit puro ainda será captura do navegador, não vídeo contínuo real."
)

cfg1, cfg2, cfg3, cfg4 = st.columns(4)
with cfg1:
    st.session_state.modo_interface = st.radio(
        "Modo", ["Desktop", "Mobile"],
        index=0 if st.session_state.modo_interface == "Desktop" else 1,
        horizontal=True,
    )
with cfg2:
    st.session_state.camera_fallback_mobile = st.toggle("Fallback mobile", value=st.session_state.camera_fallback_mobile)
with cfg3:
    st.session_state.auto_refresh = st.toggle("Atualização automática", value=st.session_state.auto_refresh)
with cfg4:
    st.session_state.refresh_seconds = st.number_input("Refresh (s)", min_value=1, max_value=10, value=int(st.session_state.refresh_seconds), step=1)

adj1, adj2, adj3, adj4 = st.columns(4)
with adj1:
    st.session_state.macro_zoom = st.slider("Zoom digital", 1.0, 4.0, float(st.session_state.macro_zoom), 0.1)
with adj2:
    st.session_state.roi_size = st.slider("Tamanho ROI", 0.05, 0.50, float(st.session_state.roi_size), 0.01)
with adj3:
    st.session_state.debounce_ms = st.number_input("Debounce (ms)", min_value=50, max_value=2000, value=int(st.session_state.debounce_ms), step=10)
with adj4:
    st.session_state.threshold_margin = st.slider("Ajuste threshold", -50.0, 50.0, float(st.session_state.threshold_margin), 0.5)

tim1, tim2, tim3 = st.columns(3)
with tim1:
    st.session_state.min_on_ms = st.number_input("Mín. ON (ms)", min_value=20, max_value=2000, value=int(st.session_state.min_on_ms), step=10)
with tim2:
    st.session_state.min_off_ms = st.number_input("Mín. OFF (ms)", min_value=20, max_value=2000, value=int(st.session_state.min_off_ms), step=10)
with tim3:
    if st.button("Limpar calibração / detector", use_container_width=True):
        st.session_state.calib_off = None
        st.session_state.calib_on = None
        reset_detector()
        st.success("Calibração e detector limpos.")

flow1, flow2, flow3 = st.columns(3)
with flow1:
    if st.button("🎯 Calibrar", use_container_width=True):
        st.session_state.flow_mode = "calibrate"
        st.rerun()
with flow2:
    if st.button("▶ Iniciar detecção", use_container_width=True):
        st.session_state.flow_mode = "detect"
        st.session_state.detector_active = True
        st.rerun()
with flow3:
    if st.button("⏹ Fechar câmera", use_container_width=True):
        st.session_state.flow_mode = "idle"
        st.session_state.detector_active = False
        st.rerun()

st.info(
    "Fluxo: 1) clique em Calibrar 2) a câmera abre nesta área 3) salve OFF/ON 4) clique em Iniciar detecção "
    "5) a câmera abre nesta mesma área e a análise ocorre nela."
)

# =============================================================================
# MAIN DETECTION AREA
# =============================================================================
st.markdown('<div class="detect-box">', unsafe_allow_html=True)
st.markdown('<div class="detect-title">📷 Área principal de detecção</div>', unsafe_allow_html=True)

capture = None
result = None

if st.session_state.flow_mode in ("calibrate", "detect"):
    capture = get_capture()
    result = analyze_capture(capture)

    if result is not None:
        # A análise acontece nesta própria área de detecção
        st.image(result["img_main"], caption="Zoom digital + ROI na área principal", use_container_width=True)
    else:
        st.info("Aguardando captura nesta área.")
else:
    st.caption("Câmera fechada. Use Calibrar ou Iniciar detecção para abrir a câmera nesta área.")

st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# SIDE STATUS
# =============================================================================
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.metric("Status", st.session_state.last_status)
with s2:
    rs = "-" if st.session_state.last_red_score is None else f"{st.session_state.last_red_score:.2f}"
    st.metric("Red Score", rs)
with s3:
    cf = "-" if st.session_state.last_confidence is None else f"{st.session_state.last_confidence:.1f}%"
    st.metric("Confiança", cf)
with s4:
    st.metric("Pulsos", st.session_state.pulse_count)

s5, s6 = st.columns(2)
with s5:
    st.metric("Detector", "ATIVO" if st.session_state.detector_active else "PARADO")
with s6:
    st.metric("Estado", st.session_state.detector_state)

if st.session_state.last_guidance:
    st.info(st.session_state.last_guidance)

if result is not None:
    st.dataframe(result["metrics"], use_container_width=True, hide_index=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Salvar OFF", use_container_width=True):
            st.session_state.calib_off = float(result["red_score"])
            st.success(f"OFF salvo: {round(float(result['red_score']), 2)}")
    with b2:
        if st.button("Salvar ON", use_container_width=True):
            st.session_state.calib_on = float(result["red_score"])
            st.success(f"ON salvo: {round(float(result['red_score']), 2)}")
    with b3:
        if st.button("Reset detector", use_container_width=True):
            reset_detector()
            st.success("Detector resetado.")

# =============================================================================
# TEMPORAL PROCESS
# =============================================================================
if result is not None:
    ts_now = time.time()
    st.session_state.frame_index += 1
    process_detector(result["status"], ts_now)
    append_timeline(ts_now, float(result["red_score"]), result["status"], float(result["confidence"]))

st.markdown("### Timeline recente")
if st.session_state.timeline:
    st.dataframe(pd.DataFrame(st.session_state.timeline), use_container_width=True, hide_index=True)
else:
    st.caption("Sem frames processados ainda.")

st.caption(
    "Limite honesto desta versão: a câmera abre/analisa nesta área, mas ainda depende de capturas do navegador. "
    "Para tempo real verdadeiro na mesma área, o próximo passo certo é WebRTC."
)

if st.session_state.auto_refresh and st.session_state.flow_mode in ("calibrate", "detect"):
    time.sleep(int(st.session_state.refresh_seconds))
    st.rerun()
