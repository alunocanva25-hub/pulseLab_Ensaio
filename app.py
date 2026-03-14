# pulselab_camera_macro_fase2.py
# -----------------------------------------------------------------------------
# PulseLab Camera Macro - Fase 2
# Agora com a atualização pedida:
# - câmera como área principal
# - macro/zoom digital direto na área da câmera
# - ROI central visível
# - calibração OFF / ON
# - lógica temporal experimental
# - debounce
# - contagem automática experimental de pulsos
#
# OBS:
# Em Streamlit puro ainda dependemos de capturas sucessivas do navegador.
# Isso valida a lógica. Para leitura contínua real, a próxima etapa é
# streamlit-webrtc + OpenCV.
# -----------------------------------------------------------------------------

from __future__ import annotations

import io
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="PulseLab Macro Camera Fase 2", page_icon="🔴", layout="wide")

# =============================================================================
# STATE
# =============================================================================
def init_state():
    defaults = {
        "macro_zoom": 2.0,
        "roi_size": 0.20,
        "calib_off": None,
        "calib_on": None,
        "ultimo_status": "NÃO ANALISADO",
        "ultimo_red_score": None,
        "ultimo_confidence": None,
        "ultima_orientacao": None,
        "auto_refresh": False,
        "refresh_seconds": 1,
        "camera_fallback_mobile": False,
        "detector_ativo": False,
        "pulse_count": 0,
        "detector_state": "IDLE",  # IDLE / OFF / ON
        "last_transition_ts": 0.0,
        "last_pulse_ts": 0.0,
        "debounce_ms": 250,
        "min_on_ms": 80,
        "min_off_ms": 80,
        "threshold_margin": 0.0,
        "timeline": [],
        "frame_index": 0,
        "last_capture_bytes": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =============================================================================
# HELPERS
# =============================================================================
def save_capture_bytes(uploaded_file):
    if uploaded_file is not None:
        st.session_state.last_capture_bytes = uploaded_file.getvalue()

def get_capture():
    if st.session_state.camera_fallback_mobile:
        up = st.file_uploader(
            "Capturar/selecionar imagem",
            type=["png", "jpg", "jpeg", "webp"],
            key="macro_fase2_uploader",
            help="Fallback para celular quando a câmera do navegador não abrir.",
        )
        save_capture_bytes(up)
    else:
        cam = st.camera_input(
            "Área da câmera",
            key="macro_fase2_camera",
            help="Use a câmera traseira, aproxime o LED e mantenha a cena estável.",
        )
        save_capture_bytes(cam)

    if st.session_state.last_capture_bytes:
        return io.BytesIO(st.session_state.last_capture_bytes)
    return None

def get_threshold():
    if st.session_state.calib_off is None or st.session_state.calib_on is None:
        return None
    return ((float(st.session_state.calib_off) + float(st.session_state.calib_on)) / 2.0) + float(st.session_state.threshold_margin)

def reset_detector():
    st.session_state.detector_ativo = False
    st.session_state.pulse_count = 0
    st.session_state.detector_state = "IDLE"
    st.session_state.last_transition_ts = 0.0
    st.session_state.last_pulse_ts = 0.0
    st.session_state.timeline = []
    st.session_state.frame_index = 0

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
    st.session_state.timeline = st.session_state.timeline[:120]

def process_detector(status: str, ts_now: float):
    if not st.session_state.detector_ativo:
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

    # MACRO / ZOOM DIGITAL NA ÁREA DA CÂMERA
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

    st.session_state.ultimo_status = status
    st.session_state.ultimo_red_score = red_score
    st.session_state.ultimo_confidence = confidence
    st.session_state.ultima_orientacao = guidance

    img_macro = Image.fromarray(crop)
    draw = ImageDraw.Draw(img_macro)
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
        {"Métrica": "Calibração OFF", "Valor": st.session_state.calib_off},
        {"Métrica": "Calibração ON", "Valor": st.session_state.calib_on},
        {"Métrica": "Orientação", "Valor": guidance},
    ])

    return {
        "img_macro": img_macro,
        "metrics": metrics,
        "red_score": red_score,
        "status": status,
        "confidence": confidence,
    }

# =============================================================================
# UI
# =============================================================================
st.title("🔴 PulseLab Camera Macro - Fase 2")
st.caption("Macro/zoom na própria área da câmera + ROI + OFF/ON + debounce + pulso automático experimental.")

top1, top2, top3, top4 = st.columns(4)
with top1:
    st.session_state.macro_zoom = st.slider("Zoom digital", 1.0, 4.0, float(st.session_state.macro_zoom), 0.1)
with top2:
    st.session_state.roi_size = st.slider("Tamanho ROI", 0.05, 0.50, float(st.session_state.roi_size), 0.01)
with top3:
    st.session_state.auto_refresh = st.toggle("Atualização automática", value=st.session_state.auto_refresh)
with top4:
    st.session_state.refresh_seconds = st.number_input("Refresh (s)", min_value=1, max_value=10, value=int(st.session_state.refresh_seconds), step=1)

cfg1, cfg2, cfg3, cfg4 = st.columns(4)
with cfg1:
    st.session_state.camera_fallback_mobile = st.toggle("Fallback mobile", value=st.session_state.camera_fallback_mobile)
with cfg2:
    st.session_state.debounce_ms = st.number_input("Debounce (ms)", min_value=50, max_value=2000, value=int(st.session_state.debounce_ms), step=10)
with cfg3:
    st.session_state.min_on_ms = st.number_input("Mín. ON (ms)", min_value=20, max_value=2000, value=int(st.session_state.min_on_ms), step=10)
with cfg4:
    st.session_state.min_off_ms = st.number_input("Mín. OFF (ms)", min_value=20, max_value=2000, value=int(st.session_state.min_off_ms), step=10)

st.session_state.threshold_margin = st.slider("Ajuste fino do threshold", -50.0, 50.0, float(st.session_state.threshold_margin), 0.5)

st.info("Fluxo sugerido: 1) aproxime o LED 2) ajuste zoom 3) ajuste ROI 4) salve OFF 5) salve ON 6) ative detector.")

capture = get_capture()
result = analyze_capture(capture)

left, right = st.columns([1.5, 1])

with left:
    st.markdown("### 📷 Área principal de detecção")
    if result is not None:
        st.image(result["img_macro"], caption="Zoom digital + ROI", use_container_width=True)
    else:
        st.info("Aguardando captura.")

with right:
    st.markdown("### Estado atual")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Status", st.session_state.ultimo_status)
    with k2:
        val = "-" if st.session_state.ultimo_red_score is None else f"{st.session_state.ultimo_red_score:.2f}"
        st.metric("Red Score", val)
    with k3:
        conf = "-" if st.session_state.ultimo_confidence is None else f"{st.session_state.ultimo_confidence:.1f}%"
        st.metric("Confiança", conf)
    with k4:
        st.metric("Pulsos", st.session_state.pulse_count)

    k5, k6 = st.columns(2)
    with k5:
        st.metric("Detector", "ATIVO" if st.session_state.detector_ativo else "PARADO")
    with k6:
        st.metric("Estado", st.session_state.detector_state)

    if st.session_state.ultima_orientacao:
        st.info(st.session_state.ultima_orientacao)

    if result is not None:
        st.dataframe(result["metrics"], use_container_width=True, hide_index=True)

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("Salvar OFF", use_container_width=True):
                st.session_state.calib_off = float(result["red_score"])
                st.success(f"OFF salvo: {round(float(result['red_score']), 2)}")
        with b2:
            if st.button("Salvar ON", use_container_width=True):
                st.session_state.calib_on = float(result["red_score"])
                st.success(f"ON salvo: {round(float(result['red_score']), 2)}")
        with b3:
            if st.button("Iniciar detector", use_container_width=True):
                st.session_state.detector_ativo = True
                st.success("Detector ativado.")
        with b4:
            if st.button("Reset detector", use_container_width=True):
                reset_detector()
                st.success("Detector resetado.")

# =============================================================================
# TEMPORAL PROCESSING
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

st.caption("Quando esta fase ficar estável, o próximo passo é integrar esse detector ao ensaio principal.")

if st.session_state.auto_refresh:
    time.sleep(int(st.session_state.refresh_seconds))
    st.rerun()
