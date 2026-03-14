# pulselab_vision_test_fase2.py
# -----------------------------------------------------------------------------
# PulseLab Vision Test - Fase 2
# Valida:
# - ROI
# - vermelho
# - OFF/ON
# - debounce
# - lógica temporal
# - contagem automática experimental de pulsos
#
# OBSERVAÇÃO IMPORTANTE
# Em Streamlit puro, não existe stream contínuo real frame a frame como num app
# de visão dedicado. Esta fase 2 usa capturas sucessivas do navegador para validar
# a lógica temporal e a máquina de estados do pulso.
#
# Para produção/uso forte em campo:
# - streamlit-webrtc
# - OpenCV
# -----------------------------------------------------------------------------

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="PulseLab Vision Test Fase 2", page_icon="🔴", layout="wide")

# =============================================================================
# ESTADO
# =============================================================================
def init_state():
    defaults = {
        "modo_interface": "Mobile",
        "camera_fallback_mobile": False,
        "priorizar_traseira": True,
        "macro_mode": True,
        "roi_x_pct": 50,
        "roi_y_pct": 50,
        "roi_w_pct": 20,
        "roi_h_pct": 20,
        "calib_off": None,
        "calib_on": None,
        "ultimo_status": "NÃO ANALISADO",
        "ultimo_red_score": None,
        "ultimo_confidence": None,
        "ultima_orientacao": None,
        "ultima_imagem_bytes": None,
        "auto_refresh": False,
        "refresh_seconds": 1,
        # fase 2
        "detector_ativo": False,
        "pulse_count": 0,
        "detector_state": "IDLE",   # IDLE / OFF / ON
        "last_transition_ts": 0.0,
        "last_pulse_ts": 0.0,
        "debounce_ms": 250,
        "min_on_ms": 80,
        "min_off_ms": 80,
        "threshold_margin": 0.0,
        "timeline": [],
        "frame_index": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =============================================================================
# HELPERS
# =============================================================================
def save_image_bytes(uploaded_file) -> None:
    if uploaded_file is not None:
        st.session_state.ultima_imagem_bytes = uploaded_file.getvalue()

def get_capture():
    if st.session_state.modo_interface == "Mobile" and st.session_state.camera_fallback_mobile:
        up = st.file_uploader(
            "Capturar/selecionar imagem",
            type=["png", "jpg", "jpeg", "webp"],
            key="vision_mobile_uploader",
            help="Fallback para celular quando a câmera do navegador não abrir corretamente.",
        )
        save_image_bytes(up)
    else:
        cam = st.camera_input(
            "Captura da câmera",
            key="vision_camera_input",
            help="Use a câmera traseira do celular, aproxime o LED e mantenha a cena estável.",
        )
        save_image_bytes(cam)

    if st.session_state.ultima_imagem_bytes:
        return io.BytesIO(st.session_state.ultima_imagem_bytes)
    return None

def crop_roi(arr: np.ndarray):
    h, w = arr.shape[:2]
    cx = int(w * (st.session_state.roi_x_pct / 100.0))
    cy = int(h * (st.session_state.roi_y_pct / 100.0))
    rw = max(8, int(w * (st.session_state.roi_w_pct / 100.0)))
    rh = max(8, int(h * (st.session_state.roi_h_pct / 100.0)))

    x1 = max(0, cx - rw // 2)
    y1 = max(0, cy - rh // 2)
    x2 = min(w, x1 + rw)
    y2 = min(h, y1 + rh)

    roi = arr[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)

def apply_macro_crop(arr: np.ndarray) -> np.ndarray:
    if not st.session_state.macro_mode:
        return arr

    h, w = arr.shape[:2]
    x1 = int(w * 0.2)
    x2 = int(w * 0.8)
    y1 = int(h * 0.2)
    y2 = int(h * 0.8)
    cropped = arr[y1:y2, x1:x2]
    return cropped if cropped.size else arr

def draw_roi_box(img: Image.Image, box):
    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
    return img2

def get_threshold():
    if st.session_state.calib_off is None or st.session_state.calib_on is None:
        return None
    base = (float(st.session_state.calib_off) + float(st.session_state.calib_on)) / 2.0
    return base + float(st.session_state.threshold_margin)

def append_timeline(ts: float, red_score: float, status: str, confidence: float):
    item = {
        "frame": st.session_state.frame_index,
        "ts": round(ts, 3),
        "red_score": round(red_score, 2),
        "status": status,
        "confidence": round(confidence, 1),
        "pulse_count": st.session_state.pulse_count,
        "detector_state": st.session_state.detector_state,
    }
    st.session_state.timeline.insert(0, item)
    st.session_state.timeline = st.session_state.timeline[:100]

def reset_detector():
    st.session_state.detector_ativo = False
    st.session_state.pulse_count = 0
    st.session_state.detector_state = "IDLE"
    st.session_state.last_transition_ts = 0.0
    st.session_state.last_pulse_ts = 0.0
    st.session_state.timeline = []
    st.session_state.frame_index = 0

def analyze_frame(file_like):
    if file_like is None:
        return None

    img = Image.open(file_like).convert("RGB")
    arr_full = np.array(img)
    arr = apply_macro_crop(arr_full)
    img_macro = Image.fromarray(arr)

    roi, box = crop_roi(arr)
    if roi.size == 0:
        return None

    r = float(np.mean(roi[:, :, 0]))
    g = float(np.mean(roi[:, :, 1]))
    b = float(np.mean(roi[:, :, 2]))
    brightness = float(np.mean(roi))
    red_score = r - ((g + b) / 2.0)

    orientation = "Captura boa"
    status = "INDETERMINADO"
    confidence = 50.0
    threshold = get_threshold()

    if threshold is not None:
        status = "LED LIGADO" if red_score >= threshold else "LED DESLIGADO"
        distance = abs(red_score - threshold)
        confidence = max(35.0, min(99.0, 45.0 + distance))
        if confidence < 60:
            orientation = "Baixa confiança - aproxime mais ou recalibre"
        elif confidence < 75:
            orientation = "Confiança média - estabilize a câmera"
        else:
            orientation = "Captura boa"
    else:
        status = "LED LIGADO" if red_score > 20 else "LED DESLIGADO"
        confidence = 78.0 if abs(red_score) > 20 else 55.0
        orientation = "Calibre OFF e ON para melhorar a leitura"

    if brightness < 20:
        orientation = "Imagem muito escura"
    elif brightness > 240:
        orientation = "Imagem muito clara / estourada"

    st.session_state.ultimo_status = status
    st.session_state.ultimo_red_score = red_score
    st.session_state.ultimo_confidence = confidence
    st.session_state.ultima_orientacao = orientation

    metrics = pd.DataFrame(
        [
            {"Métrica": "R médio", "Valor": round(r, 2)},
            {"Métrica": "G médio", "Valor": round(g, 2)},
            {"Métrica": "B médio", "Valor": round(b, 2)},
            {"Métrica": "Brilho médio", "Valor": round(brightness, 2)},
            {"Métrica": "Red Score", "Valor": round(red_score, 2)},
            {"Métrica": "Threshold atual", "Valor": None if threshold is None else round(threshold, 2)},
            {"Métrica": "Status óptico", "Valor": status},
            {"Métrica": "Confiança (%)", "Valor": round(confidence, 1)},
            {"Métrica": "Macro digital", "Valor": "Ativo" if st.session_state.macro_mode else "Desligado"},
            {"Métrica": "Calibração OFF", "Valor": st.session_state.calib_off},
            {"Métrica": "Calibração ON", "Valor": st.session_state.calib_on},
            {"Métrica": "Orientação", "Valor": orientation},
        ]
    )

    return {
        "img_macro": img_macro,
        "img_roi_box": draw_roi_box(img_macro, box),
        "metrics": metrics,
        "red_score": red_score,
        "status": status,
        "confidence": confidence,
    }

def process_detector(status: str, ts_now: float):
    if not st.session_state.detector_ativo:
        return

    current = st.session_state.detector_state
    debounce_s = float(st.session_state.debounce_ms) / 1000.0
    min_on_s = float(st.session_state.min_on_ms) / 1000.0
    min_off_s = float(st.session_state.min_off_ms) / 1000.0

    # inicialização
    if current == "IDLE":
        if status == "LED DESLIGADO":
            st.session_state.detector_state = "OFF"
            st.session_state.last_transition_ts = ts_now
        elif status == "LED LIGADO":
            st.session_state.detector_state = "ON"
            st.session_state.last_transition_ts = ts_now
        return

    # OFF -> ON
    if current == "OFF" and status == "LED LIGADO":
        if (ts_now - st.session_state.last_transition_ts) >= min_off_s:
            st.session_state.detector_state = "ON"
            st.session_state.last_transition_ts = ts_now
        return

    # ON -> OFF => conta pulso
    if current == "ON" and status == "LED DESLIGADO":
        on_duration = ts_now - st.session_state.last_transition_ts
        since_last_pulse = ts_now - st.session_state.last_pulse_ts
        if on_duration >= min_on_s and since_last_pulse >= debounce_s:
            st.session_state.pulse_count += 1
            st.session_state.last_pulse_ts = ts_now
            st.session_state.detector_state = "OFF"
            st.session_state.last_transition_ts = ts_now
        return

# =============================================================================
# TOPO
# =============================================================================
st.title("🔴 PulseLab Vision Test - Fase 2")
st.caption("Validação da lógica temporal: ROI + vermelho + OFF/ON + debounce + pulso automático experimental.")

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
cfg1, cfg2, cfg3 = st.columns(3)
with cfg1:
    st.session_state.modo_interface = st.radio(
        "Modo de interface",
        ["Desktop", "Mobile"],
        index=0 if st.session_state.modo_interface == "Desktop" else 1,
        horizontal=True,
    )
with cfg2:
    st.session_state.camera_fallback_mobile = st.toggle(
        "Fallback mobile",
        value=st.session_state.camera_fallback_mobile,
        help="Usa seletor de arquivo no celular quando a câmera do navegador não abrir.",
    )
with cfg3:
    st.session_state.priorizar_traseira = st.toggle(
        "Priorizar traseira",
        value=st.session_state.priorizar_traseira,
        help="No Streamlit puro isso é orientação de uso; o navegador decide a lente na prática.",
    )

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.session_state.macro_mode = st.toggle("Macro/zoom digital", value=st.session_state.macro_mode)
with c2:
    st.session_state.auto_refresh = st.toggle("Atualização automática", value=st.session_state.auto_refresh)
with c3:
    st.session_state.refresh_seconds = st.number_input(
        "Refresh (s)", min_value=1, max_value=10, value=int(st.session_state.refresh_seconds), step=1
    )
with c4:
    if st.button("Reset detector", use_container_width=True):
        reset_detector()
        st.success("Detector resetado.")

st.markdown("### ROI")
r1, r2, r3, r4 = st.columns(4)
with r1:
    st.session_state.roi_x_pct = st.slider("Centro X (%)", 0, 100, int(st.session_state.roi_x_pct))
with r2:
    st.session_state.roi_y_pct = st.slider("Centro Y (%)", 0, 100, int(st.session_state.roi_y_pct))
with r3:
    st.session_state.roi_w_pct = st.slider("Largura ROI (%)", 5, 80, int(st.session_state.roi_w_pct))
with r4:
    st.session_state.roi_h_pct = st.slider("Altura ROI (%)", 5, 80, int(st.session_state.roi_h_pct))

st.markdown("### Lógica temporal")
t1, t2, t3 = st.columns(3)
with t1:
    st.session_state.debounce_ms = st.number_input("Debounce (ms)", min_value=50, max_value=2000, value=int(st.session_state.debounce_ms), step=10)
with t2:
    st.session_state.min_on_ms = st.number_input("Duração mínima ON (ms)", min_value=20, max_value=2000, value=int(st.session_state.min_on_ms), step=10)
with t3:
    st.session_state.min_off_ms = st.number_input("Duração mínima OFF (ms)", min_value=20, max_value=2000, value=int(st.session_state.min_off_ms), step=10)

st.session_state.threshold_margin = st.slider("Ajuste fino do threshold", min_value=-50.0, max_value=50.0, value=float(st.session_state.threshold_margin), step=0.5)

st.info(
    "Fluxo sugerido: 1) ajuste ROI 2) salve OFF 3) salve ON 4) ative detector 5) use atualização automática para observar pulsos."
)

# =============================================================================
# CAPTURA + ANÁLISE
# =============================================================================
capture = get_capture()
result = analyze_frame(capture)

left, right = st.columns([1.4, 1])

with left:
    st.markdown('<div class="big-live-box">', unsafe_allow_html=True)
    st.markdown('<div class="live-title">📷 Área da câmera / validação visual</div>', unsafe_allow_html=True)
    if result is not None:
        st.image(result["img_roi_box"], caption="ROI sobre a imagem analisada", use_container_width=True)
    else:
        st.info("Aguardando captura.")
    st.markdown('</div>', unsafe_allow_html=True)

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
        st.metric("Estado detector", st.session_state.detector_state)
    with k6:
        st.metric("Detector", "ATIVO" if st.session_state.detector_ativo else "PARADO")

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
                if st.session_state.detector_state == "IDLE":
                    st.session_state.detector_state = "IDLE"
                st.success("Detector ativado.")
        with b4:
            if st.button("Parar detector", use_container_width=True):
                st.session_state.detector_ativo = False
                st.info("Detector parado.")

# =============================================================================
# PROCESSAMENTO TEMPORAL
# =============================================================================
if result is not None:
    ts_now = time.time()
    st.session_state.frame_index += 1
    process_detector(result["status"], ts_now)
    append_timeline(ts_now, float(result["red_score"]), result["status"], float(result["confidence"]))

st.markdown("### Timeline recente")
if st.session_state.timeline:
    df_timeline = pd.DataFrame(st.session_state.timeline)
    st.dataframe(df_timeline, use_container_width=True, hide_index=True)
else:
    st.caption("Sem frames processados ainda.")

st.markdown("### Próximo passo")
st.caption(
    "Quando a contagem automática experimental ficar estável, a próxima fase é integrar o contador no ensaio principal."
)

if st.session_state.auto_refresh:
    time.sleep(int(st.session_state.refresh_seconds))
    st.rerun()
