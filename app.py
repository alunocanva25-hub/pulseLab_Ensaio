# pulselab_vision_test.py
# -----------------------------------------------------------------------------
# PulseLab Vision Test
# Protótipo isolado para validar:
# - câmera ao vivo / captura
# - ROI
# - vermelho
# - estado OFF/ON
# - macro/zoom digital simples
#
# IMPORTANTE
# Streamlit puro não entrega stream contínuo frame a frame como um app de visão
# dedicado. Este protótipo usa captura recorrente do navegador para validar
# a lógica óptica antes de integrar ao ensaio.
#
# Próxima etapa depois deste teste:
# streamlit-webrtc + OpenCV para leitura contínua real.
# -----------------------------------------------------------------------------

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="PulseLab Vision Test", page_icon="🔴", layout="wide")

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
        "refresh_seconds": 2,
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
    """
    Tenta usar st.camera_input por padrão.
    Em mobile, pode usar fallback de upload se ativado.
    """
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
    """
    Zoom digital simples: recorta área central e amplia a relevância.
    """
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

def analyze_frame(file_like):
    if file_like is None:
        return None

    img = Image.open(file_like).convert("RGB")
    arr_full = np.array(img)
    arr = apply_macro_crop(arr_full)

    # reconstruir imagem macro para exibição coerente
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

    if st.session_state.calib_off is not None and st.session_state.calib_on is not None:
        threshold = (st.session_state.calib_off + st.session_state.calib_on) / 2.0
        status = "LED LIGADO" if red_score >= threshold else "LED DESLIGADO"
        distance = abs(red_score - threshold)
        confidence = max(35.0, min(99.0, 45.0 + distance))

        if confidence < 60:
            orientation = "Baixa confiança - aproxime mais ou recalcibre"
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

    metrics = pd.DataFrame(
        [
            {"Métrica": "R médio", "Valor": round(r, 2)},
            {"Métrica": "G médio", "Valor": round(g, 2)},
            {"Métrica": "B médio", "Valor": round(b, 2)},
            {"Métrica": "Brilho médio", "Valor": round(brightness, 2)},
            {"Métrica": "Red Score", "Valor": round(red_score, 2)},
            {"Métrica": "Status óptico", "Valor": status},
            {"Métrica": "Confiança (%)", "Valor": round(confidence, 1)},
            {"Métrica": "Macro digital", "Valor": "Ativo" if st.session_state.macro_mode else "Desligado"},
            {"Métrica": "Calibração OFF", "Valor": st.session_state.calib_off},
            {"Métrica": "Calibração ON", "Valor": st.session_state.calib_on},
            {"Métrica": "Orientação", "Valor": orientation},
        ]
    )

    st.session_state.ultimo_status = status
    st.session_state.ultimo_red_score = red_score
    st.session_state.ultimo_confidence = confidence
    st.session_state.ultima_orientacao = orientation

    return {
        "img_macro": img_macro,
        "img_roi_box": draw_roi_box(img_macro, box),
        "metrics": metrics,
        "red_score": red_score,
    }

# =============================================================================
# TOPO
# =============================================================================
st.title("🔴 PulseLab Vision Test")
st.caption("Teste isolado de ROI, vermelho e OFF/ON antes da contagem automática de pulsos.")

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
    if st.button("Limpar calibração", use_container_width=True):
        st.session_state.calib_off = None
        st.session_state.calib_on = None
        st.success("Calibração limpa.")

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

st.info(
    "Fluxo sugerido: 1) abra a câmera 2) aproxime o LED com a traseira 3) ajuste ROI 4) salve OFF 5) salve ON."
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
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Status", st.session_state.ultimo_status)
    with k2:
        val = "-" if st.session_state.ultimo_red_score is None else f"{st.session_state.ultimo_red_score:.2f}"
        st.metric("Red Score", val)
    with k3:
        conf = "-" if st.session_state.ultimo_confidence is None else f"{st.session_state.ultimo_confidence:.1f}%"
        st.metric("Confiança", conf)

    if st.session_state.ultima_orientacao:
        st.info(st.session_state.ultima_orientacao)

    if result is not None:
        st.dataframe(result["metrics"], use_container_width=True, hide_index=True)

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Salvar OFF", use_container_width=True):
                st.session_state.calib_off = float(result["red_score"])
                st.success(f"OFF salvo: {round(float(result['red_score']), 2)}")
        with b2:
            if st.button("Salvar ON", use_container_width=True):
                st.session_state.calib_on = float(result["red_score"])
                st.success(f"ON salvo: {round(float(result['red_score']), 2)}")

st.markdown("### Próximo passo")
st.caption(
    "Quando ROI + vermelho + OFF/ON estiverem estáveis, a próxima fase é adicionar estado temporal "
    "e contagem automática de pulsos."
)

if st.session_state.auto_refresh:
    time.sleep(int(st.session_state.refresh_seconds))
    st.rerun()
