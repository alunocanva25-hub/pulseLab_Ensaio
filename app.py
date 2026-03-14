
# pulselab_camera_macro_prototype.py
# ------------------------------------------------------------
# PulseLab Camera Macro Prototype
# Objetivo:
# - câmera como área principal
# - zoom/macro digital na própria área
# - ROI visível
# - análise vermelho OFF/ON
# ------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="PulseLab Macro Camera Test", layout="wide")

# -------------------------------
# STATE
# -------------------------------
if "macro_zoom" not in st.session_state:
    st.session_state.macro_zoom = 2.0

if "roi_size" not in st.session_state:
    st.session_state.roi_size = 0.2

if "calib_off" not in st.session_state:
    st.session_state.calib_off = None

if "calib_on" not in st.session_state:
    st.session_state.calib_on = None

# -------------------------------
# UI CONFIG
# -------------------------------
st.title("PulseLab - Camera Macro Detection Prototype")

col1, col2 = st.columns(2)

with col1:
    st.session_state.macro_zoom = st.slider(
        "Zoom digital (macro)",
        min_value=1.0,
        max_value=4.0,
        value=st.session_state.macro_zoom,
        step=0.1
    )

with col2:
    st.session_state.roi_size = st.slider(
        "Tamanho ROI",
        min_value=0.05,
        max_value=0.5,
        value=st.session_state.roi_size,
        step=0.01
    )

st.info("Aproxime a câmera do LED e mantenha o LED dentro da caixa verde.")

# -------------------------------
# CAMERA INPUT
# -------------------------------
camera = st.camera_input("Câmera")

if camera is not None:

    img = Image.open(camera).convert("RGB")
    arr = np.array(img)

    h, w = arr.shape[:2]

    # -------------------------------
    # MACRO ZOOM DIGITAL
    # -------------------------------
    zoom = st.session_state.macro_zoom

    crop_w = int(w / zoom)
    crop_h = int(h / zoom)

    cx = w // 2
    cy = h // 2

    x1 = cx - crop_w // 2
    y1 = cy - crop_h // 2
    x2 = cx + crop_w // 2
    y2 = cy + crop_h // 2

    crop = arr[y1:y2, x1:x2]

    img_macro = Image.fromarray(crop)

    # -------------------------------
    # ROI
    # -------------------------------
    ch, cw = crop.shape[:2]

    roi_w = int(cw * st.session_state.roi_size)
    roi_h = int(ch * st.session_state.roi_size)

    rx1 = cw // 2 - roi_w // 2
    ry1 = ch // 2 - roi_h // 2
    rx2 = cw // 2 + roi_w // 2
    ry2 = ch // 2 + roi_h // 2

    roi = crop[ry1:ry2, rx1:rx2]

    # -------------------------------
    # DETECÇÃO VERMELHO
    # -------------------------------
    r = np.mean(roi[:, :, 0])
    g = np.mean(roi[:, :, 1])
    b = np.mean(roi[:, :, 2])

    red_score = r - ((g + b) / 2)

    status = "INDEFINIDO"

    if st.session_state.calib_off is not None and st.session_state.calib_on is not None:

        threshold = (st.session_state.calib_off + st.session_state.calib_on) / 2

        if red_score > threshold:
            status = "LED LIGADO"
        else:
            status = "LED DESLIGADO"

    else:

        if red_score > 20:
            status = "LED LIGADO"
        else:
            status = "LED DESLIGADO"

    # -------------------------------
    # DRAW ROI
    # -------------------------------
    draw = ImageDraw.Draw(img_macro)
    draw.rectangle((rx1, ry1, rx2, ry2), outline=(0,255,0), width=4)

    # -------------------------------
    # DISPLAY
    # -------------------------------
    colA, colB = st.columns([2,1])

    with colA:
        st.image(img_macro, caption="Macro + ROI", use_column_width=True)

    with colB:

        st.subheader("Detecção")

        st.metric("Red Score", round(red_score,2))
        st.metric("Status", status)

        st.write("R:", round(r,2))
        st.write("G:", round(g,2))
        st.write("B:", round(b,2))

        c1, c2 = st.columns(2)

        with c1:
            if st.button("Salvar OFF"):
                st.session_state.calib_off = red_score
                st.success("OFF salvo")

        with c2:
            if st.button("Salvar ON"):
                st.session_state.calib_on = red_score
                st.success("ON salvo")

        st.write("OFF:", st.session_state.calib_off)
        st.write("ON:", st.session_state.calib_on)
