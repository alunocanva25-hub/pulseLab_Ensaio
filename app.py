import streamlit as st
from ensaio.ensaio_page import render_ensaio_page

st.set_page_config(layout="wide")

# ===== STATE =====
if "led_color_mode" not in st.session_state:
    st.session_state.led_color_mode = "VERMELHO"

if "fast_pulse_mode" not in st.session_state:
    st.session_state.fast_pulse_mode = True

# ===== UI =====
st.title("⚡ PulseLab v7 - Detector Inteligente")

with st.sidebar:
    st.header("⚙️ Configuração")

    st.session_state.led_color_mode = st.selectbox(
        "Cor do LED",
        ["VERMELHO", "AMARELO", "BRANCO", "AZUL"]
    )

    st.session_state.fast_pulse_mode = st.toggle(
        "Modo pulso rápido",
        value=True
    )

# ===== PAGE =====
render_ensaio_page()