import streamlit as st

st.set_page_config(layout="wide")

if "led_color_mode" not in st.session_state:
    st.session_state.led_color_mode = "VERMELHO"

if "fast_pulse_mode" not in st.session_state:
    st.session_state.fast_pulse_mode = True

st.title("⚡ PulseLab v7 Integrado")

with st.sidebar:
    st.header("⚙️ Detector")

    st.session_state.led_color_mode = st.selectbox(
        "Cor do LED",
        ["VERMELHO","BRANCO","AMARELO","AZUL","AUTOMÁTICO"]
    )

    st.session_state.fast_pulse_mode = st.toggle(
        "Modo pulso rápido",
        value=st.session_state.fast_pulse_mode
    )

st.success("Sistema pronto para teste")
