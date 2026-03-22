from __future__ import annotations

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from detector.processor import DetectorConfig, PulseDetectorProcessor

st.set_page_config(
    page_title="PulseLab – Detector Industrial",
    page_icon="🔴",
    layout="wide",
)

DEFAULTS = {
    "led_color_mode": "VERMELHO",
    "fast_pulse_mode": True,
    "roi_size": 0.20,
    "show_overlay": True,
    "smooth_window": 3,
    "detector_enabled": True,
    "debounce_ms": 120,
    "limiar_on": 18.0,
    "limiar_off": 8.0,
    "auto_calibrate": True,
    "target_lock": True,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🔴 PulseLab – Detector Industrial com IA")

with st.sidebar:
    st.header("Configuração do Detector")

    st.session_state.led_color_mode = st.selectbox(
        "Cor do LED",
        ["VERMELHO", "BRANCO", "AMARELO", "AZUL", "AUTOMÁTICO"],
        index=["VERMELHO", "BRANCO", "AMARELO", "AZUL", "AUTOMÁTICO"].index(
            st.session_state.led_color_mode
        ),
    )

    st.session_state.fast_pulse_mode = st.toggle(
        "Modo pulso rápido",
        value=st.session_state.fast_pulse_mode,
    )

    st.session_state.auto_calibrate = st.toggle(
        "Auto calibração",
        value=st.session_state.auto_calibrate,
    )

    st.session_state.target_lock = st.toggle(
        "Travar alvo",
        value=st.session_state.target_lock,
    )

    st.session_state.limiar_on = st.number_input(
        "Limiar ON",
        min_value=1.0,
        max_value=100.0,
        value=float(st.session_state.limiar_on),
        step=0.5,
    )

    st.session_state.limiar_off = st.number_input(
        "Limiar OFF",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.limiar_off),
        step=0.5,
    )

    st.session_state.debounce_ms = st.number_input(
        "Debounce (ms)",
        min_value=20,
        max_value=5000,
        value=int(st.session_state.debounce_ms),
        step=10,
    )

    st.session_state.smooth_window = st.slider(
        "Suavização (frames)",
        1,
        15,
        int(st.session_state.smooth_window),
        1,
    )

    st.session_state.roi_size = st.slider(
        "Tamanho ROI",
        0.10,
        0.50,
        float(st.session_state.roi_size),
        0.01,
    )

    st.session_state.detector_enabled = st.toggle(
        "Detector ativo",
        value=st.session_state.detector_enabled,
    )

    st.session_state.show_overlay = st.toggle(
        "Mostrar overlay",
        value=st.session_state.show_overlay,
    )

det_cfg = DetectorConfig(
    roi_size=float(st.session_state.roi_size),
    show_overlay=bool(st.session_state.show_overlay),
    smooth_window=int(st.session_state.smooth_window),
    detector_enabled=bool(st.session_state.detector_enabled),
    debounce_ms=int(st.session_state.debounce_ms),
    limiar_on=float(st.session_state.limiar_on),
    limiar_off=float(st.session_state.limiar_off),
    led_color_mode=str(st.session_state.led_color_mode),
    fast_pulse_mode=bool(st.session_state.fast_pulse_mode),
    auto_calibrate=bool(st.session_state.auto_calibrate),
    target_lock=bool(st.session_state.target_lock),
)

st.caption("Clique em START para abrir a câmera e testar o detector.")

ctx = webrtc_streamer(
    key="pulselab-detector-industrial-v3",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: PulseDetectorProcessor(det_cfg),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx and ctx.video_processor:
    snap = ctx.video_processor.get_snapshot()

    st.markdown("---")
    st.subheader("Diagnóstico")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Status", snap["status"])
    c2.metric("Score", snap["score"])
    c3.metric("Pulsos", snap["pulse_count"])
    c4.metric("Cor", snap["color"])
    c5.metric("Estado", snap["estado"])

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Área", snap["area"])
    c7.metric("Brilho", snap["brilho"])
    c8.metric("IA confiança", snap["ai_confidence"])
    c9.metric("Alvo válido", "SIM" if snap["target_valid"] else "NÃO")
    c10.metric("Frequência (Hz)", snap["hz"])

    c11, c12, c13, c14 = st.columns(4)
    c11.metric("Modelo", snap["model_label"])
    c12.metric("Conf. modelo", snap["model_confidence"])
    c13.metric("Limiar ON atual", snap["limiar_on"])
    c14.metric("Limiar OFF atual", snap["limiar_off"])

    st.info(f"Motivo IA: {snap['ai_reason']}")

    st.markdown("### Ensinar a IA")
    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("Salvar amostra ON", use_container_width=True):
            path = ctx.video_processor.save_current_sample("on")
            if path:
                st.success(f"Amostra salva: {path}")
            else:
                st.warning("Nenhuma ROI disponível para salvar.")

    with b2:
        if st.button("Salvar amostra OFF", use_container_width=True):
            path = ctx.video_processor.save_current_sample("off")
            if path:
                st.success(f"Amostra salva: {path}")
            else:
                st.warning("Nenhuma ROI disponível para salvar.")

    with b3:
        if st.button("Salvar RUÍDO", use_container_width=True):
            path = ctx.video_processor.save_current_sample("ruido")
            if path:
                st.success(f"Amostra salva: {path}")
            else:
                st.warning("Nenhuma ROI disponível para salvar.")

    st.markdown("---")
    st.markdown("### Como treinar o modelo")
    st.code("python -m detector.train_model", language="bash")
else:
    st.warning("Aguardando câmera.")
