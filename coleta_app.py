from __future__ import annotations

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from detector.collector_processor import CollectorConfig, PulseCollectorProcessor

st.set_page_config(
    page_title="PulseLab - Coleta de Dataset",
    page_icon="🎯",
    layout="wide",
)

DEFAULTS = {
    "roi_size": 0.20,
    "show_overlay": True,
    "led_color_mode": "VERMELHO",
    "fonte_coleta": "emissor",
    "classe_amostra": "on",
    "session_name": "sessao_1",
    "sequencia_frames": 12,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🎯 PulseLab - Coleta de Dataset da IA")

with st.sidebar:
    st.header("Configuração da coleta")

    st.session_state.led_color_mode = st.selectbox(
        "Cor alvo",
        ["VERMELHO", "BRANCO", "AMARELO", "AZUL", "AUTOMÁTICO"],
        index=["VERMELHO", "BRANCO", "AMARELO", "AZUL", "AUTOMÁTICO"].index(
            st.session_state.led_color_mode
        ),
    )

    st.session_state.roi_size = st.slider(
        "Tamanho ROI",
        0.10,
        0.50,
        float(st.session_state.roi_size),
        0.01,
    )

    st.session_state.sequencia_frames = st.slider(
        "Frames por sequência",
        4,
        24,
        int(st.session_state.sequencia_frames),
        1,
    )

    st.session_state.show_overlay = st.toggle(
        "Mostrar overlay",
        value=st.session_state.show_overlay,
    )

    st.session_state.fonte_coleta = st.selectbox(
        "Fonte",
        ["emissor", "campo_real"],
        index=["emissor", "campo_real"].index(st.session_state.fonte_coleta),
    )

    st.session_state.session_name = st.text_input(
        "Nome da sessão",
        value=st.session_state.session_name,
    )

collector_cfg = CollectorConfig(
    roi_size=float(st.session_state.roi_size),
    show_overlay=bool(st.session_state.show_overlay),
    led_color_mode=str(st.session_state.led_color_mode),
    sequence_size=int(st.session_state.sequencia_frames),
)

st.caption("Clique em START para abrir a câmera e coletar exemplos reais do ROI.")

ctx = webrtc_streamer(
    key="pulselab-coleta-dataset",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: PulseCollectorProcessor(collector_cfg),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx and ctx.video_processor:
    snap = ctx.video_processor.get_snapshot()

    st.markdown("---")
    st.subheader("Diagnóstico da coleta")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", snap["status"])
    c2.metric("Cor sugerida", snap["detected_color"])
    c3.metric("Área", snap["area"])
    c4.metric("Brilho", snap["brightness"])

    c5, c6, c7 = st.columns(3)
    c5.metric("Score", snap["score"])
    c6.metric("Frames na sequência", snap["sequence_len"])
    c7.metric("ROI pronta", "SIM" if snap["roi_ready"] else "NÃO")

    st.info(f"Observação: {snap['reason']}")

    st.markdown("### Salvar amostras")
    st.caption("Use ON/OFF/RUÍDO/ON_RAPIDO conforme o estado real do LED nesse momento.")

    b1, b2, b3, b4, b5 = st.columns(5)

    save_meta = {
        "selected_color": st.session_state.led_color_mode,
        "source": st.session_state.fonte_coleta,
        "session_name": st.session_state.session_name,
    }

    with b1:
        if st.button("Salvar ON", use_container_width=True):
            path = ctx.video_processor.save_current_sample("on", save_meta)
            st.success(f"Amostra salva: {path}") if path else st.warning("Sem ROI pronta.")

    with b2:
        if st.button("Salvar OFF", use_container_width=True):
            path = ctx.video_processor.save_current_sample("off", save_meta)
            st.success(f"Amostra salva: {path}") if path else st.warning("Sem ROI pronta.")

    with b3:
        if st.button("Salvar RUÍDO", use_container_width=True):
            path = ctx.video_processor.save_current_sample("ruido", save_meta)
            st.success(f"Amostra salva: {path}") if path else st.warning("Sem ROI pronta.")

    with b4:
        if st.button("Salvar ON_RAPIDO", use_container_width=True):
            path = ctx.video_processor.save_current_sample("on_rapido", save_meta)
            st.success(f"Amostra salva: {path}") if path else st.warning("Sem ROI pronta.")

    with b5:
        if st.button("Salvar CAMPO_REAL", use_container_width=True):
            path = ctx.video_processor.save_current_sample("campo_real", save_meta)
            st.success(f"Amostra salva: {path}") if path else st.warning("Sem ROI pronta.")

    st.markdown("---")
    st.markdown("### Próximo passo")
    st.code("python -m detector.train_model", language="bash")

else:
    st.warning("Aguardando câmera.")