from __future__ import annotations
import time
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from detector.processor import DetectorConfig, PulseDetectorProcessor
from ensaio.calculos import calcular_erro, calcular_potencia, energia_medida_wh, energia_teorica_wh
from ensaio.validacao import classificar_robustez, validar_teste

CLASSES = {"A (2%)": 4.00, "B (1%)": 1.30, "C (0.5%)": 0.70, "D (0.2%)": 0.30}
CONSTANTES = [0.100,0.200,0.300,0.400,0.500,0.600,0.625,0.900,0.960,1.000,1.250,1.500,1.800,2.000,2.400,2.800,3.000,3.125,3.600,4.800,5.400,6.250,7.200,8.000,9.600,10.800,21.600]

def now_str():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def reset_ensaio():
    st.session_state.pulsos = 0
    st.session_state.rodando = False
    st.session_state.inicio = None
    st.session_state.resultado = None

def iniciar_ensaio():
    st.session_state.pulsos = 0
    st.session_state.rodando = True
    st.session_state.inicio = time.time()
    st.session_state.resultado = None

def registrar_pulso():
    if st.session_state.rodando:
        st.session_state.pulsos += 1

def remover_pulso():
    if st.session_state.rodando and st.session_state.pulsos > 0:
        st.session_state.pulsos -= 1

def tempo_por_pulso_seg(potencia, constante):
    if potencia <= 0:
        return 0.0
    return (constante / potencia) * 3600.0

def tempo_sugerido_seg(potencia, constante):
    base = tempo_por_pulso_seg(potencia, constante) * int(st.session_state.alvo_pulsos_auto)
    base = max(base, float(st.session_state.tempo_minimo_seg))
    base = min(base, float(st.session_state.tempo_maximo_seg))
    return round(base, 2)

def tabela_parametros(potencia, constante, tempo_pulso, tempo_sug, meta_pulsos):
    robustez_prevista = classificar_robustez(int(meta_pulsos), int(meta_pulsos), float(tempo_sug))
    return pd.DataFrame([
        {"Campo": "Potência calculada (W)", "Valor": round(potencia, 2)},
        {"Campo": "Energia por pulso (Wh)", "Valor": round(constante, 4)},
        {"Campo": "Tempo por pulso (s)", "Valor": round(tempo_pulso, 2)},
        {"Campo": "Tempo sugerido (s)", "Valor": round(tempo_sug, 2)},
        {"Campo": "Tolerância da classe (%)", "Valor": f"± {st.session_state.tolerancia:.2f}"},
        {"Campo": "Robustez prevista", "Valor": robustez_prevista},
        {"Campo": "Debounce (ms)", "Valor": int(st.session_state.debounce_ms)},
    ])

def sync_pulsos_from_detector(ctx):
    if ctx and ctx.video_processor:
        snap = ctx.video_processor.get_snapshot()
        st.session_state.live_last_snapshot = snap
        if st.session_state.rodando and st.session_state.captura_modo == "IA LED ao vivo":
            st.session_state.pulsos = int(snap["pulse_count"])

def render_ensaio_page(auth_user: dict):
    st.subheader("Ensaio")
    col1, col2, col3 = st.columns(3)
    with col1:
        tensao = st.number_input("Tensão (V)", value=220.0, step=1.0)
    with col2:
        corrente = st.number_input("Corrente (A)", value=10.0, step=0.1)
    with col3:
        fp = st.number_input("Fator de potência", value=1.0, step=0.01, min_value=0.0, max_value=1.0)
    a1, a2, a3 = st.columns(3)
    with a1:
        classe_keys = list(CLASSES.keys())
        idx_classe = classe_keys.index(st.session_state.classe)
        st.session_state.classe = st.selectbox("Classe do medidor", classe_keys, index=idx_classe)
        st.session_state.tolerancia = CLASSES[st.session_state.classe]
    with a2:
        constante = st.selectbox("Constante Kh/Kd (Wh/pulso)", CONSTANTES, index=CONSTANTES.index(3.6))
    with a3:
        st.session_state.captura_modo = st.selectbox("Modo de captura", ["Manual", "IA LED ao vivo", "Tarja Eletromecânico (futuro)"])
    potencia = calcular_potencia(tensao, corrente, fp)
    t_pulso = tempo_por_pulso_seg(potencia, constante)
    t_sug = tempo_sugerido_seg(potencia, constante)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.session_state.tempo_automatico = st.toggle("Tempo automático", value=st.session_state.tempo_automatico)
    with c2:
        meta_pulsos = st.number_input("Meta de pulsos", min_value=1, value=10, step=1)
    with c3:
        if st.session_state.tempo_automatico:
            tempo_ensaio = st.number_input("Tempo de ensaio (s)", value=float(t_sug), disabled=True)
        else:
            tempo_ensaio = st.number_input("Tempo de ensaio (s)", value=max(float(t_sug), 1.0), min_value=1.0, step=1.0)
    if not st.session_state.rodando:
        st.dataframe(tabela_parametros(potencia, constante, t_pulso, t_sug, meta_pulsos), use_container_width=True, hide_index=True)
    ctx = None
    if st.session_state.captura_modo == "IA LED ao vivo" and st.session_state.camera_habilitada:
        st.markdown('<div class="card-soft">', unsafe_allow_html=True)
        st.markdown("### 🔴🟡⚪ Detector ao vivo multi-LED")
        det_cfg = DetectorConfig(
            roi_size=float(st.session_state.roi_size),
            show_overlay=bool(st.session_state.show_overlay),
            smooth_window=int(st.session_state.smooth_window),
            detector_enabled=bool(st.session_state.detector_enabled),
            debounce_ms=int(st.session_state.debounce_ms),
            limiar_on=30.0,
            limiar_off=20.0,
        )
        ctx = webrtc_streamer(
            key="pulselab-live-detector",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: PulseDetectorProcessor(det_cfg),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if ctx and ctx.video_processor:
            sync_pulsos_from_detector(ctx)
            snap = st.session_state.live_last_snapshot or ctx.video_processor.get_snapshot()
            st.markdown(f'<div style="background:#111;color:#f7ea1c;border-radius:14px;padding:10px 12px;margin-bottom:10px;font-weight:800;">Status: {snap["status"]} | Score: {snap["score"]} | Pulsos: {snap["pulse_count"]} | Cor: {snap["color"]}</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pulsos detector", snap["pulse_count"])
            m2.metric("Status", snap["status"])
            m3.metric("Score", snap["score"])
            m4.metric("Cor", snap["color"])
        else:
            st.info("Clique em START para abrir a câmera ao vivo.")
        st.markdown("</div>", unsafe_allow_html=True)
    elif st.session_state.captura_modo == "Tarja Eletromecânico (futuro)":
        st.info("Modo Tarja é um placeholder por enquanto.")
    else:
        st.info("Modo Manual ativo.")
    st.subheader("Execução do ensaio")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.session_state.rodando:
            if st.button("⏹ Finalizar", use_container_width=True):
                tempo_real = max(time.time() - st.session_state.inicio, 0.001)
                e_teorica = energia_teorica_wh(potencia, tempo_real)
                e_medida = energia_medida_wh(st.session_state.pulsos, constante)
                erro = calcular_erro(e_medida, e_teorica)
                robustez = classificar_robustez(int(meta_pulsos), int(st.session_state.pulsos), float(tempo_real))
                status = validar_teste(erro, float(st.session_state.tolerancia), robustez, int(st.session_state.pulsos))
                st.session_state.resultado = {"datahora": now_str(), "usuario": auth_user["username"], "classe": st.session_state.classe, "captura_modo": st.session_state.captura_modo, "tensao": round(tensao, 2), "corrente": round(corrente, 2), "fp": round(fp, 2), "potencia": round(potencia, 2), "constante": float(constante), "tempo_configurado": round(float(tempo_ensaio), 2), "tempo_real": round(float(tempo_real), 2), "meta_pulsos": int(meta_pulsos), "pulsos": int(st.session_state.pulsos), "energia_teorica": round(e_teorica, 4), "energia_medida": round(e_medida, 4), "erro": round(erro, 4), "tolerancia": round(float(st.session_state.tolerancia), 2), "status": status, "robustez": robustez}
                st.session_state.historico_local.insert(0, st.session_state.resultado)
                st.session_state.rodando = False
                st.rerun()
        else:
            if st.button("▶ Iniciar", use_container_width=True):
                iniciar_ensaio()
                st.rerun()
    with b2:
        if st.button("+", use_container_width=True):
            if st.session_state.captura_modo == "Manual":
                registrar_pulso()
            st.rerun()
    with b3:
        if st.button("-", use_container_width=True):
            if st.session_state.captura_modo == "Manual":
                remover_pulso()
            st.rerun()
    if st.session_state.rodando:
        if st.session_state.captura_modo == "IA LED ao vivo" and ctx and ctx.video_processor:
            sync_pulsos_from_detector(ctx)
        tempo_decorrido = time.time() - st.session_state.inicio
        energia_teorica = energia_teorica_wh(potencia, tempo_decorrido)
        energia_medida = energia_medida_wh(st.session_state.pulsos, constante)
        erro = calcular_erro(energia_medida, energia_teorica)
        robustez = classificar_robustez(int(meta_pulsos), int(st.session_state.pulsos), float(tempo_decorrido))
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Pulsos", st.session_state.pulsos)
        k2.metric("Tempo (s)", round(tempo_decorrido, 2))
        k3.metric("Energia teórica", round(energia_teorica, 4))
        k4.metric("Erro (%)", round(erro, 4))
        k5.metric("Robustez", robustez)
        progresso = min(tempo_decorrido / max(float(tempo_ensaio), 1.0), 1.0)
        restante = max(float(tempo_ensaio) - tempo_decorrido, 0.0)
        st.progress(progresso)
        st.caption(f"Tempo configurado: {round(float(tempo_ensaio),2)} s | Restante: {round(restante,2)} s")
        if st.button("🔄 Resetar ensaio", use_container_width=True):
            reset_ensaio()
            st.rerun()
        if progresso < 1.0:
            time.sleep(1)
            st.rerun()
        else:
            st.info("Tempo atingido. Você pode finalizar o teste ou continuar registrando pulsos.")
    else:
        if st.session_state.resultado is None:
            st.info("O ensaio ainda não foi iniciado.")
        else:
            rel = st.session_state.resultado
            st.subheader("Resultado do último ensaio")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Erro (%)", rel["erro"])
            c2.metric("Tolerância", f"± {rel['tolerancia']:.2f}%")
            c3.metric("Pulsos", rel["pulsos"])
            c4.metric("Tempo real (s)", rel["tempo_real"])
            c5.metric("Robustez", rel["robustez"])
            if rel["status"] == "APROVADO":
                st.success("✅ APROVADO")
            elif rel["status"] == "APROVADO COM RESSALVA":
                st.warning("⚠️ APROVADO COM RESSALVA")
            elif rel["status"] == "INVALIDO":
                st.warning("📌 INVÁLIDO")
            else:
                st.error("❌ REPROVADO")
