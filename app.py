# PulseLab - Ensaio Teste v4
# Atualizações:
# - botão QR Code nas configurações
# - campo "Tempo de ensaio" menor e ao lado de "Meta de pulsos"
# - botão para ocultar/exibir tabela de parâmetros
# - botão em configurações para upload de imagem/logo do app
# - pequena atualização de inteligência:
#   * sugestão de robustez do ensaio
#   * alerta de poucos pulsos
#   * sugestão automática mais clara

import base64
import io
import time
import socket
from datetime import datetime
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import qrcode
from PIL import Image

st.set_page_config(page_title="PulseLab Ensaio", layout="wide")

# =========================================================
# ESTADO
# =========================================================
def init_state():
    defaults = {
        "page": "ensaio",
        "pulsos": 0,
        "rodando": False,
        "inicio": None,
        "fim": None,
        "resultado": None,
        "ultimo_relatorio": None,
        "historico": [],
        "camera_habilitada": True,
        "mostrar_qrcode": False,
        "modo_interface": "Mobile",
        "tempo_automatico": True,
        "alvo_pulsos_auto": 10,
        "tempo_minimo_seg": 5,
        "tempo_maximo_seg": 600,
        "debounce_ms": 250,
        "classe": "B (1%)",
        "tolerancia": 1.30,
        "captura_modo": "Manual",
        "calibracao_aberta": False,
        "calib_off": None,
        "calib_on": None,
        "mostrar_tabela_parametros": True,
        "app_logo_bytes": None,
        "app_logo_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

CLASSES = {
    "A (2%)": 4.00,
    "B (1%)": 1.30,
    "C (0.5%)": 0.70,
    "D (0.2%)": 0.30,
}

CONSTANTES = [0.1, 0.2, 0.5, 1.0, 1.25, 1.5, 2.0, 2.4, 3.6, 4.8, 5.4, 6.25, 7.2, 8.0, 9.6, 10.8, 21.6]

# =========================================================
# FUNÇÕES
# =========================================================
def set_page(page):
    st.session_state.page = page
    st.rerun()

def reset_ensaio():
    st.session_state.pulsos = 0
    st.session_state.rodando = False
    st.session_state.inicio = None
    st.session_state.fim = None
    st.session_state.resultado = None
    st.session_state.ultimo_relatorio = None

def iniciar_ensaio():
    st.session_state.pulsos = 0
    st.session_state.rodando = True
    st.session_state.inicio = time.time()
    st.session_state.fim = None
    st.session_state.resultado = None
    st.session_state.ultimo_relatorio = None

def registrar_pulso():
    if st.session_state.rodando:
        st.session_state.pulsos += 1

def remover_pulso():
    if st.session_state.rodando and st.session_state.pulsos > 0:
        st.session_state.pulsos -= 1

def calcular_potencia(tensao, corrente, fp):
    return tensao * corrente * fp

def tempo_por_pulso_seg(potencia, constante):
    if potencia <= 0:
        return 0.0
    return (constante / potencia) * 3600.0

def tempo_sugerido_seg(potencia, constante):
    base = tempo_por_pulso_seg(potencia, constante) * int(st.session_state.alvo_pulsos_auto)
    base = max(base, float(st.session_state.tempo_minimo_seg))
    base = min(base, float(st.session_state.tempo_maximo_seg))
    return round(base, 2)

def energia_teorica_wh(potencia, tempo_seg):
    return potencia * (tempo_seg / 3600.0)

def energia_medida_wh(pulsos, constante):
    return pulsos * constante

def calcular_erro(energia_medida, energia_teorica):
    if energia_teorica <= 0:
        return 0.0
    return ((energia_medida - energia_teorica) / energia_teorica) * 100.0

def classificar_robustez(meta_pulsos, pulsos_realizados, tempo_seg):
    if pulsos_realizados >= max(10, meta_pulsos) and tempo_seg >= 10:
        return "ALTA"
    if pulsos_realizados >= max(5, meta_pulsos // 2) and tempo_seg >= 5:
        return "MÉDIA"
    return "BAIXA"

def finalizar_ensaio(tensao, corrente, fp, constante, tempo_configurado, meta_pulsos):
    st.session_state.rodando = False
    st.session_state.fim = time.time()
    tempo_real = max(st.session_state.fim - st.session_state.inicio, 0.001) if st.session_state.inicio else float(tempo_configurado)
    potencia = calcular_potencia(tensao, corrente, fp)
    e_teorica = energia_teorica_wh(potencia, tempo_real)
    e_medida = energia_medida_wh(st.session_state.pulsos, constante)
    erro = calcular_erro(e_medida, e_teorica)
    tolerancia = float(st.session_state.tolerancia)
    status = "APROVADO" if abs(erro) <= tolerancia else "REPROVADO"
    robustez = classificar_robustez(int(meta_pulsos), int(st.session_state.pulsos), float(tempo_real))

    rel = {
        "datahora": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "classe": st.session_state.classe,
        "captura_modo": st.session_state.captura_modo,
        "tensao": round(tensao, 2),
        "corrente": round(corrente, 2),
        "fp": round(fp, 2),
        "potencia": round(potencia, 2),
        "constante": float(constante),
        "tempo_configurado": round(float(tempo_configurado), 2),
        "tempo_real": round(float(tempo_real), 2),
        "meta_pulsos": int(meta_pulsos),
        "pulsos": int(st.session_state.pulsos),
        "energia_teorica": round(e_teorica, 4),
        "energia_medida": round(e_medida, 4),
        "erro": round(erro, 4),
        "tolerancia": round(tolerancia, 2),
        "status": status,
        "robustez": robustez,
    }
    st.session_state.resultado = rel
    st.session_state.ultimo_relatorio = rel
    st.session_state.historico.insert(0, rel)
    set_page("resultado")

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

def analisar_led(uploaded_file):
    if uploaded_file is None:
        return None, None
    img = Image.open(uploaded_file).convert("RGB")
    arr = np.array(img)
    r = arr[:, :, 0].mean()
    g = arr[:, :, 1].mean()
    b = arr[:, :, 2].mean()
    red_score = r - ((g + b) / 2.0)

    if st.session_state.calib_off is not None and st.session_state.calib_on is not None:
        limiar = (st.session_state.calib_off + st.session_state.calib_on) / 2.0
        status = "LED LIGADO" if red_score >= limiar else "LED DESLIGADO"
    else:
        status = "LED LIGADO" if red_score > 20 else "LED DESLIGADO"

    tabela = pd.DataFrame([
        {"Métrica": "R médio", "Valor": round(r, 2)},
        {"Métrica": "G médio", "Valor": round(g, 2)},
        {"Métrica": "B médio", "Valor": round(b, 2)},
        {"Métrica": "Red Score", "Valor": round(red_score, 2)},
        {"Métrica": "Status óptico", "Valor": status},
        {"Métrica": "Calibração OFF", "Valor": st.session_state.calib_off},
        {"Métrica": "Calibração ON", "Valor": st.session_state.calib_on},
    ])
    return img, tabela

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def render_qrcode(porta=8501):
    ip = get_local_ip()
    url = f"http://{ip}:{porta}"
    qr = qrcode.QRCode(version=2, box_size=8, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    st.image(buffer, width=220)
    st.code(url, language=None)
    st.caption("Abra esse endereço no navegador do celular, desde que celular e PC estejam na mesma rede Wi‑Fi.")

def salvar_logo(uploaded_file):
    if uploaded_file is not None:
        st.session_state.app_logo_bytes = uploaded_file.getvalue()
        st.session_state.app_logo_name = uploaded_file.name

# =========================================================
# TOPO
# =========================================================
top1, top2 = st.columns([4, 1])
with top1:
    if st.session_state.app_logo_bytes:
        b64 = base64.b64encode(st.session_state.app_logo_bytes).decode("utf-8")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;"><img src="data:image/png;base64,{b64}" style="height:48px;border-radius:8px;"><h1 style="margin:0;">PulseLab - Ensaio</h1></div>',
            unsafe_allow_html=True
        )
    else:
        st.title("⚡ PulseLab - Ensaio")
with top2:
    if st.button("⚙️ Configurações", use_container_width=True):
        set_page("config")

# =========================================================
# PÁGINAS
# =========================================================
if st.session_state.page == "ensaio":
    col1, col2, col3 = st.columns(3)
    with col1:
        tensao = st.number_input("Tensão (V)", value=220.0, step=1.0)
    with col2:
        corrente = st.number_input("Corrente (A)", value=10.0, step=0.1)
    with col3:
        fp = st.number_input("Fator de potência", value=1.0, step=0.01, min_value=0.0, max_value=1.0)

    a1, a2, a3 = st.columns(3)
    with a1:
        st.session_state.classe = st.selectbox("Classe do medidor", list(CLASSES.keys()), index=list(CLASSES.keys()).index(st.session_state.classe))
        st.session_state.tolerancia = CLASSES[st.session_state.classe]
    with a2:
        constante = st.selectbox("Constante Kh/Kd (Wh/pulso)", CONSTANTES, index=CONSTANTES.index(3.6))
    with a3:
        st.session_state.captura_modo = st.selectbox("Modo de captura", ["Manual", "LED Vermelho (diagnóstico)", "Tarja Eletromecânico (futuro)"])

    potencia = calcular_potencia(tensao, corrente, fp)
    t_pulso = tempo_por_pulso_seg(potencia, constante)
    t_sug = tempo_sugerido_seg(potencia, constante)

    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        st.session_state.tempo_automatico = st.toggle("Tempo automático", value=st.session_state.tempo_automatico)
    with c2:
        meta_pulsos = st.number_input("Meta de pulsos", min_value=1, value=10, step=1)
    with c3:
        if st.session_state.tempo_automatico:
            tempo_ensaio = st.number_input("Tempo de ensaio (s)", value=float(t_sug), disabled=True)
        else:
            tempo_ensaio = st.number_input("Tempo de ensaio (s)", value=max(float(t_sug), 1.0), min_value=1.0, step=1.0)

    t1, t2 = st.columns([1, 3])
    with t1:
        if st.button("📊 Ocultar/Exibir parâmetros", use_container_width=True):
            st.session_state.mostrar_tabela_parametros = not st.session_state.mostrar_tabela_parametros
            st.rerun()
    with t2:
        st.caption("A tabela ajuda na conferência do ensaio, mas pode ser ocultada para deixar a tela mais limpa.")

    if st.session_state.mostrar_tabela_parametros:
        st.subheader("Tabela de parâmetros automáticos")
        st.dataframe(
            tabela_parametros(potencia, constante, t_pulso, t_sug, meta_pulsos),
            use_container_width=True,
            hide_index=True
        )

    if int(meta_pulsos) < 5:
        st.warning("Meta de pulsos muito baixa. Para um ensaio mais robusto, prefira pelo menos 5 pulsos.")
    elif int(meta_pulsos) >= 10:
        st.success("Meta de pulsos boa para teste operacional mais consistente.")

    if st.session_state.captura_modo == "LED Vermelho (diagnóstico)":
        b1, b2 = st.columns([1, 3])
        with b1:
            if st.button("🎯 Calibração do LED", use_container_width=True):
                st.session_state.calibracao_aberta = not st.session_state.calibracao_aberta
                st.rerun()
        with b2:
            st.info(f"Calibração: {'ativa' if st.session_state.calibracao_aberta else 'oculta'}")

        if st.session_state.calibracao_aberta:
            foto = st.camera_input("Capturar LED") if st.session_state.camera_habilitada else None
            img, met = analisar_led(foto)
            if foto is not None and img is not None:
                st.image(img, caption="Imagem capturada", use_container_width=True)
                st.dataframe(met, use_container_width=True, hide_index=True)
                red_score = float(met.loc[met["Métrica"] == "Red Score", "Valor"].values[0])

                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    if st.button("Salvar OFF", use_container_width=True):
                        st.session_state.calib_off = red_score
                        st.rerun()
                with cc2:
                    if st.button("Salvar ON", use_container_width=True):
                        st.session_state.calib_on = red_score
                        st.rerun()
                with cc3:
                    if st.button("Limpar calibração", use_container_width=True):
                        st.session_state.calib_off = None
                        st.session_state.calib_on = None
                        st.rerun()

    st.subheader("Execução do ensaio")

    e1, e2, e3 = st.columns(3)
    with e1:
        if st.button("▶ Iniciar", use_container_width=True):
            iniciar_ensaio()
            st.rerun()
    with e2:
        if st.button("➕ Pulso", use_container_width=True):
            registrar_pulso()
            st.rerun()
    with e3:
        if st.button("➖ Remover", use_container_width=True):
            remover_pulso()
            st.rerun()

    if st.session_state.rodando:
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

        f1, f2 = st.columns(2)
        with f1:
            if st.button("⏹ Finalizar teste", use_container_width=True):
                finalizar_ensaio(tensao, corrente, fp, constante, tempo_ensaio, meta_pulsos)
        with f2:
            if st.button("🔄 Resetar ensaio", use_container_width=True):
                reset_ensaio()
                st.rerun()

        if progresso < 1.0:
            time.sleep(1)
            st.rerun()
        else:
            st.info("Tempo atingido. Você pode finalizar o teste ou continuar registrando pulsos.")
    else:
        st.info("O ensaio ainda não foi iniciado.")

elif st.session_state.page == "resultado":
    rel = st.session_state.resultado
    if not rel:
        st.warning("Nenhum resultado disponível.")
    else:
        st.subheader("Resultado do ensaio")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Erro (%)", rel["erro"])
        c2.metric("Tolerância", f"± {rel['tolerancia']:.2f}%")
        c3.metric("Pulsos", rel["pulsos"])
        c4.metric("Tempo real (s)", rel["tempo_real"])
        c5.metric("Robustez", rel["robustez"])

        if rel["status"] == "APROVADO":
            st.success("✅ APROVADO")
        else:
            st.error("❌ REPROVADO")

        st.dataframe(pd.DataFrame([
            {"Campo": "Classe", "Valor": rel["classe"]},
            {"Campo": "Modo de captura", "Valor": rel["captura_modo"]},
            {"Campo": "Tensão (V)", "Valor": rel["tensao"]},
            {"Campo": "Corrente (A)", "Valor": rel["corrente"]},
            {"Campo": "FP", "Valor": rel["fp"]},
            {"Campo": "Potência (W)", "Valor": rel["potencia"]},
            {"Campo": "Constante (Wh/pulso)", "Valor": rel["constante"]},
            {"Campo": "Tempo configurado (s)", "Valor": rel["tempo_configurado"]},
            {"Campo": "Tempo real (s)", "Valor": rel["tempo_real"]},
            {"Campo": "Meta pulsos", "Valor": rel["meta_pulsos"]},
            {"Campo": "Pulsos registrados", "Valor": rel["pulsos"]},
            {"Campo": "Energia teórica (Wh)", "Valor": rel["energia_teorica"]},
            {"Campo": "Energia medida (Wh)", "Valor": rel["energia_medida"]},
            {"Campo": "Erro (%)", "Valor": rel["erro"]},
            {"Campo": "Status", "Valor": rel["status"]},
        ]), use_container_width=True, hide_index=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        if st.button("📋 Histórico", use_container_width=True):
            set_page("historico")
    with r2:
        if st.button("🟢 Reiniciar", use_container_width=True):
            reset_ensaio()
            set_page("ensaio")
    with r3:
        if st.button("🔴 Finalizar", use_container_width=True):
            reset_ensaio()
            set_page("ensaio")

elif st.session_state.page == "historico":
    st.subheader("Histórico de ensaios")
    if not st.session_state.historico:
        st.info("Ainda não há ensaios registrados.")
    else:
        for i, item in enumerate(st.session_state.historico):
            titulo = f"{item['datahora']} | {item['status']} | erro {item['erro']}%"
            with st.expander(titulo, expanded=(i == 0)):
                st.dataframe(pd.DataFrame([{"Campo": k, "Valor": v} for k, v in item.items()]), use_container_width=True, hide_index=True)
    if st.button("⬅ Voltar para resultado", use_container_width=True):
        set_page("resultado")

elif st.session_state.page == "config":
    st.subheader("Configurações")

    st.session_state.modo_interface = st.radio(
        "Modo de interface",
        ["Desktop", "Mobile"],
        index=0 if st.session_state.modo_interface == "Desktop" else 1,
        horizontal=True
    )
    st.session_state.camera_habilitada = st.toggle("Habilitar câmera no ensaio", value=st.session_state.camera_habilitada)
    st.session_state.tempo_automatico = st.toggle("Usar tempo automático por padrão", value=st.session_state.tempo_automatico)
    st.session_state.alvo_pulsos_auto = st.number_input("Pulsos alvo do tempo automático", min_value=1, max_value=100, value=int(st.session_state.alvo_pulsos_auto), step=1)
    st.session_state.tempo_minimo_seg = st.number_input("Tempo mínimo do ensaio (s)", min_value=1, max_value=3600, value=int(st.session_state.tempo_minimo_seg), step=1)
    st.session_state.tempo_maximo_seg = st.number_input("Tempo máximo do ensaio (s)", min_value=5, max_value=7200, value=int(st.session_state.tempo_maximo_seg), step=1)
    st.session_state.debounce_ms = st.number_input("Debounce entre pulsos (ms)", min_value=50, max_value=5000, value=int(st.session_state.debounce_ms), step=10)

    st.markdown("### QR Code")
    st.session_state.mostrar_qrcode = st.toggle("Mostrar QR Code de acesso", value=st.session_state.mostrar_qrcode)
    if st.session_state.mostrar_qrcode:
        render_qrcode(8501)

    st.markdown("### Imagem / Logo do app")
    up_logo = st.file_uploader("Adicionar imagem do app", type=["png", "jpg", "jpeg", "webp"])
    if up_logo is not None:
        salvar_logo(up_logo)
        st.success("Imagem do app atualizada.")
    if st.session_state.app_logo_bytes:
        st.image(st.session_state.app_logo_bytes, width=160)
        st.caption(st.session_state.app_logo_name or "Logo carregada")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅ Voltar", use_container_width=True):
            set_page("ensaio")
    with c2:
        if st.button("🧹 Limpar calibração LED", use_container_width=True):
            st.session_state.calib_off = None
            st.session_state.calib_on = None
            st.success("Calibração limpa.")
