# app.py
from __future__ import annotations

import base64
import hashlib
import io
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import qrcode
import streamlit as st
from PIL import Image

st.set_page_config(page_title="PulseLab v5.1", page_icon="⚡", layout="wide")

DB_PATH = Path("pulselab_v51.db")

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'tecnico',
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        actor_username TEXT,
        event_type TEXT NOT NULL,
        details TEXT
    )
    """)
    conn.commit()
    return conn

CONN = get_conn()

def init_state():
    defaults = {
        "auth_user": None,
        "pulsos": 0,
        "rodando": False,
        "inicio": None,
        "fim": None,
        "resultado": None,
        "historico_local": [],
        "modo_interface": "Mobile",
        "camera_habilitada": True,
        "camera_mobile_fallback": False,
        "mostrar_qrcode": False,
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
        "macro_mode": True,
        "rear_camera_hint": True,
        "ultima_foto_led_bytes": None,
        "ultima_foto_led_nome": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

st.markdown("""
<style>
.block-container {padding-top: 0.9rem; padding-bottom: 2rem;}
.big-live-box {border: 2px solid #d9d9d9; border-radius: 18px; padding: 1rem; background: #ffffff; margin-bottom: 1rem;}
.live-title {font-size: 1.15rem; font-weight: 800; margin-bottom: 0.4rem;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #e6e6e6; border-radius: 14px; padding: 0.55rem 0.7rem;}
</style>
""", unsafe_allow_html=True)

CLASSES = {"A (2%)": 4.00, "B (1%)": 1.30, "C (0.5%)": 0.70, "D (0.2%)": 0.30}
CONSTANTES = [0.1, 0.2, 0.5, 1.0, 1.25, 1.5, 2.0, 2.4, 3.6, 4.8, 5.4, 6.25, 7.2, 8.0, 9.6, 10.8, 21.6]

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def log_event(actor: str, event_type: str, details: str = "") -> None:
    CONN.execute(
        "INSERT INTO audit_log (ts, actor_username, event_type, details) VALUES (?, ?, ?, ?)",
        (now_str(), actor, event_type, details),
    )
    CONN.commit()

def count_users() -> int:
    row = CONN.execute("SELECT COUNT(*) AS c FROM users").fetchone()
    return int(row["c"] or 0)

def get_user_by_username(username: str):
    return CONN.execute("SELECT * FROM users WHERE lower(username)=lower(?)", (username,)).fetchone()

def verify_login(username: str, password: str):
    row = get_user_by_username(username)
    if not row:
        return None
    if not int(row["is_active"]):
        return None
    if row["password_hash"] != sha256_text(password):
        return None
    return row

def create_user(username: str, full_name: str, password: str, role: str = "tecnico", is_active: bool = True):
    ts = now_str()
    CONN.execute(
        "INSERT INTO users (username, full_name, password_hash, role, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (username.strip().lower(), full_name.strip(), sha256_text(password), role, 1 if is_active else 0, ts, ts),
    )
    CONN.commit()

def update_user_status(username: str, is_active: bool):
    CONN.execute("UPDATE users SET is_active=?, updated_at=? WHERE lower(username)=lower(?)", (1 if is_active else 0, now_str(), username))
    CONN.commit()

def update_user_role(username: str, role: str):
    CONN.execute("UPDATE users SET role=?, updated_at=? WHERE lower(username)=lower(?)", (role, now_str(), username))
    CONN.commit()

def update_user_password(username: str, password: str):
    CONN.execute("UPDATE users SET password_hash=?, updated_at=? WHERE lower(username)=lower(?)", (sha256_text(password), now_str(), username))
    CONN.commit()

def list_users_df() -> pd.DataFrame:
    rows = CONN.execute("SELECT id, username, full_name, role, is_active, created_at, updated_at FROM users ORDER BY role DESC, username ASC").fetchall()
    if not rows:
        return pd.DataFrame(columns=["id", "username", "full_name", "role", "is_active", "created_at", "updated_at"])
    return pd.DataFrame([dict(r) for r in rows])

def salvar_logo(uploaded_file):
    if uploaded_file is not None:
        st.session_state.app_logo_bytes = uploaded_file.getvalue()
        st.session_state.app_logo_name = uploaded_file.name

def get_local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def render_qrcode(porta: int = 8501):
    url = f"http://{get_local_ip()}:{porta}"
    qr = qrcode.QRCode(version=2, box_size=8, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    st.image(buffer, width=220)
    st.code(url, language=None)
    st.caption("Abra no celular na mesma rede Wi‑Fi. Em HTTPS a câmera tende a funcionar melhor.")

def bootstrap_first_admin():
    st.title("⚡ PulseLab v5.1")
    st.subheader("Primeiro acesso - criar administrador master")
    st.info("Como ainda não existe usuário, crie agora o administrador principal do sistema.")
    with st.form("bootstrap_admin_form"):
        username = st.text_input("Usuário admin", placeholder="admin")
        full_name = st.text_input("Nome completo")
        password = st.text_input("Senha", type="password")
        password2 = st.text_input("Confirmar senha", type="password")
        submitted = st.form_submit_button("Criar administrador")
    if submitted:
        username = username.strip().lower()
        if not username or not full_name.strip() or not password:
            st.error("Preencha todos os campos.")
            st.stop()
        if password != password2:
            st.error("As senhas não conferem.")
            st.stop()
        try:
            create_user(username, full_name.strip(), password, role="admin", is_active=True)
            log_event(username, "bootstrap_admin", f"Administrador master criado: {username}")
            st.success("Administrador criado com sucesso. Faça login.")
            time.sleep(1)
            st.rerun()
        except sqlite3.IntegrityError:
            st.error("Esse usuário já existe.")
    st.stop()

def login_screen():
    st.title("🔒 PulseLab v5.1")
    st.subheader("Login interno")
    st.caption("Administração interna + modo campo + IA LED experimental")
    with st.form("login_form"):
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        login_submitted = st.form_submit_button("Entrar")
    if login_submitted:
        row = verify_login(username.strip().lower(), password)
        if row is None:
            st.error("Usuário, senha ou status inválido.")
        else:
            st.session_state.auth_user = {"username": row["username"], "full_name": row["full_name"], "role": row["role"]}
            log_event(row["username"], "login", "Login efetuado")
            st.rerun()
    st.stop()

if count_users() == 0:
    bootstrap_first_admin()
if st.session_state.auth_user is None:
    login_screen()

AUTH = st.session_state.auth_user
IS_ADMIN = AUTH["role"] == "admin"

def reset_ensaio():
    st.session_state.pulsos = 0
    st.session_state.rodando = False
    st.session_state.inicio = None
    st.session_state.fim = None
    st.session_state.resultado = None

def iniciar_ensaio():
    st.session_state.pulsos = 0
    st.session_state.rodando = True
    st.session_state.inicio = time.time()
    st.session_state.fim = None
    st.session_state.resultado = None
    log_event(AUTH["username"], "start_test", f"modo={st.session_state.captura_modo}")

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

def salvar_foto_led(uploaded_file):
    if uploaded_file is not None:
        st.session_state.ultima_foto_led_bytes = uploaded_file.getvalue()
        st.session_state.ultima_foto_led_nome = getattr(uploaded_file, "name", "imagem_led")

def obter_foto_led(widget_key: str):
    if not st.session_state.camera_habilitada:
        return None
    if st.session_state.camera_mobile_fallback and st.session_state.modo_interface == "Mobile":
        up = st.file_uploader(
            "Capturar/selecionar imagem do LED",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"{widget_key}_uploader",
            help="Fallback para celular quando a câmera do navegador não abrir."
        )
        salvar_foto_led(up)
        if st.session_state.ultima_foto_led_bytes:
            return io.BytesIO(st.session_state.ultima_foto_led_bytes)
        return None
    cam = st.camera_input(
        "Capturar LED",
        key=widget_key,
        help="Em celular, tente manter a câmera traseira e aproximar em modo macro quando disponível no aparelho."
    )
    salvar_foto_led(cam)
    if st.session_state.ultima_foto_led_bytes:
        return io.BytesIO(st.session_state.ultima_foto_led_bytes)
    return None

def analyze_led_image(file_like):
    if file_like is None:
        return None, None, None
    img = Image.open(file_like).convert("RGB")
    arr = np.array(img)
    if st.session_state.macro_mode:
        h, w = arr.shape[:2]
        x1, x2 = int(w * 0.2), int(w * 0.8)
        y1, y2 = int(h * 0.2), int(h * 0.8)
        arr_crop = arr[y1:y2, x1:x2]
        if arr_crop.size > 0:
            arr = arr_crop
    r = float(arr[:, :, 0].mean())
    g = float(arr[:, :, 1].mean())
    b = float(arr[:, :, 2].mean())
    red_score = r - ((g + b) / 2.0)
    guidance = "Captura boa"
    if st.session_state.calib_off is not None and st.session_state.calib_on is not None:
        threshold = (st.session_state.calib_off + st.session_state.calib_on) / 2.0
        status = "LED LIGADO" if red_score >= threshold else "LED DESLIGADO"
        distance = abs(red_score - threshold)
        confidence = min(99.0, max(40.0, 40.0 + distance))
        if confidence < 60:
            guidance = "Baixa confiança - aproxime a câmera ou recalibre"
        elif confidence < 75:
            guidance = "Captura razoável - tente estabilizar mais"
    else:
        status = "LED LIGADO" if red_score > 20 else "LED DESLIGADO"
        confidence = 55.0 if abs(red_score - 20) < 10 else 78.0
        guidance = "Calibre OFF e ON para melhorar a confiança"
    df = pd.DataFrame([
        {"Métrica": "R médio", "Valor": round(r, 2)},
        {"Métrica": "G médio", "Valor": round(g, 2)},
        {"Métrica": "B médio", "Valor": round(b, 2)},
        {"Métrica": "Red Score", "Valor": round(red_score, 2)},
        {"Métrica": "Status óptico", "Valor": status},
        {"Métrica": "Confiança (%)", "Valor": round(confidence, 1)},
        {"Métrica": "Macro/zoom digital", "Valor": "Ativo" if st.session_state.macro_mode else "Desligado"},
        {"Métrica": "Calibração OFF", "Valor": st.session_state.calib_off},
        {"Métrica": "Calibração ON", "Valor": st.session_state.calib_on},
        {"Métrica": "Orientação", "Valor": guidance},
    ])
    return img, df, red_score

left, mid, right = st.columns([3, 1.5, 1.1])
with left:
    if st.session_state.app_logo_bytes:
        b64 = base64.b64encode(st.session_state.app_logo_bytes).decode("utf-8")
        st.markdown(f'<div style="display:flex;align-items:center;gap:12px;"><img src="data:image/png;base64,{b64}" style="height:48px;border-radius:8px;"><h1 style="margin:0;">PulseLab v5.1</h1></div>', unsafe_allow_html=True)
    else:
        st.title("⚡ PulseLab v5.1")
with mid:
    st.write(f"**Usuário:** {AUTH['full_name']}")
    st.caption(f"{AUTH['username']} • {AUTH['role']}")
with right:
    if st.button("Sair", use_container_width=True):
        log_event(AUTH["username"], "logout", "Logout manual")
        st.session_state.auth_user = None
        st.rerun()

menu_options = ["Ensaio", "Histórico", "Configurações"]
if IS_ADMIN:
    menu_options.append("Admin")
selected = st.radio("Menu", menu_options, horizontal=True, label_visibility="collapsed")

if selected == "Ensaio":
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
        st.session_state.captura_modo = st.selectbox("Modo de captura", ["Manual", "IA LED Vermelho (experimental)", "Tarja Eletromecânico (futuro)"])

    potencia = calcular_potencia(tensao, corrente, fp)
    t_pulso = tempo_por_pulso_seg(potencia, constante)
    t_sug = tempo_sugerido_seg(potencia, constante)

    c1, c2, c3 = st.columns([1, 1, 0.9])
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
        p1, p2 = st.columns([1, 3])
        with p1:
            if st.button("📊 Ocultar/Exibir parâmetros", use_container_width=True):
                st.session_state.mostrar_tabela_parametros = not st.session_state.mostrar_tabela_parametros
                st.rerun()
        with p2:
            st.caption("A tabela ajuda na conferência do ensaio, mas pode ser ocultada para deixar a tela mais limpa.")
        if st.session_state.mostrar_tabela_parametros:
            st.dataframe(tabela_parametros(potencia, constante, t_pulso, t_sug, meta_pulsos), use_container_width=True, hide_index=True)

    if int(meta_pulsos) < 5:
        st.warning("Meta de pulsos muito baixa. Para um ensaio mais robusto, prefira pelo menos 5 pulsos.")
    elif int(meta_pulsos) >= 10:
        st.success("Meta de pulsos boa para teste operacional mais consistente.")

    if st.session_state.captura_modo == "IA LED Vermelho (experimental)":
        if not st.session_state.rodando:
            b1, b2 = st.columns([1, 3])
            with b1:
                if st.button("🎯 Calibração IA LED", use_container_width=True):
                    st.session_state.calibracao_aberta = not st.session_state.calibracao_aberta
                    st.rerun()
            with b2:
                st.info(f"Calibração: {'ativa' if st.session_state.calibracao_aberta else 'oculta'} | traseira padrão: {'sim' if st.session_state.rear_camera_hint else 'não'} | macro/zoom: {'ativo' if st.session_state.macro_mode else 'desligado'}")
            if st.session_state.calibracao_aberta:
                st.caption("Para o LED: use a câmera traseira do celular e aproxime ao máximo, como macro.")
                calib_file = obter_foto_led("camera_led_calib")
                img, metrics_df, red_score = analyze_led_image(calib_file)
                if img is not None:
                    st.image(img, caption="Imagem para análise", use_container_width=True)
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                    ccal1, ccal2, ccal3 = st.columns(3)
                    with ccal1:
                        if st.button("Salvar OFF", use_container_width=True):
                            st.session_state.calib_off = red_score
                            st.success(f"OFF salvo: {round(red_score, 2)}")
                    with ccal2:
                        if st.button("Salvar ON", use_container_width=True):
                            st.session_state.calib_on = red_score
                            st.success(f"ON salvo: {round(red_score, 2)}")
                    with ccal3:
                        if st.button("Limpar calibração", use_container_width=True):
                            st.session_state.calib_off = None
                            st.session_state.calib_on = None
                            st.success("Calibração limpa.")
        else:
            st.markdown('<div class="big-live-box">', unsafe_allow_html=True)
            st.markdown('<div class="live-title">🔴 IA LED - modo campo</div>', unsafe_allow_html=True)
            st.caption("Durante o ensaio, a área do LED fica em destaque. Use a câmera traseira e aproxime o LED o máximo possível.")
            live_file = obter_foto_led("camera_led_live")
            img, metrics_df, _ = analyze_led_image(live_file)
            if img is not None:
                st.image(img, caption="Captura atual do LED", use_container_width=True)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            else:
                st.info("Aguardando captura do LED.")
            st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Execução do ensaio")
    if st.session_state.modo_interface == "Mobile":
        b1, b2, b3 = st.columns(3)
    else:
        b1, b2, b3 = st.columns([1.3, 1, 1])

    with b1:
        if st.session_state.rodando:
            if st.button("⏹ Finalizar", use_container_width=True):
                tempo_real = max(time.time() - st.session_state.inicio, 0.001)
                e_teorica = energia_teorica_wh(potencia, tempo_real)
                e_medida = energia_medida_wh(st.session_state.pulsos, constante)
                erro = calcular_erro(e_medida, e_teorica)
                robustez = classificar_robustez(int(meta_pulsos), int(st.session_state.pulsos), float(tempo_real))
                status = "APROVADO" if abs(erro) <= float(st.session_state.tolerancia) else "REPROVADO"
                st.session_state.resultado = {
                    "datahora": now_str(),
                    "usuario": AUTH["username"],
                    "classe": st.session_state.classe,
                    "captura_modo": st.session_state.captura_modo,
                    "tensao": round(tensao, 2),
                    "corrente": round(corrente, 2),
                    "fp": round(fp, 2),
                    "potencia": round(potencia, 2),
                    "constante": float(constante),
                    "tempo_configurado": round(float(tempo_ensaio), 2),
                    "tempo_real": round(float(tempo_real), 2),
                    "meta_pulsos": int(meta_pulsos),
                    "pulsos": int(st.session_state.pulsos),
                    "energia_teorica": round(e_teorica, 4),
                    "energia_medida": round(e_medida, 4),
                    "erro": round(erro, 4),
                    "tolerancia": round(float(st.session_state.tolerancia), 2),
                    "status": status,
                    "robustez": robustez,
                }
                st.session_state.historico_local.insert(0, st.session_state.resultado)
                log_event(AUTH["username"], "finish_test", f"status={status}; erro={round(erro,4)}")
                st.session_state.rodando = False
                st.rerun()
        else:
            if st.button("▶ Iniciar", use_container_width=True):
                iniciar_ensaio()
                st.rerun()

    with b2:
        if st.button("+", use_container_width=True):
            registrar_pulso()
            st.rerun()
    with b3:
        if st.button("-", use_container_width=True):
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
            else:
                st.error("❌ REPROVADO")

elif selected == "Histórico":
    st.subheader("Histórico local")
    if not st.session_state.historico_local:
        st.info("Ainda não há ensaios registrados.")
    else:
        for i, item in enumerate(st.session_state.historico_local):
            titulo = f"{item['datahora']} | {item['status']} | erro {item['erro']}%"
            with st.expander(titulo, expanded=(i == 0)):
                st.dataframe(pd.DataFrame([{"Campo": k, "Valor": v} for k, v in item.items()]), use_container_width=True, hide_index=True)

elif selected == "Configurações":
    st.subheader("Configurações do app")
    st.session_state.modo_interface = st.radio("Modo de interface", ["Desktop", "Mobile"], index=0 if st.session_state.modo_interface == "Desktop" else 1, horizontal=True)
    st.session_state.camera_habilitada = st.toggle("Habilitar câmera no ensaio", value=st.session_state.camera_habilitada)
    st.session_state.camera_mobile_fallback = st.toggle("Usar fallback de câmera no celular", value=st.session_state.camera_mobile_fallback, help="Quando a câmera do navegador não abrir no celular, usa seletor de imagem.")
    st.session_state.rear_camera_hint = st.toggle("Priorizar câmera traseira (orientação de uso)", value=st.session_state.rear_camera_hint)
    st.session_state.macro_mode = st.toggle("Modo macro/zoom digital LED", value=st.session_state.macro_mode, help="Aplica recorte digital central para aproximar a análise do LED.")
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
        st.image(st.session_state.app_logo_bytes, width=140)
    st.info("Nesta versão, a câmera traseira e o macro são tratados como fluxo orientado de campo. O controle total da lente depende do navegador/dispositivo.")

elif selected == "Admin":
    if not IS_ADMIN:
        st.error("Acesso restrito.")
        st.stop()
    st.subheader("Painel administrativo")
    tab1, tab2, tab3 = st.tabs(["Usuários", "Auditoria", "Minha senha"])
    with tab1:
        st.markdown("### Cadastro interno de usuários")
        with st.form("create_user_form"):
            new_username = st.text_input("Usuário")
            new_full_name = st.text_input("Nome completo")
            new_password = st.text_input("Senha inicial", type="password")
            new_role = st.selectbox("Papel", ["tecnico", "admin"])
            new_active = st.toggle("Ativo", value=True)
            save_user = st.form_submit_button("Criar usuário")
        if save_user:
            try:
                if not new_username.strip() or not new_full_name.strip() or not new_password:
                    st.error("Preencha usuário, nome e senha.")
                else:
                    create_user(new_username.strip().lower(), new_full_name.strip(), new_password, new_role, new_active)
                    log_event(AUTH["username"], "create_user", f"user={new_username.strip().lower()}; role={new_role}; active={new_active}")
                    st.success("Usuário criado com sucesso.")
                    st.rerun()
            except sqlite3.IntegrityError:
                st.error("Esse usuário já existe.")
        st.markdown("### Lista de usuários")
        users_df = list_users_df()
        st.dataframe(users_df, use_container_width=True, hide_index=True)
        st.markdown("### Alterar usuário existente")
        usernames = [str(x) for x in users_df["username"].tolist()] if not users_df.empty else []
        if usernames:
            sel_user = st.selectbox("Selecionar usuário", usernames)
            user_row = get_user_by_username(sel_user)
            colu1, colu2, colu3 = st.columns(3)
            with colu1:
                status_now = bool(user_row["is_active"])
                new_status = st.toggle("Ativo", value=status_now, key="admin_status_toggle")
                if st.button("Salvar status", use_container_width=True):
                    update_user_status(sel_user, new_status)
                    log_event(AUTH["username"], "update_user_status", f"user={sel_user}; active={new_status}")
                    st.success("Status atualizado.")
                    st.rerun()
            with colu2:
                role_now = str(user_row["role"])
                role_idx = 0 if role_now == "tecnico" else 1
                new_user_role = st.selectbox("Papel", ["tecnico", "admin"], index=role_idx, key="admin_role_select")
                if st.button("Salvar papel", use_container_width=True):
                    update_user_role(sel_user, new_user_role)
                    log_event(AUTH["username"], "update_user_role", f"user={sel_user}; role={new_user_role}")
                    st.success("Papel atualizado.")
                    st.rerun()
            with colu3:
                new_pwd = st.text_input("Nova senha", type="password")
                if st.button("Resetar senha", use_container_width=True):
                    if not new_pwd:
                        st.error("Informe a nova senha.")
                    else:
                        update_user_password(sel_user, new_pwd)
                        log_event(AUTH["username"], "reset_password", f"user={sel_user}")
                        st.success("Senha alterada.")
                        st.rerun()
        else:
            st.info("Nenhum usuário cadastrado além do admin inicial.")
    with tab2:
        st.markdown("### Auditoria do sistema")
        rows = CONN.execute("SELECT ts, actor_username, event_type, details FROM audit_log ORDER BY id DESC LIMIT 500").fetchall()
        if rows:
            st.dataframe(pd.DataFrame([dict(r) for r in rows]), use_container_width=True, hide_index=True)
        else:
            st.info("Sem eventos auditados ainda.")
    with tab3:
        st.markdown("### Alterar minha senha")
        with st.form("my_password_form"):
            current_pwd = st.text_input("Senha atual", type="password")
            new_pwd1 = st.text_input("Nova senha", type="password")
            new_pwd2 = st.text_input("Confirmar nova senha", type="password")
            save_my_pwd = st.form_submit_button("Salvar nova senha")
        if save_my_pwd:
            user_row = verify_login(AUTH["username"], current_pwd)
            if user_row is None:
                st.error("Senha atual inválida.")
            elif new_pwd1 != new_pwd2:
                st.error("As novas senhas não conferem.")
            elif not new_pwd1:
                st.error("Informe a nova senha.")
            else:
                update_user_password(AUTH["username"], new_pwd1)
                log_event(AUTH["username"], "change_own_password", "Senha própria alterada")
                st.success("Senha atualizada com sucesso.")
