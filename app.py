from __future__ import annotations

import base64
import hashlib
import io
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import qrcode
import streamlit as st

from admin.users_admin import render_users_admin
from ensaio.ensaio_page import render_ensaio_page
from historico.historico_page import render_historico_page

st.set_page_config(page_title="PulseLab v6", page_icon="⚡", layout="wide")

APP_DIR = Path(".")
DB_PATH = APP_DIR / "pulselab_v6.db"

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
        """
        INSERT INTO users (username, full_name, password_hash, role, is_active, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (username.strip().lower(), full_name.strip(), sha256_text(password), role, 1 if is_active else 0, ts, ts),
    )
    CONN.commit()

def update_user_status(username: str, is_active: bool):
    CONN.execute("UPDATE users SET is_active=?, updated_at=? WHERE lower(username)=lower(?)",
                 (1 if is_active else 0, now_str(), username))
    CONN.commit()

def update_user_role(username: str, role: str):
    CONN.execute("UPDATE users SET role=?, updated_at=? WHERE lower(username)=lower(?)",
                 (role, now_str(), username))
    CONN.commit()

def update_user_password(username: str, password: str):
    CONN.execute("UPDATE users SET password_hash=?, updated_at=? WHERE lower(username)=lower(?)",
                 (sha256_text(password), now_str(), username))
    CONN.commit()

def list_users_df() -> pd.DataFrame:
    rows = CONN.execute("""
        SELECT id, username, full_name, role, is_active, created_at, updated_at
        FROM users ORDER BY role DESC, username ASC
    """).fetchall()
    if not rows:
        return pd.DataFrame(columns=["id","username","full_name","role","is_active","created_at","updated_at"])
    return pd.DataFrame([dict(r) for r in rows])

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
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    st.image(buf, width=220)
    st.code(url, language=None)

def init_state():
    defaults = {
        "auth_user": None,
        "app_logo_bytes": None,
        "app_logo_name": None,
        "modo_interface": "Mobile",
        "camera_habilitada": True,
        "mostrar_qrcode": False,
        "historico_local": [],
        "classe": "B (1%)",
        "tolerancia": 1.30,
        "captura_modo": "Manual",
        "tempo_automatico": True,
        "alvo_pulsos_auto": 10,
        "tempo_minimo_seg": 5,
        "tempo_maximo_seg": 600,
        "debounce_ms": 250,
        "detector_enabled": True,
        "show_overlay": True,
        "roi_size": 0.20,
        "smooth_window": 5,
        "rodando": False,
        "inicio": None,
        "resultado": None,
        "pulsos": 0,
        "live_last_snapshot": None,
        "show_params_table": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #e6e6e6; border-radius: 14px; padding: 0.5rem 0.7rem;}
.card-soft {background: #fff; border: 1px solid #e9e9e9; border-radius: 16px; padding: 1rem;}
.topbar-wrap {display:flex; align-items:center; gap:12px;}
.logo-title {margin:0;}
</style>
""", unsafe_allow_html=True)

def bootstrap_first_admin():
    st.title("⚡ PulseLab v6")
    st.subheader("Primeiro acesso - criar administrador master")
    with st.form("bootstrap_admin_form"):
        username = st.text_input("Usuário admin")
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
            st.success("Administrador criado com sucesso.")
            st.rerun()
        except sqlite3.IntegrityError:
            st.error("Esse usuário já existe.")
    st.stop()

def login_screen():
    st.title("🔒 PulseLab v6")
    st.subheader("Login interno")
    st.caption("Base modular + detector ao vivo em evolução")
    with st.form("login_form"):
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        submitted = st.form_submit_button("Entrar")
    if submitted:
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

left, mid, right = st.columns([3, 1.3, 1.2])
with left:
    if st.session_state.app_logo_bytes:
        b64 = base64.b64encode(st.session_state.app_logo_bytes).decode("utf-8")
        st.markdown(f'<div class="topbar-wrap"><img src="data:image/png;base64,{b64}" style="height:48px;border-radius:8px;"><h1 class="logo-title">PulseLab v6</h1></div>', unsafe_allow_html=True)
    else:
        st.title("⚡ PulseLab v6")

with mid:
    st.write(f"**Usuário:** {AUTH['full_name']}")
    st.caption(f"{AUTH['username']} • {AUTH['role']}")

with right:
    if st.button("Sair", use_container_width=True):
        log_event(AUTH["username"], "logout", "Logout manual")
        st.session_state.auth_user = None
        st.rerun()

with st.sidebar:
    st.header("⚙️ Configurações")
    st.session_state.modo_interface = st.radio("Modo de interface", ["Desktop", "Mobile"], index=0 if st.session_state.modo_interface == "Desktop" else 1, horizontal=True)
    st.session_state.camera_habilitada = st.toggle("Habilitar câmera", value=st.session_state.camera_habilitada)
    st.session_state.tempo_automatico = st.toggle("Tempo automático", value=st.session_state.tempo_automatico)
    st.session_state.alvo_pulsos_auto = st.number_input("Pulsos alvo auto", min_value=1, max_value=100, value=int(st.session_state.alvo_pulsos_auto), step=1)
    st.session_state.tempo_minimo_seg = st.number_input("Tempo mínimo (s)", min_value=1, max_value=3600, value=int(st.session_state.tempo_minimo_seg), step=1)
    st.session_state.tempo_maximo_seg = st.number_input("Tempo máximo (s)", min_value=5, max_value=7200, value=int(st.session_state.tempo_maximo_seg), step=1)
    st.session_state.debounce_ms = st.number_input("Debounce (ms)", min_value=50, max_value=5000, value=int(st.session_state.debounce_ms), step=10)
    st.markdown("### Detector")
    st.session_state.detector_enabled = st.toggle("Detector ativo", value=st.session_state.detector_enabled)
    st.session_state.show_overlay = st.toggle("Mostrar overlay", value=st.session_state.show_overlay)
    st.session_state.roi_size = st.slider("Tamanho ROI", 0.10, 0.50, float(st.session_state.roi_size), 0.01)
    st.session_state.smooth_window = st.slider("Suavização (frames)", 1, 15, int(st.session_state.smooth_window), 1)
    st.markdown("### QR Code")
    st.session_state.mostrar_qrcode = st.toggle("Mostrar QR Code", value=st.session_state.mostrar_qrcode)
    if st.session_state.mostrar_qrcode:
        render_qrcode(8501)
    st.markdown("### Logo")
    up_logo = st.file_uploader("Adicionar imagem do app", type=["png", "jpg", "jpeg", "webp"])
    if up_logo is not None:
        st.session_state.app_logo_bytes = up_logo.getvalue()
        st.session_state.app_logo_name = up_logo.name
        st.success("Imagem atualizada.")
    if st.session_state.app_logo_bytes:
        st.image(st.session_state.app_logo_bytes, width=140)

menu_options = ["Ensaio", "Histórico"]
if IS_ADMIN:
    menu_options.append("Admin")

selected = st.radio("Menu", menu_options, horizontal=True, label_visibility="collapsed")

if selected == "Ensaio":
    render_ensaio_page(AUTH)
elif selected == "Histórico":
    render_historico_page()
elif selected == "Admin":
    render_users_admin(auth_user=AUTH, get_user_by_username=get_user_by_username, create_user=create_user, update_user_status=update_user_status, update_user_role=update_user_role, update_user_password=update_user_password, list_users_df=list_users_df, log_event=log_event)
    st.markdown("---")
    st.subheader("Auditoria")
    rows = CONN.execute("SELECT ts, actor_username, event_type, details FROM audit_log ORDER BY id DESC LIMIT 200").fetchall()
    if rows:
        st.dataframe(pd.DataFrame([dict(r) for r in rows]), use_container_width=True, hide_index=True)
    else:
        st.info("Sem eventos auditados ainda.")
    st.markdown("---")
    st.subheader("Alterar minha senha")
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
