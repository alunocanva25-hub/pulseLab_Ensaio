# app.py
# PulseLab v5.1 - Administração interna + detector ao vivo multi-LED
# -----------------------------------------------------------------------------
# Mantém:
# - login interno com SQLite
# - bootstrap do primeiro admin
# - admin controla usuários no próprio app
# - cadastro, ativação/bloqueio, troca de senha, papéis
# - histórico e auditoria local
# - módulo de ensaio baseado na versão estável
#
# Evolui:
# - detector ao vivo com streamlit-webrtc
# - detecção multi-LED: vermelho, amarelo e branco
# - configurações jogadas para a sidebar
# - modo Manual / IA LED ao vivo / Tarja (placeholder)
# -----------------------------------------------------------------------------

from __future__ import annotations

import base64
import hashlib
import io
import sqlite3
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import qrcode
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(page_title="PulseLab v5.1", page_icon="⚡", layout="wide")

APP_DIR = Path(".")
DB_PATH = APP_DIR / "pulselab_v5.db"

# =============================================================================
# BANCO
# =============================================================================
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

    conn.execute("""
    CREATE TABLE IF NOT EXISTS app_config (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)

    conn.commit()
    return conn

CONN = get_conn()

# =============================================================================
# HELPERS
# =============================================================================
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
    CONN.execute(
        "UPDATE users SET is_active=?, updated_at=? WHERE lower(username)=lower(?)",
        (1 if is_active else 0, now_str(), username),
    )
    CONN.commit()

def update_user_role(username: str, role: str):
    CONN.execute(
        "UPDATE users SET role=?, updated_at=? WHERE lower(username)=lower(?)",
        (role, now_str(), username),
    )
    CONN.commit()

def update_user_password(username: str, password: str):
    CONN.execute(
        "UPDATE users SET password_hash=?, updated_at=? WHERE lower(username)=lower(?)",
        (sha256_text(password), now_str(), username),
    )
    CONN.commit()

def list_users_df() -> pd.DataFrame:
    rows = CONN.execute("""
        SELECT id, username, full_name, role, is_active, created_at, updated_at
        FROM users
        ORDER BY role DESC, username ASC
    """).fetchall()
    if not rows:
        return pd.DataFrame(columns=["id", "username", "full_name", "role", "is_active", "created_at", "updated_at"])
    return pd.DataFrame([dict(r) for r in rows])

# =============================================================================
# ESTADO
# =============================================================================
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
        "mostrar_qrcode": False,
        "tempo_automatico": True,
        "alvo_pulsos_auto": 10,
        "tempo_minimo_seg": 5,
        "tempo_maximo_seg": 600,
        "debounce_ms": 250,
        "classe": "B (1%)",
        "tolerancia": 1.30,
        "captura_modo": "Manual",
        "mostrar_tabela_parametros": True,
        "app_logo_bytes": None,
        "app_logo_name": None,
        "detector_enabled": True,
        "show_overlay": True,
        "roi_size": 0.20,
        "smooth_window": 5,
        "manual_plusminus_enabled": True,
        "live_last_snapshot": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =============================================================================
# UI HELPERS
# =============================================================================
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid #e6e6e6;
    border-radius: 14px;
    padding: 0.5rem 0.7rem;
}
.big-live-box {
    border: 2px solid #d9d9d9;
    border-radius: 18px;
    padding: 1rem;
    background: #ffffff;
    margin-bottom: 1rem;
}
.live-title {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.card-soft {
    background: #fff;
    border: 1px solid #e9e9e9;
    border-radius: 16px;
    padding: 1rem;
}
.top-strip {
    background: #111;
    color: #f7ea1c;
    border-radius: 14px;
    padding: 10px 12px;
    margin-bottom: 10px;
    font-weight: 800;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

CLASSES = {
    "A (2%)": 4.00,
    "B (1%)": 1.30,
    "C (0.5%)": 0.70,
    "D (0.2%)": 0.30,
}

CONSTANTES = [
    0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.625, 0.900, 0.960,
    1.000, 1.250, 1.500, 1.800, 2.000, 2.400, 2.800, 3.000, 3.125,
    3.600, 4.800, 5.400, 6.250, 7.200, 8.000, 9.600, 10.800, 21.600
]

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

def salvar_logo(uploaded_file):
    if uploaded_file is not None:
        st.session_state.app_logo_bytes = uploaded_file.getvalue()
        st.session_state.app_logo_name = uploaded_file.name

# =============================================================================
# AUTH - BOOTSTRAP
# =============================================================================
def bootstrap_first_admin():
    st.title("⚡ PulseLab v5.1")
    st.subheader("Primeiro acesso - criar administrador master")
    st.info("Como ainda não existe usuário, crie agora o administrador principal do sistema.")

    with st.form("bootstrap_admin_form"):
        username = st.text_input("Usuário admin", placeholder="admin ou seu email")
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
            st.success("Administrador criado com sucesso. Faça login abaixo.")
            time.sleep(1)
            st.rerun()
        except sqlite3.IntegrityError:
            st.error("Esse usuário já existe.")

    st.stop()

def login_screen():
    st.title("🔒 PulseLab v5.1")
    st.subheader("Login interno")
    st.caption("Administração interna + detector ao vivo multi-LED")

    with st.form("login_form"):
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        login_submitted = st.form_submit_button("Entrar")

    if login_submitted:
        row = verify_login(username.strip().lower(), password)
        if row is None:
            st.error("Usuário, senha ou status inválido.")
        else:
            st.session_state.auth_user = {
                "username": row["username"],
                "full_name": row["full_name"],
                "role": row["role"],
            }
            log_event(row["username"], "login", "Login efetuado")
            st.rerun()

    st.stop()

if count_users() == 0:
    bootstrap_first_admin()

if st.session_state.auth_user is None:
    login_screen()

AUTH = st.session_state.auth_user
IS_ADMIN = AUTH["role"] == "admin"

# =============================================================================
# DETECTOR AO VIVO
# =============================================================================
class ContadorPulso:
    def __init__(self, limiar_on=30.0, limiar_off=20.0, debounce_s=0.25, min_on_s=0.08, min_off_s=0.08):
        self.limiar_on = float(limiar_on)
        self.limiar_off = float(limiar_off)
        self.debounce_s = float(debounce_s)
        self.min_on_s = float(min_on_s)
        self.min_off_s = float(min_off_s)

        self.estado = "OFF"
        self.pulsos = 0
        self.ultimo_pulso_ts = 0.0
        self.ultima_transicao_ts = time.time()

    def atualizar_parametros(self, limiar_on, limiar_off, debounce_s, min_on_s, min_off_s):
        self.limiar_on = float(limiar_on)
        self.limiar_off = float(limiar_off)
        self.debounce_s = float(debounce_s)
        self.min_on_s = float(min_on_s)
        self.min_off_s = float(min_off_s)

    def atualizar(self, score: float):
        agora = time.time()
        pulso_confirmado = False

        if self.estado == "OFF":
            tempo_off = agora - self.ultima_transicao_ts
            if score >= self.limiar_on and tempo_off >= self.min_off_s:
                self.estado = "ON"
                self.ultima_transicao_ts = agora

        elif self.estado == "ON":
            if score <= self.limiar_off:
                tempo_on = agora - self.ultima_transicao_ts
                tempo_desde_ultimo = agora - self.ultimo_pulso_ts

                if tempo_on >= self.min_on_s and tempo_desde_ultimo >= self.debounce_s:
                    self.pulsos += 1
                    self.ultimo_pulso_ts = agora
                    pulso_confirmado = True

                self.estado = "OFF"
                self.ultima_transicao_ts = agora

        return self.estado, self.pulsos, pulso_confirmado

@dataclass
class DetectorConfig:
    roi_size: float
    show_overlay: bool
    smooth_window: int
    detector_enabled: bool
    debounce_ms: int

class PulseDetectorProcessor:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.lock = threading.Lock()
        self.buffer = deque(maxlen=max(1, int(config.smooth_window)))
        self.contador = ContadorPulso(
            limiar_on=30.0,
            limiar_off=20.0,
            debounce_s=float(config.debounce_ms) / 1000.0,
            min_on_s=0.08,
            min_off_s=0.08,
        )
        self.score = 0.0
        self.pulsos = 0
        self.status = "AGUARDANDO"
        self.last_color = "-"
        self.last_area = 0.0
        self.last_brilho = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        size = int(min(w, h) * float(self.config.roi_size))
        x = w // 2 - size // 2
        y = h // 2 - size // 2
        roi = img[y:y + size, x:x + size]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # vermelho
        red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        red2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
        red_mask = cv2.add(red1, red2)

        # amarelo
        yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))

        # branco
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))

        color_masks = [
            ("VERMELHO", red_mask),
            ("AMARELO", yellow_mask),
            ("BRANCO", white_mask),
        ]

        merged_mask = red_mask + yellow_mask + white_mask
        score_full = (np.sum(merged_mask) / (merged_mask.size * 255)) * 100.0

        melhor = None
        best_score = 0.0
        best_color = "-"

        for color_name, mask in color_masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 3:
                    continue

                mask_c = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(mask_c, [c], -1, 255, -1)
                brilho = cv2.mean(hsv[:, :, 2], mask=mask_c)[0]

                score = area * brilho
                if score > best_score:
                    best_score = score
                    melhor = (c, area, brilho)
                    best_color = color_name

        if melhor:
            c, area, brilho = melhor
            self.last_color = best_color
            self.last_area = float(area)
            self.last_brilho = float(brilho)

            if area >= 3 and brilho >= 70:
                score = score_full * 1.20
                self.status = f"{best_color}"
            else:
                score = score_full * 0.60
                self.status = f"{best_color} FRACO"
        else:
            score = score_full * 0.75
            self.status = "SEM LED"
            self.last_color = "-"
            self.last_area = 0.0
            self.last_brilho = 0.0

        self.buffer.append(score)
        score = float(np.mean(self.buffer))
        self.score = score

        if self.config.detector_enabled:
            estado, pulsos, pulso = self.contador.atualizar(score)
            self.pulsos = pulsos
            if pulso:
                self.status = "PULSO DETECTADO"
            elif estado == "ON" and self.status == "SEM LED":
                self.status = "LED ON"
            elif estado == "OFF" and self.status == "SEM LED":
                self.status = "LED OFF"

        if self.config.show_overlay:
            cv2.rectangle(img, (x, y), (x + size, y + size), (0, 255, 255), 2)
            cv2.putText(
                img,
                f"{self.status} | P:{self.pulsos}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_snapshot(self):
        with self.lock:
            return {
                "status": self.status,
                "red_score_smooth": round(self.score, 2),
                "pulse_count": int(self.pulsos),
                "color": self.last_color,
                "area": round(self.last_area, 2),
                "brilho": round(self.last_brilho, 2),
            }

# =============================================================================
# ENSAIO HELPERS
# =============================================================================
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

def sync_pulsos_from_detector(ctx):
    if ctx and ctx.video_processor:
        snap = ctx.video_processor.get_snapshot()
        st.session_state.live_last_snapshot = snap
        if st.session_state.rodando and st.session_state.captura_modo == "IA LED ao vivo":
            st.session_state.pulsos = int(snap["pulse_count"])

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

# =============================================================================
# TOPO
# =============================================================================
left, mid, right = st.columns([3, 1.3, 1.2])
with left:
    if st.session_state.app_logo_bytes:
        b64 = base64.b64encode(st.session_state.app_logo_bytes).decode("utf-8")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;"><img src="data:image/png;base64,{b64}" style="height:48px;border-radius:8px;"><h1 style="margin:0;">PulseLab v5.1</h1></div>',
            unsafe_allow_html=True
        )
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

# =============================================================================
# SIDEBAR = CONFIGURAÇÕES GERAIS + ADMIN RÁPIDO
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configurações")

    st.session_state.modo_interface = st.radio(
        "Modo de interface",
        ["Desktop", "Mobile"],
        index=0 if st.session_state.modo_interface == "Desktop" else 1,
        horizontal=True,
    )

    st.session_state.camera_habilitada = st.toggle("Habilitar câmera", value=st.session_state.camera_habilitada)
    st.session_state.tempo_automatico = st.toggle("Tempo automático", value=st.session_state.tempo_automatico)
    st.session_state.alvo_pulsos_auto = st.number_input("Pulsos alvo auto", min_value=1, max_value=100, value=int(st.session_state.alvo_pulsos_auto), step=1)
    st.session_state.tempo_minimo_seg = st.number_input("Tempo mínimo (s)", min_value=1, max_value=3600, value=int(st.session_state.tempo_minimo_seg), step=1)
    st.session_state.tempo_maximo_seg = st.number_input("Tempo máximo (s)", min_value=5, max_value=7200, value=int(st.session_state.tempo_maximo_seg), step=1)
    st.session_state.debounce_ms = st.number_input("Debounce (ms)", min_value=50, max_value=5000, value=int(st.session_state.debounce_ms), step=10)

    st.markdown("### Detector ao vivo")
    st.session_state.detector_enabled = st.toggle("Detector ativo", value=st.session_state.detector_enabled)
    st.session_state.show_overlay = st.toggle("Mostrar overlay", value=st.session_state.show_overlay)
    st.session_state.roi_size = st.slider("Tamanho ROI", 0.10, 0.50, float(st.session_state.roi_size), 0.01)
    st.session_state.smooth_window = st.slider("Suavização (frames)", 1, 15, int(st.session_state.smooth_window), 1)

    st.markdown("### QR Code")
    st.session_state.mostrar_qrcode = st.toggle("Mostrar QR Code", value=st.session_state.mostrar_qrcode)
    if st.session_state.mostrar_qrcode:
        render_qrcode(8501)

    st.markdown("### Imagem / Logo")
    up_logo = st.file_uploader("Adicionar imagem do app", type=["png", "jpg", "jpeg", "webp"])
    if up_logo is not None:
        salvar_logo(up_logo)
        st.success("Imagem do app atualizada.")
    if st.session_state.app_logo_bytes:
        st.image(st.session_state.app_logo_bytes, width=140)

    if IS_ADMIN:
        st.markdown("---")
        st.markdown("### 👤 Usuários (rápido)")
        novo = st.text_input("Novo usuário")
        senha = st.text_input("Senha usuário", type="password")
        nivel = st.selectbox("Nível", ["tecnico", "admin"])
        if st.button("Criar usuário", use_container_width=True):
            try:
                if not novo.strip() or not senha:
                    st.error("Informe usuário e senha.")
                else:
                    create_user(novo.strip().lower(), novo.strip(), senha, nivel, True)
                    log_event(AUTH["username"], "create_user_sidebar", f"user={novo.strip().lower()}; role={nivel}")
                    st.success("Usuário criado.")
            except sqlite3.IntegrityError:
                st.error("Esse usuário já existe.")

menu_options = ["Ensaio", "Histórico"]
if IS_ADMIN:
    menu_options.append("Admin")
selected = st.radio("Menu", menu_options, horizontal=True, label_visibility="collapsed")

# =============================================================================
# ENSAIO
# =============================================================================
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
        st.session_state.captura_modo = st.selectbox(
            "Modo de captura",
            ["Manual", "IA LED ao vivo", "Tarja Eletromecânico (futuro)"],
        )

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
            st.dataframe(
                tabela_parametros(potencia, constante, t_pulso, t_sug, meta_pulsos),
                use_container_width=True,
                hide_index=True,
            )

    # detector ao vivo
    ctx = None
    if st.session_state.captura_modo == "IA LED ao vivo" and st.session_state.camera_habilitada:
        st.markdown('<div class="big-live-box">', unsafe_allow_html=True)
        st.markdown('<div class="live-title">🔴🟡⚪ Detector ao vivo multi-LED</div>', unsafe_allow_html=True)
        st.caption("Detecção ao vivo para LED vermelho, amarelo e branco.")

        det_cfg = DetectorConfig(
            roi_size=float(st.session_state.roi_size),
            show_overlay=bool(st.session_state.show_overlay),
            smooth_window=int(st.session_state.smooth_window),
            detector_enabled=bool(st.session_state.detector_enabled),
            debounce_ms=int(st.session_state.debounce_ms),
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
            st.markdown(
                f'<div class="top-strip">Status: {snap["status"]} | Score: {snap["red_score_smooth"]} | Pulsos: {snap["pulse_count"]} | Cor: {snap["color"]}</div>',
                unsafe_allow_html=True
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pulsos detector", snap["pulse_count"])
            m2.metric("Status", snap["status"])
            m3.metric("Score", snap["red_score_smooth"])
            m4.metric("Cor", snap["color"])
        else:
            st.info("Clique em START para abrir a câmera ao vivo.")

        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state.captura_modo == "Tarja Eletromecânico (futuro)":
        st.info("Modo Tarja é um placeholder por enquanto.")
    else:
        st.info("Modo Manual ativo.")

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
            else:
                st.error("❌ REPROVADO")

# =============================================================================
# HISTÓRICO
# =============================================================================
elif selected == "Histórico":
    st.subheader("Histórico local da sessão / app")
    if not st.session_state.historico_local:
        st.info("Ainda não há ensaios registrados nesta base.")
    else:
        for i, item in enumerate(st.session_state.historico_local):
            titulo = f"{item['datahora']} | {item['status']} | erro {item['erro']}%"
            with st.expander(titulo, expanded=(i == 0)):
                st.dataframe(
                    pd.DataFrame([{"Campo": k, "Valor": v} for k, v in item.items()]),
                    use_container_width=True,
                    hide_index=True
                )

# =============================================================================
# ADMIN
# =============================================================================
elif selected == "Admin":
    st.subheader("Painel administrativo")
    if not IS_ADMIN:
        st.error("Acesso restrito.")
        st.stop()

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
            except sqlite3.IntegrityError:
                st.error("Esse usuário já existe.")

        st.markdown("### Lista de usuários")
        users_df = list_users_df()
        st.dataframe(users_df, use_container_width=True, hide_index=True)

        usernames = [str(x) for x in users_df["username"].tolist()] if not users_df.empty else []
        if usernames:
            st.markdown("### Alterar usuário existente")
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
                new_user_role = st.selectbox("Papel", ["tecnico", "admin"], index=0 if role_now == "tecnico" else 1, key="admin_role_select")
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

    with tab2:
        st.markdown("### Auditoria do sistema")
        rows = CONN.execute(
            "SELECT ts, actor_username, event_type, details FROM audit_log ORDER BY id DESC LIMIT 500"
        ).fetchall()
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