import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import time
from collections import deque

# =========================
# BANCO SIMPLES (SESSION)
# =========================
if "users" not in st.session_state:
    st.session_state.users = {
        "admin": {"senha": "123", "nivel": "admin"}
    }

# =========================
# LOGIN
# =========================
def login():
    st.title("🔐 PulseLab Login")

    user = st.text_input("Usuário")
    senha = st.text_input("Senha", type="password")

    if st.button("Entrar"):
        if user in st.session_state.users:
            if st.session_state.users[user]["senha"] == senha:
                st.session_state.logado = True
                st.session_state.usuario = user
                st.session_state.nivel = st.session_state.users[user]["nivel"]
                st.rerun()
        st.error("Login inválido")

# =========================
# ADMIN
# =========================
def tela_admin():
    st.subheader("👤 Gerenciar usuários")

    novo_user = st.text_input("Novo usuário")
    nova_senha = st.text_input("Senha")
    nivel = st.selectbox("Nível", ["tecnico", "admin"])

    if st.button("Cadastrar"):
        st.session_state.users[novo_user] = {
            "senha": nova_senha,
            "nivel": nivel
        }
        st.success("Usuário criado")

# =========================
# CONTADOR
# =========================
class Contador:
    def __init__(self):
        self.estado = "OFF"
        self.pulsos = 0
        self.last = 0
        self.on = 15
        self.off = 5

    def update(self, score):
        now = time.time()

        if score > self.on:
            estado = "ON"
        elif score < self.off:
            estado = "OFF"
        else:
            estado = self.estado

        pulso = False

        if estado == "ON" and self.estado == "OFF":
            if now - self.last > 0.2:
                self.pulsos += 1
                self.last = now
                pulso = True

        self.estado = estado
        return estado, self.pulsos, pulso

# =========================
# DETECTOR
# =========================
class Processor(VideoProcessorBase):
    def __init__(self):
        self.buffer = deque(maxlen=5)
        self.contador = Contador()
        self.score = 0
        self.pulsos = 0
        self.status = "AGUARDANDO"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        size = int(min(w, h) * 0.25)
        x = w//2 - size//2
        y = h//2 - size//2

        roi = img[y:y+size, x:x+size]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # cores
        red1 = cv2.inRange(hsv,(0,80,80),(10,255,255))
        red2 = cv2.inRange(hsv,(160,80,80),(180,255,255))
        yellow = cv2.inRange(hsv,(15,80,80),(35,255,255))
        white = cv2.inRange(hsv,(0,0,200),(180,40,255))

        mask = red1 + red2 + yellow + white

        score_full = (np.sum(mask)/(mask.size*255))*100

        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        melhor = None
        best = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 3: continue

            mask_c = np.zeros(mask.shape,np.uint8)
            cv2.drawContours(mask_c,[c],-1,255,-1)

            brilho = cv2.mean(hsv[:,:,2],mask=mask_c)[0]
            score = area*brilho

            if score > best:
                best = score
                melhor = (area, brilho)

        if melhor:
            area, brilho = melhor
            if area >= 3 and brilho >= 70:
                score = score_full*1.2
                self.status = "LED"
            else:
                score = score_full*0.6
                self.status = "FRACO"
        else:
            score = score_full*0.75
            self.status = "SEM LED"

        self.buffer.append(score)
        score = np.mean(self.buffer)

        self.score = score

        estado,pulsos,pulso = self.contador.update(score)
        self.pulsos = pulsos

        if pulso:
            self.status = "PULSO"

        cv2.rectangle(img,(x,y),(x+size,y+size),(0,255,255),2)
        cv2.putText(img,f"{estado} P:{pulsos}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        return av.VideoFrame.from_ndarray(img,format="bgr24")

# =========================
# APP
# =========================
if "logado" not in st.session_state:
    st.session_state.logado = False

if not st.session_state.logado:
    login()
    st.stop()

# SIDEBAR
st.sidebar.title("⚙️ Config")

modo = st.sidebar.selectbox("Modo", ["Manual","LED","Tarja"])
tempo = st.sidebar.number_input("Tempo (s)", value=60)
meta = st.sidebar.number_input("Meta pulsos", value=10)

if st.session_state.nivel == "admin":
    tela_admin()

# =========================
# TELA PRINCIPAL
# =========================
st.title("⚡ PulseLab Ensaio")

col1,col2,col3 = st.columns(3)

# CAMERA
ctx = webrtc_streamer(
    key="cam",
    video_processor_factory=Processor,
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_processor:
    col1.metric("Pulsos", ctx.video_processor.pulsos)
    col2.metric("Status", ctx.video_processor.status)
    col3.metric("Score", round(ctx.video_processor.score,2))

# BOTÕES
c1,c2,c3,c4 = st.columns(4)

if c1.button("▶ Iniciar"):
    st.session_state.start = time.time()

if c2.button("⏹ Finalizar"):
    st.session_state.start = None

if c3.button("+"):
    if ctx.video_processor:
        ctx.video_processor.contador.pulsos += 1

if c4.button("-"):
    if ctx.video_processor:
        ctx.video_processor.contador.pulsos -= 1

# TEMPO
if "start" in st.session_state and st.session_state.start:
    restante = tempo - int(time.time()-st.session_state.start)
    st.metric("Tempo restante", max(restante,0))