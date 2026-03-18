
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(page_title="PulseLab - Ensaio", page_icon="🔴", layout="wide")

# =============================================================================
# 🔐 LOGIN
# =============================================================================
if "users" not in st.session_state:
    st.session_state.users = {
        "admin": {"senha": "123", "nivel": "admin"}
    }

if "logado" not in st.session_state:
    st.session_state.logado = False

def tela_login():
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

if not st.session_state.logado:
    tela_login()
    st.stop()

# =============================================================================
# ESTADO
# =============================================================================
CONSTANTES_KDKH = [
    0.100,0.200,0.300,0.400,0.500,0.600,0.625,0.900,0.960,
    1.000,1.250,1.500,1.800,2.000,2.400,2.800,3.000,3.125,
    3.600,4.800,5.400,6.250,7.200,8.000,9.600,10.800,21.600
]

DEFAULTS = {
    "modo_tela":"config_ensaio",
    "tipo_deteccao":"Manual",
    "tempo_ensaio_s":10,
    "ensaio_inicio_ts":None,
    "meta_pulsos":10,
    "constante_kh":3.600,
    "mostrar_painel_debug":False,
    "pulsos_manuais":0,
    "ensaio_iniciado":False,
}
for k,v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================================================================
# CONTADOR
# =============================================================================
class ContadorPulso:
    def __init__(self):
        self.estado="OFF"
        self.pulsos=0
        self.last=0
        self.on=30
        self.off=20

    def atualizar(self,score):
        now=time.time()

        if score>self.on:
            estado="ON"
        elif score<self.off:
            estado="OFF"
        else:
            estado=self.estado

        pulso=False

        if estado=="ON" and self.estado=="OFF":
            if now-self.last>0.2:
                self.pulsos+=1
                self.last=now
                pulso=True

        self.estado=estado
        return estado,self.pulsos,pulso

# =============================================================================
# DETECTOR MULTI-COR
# =============================================================================
@dataclass
class DetectorConfig:
    roi_size: float
    show_overlay: bool

class PulseDetectorProcessor:
    def __init__(self,config):
        self.config=config
        self.buffer=deque(maxlen=5)
        self.contador=ContadorPulso()

        self.score=0
        self.status="AGUARDANDO"
        self.pulse_count=0

    def recv(self,frame):
        img=frame.to_ndarray(format="bgr24")
        h,w=img.shape[:2]

        size=int(min(w,h)*self.config.roi_size)
        x=w//2-size//2
        y=h//2-size//2

        roi=img[y:y+size,x:x+size]
        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

        # 🔥 MULTI LED
        red1=cv2.inRange(hsv,(0,80,80),(10,255,255))
        red2=cv2.inRange(hsv,(160,80,80),(180,255,255))
        yellow=cv2.inRange(hsv,(15,80,80),(35,255,255))
        white=cv2.inRange(hsv,(0,0,200),(180,40,255))

        mask=red1+red2+yellow+white

        score_full=(np.sum(mask)/(mask.size*255))*100

        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        melhor=None
        best=0

        for c in contours:
            area=cv2.contourArea(c)
            if area<3: continue

            mask_c=np.zeros(mask.shape,np.uint8)
            cv2.drawContours(mask_c,[c],-1,255,-1)

            brilho=cv2.mean(hsv[:,:,2],mask=mask_c)[0]
            score=area*brilho

            if score>best:
                best=score
                melhor=(area,brilho)

        # 🔥 LÓGICA HÍBRIDA
        if melhor:
            area,brilho=melhor
            if area>=3 and brilho>=70:
                score=score_full*1.2
                self.status="LED"
            else:
                score=score_full*0.6
                self.status="FRACO"
        else:
            score=score_full*0.75
            self.status="SEM LED"

        self.buffer.append(score)
        score=np.mean(self.buffer)

        self.score=score

        estado,pulsos,pulso=self.contador.atualizar(score)
        self.pulse_count=pulsos

        if pulso:
            self.status="PULSO"

        if self.config.show_overlay:
            cv2.rectangle(img,(x,y),(x+size,y+size),(0,255,255),2)
            cv2.putText(img,f"{estado} P:{pulsos}",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        return av.VideoFrame.from_ndarray(img,format="bgr24")

    def get_snapshot(self):
        return {
            "status":self.status,
            "red_score_smooth":self.score,
            "pulse_count":self.pulse_count
        }

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ Config")

    roi_size=st.slider("ROI",0.1,0.5,0.2)
    show_overlay=st.toggle("Overlay",True)

    if st.session_state.nivel=="admin":
        st.markdown("### 👤 Usuários")

        novo=st.text_input("Novo usuário")
        senha=st.text_input("Senha usuário")
        nivel=st.selectbox("Nível",["tecnico","admin"])

        if st.button("Criar usuário"):
            st.session_state.users[novo]={
                "senha":senha,
                "nivel":nivel
            }
            st.success("Usuário criado")

# =============================================================================
# CONFIG ENSAIO
# =============================================================================
st.title("⚡ PulseLab")

if not st.session_state.ensaio_iniciado:
    c1,c2,c3=st.columns(3)

    with c1:
        st.session_state.tipo_deteccao=st.selectbox("Modo",["Manual","LED","Tarja"])

    with c2:
        st.session_state.tempo_ensaio_s=st.number_input("Tempo",value=10)

    with c3:
        st.session_state.meta_pulsos=st.number_input("Meta",value=10)

    if st.button("Abrir captura"):
        st.session_state.modo_tela="captura"
        st.rerun()

# =============================================================================
# CAMERA
# =============================================================================
config=DetectorConfig(roi_size,show_overlay)

ctx=webrtc_streamer(
    key="cam",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda:PulseDetectorProcessor(config),
)

# =============================================================================
# CAPTURA
# =============================================================================
if st.session_state.modo_tela=="captura":

    snap=None
    if ctx and ctx.video_processor:
        snap=ctx.video_processor.get_snapshot()

    pulsos=snap["pulse_count"] if snap else 0
    status=snap["status"] if snap else "-"
    score=round(snap["red_score_smooth"],2) if snap else "-"

    st.markdown(f"""
    ### {st.session_state.tipo_deteccao} | {status} | Pulsos: {pulsos} | Score: {score}
    """)

    b1,b2,b3,b4=st.columns(4)

    with b1:
        if st.button("▶ Iniciar"):
            st.session_state.ensaio_iniciado=True
            st.session_state.ensaio_inicio_ts=time.time()

    with b2:
        if st.button("⏹ Finalizar"):
            st.session_state.ensaio_iniciado=False
            st.session_state.modo_tela="config_ensaio"
            st.rerun()

    with b3:
        if st.button("+"):
            if ctx.video_processor:
                ctx.video_processor.contador.pulsos+=1

    with b4:
        if st.button("-"):
            if ctx.video_processor:
                ctx.video_processor.contador.pulsos-=1