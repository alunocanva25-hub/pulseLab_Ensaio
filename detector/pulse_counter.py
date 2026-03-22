from __future__ import annotations

import time
from collections import deque


class ContadorPulso:
    def __init__(self, limiar_on=18.0, limiar_off=8.0, debounce_s=0.12):
        self.limiar_on = float(limiar_on)
        self.limiar_off = float(limiar_off)
        self.debounce_s = float(debounce_s)

        self.estado = "OFF"
        self.pulsos = 0
        self.ultimo_pulso = 0.0
        self.pulse_times = deque(maxlen=12)
        self.hz_estimado = 0.0

    def atualizar(self, score):
        agora = time.time()
        pulso_detectado = False

        if score > self.limiar_on and self.estado == "OFF":
            if agora - self.ultimo_pulso > self.debounce_s:
                self.estado = "ON"

        elif score < self.limiar_off and self.estado == "ON":
            self.estado = "OFF"
            self.pulsos += 1
            self.ultimo_pulso = agora
            self.pulse_times.append(agora)
            pulso_detectado = True

            if len(self.pulse_times) >= 2:
                intervalos = []
                pts = list(self.pulse_times)
                for i in range(1, len(pts)):
                    intervalos.append(pts[i] - pts[i - 1])
                if intervalos:
                    intervalo_medio = sum(intervalos) / len(intervalos)
                    if intervalo_medio > 0:
                        self.hz_estimado = 1.0 / intervalo_medio

        return self.estado, self.pulsos, pulso_detectado, round(self.hz_estimado, 2)