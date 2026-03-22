from __future__ import annotations

import time
from collections import deque


class ContadorPulso:
    def __init__(self, limiar_on=18, limiar_off=8, debounce_s=0.12):
        self.limiar_on = float(limiar_on)
        self.limiar_off = float(limiar_off)
        self.debounce_s = float(debounce_s)

        self.estado = "OFF"
        self.pulsos = 0
        self.ultimo_pulso = 0.0

        self.historico = deque(maxlen=10)
        self.hz = 0.0

    def atualizar(self, score):
        agora = time.time()
        pulso = False

        if score > self.limiar_on and self.estado == "OFF":
            if agora - self.ultimo_pulso > self.debounce_s:
                self.estado = "ON"

        elif score < self.limiar_off and self.estado == "ON":
            self.estado = "OFF"
            self.pulsos += 1
            self.ultimo_pulso = agora
            pulso = True

            self.historico.append(agora)

            if len(self.historico) >= 2:
                diffs = [
                    self.historico[i] - self.historico[i - 1]
                    for i in range(1, len(self.historico))
                ]
                if diffs:
                    media = sum(diffs) / len(diffs)
                    if media > 0:
                        self.hz = round(1 / media, 2)

        return self.estado, self.pulsos, pulso, self.hz