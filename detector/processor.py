from collections import deque
from dataclasses import dataclass

@dataclass
class DetectorConfig:
    led_color_mode:str="VERMELHO"
    fast_pulse_mode:bool=True

class PulseDetectorProcessor:
    def __init__(self,config):
        self.config=config
        self.buffer=deque(maxlen=3 if config.fast_pulse_mode else 6)
        self.instant=[]
        self.pulsos=0

    def process(self,score):
        self.instant.append(score)
        if len(self.instant)>2:
            self.instant.pop(0)

        instant=max(self.instant)
        self.buffer.append(score)
        smooth=sum(self.buffer)/len(self.buffer)

        if self.config.fast_pulse_mode:
            return instant*0.7 + smooth*0.3
        return smooth
