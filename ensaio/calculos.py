def calcular_potencia(tensao, corrente, fp):
    return tensao * corrente * fp

def energia_teorica_wh(potencia, tempo_seg):
    return potencia * (tempo_seg / 3600.0)

def energia_medida_wh(pulsos, constante):
    return pulsos * constante

def calcular_erro(energia_medida, energia_teorica):
    if energia_teorica <= 0:
        return 0.0
    return ((energia_medida - energia_teorica) / energia_teorica) * 100.0
