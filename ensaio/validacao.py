def classificar_robustez(meta_pulsos, pulsos_realizados, tempo_seg):
    if pulsos_realizados >= max(10, meta_pulsos) and tempo_seg >= 10:
        return "ALTA"
    if pulsos_realizados >= max(5, meta_pulsos // 2) and tempo_seg >= 5:
        return "MÉDIA"
    return "BAIXA"

def validar_teste(erro, tolerancia, robustez, pulsos):
    if abs(erro) > tolerancia:
        return "REPROVADO"
    elif pulsos < 5:
        return "INVALIDO"
    elif robustez == "BAIXA":
        return "APROVADO COM RESSALVA"
    else:
        return "APROVADO"
