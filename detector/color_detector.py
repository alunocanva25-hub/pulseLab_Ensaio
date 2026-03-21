import cv2

def build_color_masks(hsv, led_color_mode="AUTOMÁTICO"):
    led_color_mode = led_color_mode.upper()
    masks=[]
    if led_color_mode in ("VERMELHO","AUTOMÁTICO"):
        r1=cv2.inRange(hsv,(0,80,80),(10,255,255))
        r2=cv2.inRange(hsv,(160,80,80),(180,255,255))
        masks.append(("VERMELHO",cv2.add(r1,r2)))
    if led_color_mode in ("AMARELO","AUTOMÁTICO"):
        masks.append(("AMARELO",cv2.inRange(hsv,(15,80,80),(35,255,255))))
    if led_color_mode in ("BRANCO","AUTOMÁTICO"):
        masks.append(("BRANCO",cv2.inRange(hsv,(0,0,200),(180,45,255))))
    if led_color_mode in ("AZUL","AUTOMÁTICO"):
        masks.append(("AZUL",cv2.inRange(hsv,(90,80,80),(130,255,255))))
    return masks
