from __future__ import annotations

import cv2
import numpy as np

def build_color_masks(hsv):
    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
    red_mask = cv2.add(red1, red2)
    yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 45, 255))
    return [("VERMELHO", red_mask), ("AMARELO", yellow_mask), ("BRANCO", white_mask)]

def merge_masks(color_masks):
    if not color_masks:
        return None
    merged = None
    for _, mask in color_masks:
        if merged is None:
            merged = mask.copy()
        else:
            merged = cv2.add(merged, mask)
    return merged

def analyze_best_target(hsv, color_masks):
    melhor = None
    best_score = 0.0
    h, w = hsv.shape[:2]
    center_x = w / 2.0
    center_y = h / 2.0
    for color_name, mask in color_masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 3:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            dist = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
            dist_norm = dist / max(w, h)
            mask_c = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(mask_c, [c], -1, 255, -1)
            brilho = float(cv2.mean(hsv[:, :, 2], mask=mask_c)[0])
            saturacao = float(cv2.mean(hsv[:, :, 1], mask=mask_c)[0])
            perimetro = cv2.arcLength(c, True)
            circularidade = 0.0
            if perimetro > 0:
                circularidade = float((4.0 * np.pi * area) / (perimetro * perimetro))
            score = float(area) * 0.25 + brilho * 0.35 + saturacao * 0.20 + circularidade * 20.0 - dist_norm * 80.0
            if score > best_score:
                best_score = score
                melhor = {"color": color_name, "area": float(area), "brightness": brilho, "saturation": saturacao, "circularity": circularidade, "score": float(score), "bbox": (x, y, bw, bh), "mask": mask_c, "contour": c}
    return melhor
