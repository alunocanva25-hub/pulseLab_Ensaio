from __future__ import annotations

import cv2
import numpy as np


def build_color_masks(hsv, led_color_mode="AUTOMÁTICO"):
    mode = (led_color_mode or "AUTOMÁTICO").upper()
    masks = []

    if mode in ("VERMELHO", "AUTOMÁTICO"):
        r1 = cv2.inRange(hsv, (0, 120, 140), (10, 255, 255))
        r2 = cv2.inRange(hsv, (170, 120, 140), (180, 255, 255))
        masks.append(("VERMELHO", cv2.add(r1, r2)))

    if mode in ("AMARELO", "AUTOMÁTICO"):
        masks.append(("AMARELO", cv2.inRange(hsv, (18, 120, 150), (38, 255, 255))))

    if mode in ("BRANCO", "AUTOMÁTICO"):
        masks.append(("BRANCO", cv2.inRange(hsv, (0, 0, 215), (180, 40, 255))))

    if mode in ("AZUL", "AUTOMÁTICO"):
        masks.append(("AZUL", cv2.inRange(hsv, (95, 120, 120), (130, 255, 255))))

    return masks


def merge_masks(color_masks):
    if not color_masks:
        return None
    merged = None
    for _, mask in color_masks:
        merged = mask.copy() if merged is None else cv2.add(merged, mask)
    return merged


def analyze_best_target(hsv, color_masks, prev_center=None, prefer_center_weight=1.0):
    melhor = None
    best_score = -1e9

    h, w = hsv.shape[:2]
    cx0, cy0 = w / 2, h / 2
    roi_area = float(h * w)

    for color_name, mask in color_masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 3:
                continue

            if area > roi_area * 0.08:
                continue

            x, y, bw, bh = cv2.boundingRect(c)
            cx = x + bw / 2
            cy = y + bh / 2

            dist_center = ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5
            dist_center /= max(w, h)

            dist_prev = 0.0
            if prev_center:
                px, py = prev_center
                dist_prev = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                dist_prev /= max(w, h)

            mask_c = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(mask_c, [c], -1, 255, -1)

            brilho = float(cv2.mean(hsv[:, :, 2], mask=mask_c)[0])
            sat = float(cv2.mean(hsv[:, :, 1], mask=mask_c)[0])

            peri = cv2.arcLength(c, True)
            circ = 0.0
            if peri > 0:
                circ = (4 * np.pi * area) / (peri * peri)

            intensity_density = brilho / max(area, 1.0)

            score = (
                area * 0.12
                + brilho * 0.45
                + sat * 0.20
                + circ * 22
                + intensity_density * 180
                - dist_center * 55 * prefer_center_weight
                - dist_prev * 40
            )

            if score > best_score:
                best_score = score
                melhor = {
                    "color": color_name,
                    "area": float(area),
                    "brightness": brilho,
                    "saturation": sat,
                    "score": float(score),
                    "bbox": (x, y, bw, bh),
                    "center_x": float(cx),
                    "center_y": float(cy),
                    "density": float(intensity_density),
                }

    return melhor
