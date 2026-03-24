from __future__ import annotations

import cv2
import numpy as np


def frame_features(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))

    h_std = float(np.std(hsv[:, :, 0]))
    s_std = float(np.std(hsv[:, :, 1]))
    v_std = float(np.std(hsv[:, :, 2]))

    gray_mean = float(np.mean(gray))
    gray_std = float(np.std(gray))
    bright_ratio = float(np.mean(gray > 200))

    b, g, r = cv2.split(img_bgr)
    red_score = float(
        np.mean(r.astype(np.float32) - ((g.astype(np.float32) + b.astype(np.float32)) / 2.0))
    )

    return {
        "h_mean": h_mean,
        "s_mean": s_mean,
        "v_mean": v_mean,
        "h_std": h_std,
        "s_std": s_std,
        "v_std": v_std,
        "gray_mean": gray_mean,
        "gray_std": gray_std,
        "bright_ratio": bright_ratio,
        "red_score": red_score,
    }


def sequence_features(frames_bgr):
    feats = [frame_features(f) for f in frames_bgr if f is not None]
    if not feats:
        return [0.0] * 18

    gray_means = np.array([f["gray_mean"] for f in feats], dtype=np.float32)
    v_means = np.array([f["v_mean"] for f in feats], dtype=np.float32)
    red_scores = np.array([f["red_score"] for f in feats], dtype=np.float32)
    bright_ratios = np.array([f["bright_ratio"] for f in feats], dtype=np.float32)

    return [
        float(np.mean(gray_means)),
        float(np.std(gray_means)),
        float(np.max(gray_means) - np.min(gray_means)),
        float(np.mean(v_means)),
        float(np.std(v_means)),
        float(np.max(v_means) - np.min(v_means)),
        float(np.mean(red_scores)),
        float(np.std(red_scores)),
        float(np.max(red_scores) - np.min(red_scores)),
        float(np.mean(bright_ratios)),
        float(np.std(bright_ratios)),
        float(np.max(bright_ratios) - np.min(bright_ratios)),
        float(gray_means[-1]),
        float(v_means[-1]),
        float(red_scores[-1]),
        float(bright_ratios[-1]),
        float(np.max(v_means)),
        float(np.max(red_scores)),
    ]