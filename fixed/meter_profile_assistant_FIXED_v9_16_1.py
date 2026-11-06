#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# AIMeterReader - Profile Assistant
# -----------------------------------------------------------------------------
# Fixed filename : meter_profile_assistant.py
# Fixed version  : FIXED_v9.16.1_dual_hub_color_index_fix
# Base source    : meter_profile_assistant_FIXED_v9_16.py
# Fixed date     : 2026-07-06
# Purpose        : Generate analog meter profiles with human confirmation and dual hub colors.
# Notes          : Fixes STEP_LABELS/points index mismatch after adding HUB_COLOR_EDGE_COLOR.
#                  Keeps 23 GUI points aligned with create_profile color sample indexes.
# =============================================================================
"""
meter_profile_assistant_v9_2_hub_select.py

AI/CV候補 + 人確認でアナログメータープロファイルを作成するツール。

v6_hub追加仕様:
  - CENTERに加えてHUB_EDGEを指定し、中心軸（ハブ）を円で囲む。
  - NEEDLE_TIP/NEEDLE_TAILを指定し、針形状テンプレートをYAMLに保存する。
  - hub.center_xy_in_crop / radius_px / diameter_px をYAMLに保存する。
  - meter_read_v8_hub.py 側でハブ半径を中心検出・画面表示に利用する。

フロー:
  YOLO検出 -> OpenCVで円中心推定 -> 針色/中心色/角度候補を自動推定
  -> GUIで候補確認・必要箇所だけクリック修正 -> YAML保存

キー:
  1-7 : 編集対象選択
  左クリック : 選択中の点を更新
  SPACE : needle/center mask 表示切替
  ENTER : 確定
  r : 自動推定候補へ戻す
  ESC : キャンセル

実行例:
  python meter_profile_assistant_v5.py \
    --model yolo26x.pt \
    --image InputImages/IMG_0145.jpg \
    --in meter_profiles.yaml \
    --out meter_profiles.yaml \
    --profile default_pressure_meter \
    --display-scale 0.66
"""

import argparse
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

Point = Tuple[int, int]


PROFILE_ASSISTANT_VERSION = "FIXED_v9.16.1_dual_hub_color_index_fix"

class TeeStdout:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def _safe_stem(path_text: str) -> str:
    try:
        return Path(path_text).stem.replace(' ', '_')
    except Exception:
        return 'image'


STEP_LABELS = [
    "HUB_CENTER",
    "HUB_EDGE",
    "DIAL_CIRCLE_P1",
    "DIAL_CIRCLE_P2",
    "DIAL_CIRCLE_P3",
    "BEZEL_CIRCLE_P1",
    "BEZEL_CIRCLE_P2",
    "BEZEL_CIRCLE_P3",
    "NEEDLE_TIP_LONG",
    "NEEDLE_TAIL_SHORT",
    "MIN_ANGLE",
    "MID_ANGLE",
    "MAX_ANGLE",
    "NEEDLE_COLOR_LIGHT",
    "NEEDLE_COLOR_DARK",
    "HUB_COLOR_CENTER",
    "HUB_COLOR_EDGE_COLOR",
    "DIAL_COLOR_1",
    "DIAL_COLOR_2",
    "DIAL_COLOR_3",
    "BEZEL_COLOR_1",
    "BEZEL_COLOR_2",
    "BEZEL_COLOR_3",
]

STEP_COLORS = {
    "HUB_CENTER": (255, 0, 0),
    "HUB_EDGE": (255, 255, 0),
    "DIAL_CIRCLE_P1": (0, 165, 255),
    "DIAL_CIRCLE_P2": (0, 165, 255),
    "DIAL_CIRCLE_P3": (0, 165, 255),
    "BEZEL_CIRCLE_P1": (255, 0, 255),
    "BEZEL_CIRCLE_P2": (255, 0, 255),
    "BEZEL_CIRCLE_P3": (255, 0, 255),
    "NEEDLE_TIP_LONG": (0, 255, 255),
    "NEEDLE_TAIL_SHORT": (255, 128, 0),
    "MIN_ANGLE": (0, 255, 0),
    "MID_ANGLE": (0, 255, 255),
    "MAX_ANGLE": (0, 0, 255),
    "NEEDLE_COLOR_LIGHT": (255, 0, 255),
    "NEEDLE_COLOR_DARK": (255, 0, 255),
    "HUB_COLOR_CENTER": (255, 255, 0),
    "HUB_COLOR_EDGE_COLOR": (0, 255, 255),
    "DIAL_COLOR_1": (220, 220, 220),
    "DIAL_COLOR_2": (220, 220, 220),
    "DIAL_COLOR_3": (220, 220, 220),
    "BEZEL_COLOR_1": (0, 128, 255),
    "BEZEL_COLOR_2": (0, 128, 255),
    "BEZEL_COLOR_3": (0, 128, 255),
}



def circle_from_3_points(p1: Point, p2: Point, p3: Point) -> Optional[Dict[str, Any]]:
    """Return circle center/radius from 3 points, or None when points are nearly collinear."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])

    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-6:
        return None

    x1s = x1 * x1 + y1 * y1
    x2s = x2 * x2 + y2 * y2
    x3s = x3 * x3 + y3 * y3
    ux = (x1s * (y2 - y3) + x2s * (y3 - y1) + x3s * (y1 - y2)) / d
    uy = (x1s * (x3 - x2) + x2s * (x1 - x3) + x3s * (x2 - x1)) / d
    radius = math.hypot(ux - x1, uy - y1)
    if not math.isfinite(radius) or radius <= 1.0:
        return None

    return {
        "center": (int(round(ux)), int(round(uy))),
        "center_float": [float(ux), float(uy)],
        "radius": float(radius),
        "radius_px": float(radius),
        "points": [list(map(int, p1)), list(map(int, p2)), list(map(int, p3))],
    }



def _circle_or_fallback(points: List[Point], idxs: Tuple[int, int, int], fallback_center: Point, fallback_edge: Point) -> Dict[str, Any]:
    """Build a circle from 3 clicked points. If points are collinear, fall back to center+edge radius."""
    try:
        p1, p2, p3 = points[idxs[0]], points[idxs[1]], points[idxs[2]]
        circle = circle_from_3_points(p1, p2, p3)
        if circle is not None:
            return circle
    except Exception:
        pass

    radius = max(1.0, distance(fallback_center, fallback_edge))
    return {
        "center": (int(fallback_center[0]), int(fallback_center[1])),
        "center_float": [float(fallback_center[0]), float(fallback_center[1])],
        "radius": float(radius),
        "radius_px": float(radius),
        "points": [list(map(int, points[i])) for i in idxs if 0 <= i < len(points)],
        "fallback": True,
        "fallback_reason": "three_points_collinear_or_invalid",
    }

def deg_from_center(center: Point, point: Point) -> float:
    cx, cy = center
    px, py = point
    dx = px - cx
    dy = cy - py
    return math.degrees(math.atan2(dy, dx))


def normalize_deg(angle: float) -> float:
    a = angle % 360.0
    if a > 180.0:
        a -= 360.0
    return a


def point_from_angle(center: Point, angle_deg: float, radius: float) -> Point:
    cx, cy = center
    rad = math.radians(angle_deg)
    return int(round(cx + radius * math.cos(rad))), int(round(cy - radius * math.sin(rad)))


def distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def angle_delta_abs(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


def prompt_float(label: str, current: float) -> float:
    text = input(f"{label} [{current}]: ").strip()
    return float(current) if text == "" else float(text)


def prompt_str(label: str, current: str) -> str:
    text = input(f"{label} [{current}]: ").strip()
    return current if text == "" else text


def infer_default_major_step(min_value: float, max_value: float, current_ticks: Optional[List[float]] = None) -> float:
    """Return a practical default major step from existing ticks or min/max."""
    ticks = []
    if current_ticks:
        try:
            ticks = sorted({float(v) for v in current_ticks})
        except Exception:
            ticks = []
    diffs = [round(ticks[i + 1] - ticks[i], 10) for i in range(len(ticks) - 1) if abs(ticks[i + 1] - ticks[i]) > 1e-12]
    if diffs:
        # Use the smallest positive interval; this keeps 0,0.1,0.2 style meters simple.
        return float(min(diffs))
    span = abs(float(max_value) - float(min_value))
    if span <= 1e-12:
        return 1.0
    # Prefer about 5 to 10 labels.
    raw = span / 6.0
    candidates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    return min(candidates, key=lambda c: abs(math.log10(max(c, 1e-12) / max(raw, 1e-12))))


def generate_major_ticks(min_value: float, max_value: float, major_step: float) -> List[float]:
    """Generate ticks from min to max inclusive using a single interval."""
    mn = float(min_value)
    mx = float(max_value)
    step = abs(float(major_step))
    if step <= 1e-12 or abs(mx - mn) <= 1e-12:
        return [_nice_float(mn), _nice_float(mx)] if abs(mx - mn) > 1e-12 else [_nice_float(mn)]
    direction = 1.0 if mx >= mn else -1.0
    signed_step = step * direction
    ticks = []
    v = mn
    max_iter = int(abs(mx - mn) / step) + 5
    for _ in range(max_iter):
        # Stop after passing max.
        if direction > 0 and v > mx + step * 0.25:
            break
        if direction < 0 and v < mx - step * 0.25:
            break
        ticks.append(_nice_float(v))
        v += signed_step
    if not ticks or abs(ticks[0] - mn) > 1e-9:
        ticks.insert(0, _nice_float(mn))
    if abs(ticks[-1] - mx) > step * 0.25:
        ticks.append(_nice_float(mx))
    else:
        ticks[-1] = _nice_float(mx)
    # Remove duplicates while preserving order.
    out = []
    for t in ticks:
        if not out or abs(float(t) - float(out[-1])) > 1e-10:
            out.append(t)
    return out


def apply_manual_scale_values(profile: Dict[str, Any], args) -> Dict[str, Any]:
    """Apply CLI-provided min/max/step and generate major_ticks automatically."""
    has_any = any(getattr(args, name, None) is not None for name in ("min_value", "max_value", "major_step", "unit"))
    if not has_any:
        return profile
    if getattr(args, "unit", None) is not None:
        profile["unit"] = str(args.unit)
    if getattr(args, "min_value", None) is not None:
        profile["min_value"] = float(args.min_value)
    if getattr(args, "max_value", None) is not None:
        profile["max_value"] = float(args.max_value)
    current_ticks = profile.get("scale", {}).get("major_ticks", [])
    step = getattr(args, "major_step", None)
    if step is None:
        step = infer_default_major_step(profile["min_value"], profile["max_value"], current_ticks)
    profile.setdefault("scale", {})["major_step"] = float(step)
    profile["scale"]["major_ticks"] = generate_major_ticks(profile["min_value"], profile["max_value"], float(step))
    profile.setdefault("review", {})["manual_scale_note"] = "Scale generated from min_value, max_value, and major_step; individual comma-separated major_ticks were not required."
    return profile


def load_profiles(path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_profiles(path: Path, profiles: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(profiles, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def expand_bbox_xyxy(bbox: List[int], img_w: int, img_h: int, padding: int = 0) -> List[int]:
    """YOLO検出bboxを上下左右にpadding px広げ、画像範囲内にクリップする。"""
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    p = max(0, int(padding or 0))
    return [
        max(0, x1 - p),
        max(0, y1 - p),
        min(int(img_w), x2 + p),
        min(int(img_h), y2 + p),
    ]


def detect_meter(img: np.ndarray, model_path: str, class_name: str, conf_th: float, device: Optional[str], crop_padding: int = 10):
    model = YOLO(model_path)
    results = model(img, device=device) if device else model(img)
    best = None
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            name = result.names[cls]
            conf = float(box.conf[0])
            if name != class_name or conf < conf_th:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = max(0, x2 - x1) * max(0, y2 - y1)
            score = conf * area
            if best is None or score > best[0]:
                best = (score, conf, [x1, y1, x2, y2])
    if best is None:
        raise RuntimeError(f"メーター class='{class_name}' を検出できませんでした")
    _, conf, raw_bbox = best
    bbox = expand_bbox_xyxy(raw_bbox, img.shape[1], img.shape[0], crop_padding)
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2].copy()
    meta = {
        "source": "yolo_crop",
        "raw_bbox_xyxy": [int(v) for v in raw_bbox],
        "expanded_bbox_xyxy": [int(v) for v in bbox],
        "crop_padding_px": int(crop_padding),
        "crop_width_px": int(crop.shape[1]),
        "crop_height_px": int(crop.shape[0]),
        "crop_short_side_px": int(min(crop.shape[1], crop.shape[0])),
        "crop_long_side_px": int(max(crop.shape[1], crop.shape[0])),
        "source_image_width_px": int(img.shape[1]),
        "source_image_height_px": int(img.shape[0]),
    }
    print(f"yolo_crop: raw_bbox={raw_bbox} expanded_bbox={bbox} crop_padding={int(crop_padding)} crop_size=({crop.shape[1]}x{crop.shape[0]})")
    return crop, bbox, conf, meta


def resize_for_display(img: np.ndarray, display_scale: float) -> np.ndarray:
    if abs(display_scale - 1.0) < 1e-6:
        return img.copy()
    return cv2.resize(
        img,
        None,
        fx=display_scale,
        fy=display_scale,
        interpolation=cv2.INTER_AREA if display_scale < 1.0 else cv2.INTER_LINEAR,
    )


def to_display_point(point: Point, display_scale: float) -> Point:
    x, y = point
    return int(round(x * display_scale)), int(round(y * display_scale))


def to_original_point(x: int, y: int, display_scale: float, width: int, height: int) -> Point:
    ox = int(round(x / display_scale))
    oy = int(round(y / display_scale))
    return max(0, min(width - 1, ox)), max(0, min(height - 1, oy))


def detect_meter_center_hough(meter_img: np.ndarray) -> Tuple[Point, int, float]:
    h, w = meter_img.shape[:2]
    gray = cv2.cvtColor(meter_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(w, h) // 2,
        param1=80,
        param2=30,
        minRadius=int(min(w, h) * 0.25),
        maxRadius=int(min(w, h) * 0.55),
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        image_center = (w // 2, h // 2)
        scored = []
        for x, y, r in circles:
            dist = math.hypot(x - image_center[0], y - image_center[1])
            scored.append((dist - r * 0.15, int(x), int(y), int(r)))
        _, x, y, r = min(scored, key=lambda v: v[0])
        return (x, y), r, 0.80
    return (w // 2, h // 2), int(min(w, h) * 0.45), 0.35


def collect_color_samples(img: np.ndarray, points: List[Point], radius: int) -> Dict[str, np.ndarray]:
    bgr = img
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = hsv.shape[:2]
    bgr_samples, rgb_samples, hsv_samples, gray_samples = [], [], [], []
    for x, y in points:
        x1, x2 = max(0, x - radius), min(w, x + radius + 1)
        y1, y2 = max(0, y - radius), min(h, y + radius + 1)
        if x1 >= x2 or y1 >= y2:
            continue
        bgr_samples.append(bgr[y1:y2, x1:x2].reshape(-1, 3))
        rgb_samples.append(rgb[y1:y2, x1:x2].reshape(-1, 3))
        hsv_samples.append(hsv[y1:y2, x1:x2].reshape(-1, 3))
        gray_samples.append(gray[y1:y2, x1:x2].reshape(-1))
    if not hsv_samples:
        raise RuntimeError("色サンプルを取得できませんでした")
    return {
        "bgr": np.vstack(bgr_samples).astype(np.float32),
        "rgb": np.vstack(rgb_samples).astype(np.float32),
        "hsv": np.vstack(hsv_samples).astype(np.float32),
        "gray": np.concatenate(gray_samples).astype(np.float32),
    }


def classify_color(rgb_mean: np.ndarray, hsv_mean: np.ndarray, gray_mean: float) -> str:
    h, s, v = hsv_mean
    if gray_mean < 85 and v < 120:
        return "black_or_dark_gray"
    if gray_mean > 200 and s < 45:
        return "white_or_light_gray"
    if s < 45:
        return "gray"
    if h < 10 or h >= 170:
        return "red"
    if h < 25:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    if h < 160:
        return "purple"
    return "red_purple"


def color_profile_from_points(
    img: np.ndarray,
    points: List[Point],
    radius: int,
    h_tol_min: int,
    s_tol_min: int,
    v_tol_min: int,
    std_scale: float,
    gray_tol_min: int,
) -> Dict[str, Any]:
    samples = collect_color_samples(img, points, radius)
    hsv_samples = samples["hsv"]
    rgb_samples = samples["rgb"]
    bgr_samples = samples["bgr"]
    gray_samples = samples["gray"]
    hsv_mean = hsv_samples.mean(axis=0)
    hsv_std = hsv_samples.std(axis=0)
    rgb_mean = rgb_samples.mean(axis=0)
    rgb_std = rgb_samples.std(axis=0)
    bgr_mean = bgr_samples.mean(axis=0)
    gray_mean = float(gray_samples.mean())
    gray_std = float(gray_samples.std())
    hsv_tol = np.maximum(hsv_std * std_scale, np.array([h_tol_min, s_tol_min, v_tol_min], dtype=np.float32))
    hsv_lower = hsv_mean - hsv_tol
    hsv_upper = hsv_mean + hsv_tol
    gray_tol = max(gray_std * std_scale, float(gray_tol_min))
    gray_lower = clamp_int(gray_mean - gray_tol, 0, 255)
    gray_upper = clamp_int(gray_mean + gray_tol, 0, 255)
    color_name = classify_color(rgb_mean, hsv_mean, gray_mean)
    mask_mode = "dark" if color_name == "black_or_dark_gray" else "hsv"
    return {
        "sample_points": [[int(x), int(y)] for x, y in points],
        "sample_radius_px": int(radius),
        "color_name": color_name,
        "mask_mode": mask_mode,
        "rgb_mean": {"r": round(float(rgb_mean[0]), 2), "g": round(float(rgb_mean[1]), 2), "b": round(float(rgb_mean[2]), 2)},
        "rgb_std": {"r": round(float(rgb_std[0]), 2), "g": round(float(rgb_std[1]), 2), "b": round(float(rgb_std[2]), 2)},
        "bgr_mean": {"b": round(float(bgr_mean[0]), 2), "g": round(float(bgr_mean[1]), 2), "r": round(float(bgr_mean[2]), 2)},
        "gray_mean": round(gray_mean, 2),
        "gray_std": round(gray_std, 2),
        "gray_lower": gray_lower,
        "gray_upper": gray_upper,
        "hsv_mean": {"h": round(float(hsv_mean[0]), 2), "s": round(float(hsv_mean[1]), 2), "v": round(float(hsv_mean[2]), 2)},
        "hsv_std": {"h": round(float(hsv_std[0]), 2), "s": round(float(hsv_std[1]), 2), "v": round(float(hsv_std[2]), 2)},
        "hsv_lower": [clamp_int(hsv_lower[0], 0, 179), clamp_int(hsv_lower[1], 0, 255), clamp_int(hsv_lower[2], 0, 255)],
        "hsv_upper": [clamp_int(hsv_upper[0], 0, 179), clamp_int(hsv_upper[1], 0, 255), clamp_int(hsv_upper[2], 0, 255)],
    }


def make_mask_from_profile(img: np.ndarray, profile: Dict[str, Any]) -> np.ndarray:
    mode = profile.get("mask_mode", "hsv")
    if mode == "dark":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.inRange(gray, int(profile.get("gray_lower", 0)), int(profile.get("gray_upper", 100)))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(profile["hsv_lower"], dtype=np.uint8), np.array(profile["hsv_upper"], dtype=np.uint8))


def color_summary_text(profile: Dict[str, Any]) -> str:
    rgb = profile.get("rgb_mean", {})
    hsv = profile.get("hsv_mean", {})
    return (
        f"{profile.get('color_name', 'unknown')} | "
        f"RGB=({rgb.get('r', 0):.0f},{rgb.get('g', 0):.0f},{rgb.get('b', 0):.0f}) | "
        f"HSV=({hsv.get('h', 0):.0f},{hsv.get('s', 0):.0f},{hsv.get('v', 0):.0f}) | "
        f"gray={profile.get('gray_mean', 0):.0f} | mode={profile.get('mask_mode', 'hsv')}"
    )


def overlay_mask(img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    colored = np.zeros_like(out)
    colored[:] = color
    out[mask > 0] = cv2.addWeighted(out, 0.35, colored, 0.65, 0)[mask > 0]
    return out


def estimate_needle_points(meter_img: np.ndarray, center: Point, radius: int) -> List[Point]:
    h, w = meter_img.shape[:2]
    gray = cv2.cvtColor(meter_img, cv2.COLOR_BGR2GRAY)
    valid = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(valid, center, int(radius * 0.72), 255, -1)
    cv2.circle(valid, center, int(radius * 0.10), 0, -1)
    mask = cv2.inRange(gray, 0, 95)
    mask = cv2.bitwise_and(mask, valid)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx, cy = center
    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 15:
            continue
        pts = cnt.reshape(-1, 2)
        dists = [math.hypot(int(px) - cx, int(py) - cy) for px, py in pts]
        min_d, max_d = min(dists), max(dists)
        score = (max_d - min_d) + max_d * 0.2 - min_d * 0.8 + area * 0.02
        if best is None or score > best[0]:
            tip = tuple(map(int, pts[int(np.argmax(dists))]))
            best = (score, tip)
    if best is not None:
        tip = best[1]
        return [
            (int(round(cx + (tip[0] - cx) * 0.45)), int(round(cy + (tip[1] - cy) * 0.45))),
            (int(round(cx + (tip[0] - cx) * 0.75)), int(round(cy + (tip[1] - cy) * 0.75))),
        ]
    return [(max(0, int(cx - radius * 0.30)), cy), (max(0, int(cx - radius * 0.55)), cy)]


def estimate_ocr_scale(meter_img: np.ndarray) -> Dict[str, Any]:
    try:
        import pytesseract
    except Exception:
        return {"available": False, "numbers": [], "unit_candidates": [], "reason": "pytesseract is not installed"}
    gray = cv2.cvtColor(meter_img, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # OCR target is intentionally restricted to lowercase alphanumerics and meter symbols.
    # This prevents Japanese/other glyphs from being returned and stabilizes scale/unit hints.
    # Uppercase units such as MPa/kPa are normalized to lowercase after OCR.
    ocr_config = (
        "--oem 3 --psm 6 "
        "-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz.-+/°℃% "
        "-c load_system_dawg=0 -c load_freq_dawg=0"
    )
    try:
        text = pytesseract.image_to_string(th, config=ocr_config)
    except Exception as e:
        return {"available": False, "numbers": [], "unit_candidates": [], "reason": str(e), "ocr_config": ocr_config}
    text_norm = text.lower()
    nums = []
    for m in re.finditer(r"[-+]?\d+(?:\.\d+)?", text):
        try:
            nums.append(float(m.group(0)))
        except ValueError:
            pass
    unit_map = {
        "kpa": "kPa",
        "mpa": "MPa",
        "bar": "bar",
        "psi": "psi",
        "c": "°C",
        "℃": "℃",
    }
    unit_candidates = []
    for key, label in unit_map.items():
        if key in text_norm and label not in unit_candidates:
            unit_candidates.append(label)
    return {
        "available": True,
        "raw_text": text,
        "normalized_text": text_norm,
        "numbers": sorted(set(nums)),
        "unit_candidates": unit_candidates,
        "ocr_config": ocr_config,
    }



def _ocr_whitelist_config(psm: int = 6) -> str:
    return (
        f"--oem 3 --psm {int(psm)} "
        "-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz.-+/°℃% "
        "-c load_system_dawg=0 -c load_freq_dawg=0"
    )


def _parse_number_token(text: str) -> Optional[float]:
    if text is None:
        return None
    t = str(text).strip().lower()
    if not t:
        return None
    t = t.replace(",", ".")
    # Common OCR confusions. Keep this conservative to avoid inventing scale values.
    t = t.replace("o", "0")
    t = t.replace("l", "1")
    t = t.replace("i", "1")
    # Keep one signed decimal number only.
    m = re.search(r"[-+]?\d+(?:\.\d+)?|[-+]?\.\d+", t)
    if not m:
        return None
    token = m.group(0)
    try:
        return float(token)
    except ValueError:
        return None


def _nice_float(v: float) -> float:
    av = abs(float(v))
    if av >= 10:
        return round(float(v), 3)
    if av >= 1:
        return round(float(v), 4)
    return round(float(v), 5)


def _angle_to_ratio_on_scale(angle: float, min_angle: float, max_angle: float, direction: str = "clockwise") -> Tuple[float, bool]:
    """Return position ratio from MIN to MAX on the configured arc.

    ratio=0 is min, ratio=1 is max.  Values outside [0,1] are outside the arc.
    """
    a = angle % 360.0
    mn = min_angle % 360.0
    mx = max_angle % 360.0
    if str(direction).lower().startswith("counter"):
        span = (mx - mn) % 360.0
        pos = (a - mn) % 360.0
    else:
        span = (mn - mx) % 360.0
        pos = (mn - a) % 360.0
    if span <= 1e-6:
        return 0.0, False
    ratio = pos / span
    return float(ratio), (0.0 <= ratio <= 1.0)


def _ocr_number_candidates_with_boxes(meter_img: np.ndarray) -> Dict[str, Any]:
    """OCR full meter image and return numeric tokens with original-image coordinates."""
    try:
        import pytesseract
    except Exception:
        return {"available": False, "numbers": [], "reason": "pytesseract is not installed"}

    gray = cv2.cvtColor(meter_img, cv2.COLOR_BGR2GRAY)
    scale = 2.0
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    variants = []
    # Variant 1: Otsu normal
    _, th = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", th, 6))
    # Variant 2: Otsu inverted can help when black text is interpreted poorly.
    variants.append(("otsu_inv", cv2.bitwise_not(th), 6))
    # Variant 3: adaptive threshold for uneven illumination.
    ad = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 9)
    variants.append(("adaptive", ad, 6))

    all_items: List[Dict[str, Any]] = []
    raw_texts: List[str] = []
    errors: List[str] = []
    for variant_name, proc, psm in variants:
        config = _ocr_whitelist_config(psm)
        try:
            data = pytesseract.image_to_data(proc, config=config, output_type=pytesseract.Output.DICT)
            raw_texts.append(" ".join([str(x) for x in data.get("text", []) if str(x).strip()]))
        except Exception as e:
            errors.append(f"{variant_name}: {e}")
            continue
        n = len(data.get("text", []))
        for i in range(n):
            text = str(data["text"][i]).strip()
            val = _parse_number_token(text)
            if val is None:
                continue
            try:
                conf = float(data.get("conf", [0] * n)[i])
            except Exception:
                conf = 0.0
            # Tesseract uses -1 for some non-word boxes.
            if conf < -0.5:
                continue
            x = float(data["left"][i]) / scale
            y = float(data["top"][i]) / scale
            w = float(data["width"][i]) / scale
            h = float(data["height"][i]) / scale
            if w <= 0 or h <= 0:
                continue
            all_items.append({
                "text": text,
                "value": _nice_float(val),
                "confidence": round(conf, 2),
                "bbox_xywh": [round(x, 1), round(y, 1), round(w, 1), round(h, 1)],
                "center_xy": [round(x + w / 2.0, 1), round(y + h / 2.0, 1)],
                "variant": variant_name,
            })

    # De-duplicate by value and nearby center; keep best confidence/larger box.
    uniq: List[Dict[str, Any]] = []
    for item in sorted(all_items, key=lambda d: (-float(d.get("confidence", 0)), d.get("value", 0))):
        cx, cy = item["center_xy"]
        dup = False
        for old in uniq:
            if abs(float(old["value"]) - float(item["value"])) < 1e-9 and math.hypot(cx - old["center_xy"][0], cy - old["center_xy"][1]) < 14:
                dup = True
                break
        if not dup:
            uniq.append(item)
    return {
        "available": True,
        "numbers": uniq,
        "raw_text": "\n".join(raw_texts),
        "errors": errors,
        "ocr_config": _ocr_whitelist_config(6),
    }


def _choose_nearest_numeric_candidate(cands: List[Dict[str, Any]], target_pt: Point, target_ratio: float, roi_radius: float) -> Optional[Dict[str, Any]]:
    if not cands:
        return None
    tx, ty = target_pt
    scored = []
    for c in cands:
        cx, cy = c.get("center_xy", [0, 0])
        d = math.hypot(float(cx) - tx, float(cy) - ty)
        ratio = float(c.get("scale_ratio", 0.0))
        # Prefer labels close to the clicked point. Ratio is only a secondary tie-break.
        roi_penalty = 0.0 if d <= roi_radius else (d - roi_radius) * 1.5
        score = d + roi_penalty + 25.0 * abs(ratio - target_ratio) - 0.05 * float(c.get("confidence", 0.0))
        scored.append((score, d, c))
    scored.sort(key=lambda x: x[0])
    return scored[0][2]


def _estimate_step_from_values(min_value: float, max_value: float, observed_values: List[float]) -> Tuple[Optional[float], List[float], Dict[str, Any]]:
    lo = float(min_value)
    hi = float(max_value)
    if hi < lo:
        lo, hi = hi, lo
    span = hi - lo
    if span <= 0:
        return None, [], {"reason": "invalid_min_max"}
    obs = sorted(set([_nice_float(v) for v in observed_values if lo - span * 0.05 <= float(v) <= hi + span * 0.05]))

    best = None
    # Prefer 2 to 30 intervals. Analog meters usually do not need more for numbered major ticks.
    for n in range(1, 31):
        step = span / n
        if step <= 0:
            continue
        grid = [lo + k * step for k in range(n + 1)]
        # Alignment error for observed values to this grid.
        if obs:
            rel_errs = []
            hits = 0
            for v in obs:
                nearest = min(grid, key=lambda g: abs(g - v))
                e = abs(nearest - v) / max(step, 1e-9)
                rel_errs.append(e)
                if e <= 0.22:
                    hits += 1
            align = sum(rel_errs) / len(rel_errs)
            coverage_bonus = hits / max(len(obs), 1)
        else:
            align = 1.0
            coverage_bonus = 0.0
        # Prefer human-friendly interval counts and not too many labels.
        nice_n_bonus = 0.0
        if n in (2, 3, 4, 5, 6, 10, 12, 15, 20):
            nice_n_bonus = 0.08
        score = align + 0.012 * n - coverage_bonus * 0.35 - nice_n_bonus
        if best is None or score < best[0]:
            best = (score, n, step, grid, align, coverage_bonus)
    if best is None:
        return None, [], {"reason": "no_candidate"}
    _, n, step, grid, align, coverage = best
    ticks = [_nice_float(v) for v in grid]
    debug = {"interval_count": int(n), "alignment_error": round(float(align), 4), "observed_coverage": round(float(coverage), 4), "observed_values": obs}
    return _nice_float(step), ticks, debug


def estimate_click_near_ocr_scale(meter_img: np.ndarray, profile: Dict[str, Any], args) -> Dict[str, Any]:
    """Use MIN/MAX click positions to localize OCR and infer scale values/ticks.

    This intentionally does not trust whole-image OCR directly.  The nearest numeric token to
    the user-clicked MIN/MAX points becomes min/max, then other OCR numbers are accepted only
    when they are both on the MIN-MAX arc and numerically between min/max.
    """
    center = tuple(map(int, profile.get("center_point") or profile.get("geometry", {}).get("center_xy_in_crop", [0, 0])))
    radius = float(profile.get("geometry", {}).get("radius_px") or 1.0)
    min_angle = float(profile.get("min_angle", 0.0))
    max_angle = float(profile.get("max_angle", 0.0))
    direction = profile.get("scale", {}).get("direction", "clockwise")
    min_pt = point_from_angle(center, min_angle, radius * 0.82)
    max_pt = point_from_angle(center, max_angle, radius * 0.82)

    ocr = _ocr_number_candidates_with_boxes(meter_img)
    if not ocr.get("available"):
        return {"available": False, "reason": ocr.get("reason", "ocr unavailable"), "raw": ocr}

    arc_margin = float(getattr(args, "ocr_arc_margin", 0.08))
    radial_min = float(getattr(args, "ocr_radial_min", 0.35))
    radial_max = float(getattr(args, "ocr_radial_max", 1.15))
    roi_radius = max(28.0, radius * float(getattr(args, "ocr_minmax_roi_ratio", 0.20)))

    arc_candidates: List[Dict[str, Any]] = []
    ignored: List[Dict[str, Any]] = []
    for item in ocr.get("numbers", []):
        cx, cy = item.get("center_xy", [0, 0])
        pt = (int(round(float(cx))), int(round(float(cy))))
        a = normalize_deg(deg_from_center(center, pt))
        ratio, in_arc = _angle_to_ratio_on_scale(a, min_angle, max_angle, direction)
        radial_ratio = distance(center, pt) / max(radius, 1.0)
        enriched = dict(item)
        enriched.update({
            "angle": round(float(a), 3),
            "scale_ratio": round(float(ratio), 4),
            "radial_ratio": round(float(radial_ratio), 4),
            "in_scale_arc": bool(in_arc and (-arc_margin <= ratio <= 1.0 + arc_margin)),
        })
        if enriched["in_scale_arc"] and radial_min <= radial_ratio <= radial_max:
            arc_candidates.append(enriched)
        else:
            ignored.append({k: enriched.get(k) for k in ("text", "value", "center_xy", "angle", "scale_ratio", "radial_ratio", "variant")})

    min_cand = _choose_nearest_numeric_candidate(arc_candidates, min_pt, 0.0, roi_radius)
    max_cand = _choose_nearest_numeric_candidate(arc_candidates, max_pt, 1.0, roi_radius)
    if min_cand is None or max_cand is None:
        return {
            "available": False,
            "reason": "could not find numeric OCR candidates near MIN/MAX clicks",
            "raw": ocr,
            "arc_candidates": arc_candidates,
            "ignored_candidates": ignored,
            "min_click_point": [int(min_pt[0]), int(min_pt[1])],
            "max_click_point": [int(max_pt[0]), int(max_pt[1])],
        }

    min_value = float(min_cand["value"])
    max_value = float(max_cand["value"])
    if abs(max_value - min_value) < 1e-12:
        return {
            "available": False,
            "reason": "MIN/MAX OCR values are identical",
            "selected_min_candidate": min_cand,
            "selected_max_candidate": max_cand,
            "arc_candidates": arc_candidates,
        }

    lo = min(min_value, max_value)
    hi = max(min_value, max_value)
    span = hi - lo
    in_range: List[Dict[str, Any]] = []
    out_of_value_range: List[Dict[str, Any]] = []
    for c in arc_candidates:
        v = float(c["value"])
        if lo - span * 0.04 <= v <= hi + span * 0.04:
            in_range.append(c)
        else:
            out_of_value_range.append({k: c.get(k) for k in ("text", "value", "center_xy", "scale_ratio", "radial_ratio", "variant")})

    observed_values = [float(c["value"]) for c in in_range] + [min_value, max_value]
    step, ticks, step_debug = _estimate_step_from_values(min_value, max_value, observed_values)
    if step is None or not ticks:
        ticks = sorted([_nice_float(min_value), _nice_float(max_value)])

    # Match generated ticks to OCR observations for traceability.
    tick_items = []
    for tv in ticks:
        best_obs = None
        best_d = 1e9
        for c in in_range:
            d = abs(float(c["value"]) - float(tv))
            if d < best_d:
                best_d = d
                best_obs = c
        if best_obs is not None and step and best_d <= max(abs(float(step)) * 0.22, 1e-6):
            src = "ocr_arc"
            if abs(float(tv) - min_value) <= max(abs(float(step or 1)) * 0.22, 1e-6):
                src = "ocr_near_min"
            if abs(float(tv) - max_value) <= max(abs(float(step or 1)) * 0.22, 1e-6):
                src = "ocr_near_max"
            tick_items.append({"value": _nice_float(tv), "source": src, "ocr": {k: best_obs.get(k) for k in ("text", "center_xy", "confidence", "scale_ratio", "variant")}})
        else:
            tick_items.append({"value": _nice_float(tv), "source": "inferred"})

    return {
        "available": True,
        "source": "click_nearby_ocr_arc_inference",
        "min_click_point": [int(min_pt[0]), int(min_pt[1])],
        "max_click_point": [int(max_pt[0]), int(max_pt[1])],
        "selected_min_candidate": min_cand,
        "selected_max_candidate": max_cand,
        "min_value": _nice_float(min_value),
        "max_value": _nice_float(max_value),
        "major_step": step,
        "major_ticks": ticks,
        "ticks": tick_items,
        "arc_candidates": arc_candidates,
        "ignored_candidates": ignored,
        "ignored_value_out_of_range": out_of_value_range,
        "step_debug": step_debug,
        "raw": {"available": ocr.get("available"), "raw_text": ocr.get("raw_text"), "errors": ocr.get("errors"), "ocr_config": ocr.get("ocr_config")},
    }


def apply_click_near_ocr_scale(profile: Dict[str, Any], meter_img: np.ndarray, args) -> Dict[str, Any]:
    if not getattr(args, "use_ocr", False) or not getattr(args, "click_near_ocr_scale", True):
        return profile
    scale_ocr = estimate_click_near_ocr_scale(meter_img, profile, args)
    profile["scale_ocr"] = scale_ocr
    profile.setdefault("ai_candidate", {})["click_near_ocr_scale"] = scale_ocr
    if not scale_ocr.get("available"):
        print(f"click-near OCR scale: unavailable ({scale_ocr.get('reason')})")
        return profile
    profile["min_value"] = float(scale_ocr["min_value"])
    profile["max_value"] = float(scale_ocr["max_value"])
    profile.setdefault("scale", {})["major_ticks"] = [float(v) for v in scale_ocr.get("major_ticks", [])]
    if scale_ocr.get("major_step") is not None:
        profile["scale"]["major_step"] = float(scale_ocr["major_step"])
    profile.setdefault("review", {})
    profile["review"]["scale_ocr_note"] = "MIN/MAX click-near OCR selected endpoint values; arc OCR values were filtered and missing ticks inferred."
    print("click-near OCR scale applied:")
    print(f"  min_value={profile['min_value']} from {scale_ocr.get('selected_min_candidate', {}).get('text')} at {scale_ocr.get('selected_min_candidate', {}).get('center_xy')}")
    print(f"  max_value={profile['max_value']} from {scale_ocr.get('selected_max_candidate', {}).get('text')} at {scale_ocr.get('selected_max_candidate', {}).get('center_xy')}")
    print(f"  major_step={scale_ocr.get('major_step')} major_ticks={profile['scale'].get('major_ticks')}")
    return profile

def auto_profile_candidate(meter_img: np.ndarray, defaults: Dict[str, Any], args) -> Dict[str, Any]:
    center, radius, center_conf = detect_meter_center_hough(meter_img)
    min_angle = float(defaults.get("min_angle", 225.0))
    max_angle = float(defaults.get("max_angle", -45.0))
    min_pt = point_from_angle(center, min_angle, radius * 0.82)
    max_pt = point_from_angle(center, max_angle, radius * 0.82)
    hub_radius = max(6, int(radius * 0.055))
    hub_edge_pt = (min(meter_img.shape[1] - 1, center[0] + hub_radius), center[1])
    needle_points = estimate_needle_points(meter_img, center, radius)
    needle_tip_pt = needle_points[1]
    tip_angle = deg_from_center(center, needle_tip_pt)
    tail_pt = point_from_angle(center, normalize_deg(tip_angle + 180.0), radius * 0.30)
    center_color_point = center
    ocr = estimate_ocr_scale(meter_img) if args.use_ocr else {"available": False, "numbers": [], "unit_candidates": []}
    unit = defaults.get("unit", "MPa")
    min_value = defaults.get("min_value", 0.0)
    max_value = defaults.get("max_value", 2.0)
    major_ticks = defaults.get("scale", {}).get("major_ticks", [0.0, 0.5, 1.0, 1.5, 2.0])
    if ocr.get("unit_candidates"):
        unit = ocr["unit_candidates"][0]
    if ocr.get("numbers") and len(ocr["numbers"]) >= 2:
        min_value = min(ocr["numbers"])
        max_value = max(ocr["numbers"])
        major_ticks = ocr["numbers"]
    needle_prof = color_profile_from_points(meter_img, needle_points, args.sample_radius, args.h_tol_min, args.s_tol_min, args.v_tol_min, args.std_scale, args.gray_tol_min)
    center_prof = color_profile_from_points(meter_img, [center_color_point], args.sample_radius, args.h_tol_min, args.s_tol_min, args.v_tol_min, args.std_scale, args.gray_tol_min)
    span = ((max_angle - min_angle + 540.0) % 360.0) - 180.0
    mid_angle = normalize_deg(min_angle + span / 2.0)
    # v9.14: 文字盤円・ベゼル円は、ハブ中心に強制一致させず3点で定義する。
    # 初期値は上・左・右に置く。ユーザーが3点を実円周上に合わせる。
    dial_p1 = point_from_angle(center, 90.0, radius * 1.00)
    dial_p2 = point_from_angle(center, 180.0, radius * 1.00)
    dial_p3 = point_from_angle(center, 0.0, radius * 1.00)
    bezel_p1 = point_from_angle(center, 90.0, radius * 1.12)
    bezel_p2 = point_from_angle(center, 180.0, radius * 1.12)
    bezel_p3 = point_from_angle(center, 0.0, radius * 1.12)
    return {
        "points": [
            center,
            hub_edge_pt,
            dial_p1,
            dial_p2,
            dial_p3,
            bezel_p1,
            bezel_p2,
            bezel_p3,
            needle_tip_pt,
            tail_pt,
            min_pt,
            point_from_angle(center, mid_angle, radius * 0.82),
            max_pt,
            needle_points[0],
            needle_points[1],
            center,  # 16 HUB_COLOR_CENTER
            point_from_angle(center, 90.0, max(hub_radius * 0.85, radius * 0.055)),  # 17 HUB_COLOR_EDGE_COLOR
            point_from_angle(center, 90.0, radius * 0.45),
            point_from_angle(center, 180.0, radius * 0.45),
            point_from_angle(center, 0.0, radius * 0.45),
            point_from_angle(center, 90.0, radius * 1.06),
            point_from_angle(center, 180.0, radius * 1.06),
            point_from_angle(center, 0.0, radius * 1.06),
        ],
        "needle_prof": needle_prof,
        "center_prof": center_prof,
        "unit": unit,
        "min_value": min_value,
        "max_value": max_value,
        "major_ticks": major_ticks,
        "scale_direction": defaults.get("scale", {}).get("direction", "clockwise"),
        "auto": {"center_confidence": center_conf, "radius_px": radius, "hub_radius_px": hub_radius, "ocr": ocr},
    }


def draw_clicks(base: np.ndarray, points: List[Point], display_scale: float = 1.0, show_overlays: bool = True) -> np.ndarray:
    out = base.copy()
    if not show_overlays:
        return out
    if not points:
        return out
    hub_center = points[0]
    hd = to_display_point(hub_center, display_scale)
    # ハブは中心 + 外周1点。文字盤/ベゼルは3点円。
    if len(points) > 1:
        rr = int(round(distance(hub_center, points[1]) * display_scale))
        cv2.circle(out, hd, max(2, rr), STEP_COLORS["HUB_EDGE"], 2)
        cv2.line(out, hd, to_display_point(points[1], display_scale), STEP_COLORS["HUB_EDGE"], 1)
    if len(points) > 4:
        dial = circle_from_3_points(points[2], points[3], points[4])
        if dial:
            cv2.circle(out, to_display_point(dial["center"], display_scale), int(round(dial["radius"] * display_scale)), (0, 165, 255), 2)
    if len(points) > 7:
        bezel = circle_from_3_points(points[5], points[6], points[7])
        if bezel:
            cv2.circle(out, to_display_point(bezel["center"], display_scale), int(round(bezel["radius"] * display_scale)), (255, 0, 255), 2)
    if len(points) > 9:
        cv2.line(out, hd, to_display_point(points[8], display_scale), STEP_COLORS["NEEDLE_TIP_LONG"], 2)
        cv2.line(out, hd, to_display_point(points[9], display_scale), STEP_COLORS["NEEDLE_TAIL_SHORT"], 2)
    for idx, label in ((10, "MIN_ANGLE"), (11, "MID_ANGLE"), (12, "MAX_ANGLE")):
        if len(points) > idx:
            cv2.line(out, hd, to_display_point(points[idx], display_scale), STEP_COLORS[label], 1)
    for i, pt in enumerate(points):
        label = STEP_LABELS[i] if i < len(STEP_LABELS) else str(i + 1)
        color = STEP_COLORS.get(label, (255, 255, 255))
        dp = to_display_point(pt, display_scale)
        cv2.circle(out, dp, 5, color, -1)
        cv2.putText(out, str(i + 1), (dp[0] + 6, dp[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def interactive_review_profile(meter_img: np.ndarray, candidate: Dict[str, Any], args):
    h_img, w_img = meter_img.shape[:2]
    base_display = resize_for_display(meter_img, args.display_scale)
    points: List[Point] = [tuple(map(int, p)) for p in candidate["points"]]
    # v9.16.1: STEP_LABELS has 23 points. Older v9.16 candidates had only 22
    # because HUB_COLOR_EDGE_COLOR was added to labels but not to the initial point list.
    # If an old 22-point candidate is seen, insert the missing hub-edge-color sample
    # at index 16 so DIAL/BEZEL color samples keep their intended indexes.
    if len(points) == len(STEP_LABELS) - 1:
        hub_center = points[0] if points else (w_img // 2, h_img // 2)
        hub_edge = points[1] if len(points) > 1 else (hub_center[0] + 10, hub_center[1])
        r = max(6.0, distance(hub_center, hub_edge))
        points.insert(16, (int(round(hub_center[0])), int(round(hub_center[1] - r * 0.85))))
    # Pad defensively so n/left-click on item 23 cannot raise IndexError even for
    # partially generated candidates.
    while len(points) < len(STEP_LABELS):
        if len(points) == 16:
            hub_center = points[0] if points else (w_img // 2, h_img // 2)
            hub_edge = points[1] if len(points) > 1 else (hub_center[0] + 10, hub_center[1])
            r = max(6.0, distance(hub_center, hub_edge))
            points.append((int(round(hub_center[0])), int(round(hub_center[1] - r * 0.85))))
        elif len(points) >= 20:
            # Missing BEZEL_COLOR_3 in old/partial candidate; use right-side bezel sample by default.
            c = points[0] if points else (w_img // 2, h_img // 2)
            r = max(20.0, min(w_img, h_img) * 0.40)
            points.append((int(min(w_img - 1, round(c[0] + r))), int(round(c[1]))))
        else:
            points.append(points[-1] if points else (w_img // 2, h_img // 2))
    original_points = list(points)
    needle_prof = candidate["needle_prof"]
    center_prof = candidate["center_prof"]
    mask_mode = False
    overlay_mode = True
    edit_index = 0

    def regenerate_color_profiles_if_complete() -> None:
        nonlocal needle_prof, center_prof
        if len(points) >= len(STEP_LABELS):
            needle_prof = color_profile_from_points(meter_img, [points[13], points[14]], args.sample_radius, args.h_tol_min, args.s_tol_min, args.v_tol_min, args.std_scale, args.gray_tol_min)
            hub_center_prof = color_profile_from_points(meter_img, [points[15]], args.sample_radius, args.h_tol_min, args.s_tol_min, args.v_tol_min, args.std_scale, args.gray_tol_min)
            hub_edge_prof = color_profile_from_points(meter_img, [points[16]], args.sample_radius, args.h_tol_min, args.s_tol_min, args.v_tol_min, args.std_scale, args.gray_tol_min)
            center_prof = color_profile_from_points(meter_img, [points[15], points[16]], args.sample_radius, args.h_tol_min, args.s_tol_min, args.v_tol_min, args.std_scale, args.gray_tol_min)
            print("Color profile generated.")
            print("  needle:", color_summary_text(needle_prof))
            print("  hub_center:", color_summary_text(hub_center_prof))
            print("  hub_edge  :", color_summary_text(hub_edge_prof))
            print("  hub_merged:", color_summary_text(center_prof))
            print("  dial  :", color_summary_text(color_profile_from_points(meter_img, [points[17], points[18], points[19]], args.sample_radius, args.h_tol_min, args.s_tol_min, args.v_tol_min, args.std_scale, args.gray_tol_min)))
            print("  bezel :", color_summary_text(color_profile_from_points(meter_img, [points[20], points[21], points[22]], args.sample_radius, args.h_tol_min, args.s_tol_min, args.v_tol_min, args.std_scale, args.gray_tol_min)))

    def on_mouse(event, x, y, flags, param):
        nonlocal edit_index
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox, oy = to_original_point(x, y, args.display_scale, w_img, h_img)
        points[edit_index] = (ox, oy)
        print(f"updated {STEP_LABELS[edit_index]}: display=({x},{y}) original=({ox},{oy})")
        if edit_index == 0 or edit_index >= 9:
            regenerate_color_profiles_if_complete()
        edit_index = min(edit_index + 1, len(STEP_LABELS) - 1)

    cv2.namedWindow("profile_assistant_v9_1_8point", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("profile_assistant_v9_1_8point", base_display.shape[1], base_display.shape[0])
    cv2.setMouseCallback("profile_assistant_v9_1_8point", on_mouse)
    print("\nAI/CV候補を表示します。")
    print("キー: 1-9=編集対象選択 / n,p=編集対象移動 / 左クリック=更新 / o=線・円・ラベル表示ON/OFF / SPACE=マスク確認 / ENTER=確定 / r=候補へ戻す / ESC=中止")
    print("1 HUB_CENTER, 2 HUB_EDGE")
    print("3-5 DIAL_CIRCLE 3 points, 6-8 BEZEL_CIRCLE 3 points")
    print("9 NEEDLE_TIP_LONG, 10 NEEDLE_TAIL_SHORT")
    print("11 MIN_ANGLE, 12 MID_ANGLE, 13 MAX_ANGLE")
    print("14-15 NEEDLE_COLOR light/dark, 16 HUB_COLOR_CENTER, 17 HUB_COLOR_EDGE_COLOR, 18-20 DIAL_COLOR samples, 21-23 BEZEL_COLOR samples")
    print("v9.14: DIAL/BEZELは3点指定で円中心と半径を保存します。ハブ中心と強制一致させません。")
    print(f"表示倍率 display_scale={args.display_scale}")
    while True:
        frame = draw_clicks(base_display, points, args.display_scale, overlay_mode)
        if mask_mode:
            needle_mask = make_mask_from_profile(meter_img, needle_prof)
            center_mask = make_mask_from_profile(meter_img, center_prof)
            frame = overlay_mask(frame, resize_for_display(needle_mask, args.display_scale), (255, 0, 255))
            frame = overlay_mask(frame, resize_for_display(center_mask, args.display_scale), (0, 255, 255))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 86), (0, 0, 0), -1)
        cv2.putText(frame, f"Edit: {edit_index + 1} {STEP_LABELS[edit_index]} | o overlay={'ON' if overlay_mode else 'OFF'} | SPACE mask | ENTER accept | r reset | ESC cancel", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        cv2.putText(frame, f"unit={candidate.get('unit')} min={candidate.get('min_value')} max={candidate.get('max_value')} ticks={candidate.get('major_ticks')}", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        if needle_prof:
            y0 = frame.shape[0] - 72
            cv2.rectangle(frame, (10, y0), (min(frame.shape[1] - 10, 900), y0 + 62), (0, 0, 0), -1)
            cv2.putText(frame, "NEEDLE " + color_summary_text(needle_prof), (18, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, "CENTER " + color_summary_text(center_prof), (18, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imshow("profile_assistant_v9_1_8point", frame)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow("profile_assistant_v9_1_8point")
            raise RuntimeError("Profile creation canceled")
        if key in (10, 13):
            regenerate_color_profiles_if_complete()
            cv2.destroyWindow("profile_assistant_v9_1_8point")
            return points, needle_prof, center_prof
        if key == 32:
            regenerate_color_profiles_if_complete()
            mask_mode = not mask_mode
            print(f"mask_mode={mask_mode}")
        if key in (ord("o"), ord("O")):
            overlay_mode = not overlay_mode
            print(f"overlay_mode={overlay_mode}")
        if key in (ord("r"), ord("R")):
            points = list(original_points)
            edit_index = 0
            regenerate_color_profiles_if_complete()
            mask_mode = False
            overlay_mode = True
            print("reset to AI/CV candidate")
        if key in (9, ord("n"), ord("N")):
            edit_index = (edit_index + 1) % len(STEP_LABELS)
            print(f"edit target: {edit_index + 1} {STEP_LABELS[edit_index]}")
        if key in (ord("p"), ord("P")):
            edit_index = (edit_index - 1) % len(STEP_LABELS)
            print(f"edit target: {edit_index + 1} {STEP_LABELS[edit_index]}")
        if ord("1") <= key <= ord("9"):
            edit_index = key - ord("1")
            print(f"edit target: {edit_index + 1} {STEP_LABELS[edit_index]}")


def apply_profile_defaults(profile: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not defaults:
        return profile
    for key in ("type", "unit", "min_value", "max_value"):
        if key in defaults:
            profile[key] = defaults[key]
    if "scale" in defaults and isinstance(defaults["scale"], dict):
        profile.setdefault("scale", {})
        for key in ("major_ticks", "direction"):
            if key in defaults["scale"]:
                profile["scale"][key] = defaults["scale"][key]
    profile.setdefault("review", {})
    profile["review"]["notes"] = (profile["review"].get("notes", "") + " Defaults loaded from --in YAML for type/unit/min/max/scale.").strip()
    return profile


def _sample_width_on_axis(mask: np.ndarray, center: Point, angle_deg: float, distance_r: float, max_half_width: int = 40) -> float:
    h, w = mask.shape[:2]
    cx, cy = center
    rad = math.radians(angle_deg)
    ux = math.cos(rad)
    uy = -math.sin(rad)
    px = -uy
    py = ux
    x0 = cx + distance_r * ux
    y0 = cy + distance_r * uy
    if x0 < 0 or x0 >= w or y0 < 0 or y0 >= h:
        return 0.0
    width = 0
    hit_seen = False
    for sign in (-1, 1):
        for i in range(0, max_half_width + 1):
            x = int(round(x0 + sign * i * px))
            y = int(round(y0 + sign * i * py))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            hit = mask[y, x] > 0
            if hit:
                width += 1
                hit_seen = True
            elif hit_seen and i > 1:
                break
    return float(width)


def make_needle_shape_profile(center: Point, tip: Point, tail: Point, outer_radius: int, meter_img: np.ndarray, needle_prof: Dict[str, Any]) -> Dict[str, Any]:
    tip_len = distance(center, tip)
    tail_len = distance(center, tail)
    tip_angle = normalize_deg(deg_from_center(center, tip))
    tail_angle = normalize_deg(deg_from_center(center, tail))
    radius = max(float(outer_radius), 1.0)
    width_profile = {"base_px": None, "mid_px": None, "tip_px": None}
    width_ratio = {"base": None, "mid": None, "tip": None}
    try:
        mask = make_mask_from_profile(meter_img, needle_prof)
        base_r = max(tail_len * 0.20, tip_len * 0.18)
        mid_r = tip_len * 0.58
        tip_r = tip_len * 0.92
        base_w = _sample_width_on_axis(mask, center, tip_angle, base_r)
        mid_w = _sample_width_on_axis(mask, center, tip_angle, mid_r)
        tip_w = _sample_width_on_axis(mask, center, tip_angle, tip_r)
        width_profile = {"base_px": round(base_w, 2), "mid_px": round(mid_w, 2), "tip_px": round(tip_w, 2)}
        width_ratio = {"base": round(base_w / radius, 4), "mid": round(mid_w / radius, 4), "tip": round(tip_w / radius, 4)}
    except Exception:
        pass
    full_len = tip_len + tail_len
    tip_tail_ratio = tip_len / max(tail_len, 1.0)
    return {
        "source": "human_confirmed",
        "reference_center_xy_in_crop": [int(center[0]), int(center[1])],
        "reference_outer_radius_px": int(outer_radius),
        "tip_xy_in_crop": [int(tip[0]), int(tip[1])],
        "tail_xy_in_crop": [int(tail[0]), int(tail[1])],
        "tip_vector_from_center": {"dx": int(tip[0] - center[0]), "dy": int(tip[1] - center[1])},
        "tail_vector_from_center": {"dx": int(tail[0] - center[0]), "dy": int(tail[1] - center[1])},
        "tip_length_px": round(float(tip_len), 2),
        "tail_length_px": round(float(tail_len), 2),
        "full_length_px": round(float(full_len), 2),
        "tip_tail_length_ratio": round(float(tip_tail_ratio), 4),
        "tip_length_ratio": round(float(tip_len / radius), 4),
        "tail_length_ratio": round(float(tail_len / radius), 4),
        "full_length_ratio": round(float(full_len / radius), 4),
        "axis_angle_deg": round(float(tip_angle), 3),
        "tail_angle_deg": round(float(tail_angle), 3),
        "tip_to_tail_angle_diff_deg": round(float(angle_delta_abs(tip_angle, tail_angle)), 3),
        "width_profile": width_profile,
        "width_ratio": width_ratio,
        "taper_direction": "tip_is_thinner",
        "tail_is_wider": True,
        "read_time_note": "Use length/width/taper as a shape template. Do not reuse the reference angle as the current needle angle.",
    }



# -----------------------------
# Hub candidate selection for profile learning
# -----------------------------



def _circle_edge_support_simple(edges: np.ndarray, center: Point, radius: int, samples: int = 144) -> Dict[str, float]:
    """円周上にどれだけエッジが乗っているかを簡易評価する。"""
    h, w = edges.shape[:2]
    cx, cy = center
    hits = 0
    valid = 0
    for i in range(samples):
        a = 2.0 * math.pi * float(i) / float(samples)
        x = int(round(cx + float(radius) * math.cos(a)))
        y = int(round(cy + float(radius) * math.sin(a)))
        if 1 <= x < w - 1 and 1 <= y < h - 1:
            valid += 1
            patch = edges[y-1:y+2, x-1:x+2]
            if int(np.count_nonzero(patch)) > 0:
                hits += 1
    return {
        "edge_support": float(hits) / float(max(valid, 1)),
        "valid_fraction": float(valid) / float(max(samples, 1)),
    }


def _outer_candidate_feature(candidate: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    geom = profile.get("geometry", {}) or {}
    expected_center = geom.get("auto_center_xy_in_crop") or geom.get("center_xy_in_crop") or profile.get("center_point") or candidate.get("center")
    expected_radius = geom.get("auto_radius_px") or geom.get("radius_px") or candidate.get("radius")
    cx, cy = candidate.get("center", [0, 0])
    r = float(candidate.get("radius", 1))
    ex, ey = int(expected_center[0]), int(expected_center[1])
    er = max(float(expected_radius), 1.0)
    return {
        "radius_ratio_to_profile": round(r / er, 5),
        "radius_error_px": round(abs(r - er), 3),
        "distance_to_profile_center_px": round(math.hypot(cx - ex, cy - ey), 3),
        "edge_support": round(float(candidate.get("edge_support", 0.0)), 5),
        "valid_fraction": round(float(candidate.get("valid_fraction", 0.0)), 5),
    }


def detect_outer_candidates_for_selection(meter_img: np.ndarray, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """外周円候補を複数抽出する。

    v9.14.5:
      - 3点指定済みの dial/bezel 円を、それぞれ独立した候補として先頭に追加する。
      - その上でHough候補を追加し、ユーザーが dial/bezel を個別に選べるようにする。
    """
    h, w = meter_img.shape[:2]
    min_dim = min(w, h)
    gray = cv2.cvtColor(meter_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(gray, 60, 160)
    geom = profile.get("geometry", {}) or {}
    prof_center = geom.get("auto_center_xy_in_crop") or geom.get("center_xy_in_crop") or profile.get("center_point") or [w//2, h//2]
    prof_radius = int(round(float(geom.get("auto_radius_px") or geom.get("radius_px") or min_dim * 0.45)))

    candidates: List[Dict[str, Any]] = []
    pair_model = profile.get("outer_circle_model") or {}

    def append_profile_candidate(source_name: str, center_xy, radius_px, note: str) -> None:
        if center_xy is None or radius_px is None:
            return
        cx, cy = int(center_xy[0]), int(center_xy[1])
        rr = int(round(float(radius_px)))
        sup_local = _circle_edge_support_simple(edges, (cx, cy), int(rr), samples=144)
        candidates.append({
            "source": source_name,
            "center": [cx, cy],
            "radius": int(rr),
            "score": None,
            "edge_support": sup_local["edge_support"],
            "valid_fraction": sup_local["valid_fraction"],
            "note": note,
        })

    # まず、3点指定済みの dial / bezel を候補へ載せる。
    dial_model = pair_model.get("dial_face_circle") or {}
    bezel_model = pair_model.get("bezel_case_circle") or {}
    append_profile_candidate("profile_dial_estimate", dial_model.get("center_xy_in_crop"), dial_model.get("radius_px"), "current YAML/profile dial circle")
    append_profile_candidate("profile_bezel_estimate", bezel_model.get("center_xy_in_crop"), bezel_model.get("radius_px"), "current YAML/profile bezel circle")

    # 併せて、reader が優先基準として使う現在の preferred outer も候補として保持する。
    pc = [int(prof_center[0]), int(prof_center[1])]
    append_profile_candidate("profile_outer_estimate", pc, int(prof_radius), "current YAML/profile preferred outer circle")

    # 複数レンジ・複数param2でHough候補を集める。
    ranges = []
    ranges.append((int(max(8, prof_radius * 0.75)), int(max(12, prof_radius * 1.30)), "profile_radius"))
    ranges.append((int(min_dim * 0.28), int(min_dim * 0.55), "normal"))
    ranges.append((int(min_dim * 0.22), int(min_dim * 0.65), "wide"))
    seen_ranges = set()
    param2_values = [44, 38, 32, 26]
    for lo, hi, label in ranges:
        lo = max(6, min(lo, int(min_dim * 0.70)))
        hi = max(lo + 3, min(hi, int(min_dim * 0.72)))
        if (lo, hi, label) in seen_ranges:
            continue
        seen_ranges.add((lo, hi, label))
        for param2 in param2_values:
            circles = cv2.HoughCircles(
                blur,
                cv2.HOUGH_GRADIENT,
                dp=1.25,
                minDist=max(20, min_dim // 3),
                param1=90,
                param2=param2,
                minRadius=lo,
                maxRadius=hi,
            )
            if circles is None:
                continue
            circles = np.round(circles[0, :]).astype("int")
            for x, y, r in circles[:80]:
                x, y, r = int(x), int(y), int(r)
                # 中心が大きく外れた巨大仮想円は避ける。ただし手選択用なので少し緩め。
                dist_to_img_center = math.hypot(x - w/2, y - h/2)
                if dist_to_img_center > max(min_dim * 0.28, 90.0):
                    continue
                if r < min_dim * 0.20 or r > min_dim * 0.70:
                    continue
                sup = _circle_edge_support_simple(edges, (x, y), r, samples=144)
                if sup["valid_fraction"] < 0.65 or sup["edge_support"] < 0.04:
                    continue
                radius_error = abs(float(r) - float(prof_radius))
                center_error = math.hypot(x - pc[0], y - pc[1])
                score = sup["edge_support"] * 420.0 + sup["valid_fraction"] * 90.0 - radius_error * 0.85 - center_error * 0.35
                candidates.append({
                    "source": "hough_outer",
                    "center": [x, y],
                    "radius": r,
                    "score": round(float(score), 3),
                    "edge_support": sup["edge_support"],
                    "valid_fraction": sup["valid_fraction"],
                    "range_label": f"{label}/param2={param2}",
                    "note": "HoughCircles outer candidate",
                })

    # 重複整理。近い候補はスコア/edge_supportの良い方を残す。
    candidates_sorted = sorted(candidates, key=lambda c: (c.get("score") is not None, float(c.get("score") or -1e9), float(c.get("edge_support", 0.0))), reverse=True)
    uniq: List[Dict[str, Any]] = []
    for cand in candidates_sorted:
        cx, cy = cand["center"]
        r = int(cand["radius"])
        duplicate = False
        for u in uniq:
            ux, uy = u["center"]
            ur = int(u["radius"])
            if math.hypot(cx - ux, cy - uy) < max(8, min(r, ur) * 0.04) and abs(r - ur) < max(8, min(r, ur) * 0.04):
                duplicate = True
                break
        if duplicate:
            continue
        cand["features"] = _outer_candidate_feature(cand, profile)
        uniq.append(cand)
        if len(uniq) >= 12:
            break

    # 最後にプロファイル候補を先頭に戻す。ユーザーが今までの値を選びやすくする。
    uniq.sort(key=lambda c: 0 if c.get("source") in ("profile_dial_estimate", "profile_bezel_estimate", "profile_outer_estimate") else 1)
    for i, cand in enumerate(uniq, start=1):
        cand["index"] = i
    return uniq


def draw_outer_candidates_for_selection(base: np.ndarray, candidates: List[Dict[str, Any]], selected_index: int, display_scale: float, show_candidates: bool = True) -> np.ndarray:
    out = base.copy()
    if not show_candidates:
        return out
    for i, cand in enumerate(candidates):
        src = cand.get("source", "unknown")
        if src == "profile_outer_estimate":
            color = (0, 0, 255)  # red
            prefix = "P"
        else:
            color = (160, 160, 160)  # gray
            prefix = "O"
        if i == selected_index:
            color = (0, 165, 255)  # orange/yellow-ish selected
        cx, cy = to_display_point(tuple(cand["center"]), display_scale)
        rr = int(round(float(cand["radius"]) * display_scale))
        thick = 4 if i == selected_index else 2
        cv2.circle(out, (cx, cy), max(2, rr), color, thick)
        cv2.drawMarker(out, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
        feat = cand.get("features", {})
        rr_prof = feat.get("radius_ratio_to_profile", "-")
        label = f"{i+1}:{prefix} r={cand['radius']} rr={rr_prof}"
        cv2.putText(out, label, (cx + 8, max(18, cy - rr - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return out


def interactive_select_outer_candidate(meter_img: np.ndarray, profile: Dict[str, Any], args) -> Dict[str, Any]:
    """外周円候補を全表示し、ユーザーが正しい外周円を選択してYAMLへ保存する。"""
    candidates = detect_outer_candidates_for_selection(meter_img, profile)
    if not candidates:
        print("outer candidate selection skipped: no candidates")
        return profile
    h, w = meter_img.shape[:2]
    base_display = resize_for_display(meter_img, args.display_scale)
    # Hough候補があれば最良スコア候補を初期選択。なければprofile候補。
    selected = 0
    best_score = -1e18
    for i, cand in enumerate(candidates):
        if cand.get("score") is not None and float(cand.get("score")) > best_score:
            selected = i
            best_score = float(cand.get("score"))
    show_candidates = True

    def print_candidates():
        print("\n=== Outer circle candidates ===")
        print("P=profile/current estimate / O=Hough outer candidate / selected=orange")
        for i, c in enumerate(candidates):
            mark = "*" if i == selected else " "
            print(f"{mark} [{i+1}] source={c.get('source')} center={c.get('center')} r={c.get('radius')} score={c.get('score')} features={c.get('features')} range={c.get('range_label')}")

    def on_mouse(event, x, y, flags, param):
        nonlocal selected
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox, oy = to_original_point(x, y, args.display_scale, w, h)
        # 円周に近い候補を選ぶ。中心クリックでも選べるよう中心距離も少し見る。
        best_i, best_d = selected, 1e9
        for i, cand in enumerate(candidates):
            cx, cy = cand["center"]
            r = float(cand["radius"])
            d_ring = abs(math.hypot(ox - cx, oy - cy) - r)
            d_center = math.hypot(ox - cx, oy - cy) * 0.15
            d = min(d_ring, d_center)
            if d < best_d:
                best_i, best_d = i, d
        selected = best_i
        print_candidates()

    cv2.namedWindow("outer_candidate_select_v9_7", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("outer_candidate_select_v9_7", base_display.shape[1], base_display.shape[0])
    cv2.setMouseCallback("outer_candidate_select_v9_7", on_mouse)
    print("\n外周円候補を表示します。o=候補表示ON/OFF、n/Tab=次、p=前、数字=候補選択、ENTER=採用、s=スキップ、ESC=中止")
    print_candidates()
    while True:
        frame = draw_outer_candidates_for_selection(base_display, candidates, selected, args.display_scale, show_candidates)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 86), (0, 0, 0), -1)
        cv2.putText(frame, f"Outer Select: o overlay={'ON' if show_candidates else 'OFF'} | n/Tab next | p prev | 1-9 select | ENTER accept | s skip | ESC cancel", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        c = candidates[selected]
        cv2.putText(frame, f"Selected {selected+1}: {c.get('source')} center={c.get('center')} r={c.get('radius')} rr={c.get('features',{}).get('radius_ratio_to_profile')}", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
        cv2.imshow("outer_candidate_select_v9_7", frame)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow("outer_candidate_select_v9_7")
            raise RuntimeError("Outer candidate selection canceled")
        if key in (ord('s'), ord('S')):
            cv2.destroyWindow("outer_candidate_select_v9_7")
            print("outer candidate selection skipped by user")
            return profile
        if key in (10, 13):
            break
        if key in (ord('o'), ord('O')):
            show_candidates = not show_candidates
            print(f"outer_candidate_overlay={show_candidates}")
        if key in (9, ord('n'), ord('N')):
            selected = (selected + 1) % len(candidates)
            print_candidates()
        if key in (ord('p'), ord('P')):
            selected = (selected - 1) % len(candidates)
            print_candidates()
        if ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(candidates):
                selected = idx
                print_candidates()
    cv2.destroyWindow("outer_candidate_select_v9_7")

    chosen = candidates[selected]
    profile.setdefault("geometry", {})
    profile["geometry"]["auto_center_xy_in_crop"] = [int(chosen["center"][0]), int(chosen["center"][1])]
    profile["geometry"]["auto_radius_px"] = int(chosen["radius"])
    # radius_pxはMIN/MAXクリックで作った読取目盛り半径として残す。外周参照はauto_radius_pxを優先利用する。
    profile["geometry"]["outer_selection_source"] = "human_selected_candidate"
    profile["geometry"]["outer_selection_candidate_source"] = chosen.get("source")

    profile["outer_selection_model"] = {
        "source": "human_selected_candidate",
        "selected_candidate_index": int(selected + 1),
        "selected_candidate_type": chosen.get("source"),
        "selected_center_xy_in_crop": [int(chosen["center"][0]), int(chosen["center"][1])],
        "selected_radius_px": int(chosen["radius"]),
        "selection_features": chosen.get("features") or _outer_candidate_feature(chosen, profile),
        "scoring_weights": {
            "source_match_bonus": 10.0,
            "preferred_radius_ratio_weight": 85.0,
            "preferred_edge_support_weight": 120.0,
            "preferred_center_distance_weight": 0.55,
        },
        "notes": "Created by outer circle candidate selection UI. Candidate is used as profile outer reference for future scaling.",
    }
    profile.setdefault("review", {})
    profile["review"]["outer_selection_note"] = "Outer circle candidate selected by user and saved as outer_selection_model. geometry.auto_radius_px is used as outer reference."
    print(f"selected outer candidate: index={selected+1} source={chosen.get('source')} center={chosen.get('center')} r={chosen.get('radius')}")
    return profile



def _make_outer_circle_model(profile: Dict[str, Any], dial_cand: Dict[str, Any], bezel_cand: Dict[str, Any]) -> Dict[str, Any]:
    dial_r = max(float(dial_cand.get("radius", 0)), 1.0)
    bezel_r = max(float(bezel_cand.get("radius", 0)), 1.0)
    dc = tuple(map(int, dial_cand.get("center", (0, 0))))
    bc = tuple(map(int, bezel_cand.get("center", (0, 0))))
    center_offset = math.hypot(float(bc[0] - dc[0]), float(bc[1] - dc[1]))
    color_model = profile.get("outer_color_model") or {}
    colors_similar = bool(color_model.get("colors_similar", False))
    preferred = "bezel_case_circle" if colors_similar else "dial_face_circle"
    return {
        "source": "human_selected_dial_and_bezel_circles",
        "dial_face_circle": {
            "center_xy_in_crop": [int(dc[0]), int(dc[1])],
            "radius_px": int(round(dial_r)),
            "candidate_source": dial_cand.get("source"),
            "features": dial_cand.get("features") or _outer_candidate_feature(dial_cand, profile),
        },
        "bezel_case_circle": {
            "center_xy_in_crop": [int(bc[0]), int(bc[1])],
            "radius_px": int(round(bezel_r)),
            "candidate_source": bezel_cand.get("source"),
            "features": bezel_cand.get("features") or _outer_candidate_feature(bezel_cand, profile),
        },
        "bezel_to_dial_radius_ratio": round(bezel_r / dial_r, 6),
        "dial_to_bezel_radius_ratio": round(dial_r / bezel_r, 6),
        "center_offset_px": round(center_offset, 3),
        "center_offset_over_dial_radius": round(center_offset / dial_r, 6),
        "colors_similar": colors_similar,
        "preferred_outer_reference": preferred,
        "notes": "Reader should detect both dial-face and bezel/case circles, classify candidates by radius ratio and color, and use preferred_outer_reference for scale.",
    }


def draw_outer_pair_candidates(base_display: np.ndarray, candidates: List[Dict[str, Any]], dial_index: int, bezel_index: int, display_scale: float, show_candidates: bool) -> np.ndarray:
    out = base_display.copy()
    if not show_candidates:
        return out
    for i, cand in enumerate(candidates):
        src = cand.get("source", "")
        color = (160, 160, 160)
        prefix = "O"
        thick = 2
        if i == dial_index:
            color = (0, 165, 255)  # orange selected dial
            prefix = "D"
            thick = 4
        if i == bezel_index:
            color = (255, 0, 255)  # magenta selected bezel
            prefix = "B"
            thick = 4
        if i == dial_index and i == bezel_index:
            color = (0, 255, 255)
            prefix = "D/B"
            thick = 4
        cx, cy = to_display_point(tuple(cand["center"]), display_scale)
        rr = int(round(float(cand["radius"]) * display_scale))
        cv2.circle(out, (cx, cy), max(2, rr), color, thick)
        cv2.drawMarker(out, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
        feat = cand.get("features", {})
        rr_prof = feat.get("radius_ratio_to_profile", cand.get("radius_ratio_to_profile", "-"))
        label = f"{i+1}:{prefix} r={cand['radius']} rr={rr_prof}"
        cv2.putText(out, label, (cx + 8, max(18, cy - rr - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return out


def interactive_select_outer_pair_model(meter_img: np.ndarray, profile: Dict[str, Any], args) -> Dict[str, Any]:
    """白文字盤円とベゼル/ケース円の両方をユーザー選択し、半径比をYAMLへ保存する。"""
    candidates = detect_outer_candidates_for_selection(meter_img, profile)
    if not candidates:
        print("outer pair selection skipped: no candidates")
        return profile
    h, w = meter_img.shape[:2]
    base_display = resize_for_display(meter_img, args.display_scale)
    # 初期値: 小さめを文字盤、大きめをベゼル候補とする。
    sorted_idx = sorted(range(len(candidates)), key=lambda i: float(candidates[i].get("radius", 0)))
    dial_index = next((i for i, c in enumerate(candidates) if c.get("source") == "profile_dial_estimate"), sorted_idx[0])
    bezel_index = next((i for i, c in enumerate(candidates) if c.get("source") == "profile_bezel_estimate"), sorted_idx[-1])
    active = "dial"
    show_candidates = True

    def print_candidates():
        print("\n=== Outer pair candidates ===")
        print("D=selected dial-face circle / B=selected bezel-case circle / o=overlay ON/OFF")
        for i, c in enumerate(candidates):
            marks = []
            if i == dial_index:
                marks.append("D")
            if i == bezel_index:
                marks.append("B")
            mark = "/".join(marks) if marks else " "
            print(f" {mark:3s} [{i+1}] source={c.get('source')} center={c.get('center')} r={c.get('radius')} score={c.get('score')} features={c.get('features')} range={c.get('range_label')}")
        dr = float(candidates[dial_index].get("radius", 1))
        br = float(candidates[bezel_index].get("radius", 1))
        print(f" current ratio bezel/dial={br/max(dr,1):.4f} active={active}")

    def nearest_candidate(ox, oy):
        best_i, best_d = 0, 1e9
        for i, cand in enumerate(candidates):
            cx, cy = cand["center"]
            r = float(cand["radius"])
            d_ring = abs(math.hypot(ox - cx, oy - cy) - r)
            d_center = math.hypot(ox - cx, oy - cy) * 0.15
            d = min(d_ring, d_center)
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    def on_mouse(event, x, y, flags, param):
        nonlocal dial_index, bezel_index
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox, oy = to_original_point(x, y, args.display_scale, w, h)
        idx = nearest_candidate(ox, oy)
        if active == "dial":
            dial_index = idx
        else:
            bezel_index = idx
        print_candidates()

    win = "outer_pair_select_v9_11"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, base_display.shape[1], base_display.shape[0])
    cv2.setMouseCallback(win, on_mouse)
    print("\n白文字盤円(DIAL)とベゼル/ケース円(BEZEL)を選択します。")
    print("d=文字盤円を選択対象 / b=ベゼル円を選択対象 / o=候補表示ON/OFF / n,p=選択中対象を移動 / 1-9=候補番号 / ENTER=採用 / s=スキップ / ESC=中止")
    print_candidates()
    while True:
        frame = draw_outer_pair_candidates(base_display, candidates, dial_index, bezel_index, args.display_scale, show_candidates)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 108), (0, 0, 0), -1)
        cv2.putText(frame, f"Outer Pair: active={active.upper()} | d dial | b bezel | o overlay={'ON' if show_candidates else 'OFF'} | n/p | 1-9 | ENTER accept", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2)
        d = candidates[dial_index]
        b = candidates[bezel_index]
        ratio = float(b.get('radius',1)) / max(float(d.get('radius',1)), 1.0)
        cv2.putText(frame, f"DIAL #{dial_index+1} r={d.get('radius')}  BEZEL #{bezel_index+1} r={b.get('radius')}  bezel/dial={ratio:.3f}", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1)
        cv2.putText(frame, "DIAL=orange / BEZEL=magenta", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
        cv2.imshow(win, frame)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(win)
            raise RuntimeError("Outer pair selection canceled")
        if key in (ord('s'), ord('S')):
            cv2.destroyWindow(win)
            print("outer pair selection skipped by user")
            return profile
        if key in (10, 13):
            break
        if key in (ord('o'), ord('O')):
            show_candidates = not show_candidates
            print(f"outer_pair_overlay={show_candidates}")
        if key in (ord('d'), ord('D')):
            active = "dial"
            print_candidates()
        if key in (ord('b'), ord('B')):
            active = "bezel"
            print_candidates()
        if key in (9, ord('n'), ord('N')):
            if active == "dial":
                dial_index = (dial_index + 1) % len(candidates)
            else:
                bezel_index = (bezel_index + 1) % len(candidates)
            print_candidates()
        if key in (ord('p'), ord('P')):
            if active == "dial":
                dial_index = (dial_index - 1) % len(candidates)
            else:
                bezel_index = (bezel_index - 1) % len(candidates)
            print_candidates()
        if ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(candidates):
                if active == "dial":
                    dial_index = idx
                else:
                    bezel_index = idx
                print_candidates()
    cv2.destroyWindow(win)

    dial = candidates[dial_index]
    bezel = candidates[bezel_index]
    model = _make_outer_circle_model(profile, dial, bezel)
    profile["outer_circle_model"] = model
    # preferred referenceをgeometryへ反映。通常は白文字盤円、色が似ている場合はベゼル円。
    preferred = model.get("preferred_outer_reference", "dial_face_circle")
    chosen = bezel if preferred == "bezel_case_circle" else dial
    profile.setdefault("geometry", {})
    profile["geometry"]["auto_center_xy_in_crop"] = [int(chosen["center"][0]), int(chosen["center"][1])]
    profile["geometry"]["auto_radius_px"] = int(chosen["radius"])
    profile["geometry"]["outer_pair_selection_source"] = "human_selected_dial_and_bezel_circles"
    profile["geometry"]["outer_pair_preferred_reference"] = preferred
    profile.setdefault("review", {})["outer_pair_selection_note"] = "Dial-face and bezel/case outer circles were selected by user and saved as outer_circle_model."
    print(f"selected outer pair: dial#{dial_index+1} center={dial.get('center')} r={dial.get('radius')} / bezel#{bezel_index+1} center={bezel.get('center')} r={bezel.get('radius')} ratio={model.get('bezel_to_dial_radius_ratio')}")
    return profile

def _hub_candidate_feature(candidate: Dict[str, Any], profile_center: Point, expected_radius: float, outer_radius: float) -> Dict[str, Any]:
    cx, cy = candidate["center"]
    r = float(candidate["radius"])
    dx = float(cx - profile_center[0])
    dy = float(cy - profile_center[1])
    outer = max(float(outer_radius), 1.0)
    exp_r = max(float(expected_radius), 1.0)
    return {
        "radius_ratio": round(r / exp_r, 4),
        "hub_radius_over_outer_radius": round(r / outer, 5),
        "distance_to_profile_center_px": round(math.hypot(dx, dy), 3),
        "distance_to_profile_center_over_outer_radius": round(math.hypot(dx, dy) / outer, 5),
        "offset_from_profile_center_ratio": {
            "dx_over_outer_radius": round(dx / outer, 5),
            "dy_over_outer_radius": round(dy / outer, 5),
        },
        "candidate_source": candidate.get("source", "unknown"),
    }


def detect_hub_candidates_for_selection(meter_img: np.ndarray, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Profile作成時に、ハブ候補を複数出してユーザー選択できるようにする。"""
    h, w = meter_img.shape[:2]
    hub_cfg = profile.get("hub", {}) or {}
    geom = profile.get("geometry", {}) or {}
    center = tuple(map(int, hub_cfg.get("center_xy_in_crop") or profile.get("center_point") or geom.get("center_xy_in_crop") or (w // 2, h // 2)))
    expected_radius = float(hub_cfg.get("radius_px") or max(6, min(w, h) * 0.04))
    outer_radius = float(geom.get("radius_px") or geom.get("auto_radius_px") or min(w, h) * 0.45)

    candidates: List[Dict[str, Any]] = []
    candidates.append({
        "source": "profile_guided_estimate",
        "center": [int(center[0]), int(center[1])],
        "radius": int(round(expected_radius)),
        "note": "YAML/profile hub estimate",
    })

    gray = cv2.cvtColor(meter_img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((h, w), dtype=np.uint8)
    search_r = int(max(expected_radius * 2.4, outer_radius * 0.12, 32))
    cv2.circle(mask, center, search_r, 255, -1)
    roi_gray = cv2.bitwise_and(gray, gray, mask=mask)
    blur = cv2.GaussianBlur(roi_gray, (7, 7), 1.5)
    min_r = max(3, int(expected_radius * 0.55))
    max_r = max(min_r + 2, int(expected_radius * 1.75))
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(8, search_r // 3),
        param1=80,
        param2=16,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles[:12]:
            if math.hypot(x - center[0], y - center[1]) > search_r:
                continue
            cand = {"source": "hough_hub", "center": [int(x), int(y)], "radius": int(r), "note": "HoughCircles"}
            candidates.append(cand)

    # center color mask由来候補も追加
    try:
        center_cfg = profile.get("center", {})
        center_mask = make_mask_from_profile(meter_img, center_cfg)
        center_mask = cv2.bitwise_and(center_mask, mask)
        contours, _ = cv2.findContours(center_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)
            if math.hypot(x - center[0], y - center[1]) > search_r:
                continue
            if r < min_r or r > max_r:
                continue
            candidates.append({
                "source": "center_color_mask",
                "center": [int(round(x)), int(round(y))],
                "radius": int(round(max(r, 1))),
                "area": round(float(area), 2),
                "note": "center color mask",
            })
    except Exception as e:
        print(f"center_color_mask hub candidate skipped: {e}")

    # 重複を軽く整理
    uniq: List[Dict[str, Any]] = []
    for cand in candidates:
        c = tuple(cand["center"])
        r = int(cand["radius"])
        duplicate = False
        for u in uniq:
            if cand.get("source") == u.get("source") and math.hypot(c[0]-u["center"][0], c[1]-u["center"][1]) < 3 and abs(r-int(u["radius"])) < 3:
                duplicate = True
                break
        if not duplicate:
            cand["features"] = _hub_candidate_feature(cand, center, expected_radius, outer_radius)
            uniq.append(cand)
    return uniq


def draw_hub_candidates_for_selection(base: np.ndarray, candidates: List[Dict[str, Any]], selected_index: int, display_scale: float, show_candidates: bool = True) -> np.ndarray:
    out = base.copy()
    if not show_candidates:
        return out
    colors = {
        "profile_guided_estimate": (0, 0, 255),    # red
        "hough_hub": (0, 255, 0),                  # green
        "center_color_mask": (255, 255, 0),        # cyan
    }
    for i, cand in enumerate(candidates):
        src = cand.get("source", "unknown")
        color = colors.get(src, (255, 255, 255))
        cx, cy = to_display_point(tuple(cand["center"]), display_scale)
        rr = int(round(float(cand["radius"]) * display_scale))
        thick = 4 if i == selected_index else 2
        if i == selected_index:
            color = (0, 255, 255)  # selected: yellow
        cv2.circle(out, (cx, cy), max(2, rr), color, thick)
        cv2.drawMarker(out, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
        label_prefix = {"profile_guided_estimate": "P", "hough_hub": "H", "center_color_mask": "C"}.get(src, "X")
        label = f"{i+1}:{label_prefix} r={cand['radius']} rr={cand.get('features',{}).get('radius_ratio','-')}"
        cv2.putText(out, label, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def interactive_select_hub_candidate(meter_img: np.ndarray, profile: Dict[str, Any], args) -> Dict[str, Any]:
    """ハブ候補を全表示し、ユーザーが正しい候補を選択してYAMLへ学習情報を保存する。"""
    candidates = detect_hub_candidates_for_selection(meter_img, profile)
    if not candidates:
        print("hub candidate selection skipped: no candidates")
        return profile
    h, w = meter_img.shape[:2]
    base_display = resize_for_display(meter_img, args.display_scale)
    selected = 0
    show_candidates = True

    def print_candidates():
        print("\n=== Hub candidates ===")
        print("P=profile estimate / H=HoughCircles / C=center color mask / selected=yellow")
        for i, c in enumerate(candidates):
            mark = "*" if i == selected else " "
            print(f"{mark} [{i+1}] source={c.get('source')} center={c.get('center')} r={c.get('radius')} features={c.get('features')}")

    def on_mouse(event, x, y, flags, param):
        nonlocal selected
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox, oy = to_original_point(x, y, args.display_scale, w, h)
        best_i, best_d = selected, 1e9
        for i, cand in enumerate(candidates):
            d = math.hypot(ox - cand["center"][0], oy - cand["center"][1])
            if d < best_d:
                best_i, best_d = i, d
        selected = best_i
        print_candidates()

    cv2.namedWindow("hub_candidate_select_v9_2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("hub_candidate_select_v9_2", base_display.shape[1], base_display.shape[0])
    cv2.setMouseCallback("hub_candidate_select_v9_2", on_mouse)
    print("\nハブ候補を表示します。o=候補表示ON/OFF、n/Tab=次、p=前、数字=候補選択、ENTER=採用、s=スキップ、ESC=中止")
    print_candidates()
    while True:
        frame = draw_hub_candidates_for_selection(base_display, candidates, selected, args.display_scale, show_candidates)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Hub Select: o overlay={'ON' if show_candidates else 'OFF'} | n/Tab next | p prev | 1-9 select | ENTER accept | s skip | ESC cancel", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        c = candidates[selected]
        cv2.putText(frame, f"Selected {selected+1}: {c.get('source')} center={c.get('center')} r={c.get('radius')} rr={c.get('features',{}).get('radius_ratio')}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
        cv2.imshow("hub_candidate_select_v9_2", frame)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow("hub_candidate_select_v9_2")
            raise RuntimeError("Hub candidate selection canceled")
        if key in (ord('s'), ord('S')):
            cv2.destroyWindow("hub_candidate_select_v9_2")
            print("hub candidate selection skipped by user")
            return profile
        if key in (10, 13):
            break
        if key in (ord('o'), ord('O')):
            show_candidates = not show_candidates
            print(f"hub_candidate_overlay={show_candidates}")
        if key in (9, ord('n'), ord('N')):
            selected = (selected + 1) % len(candidates)
            print_candidates()
        if key in (ord('p'), ord('P')):
            selected = (selected - 1) % len(candidates)
            print_candidates()
        if ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(candidates):
                selected = idx
                print_candidates()
    cv2.destroyWindow("hub_candidate_select_v9_2")

    chosen = candidates[selected]
    old_hub_center = tuple(map(int, profile.get("hub", {}).get("center_xy_in_crop", profile.get("center_point", chosen["center"]))))
    outer_radius = float(profile.get("geometry", {}).get("radius_px") or profile.get("geometry", {}).get("auto_radius_px") or max(1, min(meter_img.shape[:2]) * 0.45))
    features = chosen.get("features") or _hub_candidate_feature(chosen, old_hub_center, float(profile.get("hub", {}).get("radius_px", chosen["radius"])), outer_radius)

    profile.setdefault("hub", {})
    profile["hub"]["center_xy_in_crop"] = [int(chosen["center"][0]), int(chosen["center"][1])]
    profile["hub"]["radius_px"] = int(chosen["radius"])
    profile["hub"]["diameter_px"] = int(chosen["radius"] * 2)
    profile["hub"]["source"] = "human_selected_candidate"
    profile["hub"]["selection_source"] = chosen.get("source")

    # edge point is kept consistent with selected radius, to the right of center.
    profile["hub"]["edge_xy_in_crop"] = [int(chosen["center"][0] + chosen["radius"]), int(chosen["center"][1])]

    profile["hub_selection_model"] = {
        "source": "human_selected_candidate",
        "selected_candidate_index": int(selected + 1),
        "selected_candidate_type": chosen.get("source"),
        "selected_center_xy_in_crop": [int(chosen["center"][0]), int(chosen["center"][1])],
        "selected_radius_px": int(chosen["radius"]),
        "selection_features": features,
        "scoring_weights": {
            "source_match_bonus": 18.0,
            "preferred_radius_ratio_weight": 45.0,
            "preferred_radius_over_outer_weight": 220.0,
            "preferred_offset_weight": 320.0,
            "needle_line_weight_scale": 0.45,
            "profile_distance_weight_scale": 0.85,
        },
        "notes": "Created by hub candidate selection UI. Candidate features are used to weight future hub selection.",
    }
    profile.setdefault("review", {})
    profile["review"]["hub_selection_note"] = "Hub candidate selected by user and saved as hub_selection_model."
    print(f"selected hub candidate: index={selected+1} source={chosen.get('source')} center={chosen.get('center')} r={chosen.get('radius')}")
    return profile



def _profile_rgb_mean_tuple(prof: Dict[str, Any]) -> Tuple[float, float, float]:
    rgb = prof.get("rgb_mean", {}) if isinstance(prof, dict) else {}
    return (float(rgb.get("r", 0.0)), float(rgb.get("g", 0.0)), float(rgb.get("b", 0.0)))


def _build_outer_color_model(dial_prof: Dict[str, Any], bezel_prof: Dict[str, Any]) -> Dict[str, Any]:
    dr, dg, db = _profile_rgb_mean_tuple(dial_prof)
    br, bg, bb = _profile_rgb_mean_tuple(bezel_prof)
    rgb_dist = math.sqrt((dr - br) ** 2 + (dg - bg) ** 2 + (db - bb) ** 2)
    dial_gray = float(dial_prof.get("gray_mean", 0.0))
    bezel_gray = float(bezel_prof.get("gray_mean", 0.0))
    gray_diff = abs(dial_gray - bezel_gray)
    # 色差が小さい場合は、白文字盤円とベゼル/ケース円を色で分けにくい。
    # その場合はフラグを立て、読取側ではベゼル/ケース円を外周基準として許容・優先できるようにする。
    similar = bool(rgb_dist < 45.0 and gray_diff < 35.0)
    return {
        "source": "human_clicked_dial_and_bezel_color",
        "dial_face_profile": dial_prof,
        "bezel_case_profile": bezel_prof,
        "rgb_distance": round(float(rgb_dist), 3),
        "gray_difference": round(float(gray_diff), 3),
        "colors_similar": similar,
        "preferred_outer_reference": "bezel_case_circle" if similar else "dial_face_circle",
        "notes": "If colors_similar is false, reader prefers a circle with dial color inside and bezel/case color outside. If true, reader may use bezel/case circle because color separation is unreliable.",
    }

def _ratio_model_from_component_points(center: Point, hub_edge: Point, dial_edge: Point, bezel_edge: Point, tip_pt: Point, tail_pt: Point) -> Dict[str, Any]:
    hub_r = max(1.0, distance(center, hub_edge))
    dial_r = max(1.0, distance(center, dial_edge))
    bezel_r = max(1.0, distance(center, bezel_edge))
    tip_len = max(1.0, distance(center, tip_pt))
    tail_len = max(1.0, distance(center, tail_pt))
    vals = {
        "hub_radius": hub_r,
        "dial_radius": dial_r,
        "bezel_radius": bezel_r,
        "needle_tip_length": tip_len,
        "needle_tail_length": tail_len,
    }
    base = hub_r
    return {
        "source": "human_clicked_component_points",
        "basis": "same_center_circular_dial_gauge",
        "raw_px": {k + "_px": round(float(v), 3) for k, v in vals.items()},
        "ratio_to_hub_radius": {k: round(float(v) / base, 6) for k, v in vals.items()},
        "ratio_to_dial_radius": {k: round(float(v) / dial_r, 6) for k, v in vals.items()},
        "bezel_to_dial_radius_ratio": round(bezel_r / dial_r, 6),
        "dial_to_bezel_radius_ratio": round(dial_r / bezel_r, 6),
        "hub_to_dial_radius_ratio": round(hub_r / dial_r, 6),
        "needle_tip_to_dial_radius_ratio": round(tip_len / dial_r, 6),
        "needle_tail_to_dial_radius_ratio": round(tail_len / dial_r, 6),
        "reliability": "highest",
        "notes": "This component ratio model is treated as the most reliable scale relationship for dial gauge reading.",
    }


def create_profile(meter_img: np.ndarray, image_path: str, bbox: List[int], yolo_conf: float, profile_name: str, points: List[Point], needle_prof: Dict[str, Any], center_prof: Dict[str, Any], candidate: Dict[str, Any], detector_name: str = "full_image_sample", detection_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    auto_center, auto_radius, center_conf = detect_meter_center_hough(meter_img)
    center = points[0]
    hub_edge = points[1]
    dial_circle = _circle_or_fallback(points, (2, 3, 4), center, points[2])
    bezel_circle = _circle_or_fallback(points, (5, 6, 7), center, points[5])
    needle_tip_pt = points[8]
    needle_tail_pt = points[9]
    min_pt = points[10]
    mid_pt = points[11]
    max_pt = points[12]
    needle_color_pts = [points[13], points[14]]
    hub_center_color_pts = [points[15]]
    hub_edge_color_pts = [points[16]]
    hub_color_pts = [points[15], points[16]]
    dial_color_pts = [points[17], points[18], points[19]]
    bezel_color_pts = [points[20], points[21], points[22]]

    needle_prof = color_profile_from_points(meter_img, needle_color_pts, 8, 12, 35, 35, 3.0, 35) if not needle_prof else needle_prof
    hub_center_prof = color_profile_from_points(meter_img, hub_center_color_pts, 8, 12, 35, 35, 3.0, 35)
    hub_edge_prof = color_profile_from_points(meter_img, hub_edge_color_pts, 8, 12, 35, 35, 3.0, 35)
    hub_prof = color_profile_from_points(meter_img, hub_color_pts, 8, 12, 35, 35, 3.0, 35)
    hub_prof = {**hub_prof, "mode": "dual_or", "primary": hub_center_prof, "secondary": hub_edge_prof, "notes": "primary is flat hub center color; secondary is sloped/outer hub color"}
    dial_prof = color_profile_from_points(meter_img, dial_color_pts, 8, 12, 35, 35, 3.0, 35)
    bezel_prof = color_profile_from_points(meter_img, bezel_color_pts, 8, 12, 35, 35, 3.0, 35)
    outer_color_model = _build_outer_color_model(dial_prof, bezel_prof)

    min_angle = normalize_deg(deg_from_center(center, min_pt))
    mid_angle = normalize_deg(deg_from_center(center, mid_pt))
    max_angle = normalize_deg(deg_from_center(center, max_pt))
    hub_radius = max(1, int(round(distance(center, hub_edge))))
    dial_center = tuple(map(int, dial_circle["center"]))
    bezel_center = tuple(map(int, bezel_circle["center"]))
    dial_radius = max(1, int(round(float(dial_circle["radius"]))))
    bezel_radius = max(1, int(round(float(bezel_circle["radius"]))))
    # 比率は高信頼情報として保持。ただし、円中心は各円ごとに保持し、中心ズレ比も保存する。
    tip_len = distance(center, needle_tip_pt)
    tail_len = distance(center, needle_tail_pt)
    component_ratio_model = {
        "source": "human_3point_dial_bezel_and_component_points",
        "reliability": "highest",
        "ratio_to_hub_radius": {
            "hub_radius": 1.0,
            "dial_radius": round(float(dial_radius) / max(float(hub_radius), 1.0), 6),
            "bezel_radius": round(float(bezel_radius) / max(float(hub_radius), 1.0), 6),
            "needle_tip_length": round(float(tip_len) / max(float(hub_radius), 1.0), 6),
            "needle_tail_length": round(float(tail_len) / max(float(hub_radius), 1.0), 6),
        },
        "bezel_to_dial_radius_ratio": round(float(bezel_radius) / max(float(dial_radius), 1.0), 6),
        "dial_to_bezel_radius_ratio": round(float(dial_radius) / max(float(bezel_radius), 1.0), 6),
        "hub_to_dial_radius_ratio": round(float(hub_radius) / max(float(dial_radius), 1.0), 6),
        "needle_tip_to_dial_radius_ratio": round(float(tip_len) / max(float(dial_radius), 1.0), 6),
        "needle_tail_to_dial_radius_ratio": round(float(tail_len) / max(float(dial_radius), 1.0), 6),
        "center_offsets": {
            "dial_center_from_hub_px": [int(dial_center[0] - center[0]), int(dial_center[1] - center[1])],
            "bezel_center_from_hub_px": [int(bezel_center[0] - center[0]), int(bezel_center[1] - center[1])],
            "dial_center_from_hub_over_dial_radius": [round((dial_center[0] - center[0]) / max(float(dial_radius), 1.0), 6), round((dial_center[1] - center[1]) / max(float(dial_radius), 1.0), 6)],
            "bezel_center_from_hub_over_bezel_radius": [round((bezel_center[0] - center[0]) / max(float(bezel_radius), 1.0), 6), round((bezel_center[1] - center[1]) / max(float(bezel_radius), 1.0), 6)],
            "dial_bezel_center_distance_over_dial_radius": round(distance(dial_center, bezel_center) / max(float(dial_radius), 1.0), 6),
        },
    }
    colors_similar = bool(outer_color_model.get("colors_similar", False))
    preferred_outer_reference = "bezel_case_circle" if colors_similar else "dial_face_circle"
    preferred_circle = bezel_circle if preferred_outer_reference == "bezel_case_circle" else dial_circle
    preferred_center = tuple(map(int, preferred_circle["center"]))
    preferred_radius = int(round(float(preferred_circle["radius"])))

    detection_meta = detection_meta or {}
    crop_w = int(meter_img.shape[1])
    crop_h = int(meter_img.shape[0])
    sample_crop_model = {
        "source": str(detection_meta.get("source") or detector_name),
        "crop_padding_px": int(detection_meta.get("crop_padding_px", 0) or 0),
        "crop_width_px": int(detection_meta.get("crop_width_px", crop_w)),
        "crop_height_px": int(detection_meta.get("crop_height_px", crop_h)),
        "crop_short_side_px": int(detection_meta.get("crop_short_side_px", min(crop_w, crop_h))),
        "crop_long_side_px": int(detection_meta.get("crop_long_side_px", max(crop_w, crop_h))),
        "raw_bbox_xyxy": [int(v) for v in (detection_meta.get("raw_bbox_xyxy") or bbox)],
        "expanded_bbox_xyxy": [int(v) for v in (detection_meta.get("expanded_bbox_xyxy") or bbox)],
        "source_image_width_px": int(detection_meta.get("source_image_width_px", crop_w)),
        "source_image_height_px": int(detection_meta.get("source_image_height_px", crop_h)),
        "notes": "This crop geometry is used by meter_read to estimate current outer-circle scale from YOLO crop size ratio.",
    }
    detection_block = {
        "detector": detector_name,
        "class_name": "clock" if detector_name == "YOLO" else "single_meter_sample",
        "confidence": round(float(yolo_conf), 4),
        "bbox_xyxy": [int(v) for v in bbox],
        "raw_bbox_xyxy": [int(v) for v in (detection_meta.get("raw_bbox_xyxy") or bbox)],
        "expanded_bbox_xyxy": [int(v) for v in (detection_meta.get("expanded_bbox_xyxy") or bbox)],
        "crop_padding_px": int(detection_meta.get("crop_padding_px", 0) or 0),
        "crop_width_px": int(detection_meta.get("crop_width_px", crop_w)),
        "crop_height_px": int(detection_meta.get("crop_height_px", crop_h)),
    }

    profile = {
        "profile_version": "9.16.1",
        "meter_id": profile_name,
        "source_image": image_path,
        "type": "dial gauge",
        "unit": candidate.get("unit", "MPa"),
        "min_value": float(candidate.get("min_value", 0.0)),
        "max_value": float(candidate.get("max_value", 2.0)),
        "center_point": [int(center[0]), int(center[1])],
        "min_angle": round(float(min_angle), 3),
        "mid_angle": round(float(mid_angle), 3),
        "max_angle": round(float(max_angle), 3),
        "detection": detection_block,
        "sample_crop_model": sample_crop_model,
        "sample_image_requirements": {"high_resolution_close_front": True, "single_meter_only": True, "same_center_assumption": "relaxed: hub, dial and bezel may have perspective-induced center offsets; centers are stored separately"},
        "scale": {"direction": candidate.get("scale_direction", "clockwise"), "major_ticks": candidate.get("major_ticks", [0.0, 0.5, 1.0, 1.5, 2.0]), "major_step": candidate.get("major_step")},
        "geometry": {"center_xy_in_crop": [int(center[0]), int(center[1])], "radius_px": int(dial_radius), "auto_center_xy_in_crop": [int(preferred_center[0]), int(preferred_center[1])], "auto_radius_px": int(preferred_radius), "dial_shape": "circular_3point", "perspective_correction": False, "outer_pair_preferred_reference": preferred_outer_reference},
        "component_ratio_model": component_ratio_model,
        "outer_color_model": outer_color_model,
        "outer_circle_model": {
            "source": "human_3point_dial_and_bezel_circles",
            "dial_face_circle": {"center_xy_in_crop": [int(dial_center[0]), int(dial_center[1])], "points_xy_in_crop": dial_circle.get("points"), "radius_px": int(dial_radius), "source": dial_circle.get("source")},
            "bezel_case_circle": {"center_xy_in_crop": [int(bezel_center[0]), int(bezel_center[1])], "points_xy_in_crop": bezel_circle.get("points"), "radius_px": int(bezel_radius), "source": bezel_circle.get("source")},
            "bezel_to_dial_radius_ratio": round(float(bezel_radius) / max(float(dial_radius), 1.0), 6),
            "dial_to_bezel_radius_ratio": round(float(dial_radius) / max(float(bezel_radius), 1.0), 6),
            "center_offset_px": round(distance(dial_center, bezel_center), 3),
            "center_offset_over_dial_radius": round(distance(dial_center, bezel_center) / max(float(dial_radius), 1.0), 6),
            "hub_to_dial_center_offset_px": [int(dial_center[0] - center[0]), int(dial_center[1] - center[1])],
            "hub_to_bezel_center_offset_px": [int(bezel_center[0] - center[0]), int(bezel_center[1] - center[1])],
            "colors_similar": colors_similar,
            "preferred_outer_reference": preferred_outer_reference,
            "notes": "Dial and bezel circles are defined by three human-selected circumference points each. Centers are not forced to match hub center.",
        },
        "hub": {"center_xy_in_crop": [int(center[0]), int(center[1])], "edge_xy_in_crop": [int(hub_edge[0]), int(hub_edge[1])], "radius_px": int(hub_radius), "diameter_px": int(hub_radius * 2), "source": "human_confirmed", "search_radius_scale": 1.8, "min_radius_scale": 0.65, "max_radius_scale": 1.45},
        "needle_shape": make_needle_shape_profile(center, needle_tip_pt, needle_tail_pt, dial_radius, meter_img, needle_prof),
        "center": {**hub_prof, "min_radius_ratio": 0.02, "max_radius_ratio": 0.15},
        "hub_color": hub_prof,
        "dial_face_color": dial_prof,
        "bezel_case_color": bezel_prof,
        "needle": {**needle_prof, "dark_gray_lower": 20, "dark_gray_upper": 170, "outer_radius_ratio": 0.43, "start_radius_scale": 1.1, "min_length_scale": 2.0, "angle_step": 0.5, "max_gap": 8, "method": "ray_cast_color_profile"},
        "ai_candidate": candidate.get("auto", {}),
        "reading": {"confidence_threshold": 0.70, "value_formula": "clockwise_linear_angle_interpolation"},
        "review": {"status": "candidate", "confirmed_by_human": False, "notes": "Generated by meter_profile_assistant_v9_16_1_dual_hub_color_index_fix. Dial/bezel circles use independent 3-point centers and component ratios are highest reliability information."},
    }
    return profile


def review_in_terminal(profile: Dict[str, Any]) -> Dict[str, Any]:
    print("\n=== メータープロファイル候補 ===")
    print(yaml.safe_dump(profile, allow_unicode=True, sort_keys=False))
    print("Enterのみで候補値を採用します。")
    profile["meter_id"] = prompt_str("meter_id", profile["meter_id"])
    profile["type"] = prompt_str("type", profile["type"])
    profile["unit"] = prompt_str("unit", profile["unit"])

    # Simple scale input: enter only MIN, MAX, and major interval.
    # major_ticks are generated automatically, so users no longer need to type 0,0.1,0.2,... manually.
    profile["min_value"] = prompt_float("min_value", profile["min_value"])
    profile["max_value"] = prompt_float("max_value", profile["max_value"])
    current_step = profile.get("scale", {}).get("major_step")
    if current_step is None:
        current_step = infer_default_major_step(profile["min_value"], profile["max_value"], profile.get("scale", {}).get("major_ticks"))
    major_step = prompt_float("major_step", float(current_step))
    profile.setdefault("scale", {})["major_step"] = float(major_step)
    profile["scale"]["major_ticks"] = generate_major_ticks(profile["min_value"], profile["max_value"], major_step)
    print(f"generated major_ticks: {profile['scale']['major_ticks']}")

    profile["min_angle"] = prompt_float("min_angle", profile["min_angle"])
    profile["max_angle"] = prompt_float("max_angle", profile["max_angle"])
    profile.setdefault("review", {})["manual_scale_note"] = "Scale generated from min_value, max_value, and major_step; individual comma-separated major_ticks were not required."
    ok = input("この内容でconfirmedにしますか？ [Y/n]: ").strip().lower()
    confirmed = ok in ("", "y", "yes")
    profile["review"]["confirmed_by_human"] = confirmed
    profile["review"]["status"] = "confirmed" if confirmed else "candidate"
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="AI/CV候補 + 人確認でメータープロファイルを作成し、ハブ円情報付きYAMLを保存するツール")
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--image", default="InputImages/FixedCamera.jpg")
    parser.add_argument("--out", default="meter_profiles.yaml", help="複数プロファイルを格納するYAML")
    parser.add_argument("--in", dest="input_profiles", default=None, help="初期値として読み込む既存プロファイルYAML")
    parser.add_argument("--profile", default="default_pressure_meter", help="YAML内のプロファイル名")
    parser.add_argument("--class-name", default="clock")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sample-radius", type=int, default=8)
    parser.add_argument("--std-scale", type=float, default=3.0)
    parser.add_argument("--h-tol-min", type=int, default=12)
    parser.add_argument("--s-tol-min", type=int, default=35)
    parser.add_argument("--v-tol-min", type=int, default=35)
    parser.add_argument("--gray-tol-min", type=int, default=35)
    parser.add_argument("--display-scale", type=float, default=0.66)
    parser.add_argument("--no-review", action="store_true", help="端末での確認修正を省略する")
    parser.add_argument("--skip-hub-selection", action="store_true", help="ハブ候補選択UIをスキップする")
    parser.add_argument("--skip-outer-selection", action="store_true", default=False, help="外周円候補選択UIをスキップする。デフォルトでは dial/bezel 選択UI を実行します")
    parser.add_argument("--enable-outer-candidate-selection", action="store_true", default=True, help="互換用オプション。v9.14.5 では外周候補選択UIはデフォルトで有効です")
    parser.add_argument("--legacy-single-outer-selection", action="store_true", help="従来の単一外周円候補選択UIを使用する")
    parser.add_argument("--no-log-file", action="store_true", help="assistantログファイルを出力しない")
    parser.add_argument("--log-dir", default="Logs", help="assistantログファイルの保存先")
    parser.add_argument("--use-yolo-crop", action=argparse.BooleanOptionalAction, default=True, help="assistant側でもYOLOでメーター検出してクロップする。デフォルト: ON。--no-use-yolo-crop で全体画像サンプルに戻せます。")
    parser.add_argument("--crop-only", action="store_true", help="互換用: YOLOを使わず画像全体をメーターとして扱う。--no-use-yolo-crop と同義。")
    parser.add_argument("--crop-padding", type=int, default=10, help="--use-yolo-crop時にYOLO検出bboxから上下左右に広げる余白[pixel]")
    parser.add_argument("--use-ocr", action="store_true", help="pytesseractがあればOCRで目盛り数字を推定する")
    parser.add_argument("--min-value", type=float, default=None, help="目盛り最小値。指定時はmajor_ticksをmin/max/stepから自動生成")
    parser.add_argument("--max-value", type=float, default=None, help="目盛り最大値。指定時はmajor_ticksをmin/max/stepから自動生成")
    parser.add_argument("--major-step", type=float, default=None, help="主目盛り間隔。例: 0.1。major_ticksは自動生成")
    parser.add_argument("--unit", type=str, default=None, help="単位。例: MPa, kPa")
    parser.add_argument("--click-near-ocr-scale", action=argparse.BooleanOptionalAction, default=True, help="--use-ocr時、MIN/MAXクリック近傍OCRと円弧上OCRからmin/max/major_ticksを推定する")
    parser.add_argument("--ocr-minmax-roi-ratio", type=float, default=0.20, help="MIN/MAXクリック近傍OCRの探索半径を外周半径比で指定")
    parser.add_argument("--ocr-arc-margin", type=float, default=0.08, help="MIN-MAX円弧OCRの角度比許容マージン")
    parser.add_argument("--ocr-radial-min", type=float, default=0.35, help="OCR数値候補として採用する最小半径比")
    parser.add_argument("--ocr-radial-max", type=float, default=1.15, help="OCR数値候補として採用する最大半径比")
    args = parser.parse_args()

    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_fh = None
    if not args.no_log_file:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_name = f"meter_profile_{PROFILE_ASSISTANT_VERSION}_{args.profile}_{_safe_stem(args.image)}.log"
        log_fh = open(log_dir / log_name, "w", encoding="utf-8")
        sys.stdout = TeeStdout(sys.__stdout__, log_fh)
        print(f"log_file: {log_dir / log_name}")
    print(f"run_info: version={PROFILE_ASSISTANT_VERSION} started_at={started_at} image={args.image} profile={args.profile}")

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    use_yolo_crop = bool(args.use_yolo_crop and not args.crop_only)
    if use_yolo_crop:
        meter_img, bbox, conf, detection_meta = detect_meter(img, args.model, args.class_name, args.conf, args.device, args.crop_padding)
        detector_name = "YOLO"
    else:
        meter_img = img.copy()
        bbox = [0, 0, img.shape[1], img.shape[0]]
        conf = 1.0
        detector_name = "full_image_sample"
        detection_meta = {
            "source": "full_image_sample",
            "raw_bbox_xyxy": [0, 0, img.shape[1], img.shape[0]],
            "expanded_bbox_xyxy": [0, 0, img.shape[1], img.shape[0]],
            "crop_padding_px": 0,
            "crop_width_px": int(img.shape[1]),
            "crop_height_px": int(img.shape[0]),
            "crop_short_side_px": int(min(img.shape[1], img.shape[0])),
            "crop_long_side_px": int(max(img.shape[1], img.shape[0])),
            "source_image_width_px": int(img.shape[1]),
            "source_image_height_px": int(img.shape[0]),
        }
        print(f"sample_image_mode: full_image bbox={bbox} use_yolo_crop=False")
    input_defaults = {}
    if args.input_profiles:
        input_path = Path(args.input_profiles)
        input_profiles = load_profiles(input_path)
        if args.profile in input_profiles:
            input_defaults = input_profiles[args.profile]
            print(f"loaded defaults from --in: {input_path} profile={args.profile}")
        else:
            print(f"--in specified but profile not found: {input_path} profile={args.profile}. Using built-in defaults.")
    candidate = auto_profile_candidate(meter_img, input_defaults, args)
    points, needle_prof, center_prof = interactive_review_profile(meter_img, candidate, args)
    profile = create_profile(meter_img, args.image, bbox, conf, args.profile, points, needle_prof, center_prof, candidate, detector_name=detector_name, detection_meta=detection_meta)
    profile = apply_profile_defaults(profile, input_defaults)
    profile = apply_click_near_ocr_scale(profile, meter_img, args)
    profile = apply_manual_scale_values(profile, args)
    if not args.skip_outer_selection:
        print("outer_pair_selection: enabled (default)")
        if args.legacy_single_outer_selection:
            profile = interactive_select_outer_candidate(meter_img, profile, args)
        else:
            profile = interactive_select_outer_pair_model(meter_img, profile, args)
    else:
        print("outer_pair_selection: skipped by option")
    if not args.skip_hub_selection:
        profile = interactive_select_hub_candidate(meter_img, profile, args)
    if not args.no_review:
        profile = review_in_terminal(profile)
    out_path = Path(args.out)
    profiles = load_profiles(out_path)
    profiles[args.profile] = profile
    save_profiles(out_path, profiles)
    print(f"saved: {out_path}")
    print(f"profile: {args.profile}")
    if log_fh is not None:
        print(f"log_saved: {log_fh.name}")
        log_fh.close()
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
