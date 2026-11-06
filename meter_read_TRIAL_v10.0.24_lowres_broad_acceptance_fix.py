#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# AIMeterReader - Meter Reader
# -----------------------------------------------------------------------------
# Fixed filename : meter_read.py
# Trial version  : v10.0.24_line_width_and_lowres_axis_rescue
# Base source    : meter_read_TRIAL_v9_33_7_preserve_hub_rescue.py
# Fixed date     : 2026-07-13
# Purpose        : Read analog meters with guarded profile-guided hub selection and profile-position-based needle polarity scoring.
# Notes          : Adds bidirectional long-needle/short-needle/full-span scoring to Hough and ray needle candidates when needle_shape exists in YAML.
# =============================================================================
"""
meter_read_v8_12_center_adjust_gui_needlehub.py

改善版メーター読取 v8.2 hub fixed。

v8_hub追加仕様:
  - meter_profile_assistant_v6_hub.py が保存した hub 情報（中心・半径・直径）を利用する。
  - ハブ検出時に期待半径/探索範囲をYAMLから制約し、反射や小円の誤検出を抑制する。
  - 値検出画面では、青丸をYAMLのハブ半径ベースで表示する。
  - 針検出で除外する中心半径は、表示半径とは分離し、検出半径も利用する。

主な改善:
  - YAML fine_adjustから中心/角度補正を自動適用
  - --save-fine-adjustで補正値をYAMLへ保存
  - --show-normalizedで極座標展開ビューを表示
  1. 中心検出を改善
     - 外周円(HoughCircle)候補
     - 中心ハブ/ネジ候補
     - YAML中心点
     を組み合わせ、画像ごとに中心を再推定する。

  2. 針検出を改善
     - 色/濃淡だけでなく、中心から伸びる線としてスコアリング
     - HoughLinesP候補
     - 放射状レイキャスト候補
     を統合して針角度を決定する。

  3. MIN/MAXは座標ではなく角度を利用
     - centerが画像ごとに補正されても、min_angle/max_angleは角度として維持する。

実行例:
  python meter_read_v6_2.py \
    --model yolo26x.pt \
    --image InputImages/IMG_0145.jpg \
    --profiles meter_profiles.yaml \
    --profile default_pressure_meter \
    --debug

YOLOを使わず画像全体をメーターとして扱う:
  python meter_read_v6_2.py \
    --image InputImages/IMG_0145.jpg \
    --profiles meter_profiles.yaml \
    --profile default_pressure_meter \
    --crop-only \
    --debug
"""

import argparse
import copy
import hashlib
import json
import math
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

Point = Tuple[int, int]

VERSION = "v10.0.24"

ADMIN_CONFIG: Dict[str, Any] = {}
ADMIN_CONFIG_META: Dict[str, Any] = {}

DEFAULT_ADMIN_CONFIG: Dict[str, Any] = {
    "admin_settings": {"schema_version": 1, "allow_user_edit": False},
    # GUI表示専用のデフォルト値。読取補正パラメーターとは分離して管理する。
    "display_defaults": {
        "meter_window": {
            "force_initial_size": True,
            "initial_width_px": 600,
            "initial_height_px": 600,
        },
        "result_window": {
            "fit_to_max_size": True,
            "max_width_px": 1280,
            "max_height_px": 800,
            "force_initial_size": True,
            "initial_width_px": 1200,
            "initial_height_px": 800,
        },
    },
    "hub_selection": {
        "profile_guard": {
            "enabled": True,
            "displacement_radius_ratio": 0.45,
            "adoption_allow_ratio": 0.70,
            "radius_deviation_trigger": 0.20,
            "real_edge_margin_min": 0.04,
            "real_edge_margin_ratio": 0.10,
            "real_color_margin_min": 2.0,
            "real_color_margin_ratio": 0.10,
            "profile_edge_advantage_min": 0.08,
            "profile_edge_advantage_ratio": 0.20,
        }
    },
    "needle_segment": {
        "tip_width": {
            "normal_outer_ratio": 0.045,
            "critical_outer_ratio": 0.075,
            "critical_center_ratio": 1.25,
            "rescue_enabled": True,
            "rescue_min_shape_score": 90.0,
            "rescue_min_length_ratio": 0.72,
            "rescue_require_centerline": True,
            "rescue_require_unique_scale_direction": True,
        }
    },
    "centerline_refine": {
        "search_deg": 6.0,
        "step_deg": 0.5,
        "low_confidence_limit_deg": 3.0,
        "small_outer_radius_px": 140,
        "low_coverage_threshold": 0.90,
        "search_edge_margin_deg": 0.10,
        "reject_delta_deg": 7.0,
    },
    "dual_scale_direction_guard": {
        "enabled": True,
        "trigger": {
            "require_both_directions_in_scale": True,
            "require_length_profile_available": True,
        },
        "acceptance": {
            "minimum_length_match_advantage": 0.12,
            "minimum_long_short_ratio": 1.35,
            "minimum_orientation_margin": 35.0,
            "prefer_longer_endpoint": True,
            "use_far_end_width_tiebreaker": True,
            "minimum_tip_length_ratio_to_expected": 0.70,
            "minimum_tail_length_ratio_to_expected": 0.55,
            "maximum_observed_ratio_relative_error": 0.45,
            "reject_suspicious_tip_widening": True,
            "maximum_tip_mid_width_ratio": 1.50,
            "low_resolution_widening_reverse_rescue": {
                "enabled": True,
                "require_outer_invalid": True,
                "outer_radius_max_px": 140,
                "minimum_wrong_side_tip_mid_width_ratio": 1.80,
                "maximum_opposite_tip_mid_width_ratio": 1.50,
            },
            "maximum_refine_flip_deg": 90.0,
        },
        "low_resolution_refine_guard": {
            "enabled": True,
            "require_direction_locked": True,
            "require_outer_invalid": True,
            "outer_radius_max_px": 140,
            "coverage_max": 0.85,
            "maximum_refine_delta_deg": 1.5,
            "minimum_score_margin": 5.0,
        },
    },
    "ray_line_disagreement_rescue": {
        "enabled": True,
        "trigger": {
            "minimum_angle_difference_deg": 3.0,
            "require_outer_valid": True,
            "outer_radius_min_px": 180,
            "outer_radius_max_px": 400,
            "require_line_in_scale": True,
            "require_length_match": True,
            "minimum_shape_score": 75.0,
            "maximum_tip_width_excess_ratio": 1.30,
        },
        "acceptance": {
            "maximum_centerline_raw_difference_deg": 2.0,
            "maximum_line_ray_difference_deg": 6.0,
            "require_profile_tip_tail_length_match": True,
        },
    },
    "line_width_rejection_rescue": {
        "enabled": True,
        "trigger": {
            "require_outer_valid": True,
            "outer_radius_min_px": 180,
            "outer_radius_max_px": 400,
            "require_line_in_scale": True,
            "require_opposite_out_of_scale": True,
            "require_profile_length_match": True,
            "minimum_shape_score": 75.0,
            "rejection_reason_only_tip_width": True,
            "minimum_line_ray_difference_deg": 8.0,
            "maximum_line_ray_difference_deg": 25.0,
            "maximum_tip_width_excess_ratio": 1.30,
        },
        "acceptance": {
            "maximum_centerline_line_difference_deg": 2.0,
        },
    },
    "yolo_duplicate_merge": {
        "enabled": True,
        "grouping": {
            "iou_threshold": 0.55,
            "smaller_containment_threshold": 0.85,
            "center_distance_ratio_max": 0.15,
            "area_ratio_min": 0.30,
        },
        "selection": {
            "yolo_confidence_weight": 100.0,
            "dial_edge_weight": 80.0,
            "pair_score_weight": 0.05,
            "radius_ratio_error_weight": 100.0,
            "image_edge_cut_penalty": 80.0,
            "oversized_crop_penalty": 40.0,
            "minimum_quality_margin": 20.0,
            "minimum_dial_fill_ratio": 0.55,
        },
        "ambiguity": {"keep_both_when_uncertain": True},
    },
    "full_scale_axis_rescue": {
        "enabled": True,
        "trigger": {
            "require_outer_invalid": True,
            "outer_radius_max_px": 140,
            "coverage_max": 0.50,
            "shape_score_max": 85.0,
            "require_suspicious_tip_widening": True,
            "require_local_search_edge_hit": False,
            "geometry_fallback": {
                "enabled": True,
                "coverage_max": 0.80,
                "minimum_centerline_offset_px": 5.0,
                "minimum_tip_mid_width_ratio": 2.0,
                "require_outer_invalid": True,
                "outer_radius_max_px": 140,
            },
        },
        "search": {
            "use_effective_scale_range": True,
            "step_deg": 0.5,
        },
        "acceptance": {
            "minimum_score_ratio": 1.80,
            "minimum_score_difference": 120.0,
            "minimum_coverage": 0.70,
            "minimum_coverage_improvement": 0.25,
            "maximum_axis_offset_px": 3.0,
            "require_profile_axis_support": True,
            "require_scale_range": True,
        },
        "geometry_fallback_acceptance": {
            "enabled": True,
            "minimum_score_ratio": 1.50,
            "minimum_score_difference": 150.0,
            "minimum_coverage": 0.88,
            "minimum_coverage_improvement": 0.20,
            "maximum_top_angle_spread_deg": 2.0,
            "minimum_scale_ratio": 0.60,
            "maximum_scale_ratio": 0.80,
            "require_outer_invalid": True,
            "outer_radius_max_px": 140,
            "allow_unavailable_offset": True,
            "allow_lowres_axis_support_substitution": True,
        },
        "low_resolution_fallback": {
            "enabled": True,
            "offset_unavailable_threshold": 900.0,
            "minimum_score_ratio": 2.50,
            "minimum_coverage": 0.90,
            "top_candidate_count": 3,
            "maximum_top_angle_spread_deg": 2.0,
            "minimum_scale_ratio": 0.35,
            "maximum_scale_ratio": 0.75,
            "require_outer_invalid": True,
            "outer_radius_max_px": 140,
        },
    },
    "broad_axis_rescue": {
        "enabled": True,
        "trigger": {
            "require_outer_invalid": True,
            "outer_radius_max_px": 140,
            "coverage_max": 0.85,
            "require_local_search_edge_hit": True,
            "require_suspicious_tip_widening": True,
            "tip_widening_ratio": 1.75,
            "tip_widening_add_px": 8.0,
        },
        "search": {
            "range_deg": 18.0,
            "step_deg": 0.5,
            "minimum_axis_separation_deg": 6.5,
            "maximum_axis_separation_deg": 18.1,
        },
        "acceptance": {
            "minimum_score_ratio": 1.20,
            "minimum_score_difference": 55.0,
            "minimum_coverage_improvement": 0.08,
            "minimum_offset_improvement_px": 1.5,
            "require_profile_axis_support": True,
        },
        "low_resolution_widening_override": {
            "enabled": True,
            "require_outer_invalid": True,
            "outer_radius_max_px": 140,
            "coverage_max": 0.95,
            "require_suspicious_tip_widening": True,
            "minimum_tip_mid_width_ratio": 2.0,
            "require_local_search_edge_hit": True,
            "minimum_local_refine_delta_deg": 3.0,
            "acceptance": {
                "minimum_score_ratio": 1.08,
                "minimum_score_difference": 20.0,
                "maximum_axis_offset_px": 6.0,
                "maximum_axis_offset_outer_ratio": 0.06,
                "bypass_normal_geometry_clear": True,
                "bypass_normal_axis_support": True,
                "require_scale_range": True,
                "require_profile_length_support": True,
                "top_candidate_count": 3,
                "maximum_top_angle_spread_deg": 2.0,
            },
        },
        "extreme_low_resolution_override": {
            "enabled": True,
            "require_outer_invalid": True,
            "outer_radius_max_px": 140,
            "coverage_max": 0.80,
            "require_local_search_edge_hit": True,
            "minimum_score_ratio": 1.60,
            "require_geometry_clear": True,
            "require_scale_range": True,
        },
    },
}

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def admin_value(path: str, default: Any = None) -> Any:
    cur: Any = ADMIN_CONFIG
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def _validate_number(cfg: Dict[str, Any], path: str, lo: float, hi: float) -> None:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ValueError(f"管理者設定に必須項目がありません: {path}")
        cur = cur[part]
    if not isinstance(cur, (int, float)) or isinstance(cur, bool):
        raise TypeError(f"管理者設定は数値である必要があります: {path}={cur!r}")
    if not (lo <= float(cur) <= hi):
        raise ValueError(f"管理者設定が範囲外です: {path}={cur} allowed={lo}..{hi}")

def validate_admin_config(cfg: Dict[str, Any]) -> None:
    if int(cfg.get("admin_settings", {}).get("schema_version", -1)) != 1:
        raise ValueError("未対応のadmin schema_versionです。対応値は1です。")
    if bool(cfg.get("admin_settings", {}).get("allow_user_edit", True)):
        raise ValueError("admin_settings.allow_user_edit は false にしてください。")
    ranges = {
        "display_defaults.meter_window.initial_width_px": (240.0, 3840.0),
        "display_defaults.meter_window.initial_height_px": (240.0, 2160.0),
        "display_defaults.result_window.max_width_px": (320.0, 7680.0),
        "display_defaults.result_window.max_height_px": (240.0, 4320.0),
        "display_defaults.result_window.initial_width_px": (320.0, 7680.0),
        "display_defaults.result_window.initial_height_px": (240.0, 4320.0),
        "hub_selection.profile_guard.displacement_radius_ratio": (0.0, 2.0),
        "hub_selection.profile_guard.adoption_allow_ratio": (0.0, 2.0),
        "hub_selection.profile_guard.radius_deviation_trigger": (0.0, 1.0),
        "needle_segment.tip_width.rescue_min_shape_score": (0.0, 200.0),
        "needle_segment.tip_width.rescue_min_length_ratio": (0.0, 2.0),
        "centerline_refine.search_deg": (0.5, 30.0),
        "centerline_refine.step_deg": (0.1, 5.0),
        "centerline_refine.low_confidence_limit_deg": (0.0, 15.0),
        "centerline_refine.small_outer_radius_px": (20.0, 2000.0),
        "centerline_refine.low_coverage_threshold": (0.0, 1.0),
        "dual_scale_direction_guard.acceptance.minimum_length_match_advantage": (0.0, 1.0),
        "dual_scale_direction_guard.acceptance.minimum_long_short_ratio": (1.0, 10.0),
        "dual_scale_direction_guard.acceptance.minimum_orientation_margin": (0.0, 10000.0),
        "dual_scale_direction_guard.acceptance.minimum_tip_length_ratio_to_expected": (0.0, 2.0),
        "dual_scale_direction_guard.acceptance.minimum_tail_length_ratio_to_expected": (0.0, 2.0),
        "dual_scale_direction_guard.acceptance.maximum_observed_ratio_relative_error": (0.0, 5.0),
        "dual_scale_direction_guard.acceptance.maximum_tip_mid_width_ratio": (0.1, 20.0),
        "dual_scale_direction_guard.acceptance.low_resolution_widening_reverse_rescue.outer_radius_max_px": (20.0, 2000.0),
        "dual_scale_direction_guard.acceptance.low_resolution_widening_reverse_rescue.minimum_wrong_side_tip_mid_width_ratio": (0.1, 20.0),
        "dual_scale_direction_guard.acceptance.low_resolution_widening_reverse_rescue.maximum_opposite_tip_mid_width_ratio": (0.1, 20.0),
        "dual_scale_direction_guard.acceptance.maximum_refine_flip_deg": (0.0, 180.0),
        "dual_scale_direction_guard.low_resolution_refine_guard.outer_radius_max_px": (20.0, 2000.0),
        "dual_scale_direction_guard.low_resolution_refine_guard.coverage_max": (0.0, 1.0),
        "dual_scale_direction_guard.low_resolution_refine_guard.maximum_refine_delta_deg": (0.0, 30.0),
        "dual_scale_direction_guard.low_resolution_refine_guard.minimum_score_margin": (0.0, 10000.0),
        "ray_line_disagreement_rescue.trigger.minimum_angle_difference_deg": (0.0, 45.0),
        "ray_line_disagreement_rescue.trigger.outer_radius_min_px": (1.0, 5000.0),
        "ray_line_disagreement_rescue.trigger.outer_radius_max_px": (1.0, 5000.0),
        "ray_line_disagreement_rescue.trigger.minimum_shape_score": (0.0, 200.0),
        "ray_line_disagreement_rescue.trigger.maximum_tip_width_excess_ratio": (1.0, 5.0),
        "ray_line_disagreement_rescue.acceptance.maximum_centerline_raw_difference_deg": (0.0, 30.0),
        "ray_line_disagreement_rescue.acceptance.maximum_line_ray_difference_deg": (0.0, 45.0),
        "line_width_rejection_rescue.trigger.outer_radius_min_px": (1.0, 5000.0),
        "line_width_rejection_rescue.trigger.outer_radius_max_px": (1.0, 5000.0),
        "line_width_rejection_rescue.trigger.minimum_shape_score": (0.0, 200.0),
        "line_width_rejection_rescue.trigger.minimum_line_ray_difference_deg": (0.0, 90.0),
        "line_width_rejection_rescue.trigger.maximum_line_ray_difference_deg": (0.0, 180.0),
        "line_width_rejection_rescue.trigger.maximum_tip_width_excess_ratio": (1.0, 5.0),
        "line_width_rejection_rescue.acceptance.maximum_centerline_line_difference_deg": (0.0, 30.0),
        "yolo_duplicate_merge.grouping.iou_threshold": (0.0, 1.0),
        "yolo_duplicate_merge.grouping.smaller_containment_threshold": (0.0, 1.0),
        "yolo_duplicate_merge.grouping.center_distance_ratio_max": (0.0, 2.0),
        "yolo_duplicate_merge.grouping.area_ratio_min": (0.0, 1.0),
        "yolo_duplicate_merge.selection.yolo_confidence_weight": (0.0, 10000.0),
        "yolo_duplicate_merge.selection.dial_edge_weight": (0.0, 10000.0),
        "yolo_duplicate_merge.selection.pair_score_weight": (0.0, 100.0),
        "yolo_duplicate_merge.selection.radius_ratio_error_weight": (0.0, 10000.0),
        "yolo_duplicate_merge.selection.image_edge_cut_penalty": (0.0, 10000.0),
        "yolo_duplicate_merge.selection.oversized_crop_penalty": (0.0, 10000.0),
        "yolo_duplicate_merge.selection.minimum_quality_margin": (0.0, 10000.0),
        "yolo_duplicate_merge.selection.minimum_dial_fill_ratio": (0.1, 1.0),
        "broad_axis_rescue.trigger.coverage_max": (0.0, 1.0),
        "broad_axis_rescue.search.range_deg": (1.0, 45.0),
        "broad_axis_rescue.search.step_deg": (0.1, 5.0),
        "broad_axis_rescue.acceptance.minimum_score_ratio": (1.0, 10.0),
        "broad_axis_rescue.acceptance.minimum_score_difference": (0.0, 10000.0),
        "broad_axis_rescue.low_resolution_widening_override.outer_radius_max_px": (20.0, 2000.0),
        "broad_axis_rescue.low_resolution_widening_override.coverage_max": (0.0, 1.0),
        "broad_axis_rescue.low_resolution_widening_override.minimum_tip_mid_width_ratio": (1.0, 20.0),
        "broad_axis_rescue.low_resolution_widening_override.minimum_local_refine_delta_deg": (0.0, 45.0),
        "broad_axis_rescue.low_resolution_widening_override.acceptance.minimum_score_ratio": (1.0, 10.0),
        "broad_axis_rescue.low_resolution_widening_override.acceptance.minimum_score_difference": (0.0, 10000.0),
        "broad_axis_rescue.low_resolution_widening_override.acceptance.maximum_axis_offset_px": (0.0, 100.0),
        "broad_axis_rescue.low_resolution_widening_override.acceptance.maximum_axis_offset_outer_ratio": (0.0, 1.0),
        "broad_axis_rescue.low_resolution_widening_override.acceptance.top_candidate_count": (1.0, 10.0),
        "broad_axis_rescue.low_resolution_widening_override.acceptance.maximum_top_angle_spread_deg": (0.0, 45.0),
        "broad_axis_rescue.extreme_low_resolution_override.minimum_score_ratio": (1.0, 10.0),
        "full_scale_axis_rescue.trigger.outer_radius_max_px": (20.0, 2000.0),
        "full_scale_axis_rescue.trigger.coverage_max": (0.0, 1.0),
        "full_scale_axis_rescue.trigger.shape_score_max": (0.0, 200.0),
        "full_scale_axis_rescue.trigger.geometry_fallback.coverage_max": (0.0, 1.0),
        "full_scale_axis_rescue.trigger.geometry_fallback.minimum_centerline_offset_px": (0.0, 100.0),
        "full_scale_axis_rescue.trigger.geometry_fallback.minimum_tip_mid_width_ratio": (1.0, 20.0),
        "full_scale_axis_rescue.trigger.geometry_fallback.outer_radius_max_px": (20.0, 2000.0),
        "full_scale_axis_rescue.search.step_deg": (0.1, 5.0),
        "full_scale_axis_rescue.acceptance.minimum_score_ratio": (1.0, 10.0),
        "full_scale_axis_rescue.acceptance.minimum_score_difference": (0.0, 10000.0),
        "full_scale_axis_rescue.acceptance.minimum_coverage": (0.0, 1.0),
        "full_scale_axis_rescue.acceptance.minimum_coverage_improvement": (0.0, 1.0),
        "full_scale_axis_rescue.acceptance.maximum_axis_offset_px": (0.0, 100.0),
        "full_scale_axis_rescue.geometry_fallback_acceptance.minimum_score_ratio": (1.0, 10.0),
        "full_scale_axis_rescue.geometry_fallback_acceptance.minimum_score_difference": (0.0, 10000.0),
        "full_scale_axis_rescue.geometry_fallback_acceptance.minimum_coverage": (0.0, 1.0),
        "full_scale_axis_rescue.geometry_fallback_acceptance.minimum_coverage_improvement": (0.0, 1.0),
        "full_scale_axis_rescue.geometry_fallback_acceptance.maximum_top_angle_spread_deg": (0.0, 30.0),
        "full_scale_axis_rescue.geometry_fallback_acceptance.minimum_scale_ratio": (0.0, 1.0),
        "full_scale_axis_rescue.geometry_fallback_acceptance.maximum_scale_ratio": (0.0, 1.0),
        "full_scale_axis_rescue.geometry_fallback_acceptance.outer_radius_max_px": (20.0, 2000.0),
    }
    fit_to_max_size = admin_value_from(cfg, "display_defaults.result_window.fit_to_max_size")
    if not isinstance(fit_to_max_size, bool):
        raise TypeError("display_defaults.result_window.fit_to_max_size は true/false で指定してください。")
    meter_force_initial_size = admin_value_from(cfg, "display_defaults.meter_window.force_initial_size")
    if not isinstance(meter_force_initial_size, bool):
        raise TypeError("display_defaults.meter_window.force_initial_size は true/false で指定してください。")
    result_force_initial_size = admin_value_from(cfg, "display_defaults.result_window.force_initial_size")
    if not isinstance(result_force_initial_size, bool):
        raise TypeError("display_defaults.result_window.force_initial_size は true/false で指定してください。")
    for path, (lo, hi) in ranges.items():
        _validate_number(cfg, path, lo, hi)
    if float(admin_value_from(cfg, "centerline_refine.low_confidence_limit_deg")) > float(admin_value_from(cfg, "centerline_refine.search_deg")):
        raise ValueError("low_confidence_limit_deg は search_deg 以下にしてください。")

def admin_value_from(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        cur = cur[part]
    return cur

def load_admin_config(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    source = "built_in_defaults"
    raw: Dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        source = "file"
    cfg = _deep_merge(DEFAULT_ADMIN_CONFIG, raw)
    validate_admin_config(cfg)
    canonical = json.dumps(cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    meta = {"path": str(path), "source": source, "schema_version": cfg["admin_settings"]["schema_version"], "sha256": digest}
    return cfg, meta


# -----------------------------
# YAML / utility
# -----------------------------

def load_profiles(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"プロファイルファイルがありません: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data:
        raise RuntimeError(f"プロファイルファイルが空です: {path}")
    return data


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def save_profiles(path: Path, profiles: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            profiles,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )





def get_profile_fine_adjust(profile: Dict[str, Any]) -> Dict[str, Any]:
    """YAML内のfine_adjustを読む。無ければ0補正。"""
    fa = profile.get("fine_adjust", {})
    if fa is None:
        fa = {}

    return {
        "center_offset_x": int(fa.get("center_offset_x", 0)),
        "center_offset_y": int(fa.get("center_offset_y", 0)),
        "min_angle_offset": float(fa.get("min_angle_offset", 0.0)),
        "max_angle_offset": float(fa.get("max_angle_offset", 0.0)),
    }


def resolve_fine_adjust(args, profile: Dict[str, Any]) -> Dict[str, Any]:
    """YAML値を基本に、CLI指定があれば上書きする。"""
    fa = get_profile_fine_adjust(profile)

    if args.center_offset_x is not None:
        fa["center_offset_x"] = int(args.center_offset_x)
    if args.center_offset_y is not None:
        fa["center_offset_y"] = int(args.center_offset_y)
    if args.min_angle_offset is not None:
        fa["min_angle_offset"] = float(args.min_angle_offset)
    if args.max_angle_offset is not None:
        fa["max_angle_offset"] = float(args.max_angle_offset)

    return fa


def save_fine_adjust_to_profile(
    profiles: Dict[str, Any],
    profile_name: str,
    fine_adjust: Dict[str, Any],
    path: Path,
) -> None:
    """現在の補正値をYAMLへ保存する。"""
    if profile_name not in profiles:
        raise KeyError(f"指定プロファイルがありません: {profile_name}")

    profiles[profile_name].setdefault("fine_adjust", {})
    profiles[profile_name]["fine_adjust"]["center_offset_x"] = int(fine_adjust["center_offset_x"])
    profiles[profile_name]["fine_adjust"]["center_offset_y"] = int(fine_adjust["center_offset_y"])
    profiles[profile_name]["fine_adjust"]["min_angle_offset"] = float(fine_adjust["min_angle_offset"])
    profiles[profile_name]["fine_adjust"]["max_angle_offset"] = float(fine_adjust["max_angle_offset"])

    profiles[profile_name].setdefault("review", {})
    profiles[profile_name]["review"]["fine_adjust_note"] = (
        "center offsets are pixels in crop image coordinates; angle offsets are degrees."
    )

    save_profiles(path, profiles)


def calc_angle(center: Point, point: Point) -> float:
    cx, cy = center
    px, py = point
    dx = px - cx
    dy = cy - py
    return math.degrees(math.atan2(dy, dx))


def normalize_360(angle: float) -> float:
    return angle % 360.0


def angle_delta_abs(a: float, b: float) -> float:
    """最小角度差 0..180"""
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return d


def angle_diff_deg(a: float, b: float) -> float:
    """符号付き最小角度差 a-b を -180..180 deg で返す。"""
    return float((a - b + 180.0) % 360.0 - 180.0)


class TeeStream:
    """stdout/stderrをコンソールとログファイルに同時出力する簡易tee。"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def angle_to_value(angle_deg: float, profile: Dict[str, Any]) -> float:
    min_value = float(profile.get("min_value", profile.get("scale", {}).get("min_value", 0.0)))
    max_value = float(profile.get("max_value", profile.get("scale", {}).get("max_value", 1.0)))

    min_angle = float(profile.get("min_angle", profile.get("angle", {}).get("min_angle_deg", 225.0)))
    max_angle = float(profile.get("max_angle", profile.get("angle", {}).get("max_angle_deg", -45.0)))

    direction = profile.get("scale", {}).get("direction", profile.get("angle", {}).get("reading_direction", "clockwise"))

    angle = normalize_360(angle_deg)
    min_a = normalize_360(min_angle)
    max_a = normalize_360(max_angle)

    if direction == "clockwise":
        total = (min_a - max_a) % 360.0
        pos = (min_a - angle) % 360.0
    else:
        total = (max_a - min_a) % 360.0
        pos = (angle - min_a) % 360.0

    ratio = pos / total if total != 0 else 0.0
    ratio = max(0.0, min(1.0, ratio))
    return min_value + ratio * (max_value - min_value)


def point_from_angle(center: Point, angle_deg: float, radius: float) -> Point:
    rad = math.radians(angle_deg)
    return (
        int(round(center[0] + radius * math.cos(rad))),
        int(round(center[1] - radius * math.sin(rad))),
    )


def safe_int_list(value, default=None):
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, dict):
        # supports {"h":..., "s":..., "v":...}
        keys = ["h", "s", "v"]
        if all(k in value for k in keys):
            return [int(value[k]) for k in keys]
    return default


def get_profile_outer_reference(profile: Dict[str, Any]) -> Dict[str, Any]:
    """プロファイル作成時の外周円情報を取得する。

    ハブ半径は撮影距離やYOLO切り出しサイズが変わると画像上のpx値も変わるため、
    読取時外周半径 / 設定時外周半径 の比率でスケール補正する。
    """
    geom = profile.get("geometry", {}) or {}

    # v9.16: assistantで白文字盤円とベゼル/ケース円の両方を選択している場合、
    # preferred_outer_referenceに応じた円をプロファイル外周基準として使う。
    pair_model = profile.get("outer_circle_model") or {}
    preferred = str(pair_model.get("preferred_outer_reference") or geom.get("outer_pair_preferred_reference") or "")
    if preferred in ("dial_face_circle", "bezel_case_circle"):
        key = preferred
        circ = pair_model.get(key) or {}
        c0 = circ.get("center_xy_in_crop")
        r0 = circ.get("radius_px")
        if c0 is not None and r0 is not None:
            center = c0
            radius = r0
        else:
            center = geom.get("auto_center_xy_in_crop") or geom.get("center_xy_in_crop") or profile.get("center_point")
            radius = geom.get("auto_radius_px")
    else:
        center = geom.get("auto_center_xy_in_crop") or geom.get("center_xy_in_crop") or profile.get("center_point")
        radius = geom.get("auto_radius_px")
    if radius is None:
        # 旧形式ではradius_pxしか無い場合がある。これは目盛り半径のこともあるため第二候補。
        radius = geom.get("radius_px")
    if radius is None:
        radius = profile.get("ai_candidate", {}).get("radius_px")

    out: Dict[str, Any] = {}
    if center is not None and len(center) >= 2:
        out["center"] = (int(center[0]), int(center[1]))
    if radius is not None:
        try:
            out["radius"] = max(1, int(round(float(radius))))
        except Exception:
            pass
    return out


def get_profile_hub(
    profile: Dict[str, Any],
    current_outer_center: Optional[Point] = None,
    current_outer_radius: Optional[int] = None,
) -> Dict[str, Any]:
    """YAMLに保存されたハブ円情報を読み出す。

    v8.3:
      - 設定時外周円と読取時外周円の半径比でhub.radius_pxをスケール補正する。
      - hub中心も、設定時外周中心からの相対位置を同じ比率で読取時外周中心へ写像する。

    meter_profile_assistant_v6_hub.py では、ユーザーが CENTER と HUB_EDGE を
    指定することで hub.center_xy_in_crop / radius_px / diameter_px を保存する。
    古いYAMLとの後方互換のため、情報が無ければ空dictを返す。
    """
    hub = profile.get("hub", {}) or {}
    center = hub.get("center_xy_in_crop")
    radius = hub.get("radius_px")

    if center is None:
        # 後方互換: geometry/center_point をハブ中心の暫定値として扱う
        center = profile.get("geometry", {}).get("center_xy_in_crop") or profile.get("center_point")
    if radius is None:
        # 古いYAMLではハブ半径が無いのでNoneのまま返す
        radius = hub.get("diameter_px")
        if radius is not None:
            radius = float(radius) / 2.0

    out: Dict[str, Any] = {}
    raw_radius = None
    if radius is not None:
        raw_radius = max(1.0, float(radius))

    ref = get_profile_outer_reference(profile)
    ref_radius = ref.get("radius")
    scale = 1.0
    if raw_radius is not None and current_outer_radius is not None and ref_radius:
        scale = float(current_outer_radius) / max(float(ref_radius), 1.0)

    # hub中心は、設定時外周中心からの相対オフセットをスケールして、
    # 読取時外周中心へ移す。これにより画角・切り出しサイズ変化に強くする。
    if center is not None and len(center) >= 2:
        raw_center = (int(center[0]), int(center[1]))
        mapped_center = raw_center
        if current_outer_center is not None and ref.get("center") is not None:
            ref_center = ref["center"]
            dx = (raw_center[0] - ref_center[0]) * scale
            dy = (raw_center[1] - ref_center[1]) * scale
            mapped_center = (
                int(round(current_outer_center[0] + dx)),
                int(round(current_outer_center[1] + dy)),
            )
        out["center"] = mapped_center
        out["raw_center"] = raw_center

    if raw_radius is not None:
        scaled_radius = max(1, int(round(raw_radius * scale)))
        out["radius"] = scaled_radius
        out["diameter"] = int(scaled_radius * 2)
        out["raw_radius"] = int(round(raw_radius))
        out["raw_diameter"] = int(round(raw_radius * 2))

    out["source"] = hub.get("source", "profile")
    out["search_radius_scale"] = float(hub.get("search_radius_scale", 1.8))
    out["min_radius_scale"] = float(hub.get("min_radius_scale", 0.65))
    out["max_radius_scale"] = float(hub.get("max_radius_scale", 1.45))
    out["outer_radius_scale"] = float(scale)
    out["profile_outer_radius"] = int(ref_radius) if ref_radius else None
    out["current_outer_radius"] = int(current_outer_radius) if current_outer_radius is not None else None
    return out


def get_scaled_needle_shape(profile: Dict[str, Any], current_outer_radius: Optional[int] = None) -> Dict[str, Any]:
    """YAMLのneedle_shapeを読取時の外周半径に合わせてスケールする。

    needle_shapeは、プロファイル作成時の針先/針後方クリックから得た
    長さ比・幅・テーパー方向のテンプレートであり、参照時の針角度そのものは
    読取時の針角度としては使わない。
    """
    ns = profile.get("needle_shape", {}) or {}
    out: Dict[str, Any] = {"available": False}
    if not isinstance(ns, dict):
        return out

    ref = get_profile_outer_reference(profile)
    ref_radius = ns.get("reference_outer_radius_px") or ref.get("radius")
    scale = 1.0
    if current_outer_radius is not None and ref_radius:
        try:
            scale = float(current_outer_radius) / max(float(ref_radius), 1.0)
        except Exception:
            scale = 1.0

    def _float_or_none(v):
        try:
            return float(v)
        except Exception:
            return None

    tip_ratio = _float_or_none(ns.get("tip_length_ratio"))
    tail_ratio = _float_or_none(ns.get("tail_length_ratio"))
    full_ratio = _float_or_none(ns.get("full_length_ratio"))
    if tip_ratio is None and ns.get("tip_length_px") is not None and ref_radius:
        tip_ratio = float(ns.get("tip_length_px")) / max(float(ref_radius), 1.0)
    if tail_ratio is None and ns.get("tail_length_px") is not None and ref_radius:
        tail_ratio = float(ns.get("tail_length_px")) / max(float(ref_radius), 1.0)
    if full_ratio is None:
        if ns.get("full_length_px") is not None and ref_radius:
            full_ratio = float(ns.get("full_length_px")) / max(float(ref_radius), 1.0)
        elif tip_ratio is not None and tail_ratio is not None:
            full_ratio = float(tip_ratio) + float(tail_ratio)

    out.update({
        "available": bool(tip_ratio is not None or tail_ratio is not None or full_ratio is not None),
        "scale": float(scale),
        "profile_outer_radius": int(ref_radius) if ref_radius else None,
        "current_outer_radius": int(current_outer_radius) if current_outer_radius is not None else None,
        "tip_length_ratio": tip_ratio,
        "tail_length_ratio": tail_ratio,
        "full_length_ratio": full_ratio,
        "tip_tail_length_ratio": _float_or_none(ns.get("tip_tail_length_ratio")),
        "tip_length_px_scaled": float(tip_ratio * current_outer_radius) if tip_ratio is not None and current_outer_radius is not None else None,
        "tail_length_px_scaled": float(tail_ratio * current_outer_radius) if tail_ratio is not None and current_outer_radius is not None else None,
        "full_length_px_scaled": float(full_ratio * current_outer_radius) if full_ratio is not None and current_outer_radius is not None else None,
        "taper_direction": ns.get("taper_direction", "tip_is_thinner"),
        "tail_is_wider": bool(ns.get("tail_is_wider", True)),
    })

    wr = ns.get("width_ratio", {}) or {}
    wp = ns.get("width_profile", {}) or {}
    width_scaled = {}
    for key in ("base", "mid", "tip"):
        val = _float_or_none(wr.get(key))
        if val is None:
            val = _float_or_none(wp.get(key + "_px"))
            if val is not None and ref_radius:
                val = val / max(float(ref_radius), 1.0)
        width_scaled[key + "_width_px_scaled"] = float(val * current_outer_radius) if val is not None and current_outer_radius is not None else None
    out["width_scaled"] = width_scaled
    return out


# -----------------------------
# YOLO crop
# -----------------------------

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


def detect_meters(img: np.ndarray, model_path: str, class_name: str, conf_th: float, device: Optional[str], crop_padding: int = 10):
    """YOLOで複数メーターを検出し、左上→右下順で返す。bboxはpaddingで拡張する。"""
    model = YOLO(model_path)
    results = model(img, device=device) if device else model(img)

    detections = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            name = result.names[cls]
            conf = float(box.conf[0])
            if name != class_name or conf < conf_th:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x2 <= x1 or y2 <= y1:
                continue

            raw_bbox = [x1, y1, x2, y2]
            expanded_bbox = expand_bbox_xyxy(raw_bbox, img.shape[1], img.shape[0], crop_padding)
            ex1, ey1, ex2, ey2 = expanded_bbox
            cx = (ex1 + ex2) / 2.0
            cy = (ey1 + ey2) / 2.0
            area = (ex2 - ex1) * (ey2 - ey1)
            detections.append(
                {
                    "bbox": expanded_bbox,
                    "raw_bbox": raw_bbox,
                    "crop_padding": int(crop_padding),
                    "conf": conf,
                    "center": (cx, cy),
                    "area": area,
                }
            )

    if not detections:
        raise RuntimeError(f"メーター class='{class_name}' を検出できませんでした")

    # 読取順は「左を第一優先、上を第二優先」とする。
    # つまり、画像の左側にあるメーターから順に並べ、
    # X位置が同じ場合は上側のメーターを先にする。
    detections.sort(key=lambda d: (d["center"][0], d["center"][1]))

    total = len(detections)
    for i, d in enumerate(detections, start=1):
        d["meter_id"] = f"M{i:02d}_{total:02d}"
        x1, y1, x2, y2 = d["bbox"]
        d["crop"] = img[y1:y2, x1:x2].copy()
        print(f"yolo_crop[{d['meter_id']}]: raw_bbox={d['raw_bbox']} expanded_bbox={d['bbox']} crop_padding={d['crop_padding']}")

    return detections



def _bbox_metrics(a: List[int], b: List[int]) -> Dict[str, float]:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = max(1.0, area_a + area_b - inter)
    small = min(area_a, area_b)
    large = max(area_a, area_b)
    ca = ((ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5)
    cb = ((bx1 + bx2) * 0.5, (by1 + by2) * 0.5)
    center_distance = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
    large_short_side = max(1.0, min(max(ax2-ax1, bx2-bx1), max(ay2-ay1, by2-by1)))
    return {
        "iou": inter / union,
        "smaller_containment": inter / small,
        "area_ratio": small / large,
        "center_distance_ratio": center_distance / large_short_side,
    }


def _are_duplicate_meter_boxes(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, Dict[str, float]]:
    cfg = admin_value("yolo_duplicate_merge.grouping", {}) or {}
    m = _bbox_metrics(a.get("raw_bbox", a["bbox"]), b.get("raw_bbox", b["bbox"]))
    duplicate = (
        m["center_distance_ratio"] <= float(cfg.get("center_distance_ratio_max", 0.15))
        and m["area_ratio"] >= float(cfg.get("area_ratio_min", 0.30))
        and (
            m["iou"] >= float(cfg.get("iou_threshold", 0.55))
            or m["smaller_containment"] >= float(cfg.get("smaller_containment_threshold", 0.85))
        )
    )
    return bool(duplicate), m


def _crop_quality_score(d: Dict[str, Any], info: Dict[str, Any], image_shape: Tuple[int, ...]) -> Tuple[float, Dict[str, float]]:
    cfg = admin_value("yolo_duplicate_merge.selection", {}) or {}
    conf = float(d.get("conf", 0.0))
    om = info.get("outer_meta") or {}
    edge = float(om.get("edge_support") or 0.0)
    pair_score = float(om.get("outer_pair_score") or 0.0)
    ratio = om.get("outer_pair_ratio")
    expected = om.get("outer_pair_expected_ratio")
    ratio_err = abs(float(ratio) - float(expected)) if ratio is not None and expected is not None else 1.0
    bbox = d.get("bbox") or [0, 0, image_shape[1], image_shape[0]]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = int(image_shape[0]), int(image_shape[1])
    edge_cut = 1.0 if (x1 <= 1 or y1 <= 1 or x2 >= w-1 or y2 >= h-1) else 0.0
    crop_short = max(1.0, float(min(x2-x1, y2-y1)))
    dial_r = float(info.get("outer_radius") or 0.0)
    dial_fill = (2.0 * dial_r) / crop_short
    min_fill = float(cfg.get("minimum_dial_fill_ratio", 0.55))
    oversized = max(0.0, min_fill - dial_fill) / max(min_fill, 1e-6)
    components = {
        "confidence": conf * float(cfg.get("yolo_confidence_weight", 100.0)),
        "dial_edge": edge * float(cfg.get("dial_edge_weight", 80.0)),
        "pair_score": pair_score * float(cfg.get("pair_score_weight", 0.05)),
        "radius_ratio_penalty": ratio_err * float(cfg.get("radius_ratio_error_weight", 100.0)),
        "image_edge_cut_penalty": edge_cut * float(cfg.get("image_edge_cut_penalty", 80.0)),
        "oversized_crop_penalty": oversized * float(cfg.get("oversized_crop_penalty", 40.0)),
    }
    score = components["confidence"] + components["dial_edge"] + components["pair_score"] \
        - components["radius_ratio_penalty"] - components["image_edge_cut_penalty"] - components["oversized_crop_penalty"]
    components["dial_fill_ratio"] = dial_fill
    components["score"] = score
    return float(score), components


def consolidate_duplicate_meter_results(
    detections: List[Dict[str, Any]],
    processed: List[Dict[str, Any]],
    image_shape: Tuple[int, ...],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """同一メーターの入れ子状YOLO検出をグループ化し、品質差が明確な場合だけ統合する。"""
    if not bool(admin_value("yolo_duplicate_merge.enabled", True)) or len(detections) < 2:
        return processed, []
    n = len(detections)
    parent = list(range(n))
    pair_logs: List[Dict[str, Any]] = []
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for i in range(n):
        for j in range(i+1, n):
            dup, metrics = _are_duplicate_meter_boxes(detections[i], detections[j])
            if dup:
                union(i, j)
            pair_logs.append({"i": i, "j": j, "duplicate": dup, **metrics})
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    kept: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    min_margin = float(admin_value("yolo_duplicate_merge.selection.minimum_quality_margin", 20.0))
    keep_both = bool(admin_value("yolo_duplicate_merge.ambiguity.keep_both_when_uncertain", True))
    for idxs in groups.values():
        if len(idxs) == 1:
            kept.append(processed[idxs[0]])
            continue
        scored = []
        for idx in idxs:
            score, comp = _crop_quality_score(detections[idx], processed[idx]["info"], image_shape)
            scored.append((score, idx, comp))
        scored.sort(reverse=True, key=lambda x: x[0])
        margin = scored[0][0] - scored[1][0]
        if margin >= min_margin or not keep_both:
            chosen = scored[0][1]
            kept.append(processed[chosen])
            suppressed = [processed[x[1]]["meter_id"] for x in scored[1:]]
            decisions.append({
                "members": [processed[x[1]]["meter_id"] for x in scored],
                "chosen": processed[chosen]["meter_id"],
                "suppressed": suppressed,
                "quality_margin": margin,
                "ambiguous": False,
                "scores": {processed[x[1]]["meter_id"]: {"score": x[0], **x[2]} for x in scored},
            })
        else:
            for _, idx, _ in scored:
                kept.append(processed[idx])
            decisions.append({
                "members": [processed[x[1]]["meter_id"] for x in scored],
                "chosen": None,
                "suppressed": [],
                "quality_margin": margin,
                "ambiguous": True,
                "scores": {processed[x[1]]["meter_id"]: {"score": x[0], **x[2]} for x in scored},
            })
    # 重複統合後に、最終的な読取順へ並べ直す。
    # 左を第一優先、上を第二優先とする。
    kept.sort(
        key=lambda item: (
            float(item["source_detection"]["center"][0]),
            float(item["source_detection"]["center"][1]),
        )
    )
    total = len(kept)
    for i, item in enumerate(kept, start=1):
        item["meter_id"] = f"M{i:02d}_{total:02d}"
        item["info"]["meter_id"] = item["meter_id"]
    return kept, decisions


def detect_meter(img: np.ndarray, model_path: str, class_name: str, conf_th: float, device: Optional[str], crop_padding: int = 10):
    """後方互換: 最初の1個だけ返す。"""
    detections = detect_meters(img, model_path, class_name, conf_th, device, crop_padding)
    d = detections[0]
    return d["crop"], d["bbox"], d["conf"]


# -----------------------------
# Center detection
# -----------------------------

def _bbox_size_xyxy(bbox: Any) -> Tuple[Optional[float], Optional[float]]:
    try:
        if bbox is None or len(bbox) < 4:
            return None, None
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        return max(1.0, x2 - x1), max(1.0, y2 - y1)
    except Exception:
        return None, None


def _profile_crop_scale_info(profile: Dict[str, Any], current_w: int, current_h: int) -> Dict[str, Any]:
    """プロファイル作成時のYOLO cropサイズと現在cropサイズの比率を推定する。

    優先順:
      1) assistant v9.14.4+ が保存した sample_crop_model.crop_width/height
      2) detection.crop_width/height
      3) detection.expanded_bbox_xyxy
      4) detection.bbox_xyxy（旧形式互換）

    X/Y比率も保持し、中心座標のスケーリングには scale_x/scale_y を使う。
    半径の代表スケールは、縦横比差が小さい場合は平均、差が大きい場合は短辺比を採用する。
    """
    out: Dict[str, Any] = {
        "available": False,
        "ref_w": None,
        "ref_h": None,
        "cur_w": float(current_w),
        "cur_h": float(current_h),
        "scale_x": None,
        "scale_y": None,
        "scale_short": None,
        "scale_long": None,
        "scale": None,
        "method": None,
        "source": None,
    }
    try:
        scm = profile.get("sample_crop_model") or {}
        det = profile.get("detection") or {}

        ref_w = ref_h = None
        source = None
        if scm.get("crop_width_px") and scm.get("crop_height_px"):
            ref_w = float(scm.get("crop_width_px"))
            ref_h = float(scm.get("crop_height_px"))
            source = "sample_crop_model.crop_width_height"
        elif det.get("crop_width_px") and det.get("crop_height_px"):
            ref_w = float(det.get("crop_width_px"))
            ref_h = float(det.get("crop_height_px"))
            source = "detection.crop_width_height"
        else:
            ref_w, ref_h = _bbox_size_xyxy(det.get("expanded_bbox_xyxy"))
            if ref_w and ref_h:
                source = "detection.expanded_bbox_xyxy"
            else:
                ref_w, ref_h = _bbox_size_xyxy(det.get("bbox_xyxy"))
                if ref_w and ref_h:
                    source = "detection.bbox_xyxy"

        cur_w = max(1.0, float(current_w))
        cur_h = max(1.0, float(current_h))
        if ref_w is None or ref_h is None or ref_w <= 0 or ref_h <= 0:
            return out

        scale_x = cur_w / ref_w
        scale_y = cur_h / ref_h
        scale_short = min(cur_w, cur_h) / max(1.0, min(ref_w, ref_h))
        scale_long = max(cur_w, cur_h) / max(1.0, max(ref_w, ref_h))
        anisotropy = abs(scale_x - scale_y) / max(min(scale_x, scale_y), 1e-6)
        if anisotropy <= 0.12:
            scale = (scale_x + scale_y) / 2.0
            method = "mean_xy"
        else:
            scale = scale_short
            method = "short_side"

        if not (0.25 <= scale <= 4.0):
            return out

        out.update({
            "available": True,
            "ref_w": float(ref_w),
            "ref_h": float(ref_h),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "scale_short": float(scale_short),
            "scale_long": float(scale_long),
            "scale": float(scale),
            "method": method,
            "source": source,
        })
        return out
    except Exception:
        return out


def _profile_crop_scale_hint(profile: Dict[str, Any], current_w: int, current_h: int) -> Optional[float]:
    info = _profile_crop_scale_info(profile, current_w, current_h)
    if info.get("available"):
        return float(info.get("scale"))
    return None


def _circle_edge_support(edges: np.ndarray, center: Point, radius: int, samples: int = 96) -> Dict[str, float]:
    """円周上に実エッジがどの程度存在するかを評価する。"""
    h, w = edges.shape[:2]
    cx, cy = center
    valid = 0
    hits = 0
    # 近傍を少し見ることで、Hough円と実エッジの数px差を許容する。
    for i in range(samples):
        a = 2.0 * math.pi * i / samples
        x = int(round(cx + radius * math.cos(a)))
        y = int(round(cy + radius * math.sin(a)))
        if x < 2 or x >= w - 2 or y < 2 or y >= h - 2:
            continue
        valid += 1
        patch = edges[y-2:y+3, x-2:x+3]
        if np.any(patch > 0):
            hits += 1
    valid_fraction = valid / max(samples, 1)
    support = hits / max(valid, 1)
    return {"valid_fraction": float(valid_fraction), "edge_support": float(support), "edge_hits": float(hits)}




def _circle_dial_region_score(gray: np.ndarray, center: Point, radius: int, samples: int = 144) -> Dict[str, float]:
    """外周円候補が「白い文字盤の外周」らしいかを評価する。

    黒いケース外周を拾った場合、円のすぐ内側は黒いベゼルになりやすい。
    白い文字盤の円を拾った場合、円のすぐ内側は白～薄灰の文字盤になりやすい。
    そのため、円の内側直近リングの明るさを強く評価し、外側直近リングの暗さも補助評価する。
    """
    h, w = gray.shape[:2]
    cx, cy = center
    r = max(int(radius), 1)
    inner_offsets = [max(3, int(round(r * p))) for p in (0.018, 0.030, 0.045, 0.060)]
    outer_offsets = [max(3, int(round(r * p))) for p in (0.018, 0.035, 0.055)]

    inner_vals = []
    outer_vals = []
    valid_inner = 0
    valid_outer = 0
    for i in range(samples):
        a = 2.0 * math.pi * i / samples
        ca = math.cos(a)
        sa = math.sin(a)
        for off in inner_offsets:
            rr = max(1, r - off)
            x = int(round(cx + rr * ca))
            y = int(round(cy + rr * sa))
            if 0 <= x < w and 0 <= y < h:
                valid_inner += 1
                inner_vals.append(int(gray[y, x]))
        for off in outer_offsets:
            rr = r + off
            x = int(round(cx + rr * ca))
            y = int(round(cy + rr * sa))
            if 0 <= x < w and 0 <= y < h:
                valid_outer += 1
                outer_vals.append(int(gray[y, x]))

    if inner_vals:
        inner_arr = np.array(inner_vals, dtype=np.float32)
        inner_mean = float(np.mean(inner_arr))
        inner_bright_ratio = float(np.mean(inner_arr >= 145.0))
        inner_dark_ratio = float(np.mean(inner_arr <= 80.0))
    else:
        inner_mean = 0.0
        inner_bright_ratio = 0.0
        inner_dark_ratio = 1.0

    if outer_vals:
        outer_arr = np.array(outer_vals, dtype=np.float32)
        outer_mean = float(np.mean(outer_arr))
        outer_dark_ratio = float(np.mean(outer_arr <= 100.0))
    else:
        outer_mean = 0.0
        outer_dark_ratio = 0.0

    # 0～1程度のダイヤルらしさ。内側が明るいことを最重要、黒ベゼルらしさを減点。
    dial_white_score = (
        inner_bright_ratio * 1.20
        + min(max((inner_mean - 95.0) / 105.0, 0.0), 1.0) * 0.55
        + outer_dark_ratio * 0.25
        - inner_dark_ratio * 0.95
    )
    dial_white_score = float(max(0.0, min(1.6, dial_white_score)))

    return {
        "dial_white_score": dial_white_score,
        "inner_mean": inner_mean,
        "inner_bright_ratio": inner_bright_ratio,
        "inner_dark_ratio": inner_dark_ratio,
        "outer_mean": outer_mean,
        "outer_dark_ratio": outer_dark_ratio,
        "valid_inner_fraction": float(valid_inner) / max(float(samples * len(inner_offsets)), 1.0),
        "valid_outer_fraction": float(valid_outer) / max(float(samples * len(outer_offsets)), 1.0),
    }


def _profile_bgr_mean(profile_part: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not isinstance(profile_part, dict):
        return None
    bgr = profile_part.get("bgr_mean") or {}
    try:
        return np.array([float(bgr.get("b")), float(bgr.get("g")), float(bgr.get("r"))], dtype=np.float32)
    except Exception:
        return None


def _color_match_from_distance(dist: float, soft_limit: float = 80.0) -> float:
    return float(max(0.0, min(1.0, 1.0 - float(dist) / max(float(soft_limit), 1.0))))


def _sample_circle_color_rings(img_bgr: np.ndarray, center: Point, radius: int, samples: int = 144) -> Dict[str, Any]:
    """外周候補の内側/外側リングのBGR色を取得する。"""
    h, w = img_bgr.shape[:2]
    cx, cy = center
    r = max(int(radius), 1)
    inner_offsets = [max(3, int(round(r * p))) for p in (0.025, 0.045, 0.070, 0.095)]
    outer_offsets = [max(3, int(round(r * p))) for p in (0.020, 0.040, 0.065)]
    inner_vals = []
    outer_vals = []
    for i in range(samples):
        a = 2.0 * math.pi * float(i) / float(samples)
        ca = math.cos(a)
        sa = math.sin(a)
        for off in inner_offsets:
            rr = max(1, r - off)
            x = int(round(cx + rr * ca))
            y = int(round(cy + rr * sa))
            if 0 <= x < w and 0 <= y < h:
                inner_vals.append(img_bgr[y, x].astype(np.float32))
        for off in outer_offsets:
            rr = r + off
            x = int(round(cx + rr * ca))
            y = int(round(cy + rr * sa))
            if 0 <= x < w and 0 <= y < h:
                outer_vals.append(img_bgr[y, x].astype(np.float32))
    result: Dict[str, Any] = {"available": False}
    if inner_vals:
        inner_arr = np.vstack(inner_vals).astype(np.float32)
        result["inner_bgr_mean"] = np.mean(inner_arr, axis=0)
        result["inner_bgr_std"] = np.std(inner_arr, axis=0)
    if outer_vals:
        outer_arr = np.vstack(outer_vals).astype(np.float32)
        result["outer_bgr_mean"] = np.mean(outer_arr, axis=0)
        result["outer_bgr_std"] = np.std(outer_arr, axis=0)
    result["available"] = bool(inner_vals or outer_vals)
    return result


def _outer_color_model_score(img_bgr: np.ndarray, center: Point, radius: int, profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """assistantで取得した文字盤色/ベゼル色を使って外周候補を評価する。"""
    if not isinstance(profile, dict):
        return {"available": False, "reason": "no_profile"}
    model = profile.get("outer_color_model") or {}
    dial_prof = model.get("dial_face_profile") or profile.get("dial_face") or {}
    bezel_prof = model.get("bezel_case_profile") or profile.get("bezel_case") or {}
    dial_bgr = _profile_bgr_mean(dial_prof)
    bezel_bgr = _profile_bgr_mean(bezel_prof)
    if dial_bgr is None or bezel_bgr is None:
        return {"available": False, "reason": "no_dial_or_bezel_color_profile"}
    rings = _sample_circle_color_rings(img_bgr, center, radius, samples=144)
    if not rings.get("available"):
        return {"available": False, "reason": "no_ring_samples"}
    inner_mean = rings.get("inner_bgr_mean")
    outer_mean = rings.get("outer_bgr_mean")
    if inner_mean is None:
        return {"available": False, "reason": "no_inner_ring_samples"}
    inner_dial_dist = float(np.linalg.norm(inner_mean - dial_bgr))
    inner_bezel_dist = float(np.linalg.norm(inner_mean - bezel_bgr))
    inner_dial_match = _color_match_from_distance(inner_dial_dist, 90.0)
    inner_bezel_match = _color_match_from_distance(inner_bezel_dist, 90.0)
    outer_bezel_match = 0.0
    outer_dial_match = 0.0
    outer_bezel_dist = None
    outer_dial_dist = None
    if outer_mean is not None:
        outer_bezel_dist = float(np.linalg.norm(outer_mean - bezel_bgr))
        outer_dial_dist = float(np.linalg.norm(outer_mean - dial_bgr))
        outer_bezel_match = _color_match_from_distance(outer_bezel_dist, 90.0)
        outer_dial_match = _color_match_from_distance(outer_dial_dist, 90.0)
    colors_similar = bool(model.get("colors_similar", False))
    preferred = str(model.get("preferred_outer_reference") or ("bezel_case_circle" if colors_similar else "dial_face_circle"))
    if colors_similar or preferred == "bezel_case_circle":
        # 色で文字盤円とベゼル円を分離しにくい場合は、ベゼル/ケース色が内側に多い円を許容・優先する。
        score = inner_bezel_match * 0.75 + max(outer_bezel_match, inner_bezel_match) * 0.25
        mode = "bezel_case_circle"
    else:
        # 通常は「円のすぐ内側が文字盤色、すぐ外側がベゼル/ケース色」の候補を優先。
        score = inner_dial_match * 0.72 + outer_bezel_match * 0.28 - inner_bezel_match * 0.18
        score = float(max(0.0, min(1.0, score)))
        mode = "dial_face_circle"
    return {
        "available": True,
        "mode": mode,
        "colors_similar": colors_similar,
        "preferred_outer_reference": preferred,
        "outer_color_score": round(float(score), 5),
        "inner_dial_match": round(float(inner_dial_match), 5),
        "inner_bezel_match": round(float(inner_bezel_match), 5),
        "outer_bezel_match": round(float(outer_bezel_match), 5),
        "outer_dial_match": round(float(outer_dial_match), 5),
        "inner_dial_distance": round(float(inner_dial_dist), 3),
        "inner_bezel_distance": round(float(inner_bezel_dist), 3),
        "outer_bezel_distance": round(float(outer_bezel_dist), 3) if outer_bezel_dist is not None else None,
        "outer_dial_distance": round(float(outer_dial_dist), 3) if outer_dial_dist is not None else None,
    }



def _outer_pair_model(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    return profile.get("outer_circle_model") or {}


def _outer_pair_expected(profile: Optional[Dict[str, Any]], current_w: int, current_h: int) -> Dict[str, Any]:
    """assistantで保存した白文字盤円/ベゼル円の期待半径・中心オフセットを現在cropへスケールする。

    v9.23:
      - assistant/read の YOLO crop サイズ比から半径をスケーリングする。
      - 中心座標は scale_x / scale_y でそれぞれスケーリングする。
      - dial->bezel オフセットも現在cropへ写像して、ペア選択に利用する。
    """
    model = _outer_pair_model(profile)
    if not model:
        return {"available": False}
    crop_scale_info = _profile_crop_scale_info(profile or {}, current_w, current_h)
    crop_scale = float(crop_scale_info.get("scale") or 1.0)
    scale_x = float(crop_scale_info.get("scale_x") or crop_scale)
    scale_y = float(crop_scale_info.get("scale_y") or crop_scale)
    result: Dict[str, Any] = {
        "available": True,
        "crop_scale": float(crop_scale),
        "crop_scale_info": crop_scale_info,
        "preferred_outer_reference": model.get("preferred_outer_reference", "dial_face_circle"),
        "colors_similar": bool(model.get("colors_similar", False)),
        "bezel_to_dial_radius_ratio": model.get("bezel_to_dial_radius_ratio"),
    }
    centers: Dict[str, Optional[Tuple[int, int]]] = {}
    raw_centers: Dict[str, Optional[Tuple[float, float]]] = {}
    for key in ("dial_face_circle", "bezel_case_circle"):
        circ = model.get(key) or {}
        try:
            r = float(circ.get("radius_px")) * float(crop_scale)
            c = circ.get("center_xy_in_crop")
            if c and len(c) >= 2:
                raw_cx = float(c[0])
                raw_cy = float(c[1])
                center = (int(round(raw_cx * scale_x)), int(round(raw_cy * scale_y)))
                raw_centers[key] = (raw_cx, raw_cy)
            else:
                center = None
                raw_centers[key] = None
            centers[key] = center
            result[key] = {"radius": float(r), "center": center}
        except Exception:
            centers[key] = None
            raw_centers[key] = None
    d = centers.get("dial_face_circle")
    b = centers.get("bezel_case_circle")
    raw_d = raw_centers.get("dial_face_circle")
    raw_b = raw_centers.get("bezel_case_circle")
    if d is not None and b is not None:
        result["expected_bezel_from_dial_offset"] = (float(b[0] - d[0]), float(b[1] - d[1]))
        dial_r = (result.get("dial_face_circle") or {}).get("radius") or 1.0
        result["expected_center_offset_over_dial_radius"] = math.hypot(float(b[0] - d[0]), float(b[1] - d[1])) / max(float(dial_r), 1.0)
    elif raw_d is not None and raw_b is not None:
        raw_dx = float(raw_b[0] - raw_d[0])
        raw_dy = float(raw_b[1] - raw_d[1])
        result["expected_bezel_from_dial_offset"] = (raw_dx * scale_x, raw_dy * scale_y)
        dial_r = (result.get("dial_face_circle") or {}).get("radius") or 1.0
        result["expected_center_offset_over_dial_radius"] = math.hypot(raw_dx * scale_x, raw_dy * scale_y) / max(float(dial_r), 1.0)
    return result


def _outer_pair_expected_role_geometry(profile: Optional[Dict[str, Any]], role: str, current_w: int, current_h: int) -> Dict[str, Any]:
    exp = _outer_pair_expected(profile, current_w, current_h)
    if not exp.get("available"):
        return {"available": False, "role": role}
    circ = exp.get(role) or {}
    center = circ.get("center")
    radius = circ.get("radius")
    if center is None or radius is None:
        return {"available": False, "role": role}
    return {
        "available": True,
        "role": role,
        "center": (int(center[0]), int(center[1])),
        "radius": float(radius),
    }


def _classify_outer_candidate_by_pair(profile: Optional[Dict[str, Any]], r: float, current_w: int, current_h: int) -> Dict[str, Any]:
    exp = _outer_pair_expected(profile, current_w, current_h)
    if not exp.get("available"):
        return {"available": False, "outer_pair_class": None}
    dial = exp.get("dial_face_circle") or {}
    bezel = exp.get("bezel_case_circle") or {}
    dial_r = float(dial.get("radius", 0.0) or 0.0)
    bezel_r = float(bezel.get("radius", 0.0) or 0.0)
    scores = {}
    if dial_r > 1.0:
        scores["dial_face_circle"] = abs(float(r) - dial_r) / dial_r
    if bezel_r > 1.0:
        scores["bezel_case_circle"] = abs(float(r) - bezel_r) / bezel_r
    if not scores:
        return {"available": False, "outer_pair_class": None}
    cls = min(scores, key=lambda k: scores[k])
    preferred = str(exp.get("preferred_outer_reference") or "dial_face_circle")
    return {
        "available": True,
        "outer_pair_class": cls,
        "preferred_outer_reference": preferred,
        "class_error": round(float(scores[cls]), 6),
        "dial_error": round(float(scores.get("dial_face_circle", 999.0)), 6),
        "bezel_error": round(float(scores.get("bezel_case_circle", 999.0)), 6),
        "dial_expected_radius": round(float(dial_r), 3) if dial_r > 1.0 else None,
        "bezel_expected_radius": round(float(bezel_r), 3) if bezel_r > 1.0 else None,
        "is_preferred_class": bool(cls == preferred),
    }

def _build_outer_circle_search_ranges(
    w: int,
    h: int,
    profile: Optional[Dict[str, Any]],
) -> List[Tuple[int, int, str]]:
    """外周円探索の半径レンジを複数作る。"""
    min_dim = min(w, h)
    ranges: List[Tuple[int, int, str]] = []

    def append_expected_range(expected_r: float, label: str, tight: float, relax: float) -> None:
        if expected_r is None or expected_r <= 0:
            return
        for width, suffix in ((tight, "tight"), (relax, "relaxed")):
            lo = int(max(8, expected_r * (1.0 - width)))
            hi = int(max(lo + 2, expected_r * (1.0 + width)))
            lo = max(8, min(lo, int(min_dim * 0.72)))
            hi = max(lo + 2, min(hi, int(min_dim * 0.72)))
            ranges.append((lo, hi, f"{label}_{suffix}"))

    # 1) assistantで保存された dial/bezel 半径を最優先探索する。
    if profile is not None:
        pair_exp = _outer_pair_expected(profile, w, h)
        if pair_exp.get("available"):
            for key, label in (("dial_face_circle", "pair_dial_face_radius"), ("bezel_case_circle", "pair_bezel_case_radius")):
                circ = pair_exp.get(key) or {}
                er = circ.get("radius")
                if er:
                    append_expected_range(float(er), label, tight=0.12, relax=0.20)
        ref = get_profile_outer_reference(profile)
        ref_r = ref.get("radius")
        crop_scale = _profile_crop_scale_hint(profile, w, h)
        if ref_r and crop_scale:
            append_expected_range(float(ref_r) * float(crop_scale), "profile_scaled_radius", tight=0.10, relax=0.18)

    # 2) 通常レンジ。YOLO crop内ではメーター外周は概ねcrop短辺の30～55%。
    ranges.append((int(min_dim * 0.30), int(min_dim * 0.55), "normal"))

    # 3) 候補不足時のやや広いレンジ。ただし異常な巨大円は後段の妥当性で弾く。
    ranges.append((int(min_dim * 0.24), int(min_dim * 0.62), "wide"))

    # 重複・不正レンジの除去
    uniq = []
    seen = set()
    for lo, hi, label in ranges:
        lo = max(4, int(lo))
        hi = max(lo + 2, int(hi))
        key = (lo, hi)
        if key not in seen:
            uniq.append((lo, hi, label))
            seen.add(key)
    return uniq


def _evaluate_outer_circle_candidate(
    img_bgr: np.ndarray,
    gray: np.ndarray,
    edges: np.ndarray,
    x: int,
    y: int,
    r: int,
    expected_center: Optional[Point],
    profile: Optional[Dict[str, Any]],
    range_label: str,
) -> Optional[Dict[str, Any]]:
    """外周円候補を妥当性チェックし、スコア付き候補として返す。"""
    h, w = gray.shape[:2]
    min_dim = min(w, h)
    img_center = (w // 2, h // 2)

    # 中心が画像から大きく外れる円は、今回のような巨大仮想円になりやすい。
    margin_x = max(12, int(w * 0.08))
    margin_y = max(12, int(h * 0.08))
    if x < -margin_x or x > w + margin_x or y < -margin_y or y > h + margin_y:
        return None

    # YOLO cropでは外周中心は画像中央付近にあるのが自然。
    dist_img = math.hypot(x - img_center[0], y - img_center[1])
    max_center_dist = max(min_dim * 0.23, 80.0)
    if dist_img > max_center_dist:
        # expected_center が妥当で、そちらには近い場合だけ救済する。
        rescue = False
        if expected_center is not None:
            ex, ey = expected_center
            if 0 <= ex < w and 0 <= ey < h:
                dist_exp = math.hypot(x - ex, y - ey)
                rescue = dist_exp <= max(min_dim * 0.18, 70.0)
        if not rescue:
            return None

    # 半径がcrop短辺に対して大きすぎる/小さすぎるものを除外。
    if r < min_dim * 0.22 or r > min_dim * 0.64:
        return None

    support = _circle_edge_support(edges, (x, y), int(r), samples=96)
    valid_fraction = support["valid_fraction"]
    edge_support = support["edge_support"]

    # 円周の多くが画像外になる候補は、外周円ではなく巨大仮想円の可能性が高い。
    if valid_fraction < 0.70:
        return None

    # エッジ支持が低い候補は棄却。ただし円周が淡い場合もあるため厳しすぎない値にする。
    if edge_support < 0.08:
        return None

    dial_region = _circle_dial_region_score(gray, (int(x), int(y)), int(r), samples=144)
    dial_white_score = float(dial_region.get("dial_white_score", 0.0))
    outer_color = _outer_color_model_score(img_bgr, (int(x), int(y)), int(r), profile)
    outer_color_score = float(outer_color.get("outer_color_score", 0.0)) if outer_color.get("available") else 0.0
    outer_pair = _classify_outer_candidate_by_pair(profile, float(r), w, h)
    outer_pair_class = outer_pair.get("outer_pair_class")
    preferred_outer_reference = outer_pair.get("preferred_outer_reference")
    pair_class_bonus = 0.0
    role_center_error_px = None
    role_center_error = None
    role_radius_error = None
    if outer_pair.get("available"):
        class_error = float(outer_pair.get("class_error", 1.0))
        pair_class_bonus += max(0.0, 1.0 - class_error / 0.18) * 220.0
        if outer_pair_class == preferred_outer_reference:
            pair_class_bonus += 260.0
        else:
            # 白文字盤円基準プロファイルでベゼル外周を選びにくくする。
            pair_class_bonus -= 240.0

        # assistantで選択した dial/bezel 円の中心・半径に近い候補をさらに優先する。
        role_geom = _outer_pair_expected_role_geometry(profile, str(outer_pair_class), w, h)
        if role_geom.get("available"):
            er = max(float(role_geom.get("radius") or 1.0), 1.0)
            ecx, ecy = role_geom.get("center")
            role_center_error_px = float(math.hypot(x - ecx, y - ecy))
            role_center_error = role_center_error_px / er
            role_radius_error = abs(float(r) - er) / er
            pair_class_bonus += max(0.0, 1.0 - role_center_error / 0.10) * 220.0
            pair_class_bonus += max(0.0, 1.0 - role_radius_error / 0.10) * 180.0
            pair_class_bonus -= role_center_error * 120.0
            pair_class_bonus -= role_radius_error * 80.0

    # 黒いケース外周は、円のすぐ内側が暗くなりやすい。
    # 白い文字盤の円を優先したいため、内側が極端に暗い候補は棄却する。
    # ただし照明条件の違いを考慮し、edge_supportが非常に高い候補はスコア減点に留める。
    if dial_white_score < 0.10 and edge_support < 0.22:
        return None

    ref_radius = None
    crop_scaled_ref_radius = None
    if profile is not None:
        ref = get_profile_outer_reference(profile)
        ref_radius = ref.get("radius")
        crop_scale = _profile_crop_scale_hint(profile, w, h)
        if ref_radius and crop_scale:
            crop_scaled_ref_radius = float(ref_radius) * float(crop_scale)

    # プロファイルから現在cropでの想定半径が分かる場合、極端に違う候補は弾く。
    radius_penalty = 0.0
    if crop_scaled_ref_radius:
        ratio = float(r) / max(float(crop_scaled_ref_radius), 1.0)
        if ratio < 0.58 or ratio > 1.75:
            return None
        radius_penalty = abs(float(r) - float(crop_scaled_ref_radius))
    elif ref_radius:
        # cropサイズ情報がない場合は弱い制約に留める。
        ratio = float(r) / max(float(ref_radius), 1.0)
        if ratio < 0.45 or ratio > 2.20:
            return None
        radius_penalty = abs(float(r) - float(ref_radius)) * 0.25

    radius_ratio_to_profile = None
    if crop_scaled_ref_radius:
        radius_ratio_to_profile = float(r) / max(float(crop_scaled_ref_radius), 1.0)
    elif ref_radius:
        radius_ratio_to_profile = float(r) / max(float(ref_radius), 1.0)

    # スコア: 中央に近く、想定半径に近く、円周エッジ支持が高いものを優先。
    # v9.12: 黒い外枠側の大きな円へ吸着する問題を抑えるため、プロファイル半径との差を強めに見る。
    score = (
        edge_support * 360.0
        + valid_fraction * 80.0
        + dial_white_score * 190.0
        + outer_color_score * 360.0
        + pair_class_bonus
        - dist_img * 0.55
        - radius_penalty * 1.15
        + float(r) * 0.010
    )

    return {
        "center": (int(x), int(y)),
        "radius": int(r),
        "score": float(score),
        "radius_ratio_to_profile": float(radius_ratio_to_profile) if radius_ratio_to_profile is not None else None,
        "radius_penalty": float(radius_penalty),
        "edge_support": float(edge_support),
        "valid_fraction": float(valid_fraction),
        "dist_to_image_center": float(dist_img),
        "range_label": range_label,
        "crop_scaled_ref_radius": crop_scaled_ref_radius,
        "dial_white_score": float(dial_white_score),
        "dial_inner_mean": float(dial_region.get("inner_mean", 0.0)),
        "dial_inner_bright_ratio": float(dial_region.get("inner_bright_ratio", 0.0)),
        "dial_inner_dark_ratio": float(dial_region.get("inner_dark_ratio", 0.0)),
        "dial_outer_dark_ratio": float(dial_region.get("outer_dark_ratio", 0.0)),
        "outer_color_score": float(outer_color_score),
        "outer_color_debug": outer_color,
        "outer_pair_class": outer_pair_class,
        "outer_pair_preferred": preferred_outer_reference,
        "outer_pair_debug": outer_pair,
        "outer_pair_class_bonus": float(pair_class_bonus),
        "outer_pair_role_center_error_px": float(role_center_error_px) if role_center_error_px is not None else None,
        "outer_pair_role_center_error": float(role_center_error) if role_center_error is not None else None,
        "outer_pair_role_radius_error": float(role_radius_error) if role_radius_error is not None else None,
    }



def _profile_scaled_outer_fallback(profile: Optional[Dict[str, Any]], w: int, h: int) -> Optional[Dict[str, Any]]:
    """Houghで有効候補が出ない場合の最終fallback。

    assistantで確定した preferred outer reference を、sample crop -> current crop の
    BBOXサイズ比で現在画像へ写像して返す。
    """
    if not isinstance(profile, dict):
        return None
    ref = get_profile_outer_reference(profile)
    center = ref.get("center")
    radius = ref.get("radius")
    if center is None or radius is None:
        return None
    scale_info = _profile_crop_scale_info(profile, w, h)
    if not scale_info.get("available"):
        return None
    scale = float(scale_info.get("scale") or 1.0)
    scale_x = float(scale_info.get("scale_x") or scale)
    scale_y = float(scale_info.get("scale_y") or scale)
    cx = int(round(float(center[0]) * scale_x))
    cy = int(round(float(center[1]) * scale_y))
    rr = int(round(float(radius) * scale))
    cx = int(clamp(cx, 0, max(0, w - 1)))
    cy = int(clamp(cy, 0, max(0, h - 1)))
    rr = int(clamp(rr, 4, max(6, int(min(w, h) * 0.72))))
    pair_dbg = _classify_outer_candidate_by_pair(profile, float(rr), w, h)
    return {
        "center": (cx, cy),
        "radius": rr,
        "score": -1.0,
        "radius_ratio_to_profile": 1.0,
        "radius_penalty": 0.0,
        "edge_support": 0.0,
        "valid_fraction": 1.0,
        "dist_to_image_center": float(math.hypot(cx - (w // 2), cy - (h // 2))),
        "range_label": "profile_scaled_fallback",
        "crop_scaled_ref_radius": float(rr),
        "dial_white_score": 0.0,
        "dial_inner_mean": 0.0,
        "dial_inner_bright_ratio": 0.0,
        "dial_inner_dark_ratio": 0.0,
        "dial_outer_dark_ratio": 0.0,
        "outer_color_score": 0.0,
        "outer_color_debug": {"available": False, "reason": "fallback"},
        "outer_pair_class": pair_dbg.get("outer_pair_class"),
        "outer_pair_preferred": pair_dbg.get("preferred_outer_reference"),
        "outer_pair_debug": pair_dbg,
        "outer_pair_class_bonus": 0.0,
        "candidate_count": 0,
        "candidate_count_raw": 0,
        "source": "profile_scaled_outer_fallback",
        "fallback_reason": "no_valid_hough_outer_candidate",
        "crop_scale_info": scale_info,
    }


def _select_outer_pair_by_ratio_and_center(deduped: List[Dict[str, Any]], profile: Optional[Dict[str, Any]], w: int, h: int) -> Optional[Dict[str, Any]]:
    """白文字盤円とベゼル円のペアを選び、preferred_outer_reference側の円を外周基準として返す。

    v9.19:
      - assistantの3点指定により、dial/bezel中心は必ずしも一致しない。
      - 半径比に加え、dial->bezel中心オフセットがプロファイルと近いかを評価する。
      - ハブ半径から外周を補正する処理は行わない。
    """
    if not deduped or not isinstance(profile, dict):
        return None
    pair_model = profile.get("outer_circle_model") or {}
    ratio_model = profile.get("component_ratio_model") or {}
    expected_ratio = None
    try:
        expected_ratio = float(pair_model.get("bezel_to_dial_radius_ratio") or ratio_model.get("bezel_to_dial_radius_ratio"))
    except Exception:
        expected_ratio = None
    if expected_ratio is None or expected_ratio <= 1.0:
        return None
    preferred = str(pair_model.get("preferred_outer_reference") or profile.get("geometry", {}).get("outer_pair_preferred_reference") or "dial_face_circle")
    expected = _outer_pair_expected(profile, w, h)
    exp_offset = expected.get("expected_bezel_from_dial_offset") if expected.get("available") else None

    enriched = []
    for cand in deduped:
        c = dict(cand)
        if not c.get("outer_pair_class"):
            cls = _classify_outer_candidate_by_pair(profile, float(c.get("radius", 0)), w, h)
            c.update({
                "outer_pair_class": cls.get("outer_pair_class"),
                "outer_pair_preferred": cls.get("preferred_outer_reference"),
                "outer_pair_debug": cls,
            })
        enriched.append(c)

    dial_candidates = [c for c in enriched if c.get("outer_pair_class") == "dial_face_circle"]
    bezel_candidates = [c for c in enriched if c.get("outer_pair_class") == "bezel_case_circle"]
    if not dial_candidates:
        dial_candidates = sorted(enriched, key=lambda c: float(c.get("radius", 0)))[:3]
    if not bezel_candidates:
        bezel_candidates = sorted(enriched, key=lambda c: float(c.get("radius", 0)), reverse=True)[:3]
    dial_candidates = sorted(dial_candidates, key=lambda c: float(c.get("score", 0)), reverse=True)[:3]
    bezel_candidates = sorted(bezel_candidates, key=lambda c: float(c.get("score", 0)), reverse=True)[:3]

    best_pair = None
    best_pair_score = -1e18
    min_dim = min(w, h)
    for d in dial_candidates:
        for b in bezel_candidates:
            dr = float(d.get("radius", 0))
            br = float(b.get("radius", 0))
            if dr <= 1 or br <= 1 or br <= dr:
                continue
            dc = d.get("center", (0, 0))
            bc = b.get("center", (0, 0))
            obs_dx = float(bc[0] - dc[0])
            obs_dy = float(bc[1] - dc[1])
            center_dist = math.hypot(obs_dx, obs_dy)
            ratio = br / dr
            ratio_err = abs(ratio - expected_ratio) / max(expected_ratio, 1e-6)
            # 3点指定モデルでは中心一致ではなく、保存済み中心オフセットの再現性を見る。
            if exp_offset is not None:
                ex, ey = float(exp_offset[0]), float(exp_offset[1])
                offset_err_px = math.hypot(obs_dx - ex, obs_dy - ey)
                offset_err = offset_err_px / max(dr, 1.0)
                expected_offset_px = math.hypot(ex, ey)
            else:
                offset_err_px = center_dist
                offset_err = center_dist / max(dr, 1.0)
                expected_offset_px = 0.0

            exp_dial = expected.get("dial_face_circle") or {}
            exp_bezel = expected.get("bezel_case_circle") or {}
            exp_dc = exp_dial.get("center")
            exp_dr = float(exp_dial.get("radius") or 0.0)
            exp_bc = exp_bezel.get("center")
            exp_br = float(exp_bezel.get("radius") or 0.0)
            if exp_dc is not None and exp_dr > 1.0:
                dial_center_err_px = math.hypot(float(dc[0] - exp_dc[0]), float(dc[1] - exp_dc[1]))
                dial_center_err = dial_center_err_px / max(exp_dr, 1.0)
                dial_radius_err = abs(dr - exp_dr) / max(exp_dr, 1.0)
            else:
                dial_center_err_px = 0.0
                dial_center_err = 0.0
                dial_radius_err = 0.0
            if exp_bc is not None and exp_br > 1.0:
                bezel_center_err_px = math.hypot(float(bc[0] - exp_bc[0]), float(bc[1] - exp_bc[1]))
                bezel_center_err = bezel_center_err_px / max(exp_br, 1.0)
                bezel_radius_err = abs(br - exp_br) / max(exp_br, 1.0)
            else:
                bezel_center_err_px = 0.0
                bezel_center_err = 0.0
                bezel_radius_err = 0.0

            pair_score = (
                float(d.get("score", 0)) * 0.24
                + float(b.get("score", 0)) * 0.24
                - ratio_err * 900.0
                - offset_err * 850.0
                - dial_center_err * 380.0
                - bezel_center_err * 380.0
                - dial_radius_err * 260.0
                - bezel_radius_err * 260.0
                + max(0.0, 1.0 - ratio_err / 0.08) * 450.0
                + max(0.0, 1.0 - offset_err / 0.04) * 450.0
                + max(0.0, 1.0 - dial_center_err / 0.08) * 280.0
                + max(0.0, 1.0 - bezel_center_err / 0.08) * 280.0
                + max(0.0, 1.0 - dial_radius_err / 0.08) * 180.0
                + max(0.0, 1.0 - bezel_radius_err / 0.08) * 180.0
            )
            if offset_err > 0.12:
                pair_score -= 650.0
            if ratio_err > 0.18:
                pair_score -= 650.0
            # 明らかに離れすぎるペアは除外寄りにする。3点指定でも中心差は通常半径の十数%以下を想定。
            if center_dist > max(36.0, min_dim * 0.10):
                pair_score -= 450.0
            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_pair = (d, b, ratio, ratio_err, center_dist, offset_err, offset_err_px, expected_offset_px, exp_offset)
    if best_pair is None:
        return None
    d, b, ratio, ratio_err, center_dist, offset_err, offset_err_px, expected_offset_px, exp_offset = best_pair
    selected = b if preferred == "bezel_case_circle" else d
    selected = dict(selected)
    selected["outer_pair_selected_by_ratio_model"] = True
    selected["outer_pair_selected_role"] = preferred
    selected["outer_pair_score"] = float(best_pair_score)
    selected["outer_pair_dial_candidate"] = {"center": d.get("center"), "radius": d.get("radius"), "score": d.get("score")}
    selected["outer_pair_bezel_candidate"] = {"center": b.get("center"), "radius": b.get("radius"), "score": b.get("score")}
    selected["outer_pair_ratio"] = float(ratio)
    selected["outer_pair_expected_ratio"] = float(expected_ratio)
    selected["outer_pair_ratio_error"] = float(ratio_err)
    selected["outer_pair_center_distance_px"] = float(center_dist)
    selected["outer_pair_expected_center_offset_px"] = exp_offset
    selected["outer_pair_expected_center_offset_length_px"] = float(expected_offset_px)
    selected["outer_pair_center_offset_error_px"] = float(offset_err_px)
    selected["outer_pair_center_offset_error"] = float(offset_err)
    selected["outer_pair_center_error"] = float(offset_err)
    selected["outer_pair_dial_center_error"] = float(dial_center_err)
    selected["outer_pair_bezel_center_error"] = float(bezel_center_err)
    selected["outer_pair_dial_radius_error"] = float(dial_radius_err)
    selected["outer_pair_bezel_radius_error"] = float(bezel_radius_err)
    selected["outer_pair_preferred"] = preferred
    selected["outer_pair_selection_model"] = "radius_ratio_plus_3point_center_offset"
    selected["_selected_center_radius_key"] = (int(selected["center"][0]), int(selected["center"][1]), int(selected["radius"]))
    return selected


def detect_outer_circle(
    meter_img: np.ndarray,
    expected_center: Optional[Point] = None,
    profile: Optional[Dict[str, Any]] = None,
):
    """外周円から中心・半径を推定する。

    v8.6:
      - 外周円探索だけ縮小画像で実行し、HoughCirclesを高速化する。
      - 候補数を制限し、円周エッジ評価のサンプリング点数も削減する。
      - 良好な候補が見つかった場合は早期終了する。
      - 結果は元画像座標へ戻して返すため、後段の中心/針検出は元画像で実行できる。
    """
    t0 = time.perf_counter()
    h0, w0 = meter_img.shape[:2]
    max_outer_search_dim = 640
    max_dim0 = max(w0, h0)
    scale = 1.0

    if max_dim0 > max_outer_search_dim:
        scale = float(max_outer_search_dim) / float(max_dim0)
        search_img = cv2.resize(
            meter_img,
            (max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        search_img = meter_img

    hs, ws = search_img.shape[:2]
    inv_scale = 1.0 / max(scale, 1e-9)

    scaled_expected_center = None
    if expected_center is not None:
        scaled_expected_center = (
            int(round(expected_center[0] * scale)),
            int(round(expected_center[1] * scale)),
        )

    gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(gray, 60, 160)

    candidates: List[Dict[str, Any]] = []
    ranges = _build_outer_circle_search_ranges(ws, hs, profile)
    # 速度重視。最初は候補数を絞る。見つからない場合だけ緩める。
    param2_values = [42, 36, 30]
    max_circles_per_call = 60
    early_exit = False

    for min_r, max_r, range_label in ranges:
        # v9.12: 外周円候補を一覧表示するため、早期終了せず複数レンジを探索する。
        for param2 in param2_values:
            try:
                circles = cv2.HoughCircles(
                    blur,
                    cv2.HOUGH_GRADIENT,
                    dp=1.25,
                    minDist=max(20, min(ws, hs) // 3),
                    param1=90,
                    param2=param2,
                    minRadius=int(min_r),
                    maxRadius=int(max_r),
                )
            except Exception:
                circles = None

            if circles is None:
                continue

            circles = np.round(circles[0, :]).astype("int")
            if len(circles) > max_circles_per_call:
                img_center = (ws // 2, hs // 2)
                # 画像中央に近く、レンジ中央値に近い候補を優先評価する。
                expected_r = (float(min_r) + float(max_r)) / 2.0
                scored_for_limit = []
                for x, y, r in circles:
                    d = math.hypot(int(x) - img_center[0], int(y) - img_center[1])
                    rd = abs(float(r) - expected_r)
                    scored_for_limit.append((d + rd * 0.35, x, y, r))
                scored_for_limit.sort(key=lambda v: v[0])
                circles = np.array([[x, y, r] for _, x, y, r in scored_for_limit[:max_circles_per_call]], dtype=int)

            local_valid: List[Dict[str, Any]] = []
            for x, y, r in circles:
                cand = _evaluate_outer_circle_candidate(
                    img_bgr=search_img,
                    gray=gray,
                    edges=edges,
                    x=int(x),
                    y=int(y),
                    r=int(r),
                    expected_center=scaled_expected_center,
                    profile=profile,
                    range_label=f"{range_label}/param2={param2}",
                )
                if cand is not None:
                    local_valid.append(cand)
                    candidates.append(cand)

            if local_valid:
                best_local = max(local_valid, key=lambda c: c["score"])
                # 十分良い候補が見つかったら、以降の緩い条件探索を止める。
                if (
                    (range_label == "profile_scaled_radius" and best_local.get("edge_support", 0.0) >= 0.18)
                    or best_local.get("edge_support", 0.0) >= 0.24
                    or best_local.get("score", 0.0) >= 150.0
                ):
                    # v9.12: 候補比較用に探索は継続。フラグは互換のため残す。
                    early_exit = False

    if not candidates:
        fallback = _profile_scaled_outer_fallback(profile, ws, hs)
        if fallback is None:
            return None
        best = dict(fallback)
        sx, sy = best["center"]
        sr = best["radius"]
        best["center_scaled"] = (int(sx), int(sy))
        best["radius_scaled"] = int(sr)
        best["center"] = (int(round(float(sx) * inv_scale)), int(round(float(sy) * inv_scale)))
        best["radius"] = int(round(float(sr) * inv_scale))
        best["candidate_details"] = []
        best["outer_pair_top3_by_role"] = {"dial_face_circle": [], "bezel_case_circle": []}
        best["source"] = best.get("source") or "profile_scaled_outer_fallback"
        best["outer_search_scale"] = float(scale)
        best["crop_scale_info"] = _profile_crop_scale_info(profile or {}, w0, h0) if isinstance(profile, dict) else {"available": False}
        best["outer_search_size"] = (int(ws), int(hs))
        best["outer_search_elapsed_ms"] = 0.0
        best["outer_search_early_exit"] = False
        return best

    # 近い候補を重複除去し、最良スコアを残す。
    deduped: List[Dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: c["score"], reverse=True):
        cx, cy = cand["center"]
        r = cand["radius"]
        duplicate = False
        for kept in deduped:
            kx, ky = kept["center"]
            kr = kept["radius"]
            if math.hypot(cx - kx, cy - ky) < max(8, min(r, kr) * 0.04) and abs(r - kr) < max(8, min(r, kr) * 0.04):
                duplicate = True
                break
        if not duplicate:
            deduped.append(cand)

    pair_selected = _select_outer_pair_by_ratio_and_center(deduped, profile, ws, hs)
    if pair_selected is not None:
        best = pair_selected
    else:
        best = max(deduped, key=lambda c: c["score"])

    # 縮小画像座標から元画像座標へ戻す。
    sx, sy = best["center"]
    sr = best["radius"]
    selected_key_scaled = (int(sx), int(sy), int(sr))
    best = dict(best)
    best["center_scaled"] = (int(sx), int(sy))
    best["radius_scaled"] = int(sr)
    best["center"] = (
        int(round(float(sx) * inv_scale)),
        int(round(float(sy) * inv_scale)),
    )
    best["radius"] = int(round(float(sr) * inv_scale))
    # v9.12: 外周円候補一覧を元画像座標へ戻して保持する。
    outer_candidate_details = []
    for idx, cand in enumerate(deduped, start=1):
        dc = dict(cand)
        csx, csy = dc.get("center", (0, 0))
        csr = int(dc.get("radius", 0))
        dc["index"] = idx
        dc["center_scaled"] = (int(csx), int(csy))
        dc["radius_scaled"] = int(csr)
        dc["center"] = (int(round(float(csx) * inv_scale)), int(round(float(csy) * inv_scale)))
        dc["radius"] = int(round(float(csr) * inv_scale))
        dc["selected"] = (int(csx), int(csy), int(csr)) == selected_key_scaled
        # 最大3候補ずつの確認用に分類を明示する。
        if dc.get("outer_pair_class") is None:
            cls_dbg = _classify_outer_candidate_by_pair(profile, float(dc.get("radius_scaled", dc.get("radius", 0))), ws, hs)
            dc["outer_pair_class"] = cls_dbg.get("outer_pair_class")
            dc["outer_pair_debug"] = cls_dbg
        outer_candidate_details.append(dc)

    best["candidate_count"] = len(deduped)
    best["candidate_count_raw"] = len(candidates)
    # role別上位3件の概略をログへ出すため保持
    best["outer_pair_top3_by_role"] = {
        "dial_face_circle": [
            {"center": c.get("center"), "radius": c.get("radius"), "score": c.get("score")}
            for c in sorted([c for c in outer_candidate_details if c.get("outer_pair_class") == "dial_face_circle"], key=lambda x: float(x.get("score", 0)), reverse=True)[:3]
        ],
        "bezel_case_circle": [
            {"center": c.get("center"), "radius": c.get("radius"), "score": c.get("score")}
            for c in sorted([c for c in outer_candidate_details if c.get("outer_pair_class") == "bezel_case_circle"], key=lambda x: float(x.get("score", 0)), reverse=True)[:3]
        ],
    }
    best["candidate_details"] = outer_candidate_details
    best["source"] = best.get("source") or "outer_circle_color_model_no_hub_scale"
    best["outer_search_scale"] = float(scale)
    best["crop_scale_info"] = _profile_crop_scale_info(profile or {}, w0, h0) if isinstance(profile, dict) else {"available": False}
    best["outer_search_size"] = [int(ws), int(hs)]
    best["outer_search_elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    best["outer_search_early_exit"] = bool(early_exit)
    return best



def _point_to_line_distance(point: Point, line_xyxy: List[int]) -> float:
    """点から無限直線への距離[pixel]を返す。"""
    px, py = point
    x1, y1, x2, y2 = [float(v) for v in line_xyxy]
    dx = x2 - x1
    dy = y2 - y1
    denom = math.hypot(dx, dy)
    if denom < 1e-6:
        return 1e9
    return abs(dx * (y1 - py) - (x1 - px) * dy) / denom




def _angle_unit_image(angle_deg: float) -> Tuple[float, float]:
    """calc_angle系の角度を画像座標系の単位ベクトルへ変換する。"""
    rad = math.radians(float(angle_deg))
    return math.cos(rad), -math.sin(rad)


def _line_xyxy_from_center_angle(center: Point, angle_deg: float, r1: float, r2: float) -> List[int]:
    ux, uy = _angle_unit_image(angle_deg)
    cx, cy = center
    return [
        int(round(cx + r1 * ux)), int(round(cy + r1 * uy)),
        int(round(cx + r2 * ux)), int(round(cy + r2 * uy)),
    ]


def _refine_needle_centerline_from_mask(
    mask: np.ndarray,
    center: Point,
    angle_deg: float,
    center_radius: int,
    outer_radius: int,
    half_width: Optional[int] = None,
    min_points: int = 10,
) -> Optional[Dict[str, Any]]:
    """針マスクの太さを考慮し、針幅の中央を通る中心線へ補正する。

    HoughLinesPは針の片側エッジを拾いやすく、線が針中央からずれることがある。
    そこで、候補角度方向に沿って複数断面を取り、各断面内の白画素幅の中央値を
    「その断面における針の中央」として集める。最後に、その中心点群を使って
    針中央線の角度・先端を推定する。
    """
    h, w = mask.shape[:2]
    cx, cy = center
    ux, uy = _angle_unit_image(angle_deg)
    # 直交方向。画像座標の単位ベクトル。
    nx, ny = -uy, ux

    if half_width is None:
        half_width = int(max(12, min(60, outer_radius * 0.075)))
    half_width = int(max(6, half_width))

    start_r = float(max(center_radius * 1.05, outer_radius * 0.035))
    end_r = float(outer_radius * 0.88)
    if end_r <= start_r + 8:
        return None

    samples: List[Tuple[float, float, float, float]] = []  # x,y,r,width
    # 近すぎる根元はハブ・太い尾側の影響が強いので、先端側まで広く見る。
    for r in np.linspace(start_r, end_r, 96):
        x0 = cx + r * ux
        y0 = cy + r * uy
        if x0 < -half_width or x0 >= w + half_width or y0 < -half_width or y0 >= h + half_width:
            continue
        hits = []
        for t in range(-half_width, half_width + 1):
            x = int(round(x0 + t * nx))
            y = int(round(y0 + t * ny))
            if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
                hits.append(t)
        if not hits:
            continue
        # 断面に複数物体がある場合は、中心線に一番近い連続成分を選ぶ。
        groups = []
        s = e = hits[0]
        prev = hits[0]
        for v in hits[1:]:
            if v <= prev + 1:
                e = v
            else:
                groups.append((s, e))
                s = e = v
            prev = v
        groups.append((s, e))
        # 中心に近いが、幅も一定以上ある成分を優先する。
        g = min(groups, key=lambda ge: (abs((ge[0] + ge[1]) * 0.5), -float(ge[1] - ge[0] + 1)))
        mid_t = (g[0] + g[1]) * 0.5
        width = float(g[1] - g[0] + 1)
        px = x0 + mid_t * nx
        py = y0 + mid_t * ny
        samples.append((px, py, float(r), width))

    if len(samples) < min_points:
        return None

    pts = np.array([[p[0], p[1]] for p in samples], dtype=np.float32)
    try:
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    except Exception:
        return None

    vx, vy = float(vx), float(vy)
    # 元の候補角度に近い向きへ揃える。
    if vx * ux + vy * uy < 0:
        vx, vy = -vx, -vy

    # 中心を基準に、中心線点群のうち前方側の最大射影を針先距離にする。
    projs = []
    widths = []
    for px, py, r, width in samples:
        proj = (px - cx) * vx + (py - cy) * vy
        if proj > 0:
            projs.append(float(proj))
            widths.append(float(width))
    if not projs:
        return None
    tip_r = max(projs)
    # 極端な外れを避ける。
    tip_r = float(max(start_r, min(tip_r, outer_radius * 0.92)))
    tip = (int(round(cx + tip_r * vx)), int(round(cy + tip_r * vy)))
    angle = normalize_360(calc_angle(center, tip))

    # 幅プロファイル。centerline断面の実幅を使う。
    proj_width_pairs = sorted(zip(projs, widths), key=lambda v: v[0])
    def width_at_ratio(ratio: float) -> float:
        if not proj_width_pairs:
            return 999.0
        target = tip_r * ratio
        return min(proj_width_pairs, key=lambda v: abs(v[0] - target))[1]
    base_w = width_at_ratio(0.25)
    mid_w = width_at_ratio(0.65)
    tip_w = width_at_ratio(0.92)

    # 画面表示・ハブ評価用の中心線。中心を通る線として描く。
    line = _line_xyxy_from_center_angle(center, angle, max(center_radius * 0.5, outer_radius * 0.02), tip_r)
    return {
        "centerline_available": True,
        "centerline_angle": float(angle),
        "centerline_tip": tip,
        "centerline_tip_r": float(tip_r),
        "centerline_line": line,
        "centerline_sample_count": int(len(samples)),
        "centerline_tip_width": float(tip_w),
        "centerline_mid_width": float(mid_w),
        "centerline_base_width": float(base_w),
        "centerline_half_width": int(half_width),
    }


def refine_tip_angle_by_mask_center_symmetry(
    mask: np.ndarray,
    center: Point,
    base_angle: float,
    outer_radius: int,
    center_radius: int,
    search_deg: float = 6.0,
    step_deg: float = 0.5,
) -> Dict[str, Any]:
    """針のエッジ寄り角度を、針幅の中央を通る角度へ微調整する。

    Hough/centerlineが針の片側エッジを拾った場合、profile_guided_tip_extensionで
    その角度のまま先端を延長すると、先端位置が上下にずれる。ここでは候補角度の
    周辺を探索し、各断面で針マスクの中心が中心線上(t=0)に来る角度を優先する。
    """
    h, w = mask.shape[:2]
    cx, cy = center
    if mask is None or outer_radius <= 0:
        return {"available": False, "reason": "invalid_input", "base_angle": float(base_angle)}

    half_width = int(max(12, min(55, outer_radius * 0.055)))
    start_r = float(max(center_radius * 1.35, outer_radius * 0.07))
    end_r = float(min(outer_radius * 0.88, max(start_r + 20.0, outer_radius * 0.82)))
    if end_r <= start_r + 10:
        return {"available": False, "reason": "invalid_radius_range", "base_angle": float(base_angle)}

    best = None
    candidates = []
    n_steps = int(round((2.0 * search_deg) / max(step_deg, 1e-6)))
    for i in range(n_steps + 1):
        angle = float(base_angle - search_deg + i * step_deg)
        ux, uy = _angle_unit_image(angle)
        nx, ny = -uy, ux
        mids = []
        widths = []
        rs = []
        continuity = 0
        prev_hit = False
        max_continuity = 0
        for r in np.linspace(start_r, end_r, 72):
            x0 = cx + r * ux
            y0 = cy + r * uy
            if x0 < -half_width or x0 >= w + half_width or y0 < -half_width or y0 >= h + half_width:
                continue
            hits = []
            for t in range(-half_width, half_width + 1):
                x = int(round(x0 + t * nx))
                y = int(round(y0 + t * ny))
                if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
                    hits.append(t)
            if not hits:
                if prev_hit:
                    max_continuity = max(max_continuity, continuity)
                prev_hit = False
                continuity = 0
                continue
            groups = []
            s0 = e0 = hits[0]
            prev = hits[0]
            for v in hits[1:]:
                if v <= prev + 1:
                    e0 = v
                else:
                    groups.append((s0, e0))
                    s0 = e0 = v
                prev = v
            groups.append((s0, e0))
            # 中心線に最も近い連結成分を選ぶ。文字や目盛りが近くにある場合の混入を抑える。
            g = min(groups, key=lambda ge: (abs((ge[0] + ge[1]) * 0.5), -float(ge[1] - ge[0] + 1)))
            mid_t = (g[0] + g[1]) * 0.5
            width = float(g[1] - g[0] + 1)
            # 幅が大きすぎるものは文字やハブ周辺混入の可能性があるため弱く扱う。
            if width > max(half_width * 1.5, center_radius * 1.8):
                continue
            mids.append(float(mid_t))
            widths.append(width)
            rs.append(float(r))
            prev_hit = True
            continuity += 1
            max_continuity = max(max_continuity, continuity)
        if len(mids) < 8:
            continue
        mids_abs = [abs(v) for v in mids]
        mean_abs_mid = float(np.mean(mids_abs))
        median_abs_mid = float(np.median(mids_abs))
        coverage = float(len(mids)) / 72.0
        far_r = float(max(rs)) if rs else 0.0
        mean_width = float(np.mean(widths)) if widths else 0.0
        width_var = float(np.std(widths)) if len(widths) >= 2 else 0.0
        # 中心線からのズレが小さく、連続して遠くまで針が存在する角度を優先。
        score = (
            coverage * 260.0
            + (far_r / max(float(outer_radius), 1.0)) * 160.0
            + float(max_continuity) * 2.0
            - mean_abs_mid * 12.0
            - median_abs_mid * 6.0
            - width_var * 1.5
        )
        cand = {
            "angle": normalize_360(angle),
            "score": float(score),
            "coverage": float(coverage),
            "sample_count": int(len(mids)),
            "mean_abs_center_offset_px": float(mean_abs_mid),
            "median_abs_center_offset_px": float(median_abs_mid),
            "far_r": float(far_r),
            "mean_width": float(mean_width),
            "width_std": float(width_var),
            "max_continuity": int(max_continuity),
        }
        candidates.append(cand)
        if best is None or cand["score"] > best["score"]:
            best = cand

    if best is None:
        return {"available": False, "reason": "no_valid_angle", "base_angle": float(base_angle)}
    delta = angle_diff_deg(best["angle"], base_angle)
    return {
        "available": True,
        "base_angle": float(normalize_360(base_angle)),
        "refined_angle": float(best["angle"]),
        "delta_deg": float(delta),
        "search_deg": float(search_deg),
        "step_deg": float(step_deg),
        "selected": best,
        "top_candidates": sorted(candidates, key=lambda c: c["score"], reverse=True)[:7],
    }


def _detect_needle_lines_for_hub_scoring(
    meter_img: np.ndarray,
    approx_center: Point,
    outer_radius: int,
    profile: Dict[str, Any],
    max_lines: int = 8,
) -> List[Dict[str, Any]]:
    """ハブ候補評価用の仮針中央線を検出する。

    v8.13:
      HoughLinesPで針のエッジ線を拾った後、針マスクの断面幅から中心線へ補正する。
      これにより、ハブ候補評価で使う水色の線が、針の片側エッジではなく
      針幅の中央を通りやすくなる。
    """
    h, w = meter_img.shape[:2]
    gray = cv2.cvtColor(meter_img, cv2.COLOR_BGR2GRAY)
    dark_upper = int(profile.get("needle", {}).get("dark_gray_upper", 165))
    dark = cv2.inRange(gray, 0, max(120, min(190, dark_upper)))
    edges = cv2.Canny(gray, 60, 160)
    combined = cv2.bitwise_or(dark, edges)

    valid = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(valid, approx_center, int(max(outer_radius * 0.92, 20)), 255, -1)
    cv2.circle(valid, approx_center, int(max(outer_radius * 0.015, 3)), 0, -1)
    combined = cv2.bitwise_and(combined, valid)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # v9.7.1 fix:
    # _detect_needle_lines_for_hub_scoring() is used before the final needle
    # connected-mask extraction step.  v9.7 accidentally attached
    # `connected_debug` to each candidate without defining it in this scope,
    # causing a NameError during hub detection.
    # Keep a small debug payload here; the final needle path still computes
    # the real connected-mask debug information later.
    connected_debug = {
        "available": False,
        "stage": "hub_scoring_probe",
        "reason": "not_computed_in_hub_scoring",
    }

    lines = cv2.HoughLinesP(
        combined,
        rho=1,
        theta=np.pi / 180,
        threshold=24,
        minLineLength=max(28, int(outer_radius * 0.16)),
        maxLineGap=max(14, int(outer_radius * 0.055)),
    )
    if lines is None:
        return []

    cx, cy = approx_center
    candidates: List[Dict[str, Any]] = []
    for line in lines[:80]:
        x1, y1, x2, y2 = map(int, line[0])
        length = math.hypot(x2 - x1, y2 - y1)
        if length < max(28, outer_radius * 0.14):
            continue
        dist_line_center = _point_to_line_distance(approx_center, [x1, y1, x2, y2])
        if dist_line_center > max(outer_radius * 0.12, 48.0):
            continue
        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)
        max_d = max(d1, d2)
        min_d = min(d1, d2)

        raw_angle = normalize_360(calc_angle(approx_center, (x1, y1) if d1 > d2 else (x2, y2)))
        centerline = _refine_needle_centerline_from_mask(
            combined,
            center=approx_center,
            angle_deg=raw_angle,
            center_radius=int(max(outer_radius * 0.025, 5)),
            outer_radius=outer_radius,
            half_width=int(max(12, min(55, outer_radius * 0.07))),
            min_points=8,
        )
        if centerline is not None:
            line_xyxy = centerline["centerline_line"]
            dist_line_center = _point_to_line_distance(approx_center, line_xyxy)
            length = float(centerline["centerline_tip_r"])
            max_d = length
            score_centerline_bonus = 80.0 + float(centerline["centerline_sample_count"]) * 1.2
        else:
            line_xyxy = [x1, y1, x2, y2]
            score_centerline_bonus = 0.0

        score = length * 1.4 + max_d * 0.6 - dist_line_center * 6.0 - max(0.0, min_d - outer_radius * 0.28) * 2.0 + score_centerline_bonus
        cand = {
            "line": [int(v) for v in line_xyxy],
            "raw_line": [x1, y1, x2, y2],
            "length": float(length),
            "dist_to_approx_center": float(dist_line_center),
            "score": float(score),
            "centerline_used": bool(centerline is not None),
            "connected_mask_debug": connected_debug,
        }
        if centerline is not None:
            cand.update(centerline)
        candidates.append(cand)

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:max_lines]

def _needle_line_support_for_hub(candidate_center: Point, needle_lines: List[Dict[str, Any]], expected_radius: Optional[float], outer_radius: int) -> Dict[str, Any]:
    """ハブ候補中心が仮針中央線上に近いかを評価する。"""
    if not needle_lines:
        return {"available": False, "distance": None, "allow": None, "penalty": 0.0, "line": None}
    best = None
    for ln in needle_lines:
        d = _point_to_line_distance(candidate_center, ln["line"])
        v = {**ln, "distance_to_hub": float(d)}
        if best is None or d < best["distance_to_hub"]:
            best = v
    allow = max((float(expected_radius) * 0.42) if expected_radius else 0.0, float(outer_radius) * 0.018, 8.0)
    distance = float(best["distance_to_hub"])
    # 絶対条件ではなくスコア要素。大きく外れた時だけ強く減点する。
    penalty = max(0.0, distance - allow) * 3.2
    bonus = -min(distance, allow) * 0.25
    return {"available": True, "distance": distance, "allow": float(allow), "penalty": float(penalty + bonus), "line": best.get("line")}


def _hub_selection_model_score_adjust(
    profile: Dict[str, Any],
    candidate_source: str,
    candidate_center: Point,
    candidate_radius: float,
    expected_center: Optional[Point],
    expected_radius: Optional[float],
    outer_radius: float,
) -> Tuple[float, Dict[str, Any]]:
    """人が選択したハブ候補の特徴を使い、候補スコアを補正する。小さいほど良い。"""
    model = profile.get("hub_selection_model", {}) or {}
    features = model.get("selection_features", {}) or {}
    weights = model.get("scoring_weights", {}) or {}
    if not features:
        return 0.0, {"available": False}

    outer = max(float(outer_radius), 1.0)
    exp_r = max(float(expected_radius), 1.0) if expected_radius is not None else max(float(candidate_radius), 1.0)
    cx, cy = candidate_center
    ex, ey = expected_center if expected_center is not None else candidate_center

    radius_ratio = float(candidate_radius) / exp_r
    radius_over_outer = float(candidate_radius) / outer
    dx_ratio = float(cx - ex) / outer
    dy_ratio = float(cy - ey) / outer

    penalty = 0.0
    details = {
        "available": True,
        "candidate_radius_ratio": radius_ratio,
        "candidate_radius_over_outer": radius_over_outer,
        "candidate_offset_ratio": {"dx": dx_ratio, "dy": dy_ratio},
    }

    pref_source = model.get("selected_candidate_type") or features.get("candidate_source")
    if pref_source and candidate_source == pref_source:
        bonus = float(weights.get("source_match_bonus", 18.0))
        penalty -= bonus
        details["source_match_bonus"] = bonus
    elif pref_source:
        # source mismatch is not fatal; just no bonus.
        details["source_match_bonus"] = 0.0

    if "radius_ratio" in features:
        pref = float(features["radius_ratio"])
        w = float(weights.get("preferred_radius_ratio_weight", 45.0))
        v = abs(radius_ratio - pref) * w
        penalty += v
        details["preferred_radius_ratio"] = pref
        details["preferred_radius_ratio_penalty"] = v

    if "hub_radius_over_outer_radius" in features:
        pref = float(features["hub_radius_over_outer_radius"])
        w = float(weights.get("preferred_radius_over_outer_weight", 220.0))
        v = abs(radius_over_outer - pref) * w
        penalty += v
        details["preferred_radius_over_outer"] = pref
        details["preferred_radius_over_outer_penalty"] = v

    off = features.get("offset_from_profile_center_ratio", {}) or {}
    if "dx_over_outer_radius" in off and "dy_over_outer_radius" in off:
        pref_dx = float(off["dx_over_outer_radius"])
        pref_dy = float(off["dy_over_outer_radius"])
        w = float(weights.get("preferred_offset_weight", 320.0))
        v = math.hypot(dx_ratio - pref_dx, dy_ratio - pref_dy) * w
        penalty += v
        details["preferred_offset"] = {"dx": pref_dx, "dy": pref_dy}
        details["preferred_offset_penalty"] = v

    return float(penalty), details


def _profile_bgr_mean(cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return BGR mean from a profile color block if available."""
    if not isinstance(cfg, dict):
        return None
    bgr = cfg.get("bgr_mean") or {}
    try:
        if all(k in bgr for k in ("b", "g", "r")):
            return np.array([float(bgr["b"]), float(bgr["g"]), float(bgr["r"])], dtype=np.float32)
    except Exception:
        pass
    rgb = cfg.get("rgb_mean") or {}
    try:
        if all(k in rgb for k in ("r", "g", "b")):
            return np.array([float(rgb["b"]), float(rgb["g"]), float(rgb["r"])], dtype=np.float32)
    except Exception:
        pass
    return None


def _mean_bgr_in_circle(img: np.ndarray, center: Point, radius: float, inner_scale: float = 0.70) -> Optional[np.ndarray]:
    """Mean BGR color inside a circle, excluding a tiny center if possible."""
    h, w = img.shape[:2]
    x, y = int(round(center[0])), int(round(center[1]))
    r = max(2, int(round(radius * inner_scale)))
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    vals = img[mask > 0]
    if vals.size == 0:
        return None
    return vals.reshape(-1, 3).mean(axis=0).astype(np.float32)


def _hub_color_similarity_score(img: np.ndarray, center: Point, radius: float, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Score candidate by how naturally its interior color matches the hub color profile.

    Lower penalty is better. This intentionally uses hub/center color only; needle-line
    information is not used here because needle roots can bias hub selection.
    """
    cfg = profile.get("hub_color") or profile.get("center") or {}
    ref = _profile_bgr_mean(cfg)
    mean = _mean_bgr_in_circle(img, center, radius, inner_scale=0.68)
    if ref is None or mean is None:
        return {"available": False, "distance": None, "penalty": 0.0, "mean_bgr": None}
    dist = float(np.linalg.norm(mean - ref))
    # 0-20 is usually very good; distances above 80 are suspicious.
    penalty = min(dist, 120.0) * 0.42
    return {
        "available": True,
        "distance": dist,
        "penalty": float(penalty),
        "mean_bgr": [float(mean[0]), float(mean[1]), float(mean[2])],
        "ref_bgr": [float(ref[0]), float(ref[1]), float(ref[2])],
    }


def _circle_edge_support_score(gray: np.ndarray, center: Point, radius: float) -> Dict[str, Any]:
    """Evaluate whether a circle candidate has edge support on its circumference."""
    h, w = gray.shape[:2]
    x0, y0 = float(center[0]), float(center[1])
    r = float(radius)
    if r <= 1:
        return {"available": False, "support": 0.0, "penalty": 25.0, "mean_edge": 0.0}
    edges = cv2.Canny(gray, 60, 140)
    vals = []
    hits = 0
    total = 0
    for deg in range(0, 360, 4):
        th = math.radians(deg)
        x = int(round(x0 + r * math.cos(th)))
        y = int(round(y0 + r * math.sin(th)))
        if 1 <= x < w - 1 and 1 <= y < h - 1:
            patch = edges[y-1:y+2, x-1:x+2]
            v = int(patch.max())
            vals.append(v)
            hits += 1 if v > 0 else 0
            total += 1
    if total <= 0:
        return {"available": False, "support": 0.0, "penalty": 25.0, "mean_edge": 0.0}
    support = hits / float(total)
    mean_edge = float(np.mean(vals)) if vals else 0.0
    # Low support means the candidate is not really circular.
    penalty = max(0.0, 0.32 - support) * 90.0
    return {"available": True, "support": float(support), "penalty": float(penalty), "mean_edge": mean_edge}


def _scaled_hub_adoption_distance(
    profile_hub_radius: Optional[float],
    outer_radius: float,
    image_w: int,
    image_h: int,
) -> float:
    """Return scale-adaptive distance allowance for accepting a real hub candidate.

    This replaces the old fixed lower bound such as 24 px. The allowance follows
    the current crop scale, so small meter crops stay strict and large crops allow
    proportionally larger perspective/outer-circle mapping errors.
    """
    min_dim = max(1.0, float(min(image_w, image_h)))
    outer = max(1.0, float(outer_radius))
    if profile_hub_radius is not None and float(profile_hub_radius) > 0:
        phr = float(profile_hub_radius)
        return float(max(phr * 0.82, outer * 0.035, min_dim * 0.020))
    return float(max(outer * 0.075, min_dim * 0.030))


def _hub_needle_connection_score(
    candidate_center: Point,
    candidate_radius: float,
    needle_lines: List[Dict[str, Any]],
    outer_radius: float,
) -> Dict[str, Any]:
    """Evaluate whether a real hub candidate naturally connects to the provisional needle axis."""
    if not needle_lines:
        return {"available": False, "distance": None, "allow": None, "connected": None, "score": 0.0}
    best = None
    for ln in needle_lines:
        d = _point_to_line_distance(candidate_center, ln.get("line", [0, 0, 0, 0]))
        item = {"distance": float(d), "line": ln.get("line"), "line_score": ln.get("score")}
        if best is None or d < best["distance"]:
            best = item
    radius = max(1.0, float(candidate_radius))
    allow = max(radius * 0.85, float(outer_radius) * 0.018)
    distance = float(best["distance"]) if best else 1e9
    connected = bool(distance <= allow)
    score = float(max(0.0, 1.0 - distance / max(allow, 1.0)))
    return {
        "available": True,
        "distance": distance,
        "allow": float(allow),
        "connected": connected,
        "score": score,
        "line": best.get("line") if best else None,
        "line_score": best.get("line_score") if best else None,
    }


def _hub_natural_score(
    meter_img: np.ndarray,
    gray: np.ndarray,
    candidate_center: Point,
    candidate_radius: float,
    expected_center: Optional[Point],
    expected_radius: Optional[float],
    approx_center: Point,
    outer_radius: float,
    profile: Dict[str, Any],
    candidate_source: str,
) -> Tuple[float, Dict[str, Any]]:
    """Natural hub score: circle + color + size + position.

    Lower score is better. Needle line support is intentionally not included as a
    primary cue. This implements the policy: choose it because it looks like a hub,
    not because it lies on a provisional needle line.
    """
    cx, cy = float(candidate_center[0]), float(candidate_center[1])
    er = float(expected_radius) if expected_radius else None
    ec = expected_center if expected_center is not None else approx_center
    dist_profile = math.hypot(cx - float(ec[0]), cy - float(ec[1]))
    dist_outer_center = math.hypot(cx - float(approx_center[0]), cy - float(approx_center[1]))

    radius_ratio = (float(candidate_radius) / max(er, 1.0)) if er else None
    if er:
        # Strong but not absolute size preference. A small hub radius is noisy, so use
        # smooth penalties instead of hard replacement/scaling.
        radius_penalty = abs(float(candidate_radius) - er) * 2.8
        ratio_penalty = abs(radius_ratio - 1.0) * er * 7.0
        if radius_ratio > 1.22:
            ratio_penalty += (radius_ratio - 1.22) * er * 24.0
        if radius_ratio < 0.78:
            ratio_penalty += (0.78 - radius_ratio) * er * 24.0
    else:
        radius_penalty = 0.0
        ratio_penalty = 0.0

    # Position: close to profile-guided hub center is important, but allows modest
    # deviations caused by perspective and outer-circle shifts.
    center_allow = max((er or 16.0) * 0.55, float(outer_radius) * 0.018, 10.0)
    center_penalty = max(0.0, dist_profile - center_allow) * 3.2 + min(dist_profile, center_allow) * 0.25
    outer_center_penalty = max(0.0, dist_outer_center - max(float(outer_radius) * 0.08, 18.0)) * 0.6

    color_info = _hub_color_similarity_score(meter_img, candidate_center, candidate_radius, profile)
    edge_info = _circle_edge_support_score(gray, candidate_center, candidate_radius)

    selection_penalty, selection_detail = _hub_selection_model_score_adjust(
        profile=profile,
        candidate_source=candidate_source,
        candidate_center=(int(round(cx)), int(round(cy))),
        candidate_radius=float(candidate_radius),
        expected_center=expected_center,
        expected_radius=expected_radius,
        outer_radius=float(outer_radius),
    )
    # Human selection model is useful, but should not dominate natural geometry/color.
    selection_penalty *= 0.65

    # v9.22: the profile-guided estimate P is useful as a reference/fallback,
    # but it is not an observed circle in the current image. Add a small structural
    # penalty so an actual Hough/color candidate with plausible natural evidence can win.
    profile_fallback_penalty = 0.0
    if candidate_source == "profile_guided_estimate":
        profile_fallback_penalty = max((er or 20.0) * 0.85, float(outer_radius) * 0.035)

    score = (
        radius_penalty
        + ratio_penalty
        + center_penalty
        + outer_center_penalty
        + float(color_info.get("penalty", 0.0))
        + float(edge_info.get("penalty", 0.0))
        + selection_penalty
        + profile_fallback_penalty
    )
    details = {
        "natural_score": float(score),
        "dist_profile": float(dist_profile),
        "dist_outer_center": float(dist_outer_center),
        "center_allow": float(center_allow),
        "center_penalty": float(center_penalty),
        "outer_center_penalty": float(outer_center_penalty),
        "radius_ratio": None if radius_ratio is None else float(radius_ratio),
        "radius_penalty": float(radius_penalty),
        "ratio_penalty": float(ratio_penalty),
        "hub_color": color_info,
        "circle_edge": edge_info,
        "selection_model_penalty": float(selection_penalty),
        "selection_model_detail": selection_detail,
        "profile_fallback_penalty": float(profile_fallback_penalty),
    }
    return float(score), details


def detect_center_hub(meter_img: np.ndarray, approx_center: Point, outer_radius: int, profile: Dict[str, Any]):
    """Detect center hub using natural hub evidence: circle, color, size, position.

    v9.27_scale_adaptive_real_hub_acceptance:
      - profile_guided_hub_center is a reference/fallback, not the fixed answer.
      - Real image hub candidates are preferred over P only when radius, color, circular edge,
        and needle-axis connectivity are all natural enough.
      - Hub adoption distance is scale-adaptive from current crop/outer radius, not a fixed 24 px.
      - Assistant-selected hub candidate remains feature guidance, not an absolute coordinate.
      - Keep all hub candidates visible in debug output.
    """
    h, w = meter_img.shape[:2]
    gray = cv2.cvtColor(meter_img, cv2.COLOR_BGR2GRAY)

    hub_cfg = get_profile_hub(profile, current_outer_center=approx_center, current_outer_radius=outer_radius)
    expected_center = hub_cfg.get("center", approx_center)
    expected_radius = hub_cfg.get("radius")

    search_center = expected_center
    if math.hypot(search_center[0] - approx_center[0], search_center[1] - approx_center[1]) > max(outer_radius * 0.20, 60):
        search_center = approx_center

    # Keep provisional needle lines only for logging. They are intentionally not a
    # strong scoring signal because needle roots/edges can bias hub choice.
    needle_lines_for_hub = _detect_needle_lines_for_hub_scoring(
        meter_img=meter_img,
        approx_center=expected_center if expected_center is not None else approx_center,
        outer_radius=outer_radius,
        profile=profile,
    )

    if expected_radius is not None:
        search_r = int(max(
            expected_radius * hub_cfg.get("search_radius_scale", 1.8),
            outer_radius * 0.07,
            _scaled_hub_adoption_distance(expected_radius, outer_radius, w, h) * 1.20,
        ))
        min_r = max(3, int(expected_radius * hub_cfg.get("min_radius_scale", 0.65)))
        max_r = max(min_r + 2, int(expected_radius * hub_cfg.get("max_radius_scale", 1.45)))
    else:
        search_r = int(max(outer_radius * 0.18, 30))
        min_r = max(4, int(outer_radius * 0.025))
        max_r = max(min_r + 2, int(outer_radius * 0.090))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, search_center, search_r, 255, -1)

    roi_gray = cv2.bitwise_and(gray, gray, mask=mask)
    blur = cv2.GaussianBlur(roi_gray, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, search_r // 2),
        param1=80,
        param2=18,
        minRadius=min_r,
        maxRadius=max_r,
    )

    candidates: List[Tuple[float, int, int, int, str]] = []
    debug_candidates: List[Dict[str, Any]] = []

    def add_candidate(source: str, center: Point, radius: float, note: str, extra: Optional[Dict[str, Any]] = None):
        x, y = int(round(center[0])), int(round(center[1]))
        r = int(max(round(radius), 1))
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        if math.hypot(x - search_center[0], y - search_center[1]) > search_r * 1.15 and source != "profile_guided_estimate":
            return
        score, details = _hub_natural_score(
            meter_img=meter_img,
            gray=gray,
            candidate_center=(x, y),
            candidate_radius=float(r),
            expected_center=expected_center,
            expected_radius=expected_radius,
            approx_center=approx_center,
            outer_radius=float(outer_radius),
            profile=profile,
            candidate_source=source,
        )
        support = _needle_line_support_for_hub((x, y), needle_lines_for_hub, expected_radius, outer_radius)
        needle_connection = _hub_needle_connection_score((x, y), float(r), needle_lines_for_hub, float(outer_radius))
        item = {
            "source": source,
            "center": [x, y],
            "radius": r,
            "score": float(score),
            "selected": False,
            "note": note,
            "natural_score_detail": details,
            "dist_search": float(math.hypot(x - search_center[0], y - search_center[1])),
            "radius_ratio": details.get("radius_ratio"),
            "radius_penalty": details.get("radius_penalty"),
            "radius_ratio_penalty": details.get("ratio_penalty"),
            "profile_distance": details.get("dist_profile"),
            "profile_distance_penalty": details.get("center_penalty"),
            "hub_color_distance": (details.get("hub_color") or {}).get("distance"),
            "hub_color_penalty": (details.get("hub_color") or {}).get("penalty"),
            "circle_edge_support": (details.get("circle_edge") or {}).get("support"),
            "circle_edge_penalty": (details.get("circle_edge") or {}).get("penalty"),
            "selection_model_penalty": details.get("selection_model_penalty"),
            "selection_model_detail": details.get("selection_model_detail"),
            "needle_line_distance": support.get("distance"),
            "needle_line_allow": support.get("allow"),
            "needle_line_penalty_debug_only": support.get("penalty"),
            "needle_connection": needle_connection,
            "needle_connection_distance": needle_connection.get("distance"),
            "needle_connection_allow": needle_connection.get("allow"),
            "needle_connection_connected": needle_connection.get("connected"),
            "needle_connection_score": needle_connection.get("score"),
        }
        if extra:
            item.update(extra)
        candidates.append((float(score), x, y, r, source))
        debug_candidates.append(item)

    # Profile-guided hub is not just a display item anymore; it is a real fallback
    # candidate scored by the same natural metrics. This prevents a weak Hough circle
    # from replacing a good human/profile hub estimate.
    if expected_center is not None and expected_radius is not None:
        add_candidate(
            "profile_guided_estimate",
            (int(expected_center[0]), int(expected_center[1])),
            float(expected_radius),
            "YAML hub scaled by outer circle; scored as natural fallback",
        )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            add_candidate("hough_hub", (int(x), int(y)), float(r), "circle detection")

    # Color-mask candidate from hub/center color. It can help when edge circle is weak.
    center_cfg = profile.get("hub_color") or profile.get("center", {})
    try:
        center_mask = make_color_mask(meter_img, center_cfg)
        center_mask = cv2.bitwise_and(center_mask, mask)
        contours, _ = cv2.findContours(center_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)
            if expected_radius is not None and not (min_r <= float(r) <= max_r):
                continue
            add_candidate("center_color_mask", (int(x), int(y)), float(max(r, 1)), "hub color mask", {"area": float(area)})
    except Exception as e:
        # Avoid failing meter reading because color-mask support is unavailable.
        debug_candidates.append({
            "source": "center_color_mask_error",
            "center": None,
            "radius": None,
            "score": None,
            "selected": False,
            "note": str(e),
        })

    # De-duplicate near-identical candidates by keeping the lowest natural score.
    dedup: Dict[Tuple[int, int, int, str], Dict[str, Any]] = {}
    dedup_candidates: List[Tuple[float, int, int, int, str]] = []
    # Keep simple: select from candidates directly, debug may contain duplicates.

    if candidates:
        # v9.21: profile_guided_estimate is a reference/fallback.
        # If an image-based candidate (Hough/color mask) is reasonably natural, prefer it
        # over the pure profile estimate even when the profile estimate has a slightly
        # lower mathematical score. This prevents a shifted P candidate from winning
        # against a real hub circle in images where perspective/needle position changes.
        profile_candidates = [c for c in candidates if c[4] == "profile_guided_estimate"]
        real_candidates = [c for c in candidates if c[4] != "profile_guided_estimate"]
        profile_best_score = min([c[0] for c in profile_candidates], default=None)

        def _find_debug_for_candidate(c):
            sc, cx0, cy0, rr0, src0 = c
            for dc0 in debug_candidates:
                if dc0.get("source") == src0 and dc0.get("center") == [int(cx0), int(cy0)] and int(dc0.get("radius") or -1) == int(rr0):
                    return dc0
            return None

        def _real_candidate_is_credible(c):
            dc0 = _find_debug_for_candidate(c) or {}
            rr_ratio = dc0.get("radius_ratio")
            prof_dist = float(dc0.get("profile_distance") or 9999.0)
            edge_support = dc0.get("circle_edge_support")
            color_info = (dc0.get("natural_score_detail") or {}).get("hub_color") or {}
            color_available = bool(color_info.get("available", False))
            color_distance = color_info.get("distance")
            color_penalty = dc0.get("hub_color_penalty")
            needle_conn = dc0.get("needle_connection") or {}
            er = float(expected_radius or max(c[3], 1))
            source = c[4]
            adoption_allow = _scaled_hub_adoption_distance(expected_radius, outer_radius, w, h)

            gates = {
                "radius_ok": True,
                "distance_ok": True,
                "color_ok": True,
                "edge_ok": True,
                "needle_connection_ok": True,
                "adoption_allow_px": float(adoption_allow),
            }

            # 1) Radius must be broadly consistent with the scaled profile hub.
            if rr_ratio is not None and not (0.70 <= float(rr_ratio) <= 1.36):
                gates["radius_ok"] = False

            # 2) The profile-guided hub is the reference. Distance allowance scales with
            # current crop/outer size instead of using a fixed 24 px threshold.
            # v9.33.6: a real image hub with strong color, circular edge and radius evidence
            # may exceed the normal distance allowance slightly. This rescue is deliberately
            # narrow so weak Hough reflections are not promoted merely by relaxing distance.
            normal_distance_ok = bool(prof_dist <= adoption_allow)
            edge_value = float(edge_support) if edge_support is not None else 0.0
            color_penalty_value = float(color_penalty) if color_penalty is not None else 1e9
            radius_ratio_value = float(rr_ratio) if rr_ratio is not None else -1.0
            strong_visual_hub = bool(
                color_available
                and edge_value >= 0.30
                and color_penalty_value <= 12.0
                and 0.75 <= radius_ratio_value <= 1.30
            )
            rescue_allow = float(adoption_allow) * 1.25 + 3.0
            rescued_distance_ok = bool(
                (not normal_distance_ok)
                and strong_visual_hub
                and prof_dist <= rescue_allow
            )
            gates["distance_ok"] = bool(normal_distance_ok or rescued_distance_ok)
            gates["distance_normal_ok"] = bool(normal_distance_ok)
            gates["distance_rescue_applied"] = bool(rescued_distance_ok)
            gates["distance_rescue_eligible"] = bool(strong_visual_hub)
            gates["distance_rescue_allow_px"] = float(rescue_allow)
            gates["distance_rescue_policy"] = (
                "edge>=0.30 color_penalty<=12 radius_ratio=0.75..1.30 "
                "and distance<=normal*1.25+3px"
            )

            # 3) Color must look like the learned hub color. If the color model is not
            # available, do not promote a real candidate over profile_guided_hub_center.
            if not color_available:
                gates["color_ok"] = False
            else:
                # Use a soft threshold from the measured profile size; this avoids a
                # hard-coded adoption-distance behavior while still rejecting unlike colors.
                color_limit = max(68.0, er * 4.5)
                if color_distance is not None and float(color_distance) > color_limit:
                    gates["color_ok"] = False
                if color_penalty is not None and float(color_penalty) > color_limit * 0.62:
                    gates["color_ok"] = False

            # 4) A real hub candidate should have a circular edge.
            # v9.33.3: Hough circles are prone to false positives on reflections and
            # partial arcs. Require a stronger minimum edge only for hough_hub so
            # weak candidates such as IMG_0148 edge=0.067 fall back to the profile
            # center, while center_color_mask candidates used by IMG_0150 remain
            # eligible under their existing softer edge threshold.
            if edge_support is None:
                gates["edge_ok"] = False
            else:
                edge_min = 0.10 if source == "hough_hub" else 0.012
                gates["edge_min_required"] = float(edge_min)
                if float(edge_support) < edge_min:
                    gates["edge_ok"] = False

            # 5) The hub must naturally connect to the provisional needle axis. If the
            # connection cannot be evaluated, keep the profile-guided center instead.
            if not needle_conn.get("available") or needle_conn.get("connected") is False:
                gates["needle_connection_ok"] = False

            credible = bool(all(gates[k] for k in ("radius_ok", "distance_ok", "color_ok", "edge_ok", "needle_connection_ok")))
            dc0["real_candidate_gates"] = gates
            dc0["real_candidate_credible"] = credible
            return credible

        def _real_candidate_ranking_score(c):
            """Rank credible real hub candidates.

            Natural score alone can prefer a Hough circle that is close to the
            profile center but visually weak.  For final real-hub selection, prefer
            candidates with stronger circular edge support and closer learned hub
            color; lightly favor center_color_mask because it comes directly from
            the learned hub colors.  Profile distance is still used, but only as a
            soft term once the candidate has already passed the credibility gates.
            """
            natural_score, cx0, cy0, rr0, src0 = c
            dc0 = _find_debug_for_candidate(c) or {}
            er = float(expected_radius or max(rr0, 1))
            adoption_allow = max(float(_scaled_hub_adoption_distance(expected_radius, outer_radius, w, h)), 1.0)
            prof_dist = float(dc0.get("profile_distance") or 0.0)
            rr_ratio = dc0.get("radius_ratio")
            rr_dev = abs(float(rr_ratio) - 1.0) if rr_ratio is not None else 0.5
            edge_support = float(dc0.get("circle_edge_support") or 0.0)
            color_penalty = dc0.get("hub_color_penalty")
            if color_penalty is None:
                color_penalty = ((dc0.get("natural_score_detail") or {}).get("hub_color") or {}).get("penalty")
            color_penalty = float(color_penalty if color_penalty is not None else 50.0)

            # Base quality terms. Lower is better.
            dist_term = 8.0 * min(prof_dist / adoption_allow, 1.35)
            radius_term = 20.0 * min(rr_dev, 0.55)
            color_term = 0.42 * min(color_penalty, max(80.0, er * 3.0))
            edge_bonus = -34.0 * min(edge_support, 0.80)
            source_bias = -7.0 if src0 == "center_color_mask" else 0.0

            weak_hough_penalty = 0.0
            if src0 == "hough_hub":
                if edge_support < 0.18:
                    weak_hough_penalty += 24.0
                elif edge_support < 0.25:
                    weak_hough_penalty += 10.0
                # If a Hough circle has weak edge/color evidence, do not let mere
                # proximity to the profile center dominate the visual hub evidence.
                if color_penalty > 12.0 and edge_support < 0.22:
                    weak_hough_penalty += 12.0

            # A very good color-mask contour is often the real visible hub even when
            # its radius is slightly larger because it includes the sloped rim.
            strong_color_mask_bonus = 0.0
            if src0 == "center_color_mask" and color_penalty <= 8.0 and edge_support >= 0.22:
                strong_color_mask_bonus -= 10.0

            ranking_score = dist_term + radius_term + color_term + edge_bonus + source_bias + weak_hough_penalty + strong_color_mask_bonus
            dc0["candidate_ranking_v9_33_2"] = {
                "ranking_score": float(ranking_score),
                "dist_term": float(dist_term),
                "radius_term": float(radius_term),
                "color_term": float(color_term),
                "edge_bonus": float(edge_bonus),
                "source_bias": float(source_bias),
                "weak_hough_penalty": float(weak_hough_penalty),
                "strong_color_mask_bonus": float(strong_color_mask_bonus),
                "natural_score": float(natural_score),
                "policy": "edge_color_priority_after_strict_gate_hough_min_edge_0_10",
            }
            dc0["ranking_score"] = float(ranking_score)
            return ranking_score

        credible_real = [c for c in real_candidates if _real_candidate_is_credible(c)]

        # v10.0.1 hub profile evidence guard:
        # A real-image candidate that is noticeably displaced from the profile-guided
        # center may replace the profile center only when its color AND circular-edge
        # evidence clearly exceed the evidence measured at the profile position.
        # This prevents a merely acceptable Hough circle from displacing a stronger
        # profile-position hub, as occurred in A0001M00298.
        profile_best = min(profile_candidates, key=lambda v: v[0]) if profile_candidates else None
        profile_dbg = _find_debug_for_candidate(profile_best) if profile_best is not None else None

        def _real_candidate_clearly_beats_profile(c):
            dc0 = _find_debug_for_candidate(c) or {}
            gates = dc0.get("real_candidate_gates") or {}
            prof_dist = float(dc0.get("profile_distance") or 9999.0)
            er = float(expected_radius or max(c[3], 1))
            adoption_allow = max(float(_scaled_hub_adoption_distance(expected_radius, outer_radius, w, h)), 1.0)
            displacement_gate = max(
                er * float(admin_value("hub_selection.profile_guard.displacement_radius_ratio", 0.45)),
                adoption_allow * float(admin_value("hub_selection.profile_guard.adoption_allow_ratio", 0.70)),
            )

            real_edge = float(dc0.get("circle_edge_support") or 0.0)
            real_color = dc0.get("hub_color_penalty")
            real_color = float(real_color) if real_color is not None else 1e9

            p_dbg = profile_dbg or {}
            profile_edge = float(p_dbg.get("circle_edge_support") or 0.0)
            profile_color = p_dbg.get("hub_color_penalty")
            profile_color = float(profile_color) if profile_color is not None else 1e9

            noticeably_displaced = bool(prof_dist > displacement_gate)

            # v10.0.2: Trigger the profile-evidence comparison not only by center
            # displacement, but also when the detected radius is substantially
            # different from the scaled profile radius, or when the profile position
            # has clearly stronger circular-edge evidence. This covers cases such as
            # A0001M00307 where the Hough center is only about 7 px away, but the
            # detected radius is about 36% too large and its edge support is much weaker.
            rr_ratio = dc0.get("radius_ratio")
            rr_ratio = float(rr_ratio) if rr_ratio is not None else 1.0
            radius_deviation = abs(rr_ratio - 1.0)
            radius_guard_threshold = float(admin_value("hub_selection.profile_guard.radius_deviation_trigger", 0.20))
            radius_mismatch = bool(radius_deviation >= radius_guard_threshold)

            edge_margin = max(
                float(admin_value("hub_selection.profile_guard.real_edge_margin_min", 0.04)),
                profile_edge * float(admin_value("hub_selection.profile_guard.real_edge_margin_ratio", 0.10)),
            )
            color_margin = max(
                float(admin_value("hub_selection.profile_guard.real_color_margin_min", 2.0)),
                min(profile_color, 50.0) * float(admin_value("hub_selection.profile_guard.real_color_margin_ratio", 0.10)),
            )
            edge_clearly_better = bool(real_edge >= profile_edge + edge_margin)
            color_clearly_better = bool(real_color <= profile_color - color_margin)

            profile_edge_advantage_margin = max(
                float(admin_value("hub_selection.profile_guard.profile_edge_advantage_min", 0.08)),
                real_edge * float(admin_value("hub_selection.profile_guard.profile_edge_advantage_ratio", 0.20)),
            )
            profile_edge_clearly_stronger = bool(
                profile_edge >= real_edge + profile_edge_advantage_margin
            )

            # When the profile evidence is unavailable, retain the existing credible gate.
            profile_evidence_available = bool(profile_dbg is not None and profile_color < 1e8)
            guard_triggered = bool(
                noticeably_displaced
                or radius_mismatch
                or profile_edge_clearly_stronger
            )
            guard_passed = bool(
                (not guard_triggered)
                or (not profile_evidence_available)
                or (edge_clearly_better and color_clearly_better)
            )

            trigger_reasons = []
            if noticeably_displaced:
                trigger_reasons.append("center_displacement")
            if radius_mismatch:
                trigger_reasons.append("radius_mismatch")
            if profile_edge_clearly_stronger:
                trigger_reasons.append("profile_edge_stronger")

            comparison = {
                "profile_guard_applied": bool(guard_triggered and profile_evidence_available),
                "profile_guard_passed": bool(guard_passed),
                "profile_guard_triggered": bool(guard_triggered),
                "profile_guard_trigger_reasons": trigger_reasons,
                "profile_distance_px": float(prof_dist),
                "displacement_gate_px": float(displacement_gate),
                "real_radius_ratio": float(rr_ratio),
                "radius_deviation": float(radius_deviation),
                "radius_guard_threshold": float(radius_guard_threshold),
                "radius_mismatch": bool(radius_mismatch),
                "real_edge_support": float(real_edge),
                "profile_edge_support": float(profile_edge),
                "required_edge_margin": float(edge_margin),
                "edge_clearly_better": bool(edge_clearly_better),
                "profile_edge_advantage_margin": float(profile_edge_advantage_margin),
                "profile_edge_clearly_stronger": bool(profile_edge_clearly_stronger),
                "real_color_penalty": None if real_color >= 1e8 else float(real_color),
                "profile_color_penalty": None if profile_color >= 1e8 else float(profile_color),
                "required_color_margin": float(color_margin),
                "color_clearly_better": bool(color_clearly_better),
                "policy": "guard_on_displacement_or_radius_mismatch_or_stronger_profile_edge_then_require_real_to_beat_profile_in_both_edge_and_color",
            }
            gates["profile_evidence_guard"] = comparison
            dc0["real_candidate_gates"] = gates
            dc0["profile_evidence_guard"] = comparison
            dc0["real_candidate_credible_after_profile_guard"] = bool(guard_passed)
            return guard_passed

        credible_real_after_profile_guard = [c for c in credible_real if _real_candidate_clearly_beats_profile(c)]

        # Populate ranking debug for all real candidates, including rejected ones, so
        # logs explain why a visually plausible candidate was or was not selected.
        for _cand in real_candidates:
            try:
                _real_candidate_ranking_score(_cand)
            except Exception:
                pass

        selection_reason = "lowest_natural_score"
        if credible_real_after_profile_guard:
            # Among candidates that passed both the natural gates and the profile
            # evidence guard, prefer stronger edge/color evidence.
            best_real = min(credible_real_after_profile_guard, key=_real_candidate_ranking_score)
            best_score, x, y, r, _source = best_real
            selection_reason = "credible_real_hub_selected_after_profile_evidence_guard"
        elif profile_best is not None:
            best_score, x, y, r, _source = profile_best
            selection_reason = (
                "profile_preserved_real_candidate_did_not_clearly_beat_profile_evidence"
                if credible_real else
                "profile_fallback_no_credible_real_hub_candidate"
            )
        else:
            best_score, x, y, r, _source = min(candidates, key=lambda v: v[0])

        for dc in debug_candidates:
            if dc.get("source") == _source and dc.get("center") == [int(x), int(y)] and int(dc.get("radius") or -1) == int(r):
                dc["selected"] = True
                dc["selection_reason"] = selection_reason
                break
        support = _needle_line_support_for_hub((int(x), int(y)), needle_lines_for_hub, expected_radius, outer_radius)
        detect_center_hub.last_debug = {
            "version": VERSION,
            "hub_scoring_policy": "natural_gate_then_multifactor_profile_evidence_guard",
            "hub_profile_candidate_role": "reference_and_fallback_only",
            "assistant_hub_selection_role": "feature_guidance_not_fixed_coordinate",
            "hub_selection_reason": selection_reason,
            "credible_real_candidate_count": len(credible_real),
            "credible_real_after_profile_guard_count": len(credible_real_after_profile_guard),
            "profile_best_score": None if profile_best_score is None else float(profile_best_score),
            "needle_line_available": bool(support.get("available", False)),
            "needle_line_distance": support.get("distance"),
            "needle_line_allow": support.get("allow"),
            "needle_line": support.get("line"),
            "needle_line_count": len(needle_lines_for_hub),
            "needle_line_is_debug_only": False,
            "hub_adoption_distance_policy": "scale_adaptive_with_strong_visual_distance_rescue",
            "selected_source": _source,
            "selected_score": float(best_score),
            "hub_candidates": sorted(debug_candidates, key=lambda d: (float("inf") if d.get("score") is None else float(d.get("score")))),
            "selected_center": [int(x), int(y)],
            "selected_radius": int(r),
            "expected_center": [int(expected_center[0]), int(expected_center[1])] if expected_center is not None else None,
            "expected_radius": None if expected_radius is None else int(round(expected_radius)),
        }
        return (x, y), r

    if expected_center is not None and expected_radius is not None and hub_cfg.get("source") in ("human_confirmed", "profile", "human_selected_candidate"):
        detect_center_hub.last_debug = {
            "version": VERSION,
            "hub_scoring_policy": "profile_fallback_after_no_candidates",
            "needle_line_available": bool(needle_lines_for_hub),
            "needle_line_distance": None,
            "needle_line_allow": None,
            "needle_line": needle_lines_for_hub[0].get("line") if needle_lines_for_hub else None,
            "needle_line_count": len(needle_lines_for_hub),
            "needle_line_is_debug_only": True,
            "selected_source": "profile_fallback",
            "selected_score": None,
            "hub_candidates": debug_candidates,
            "selected_center": [int(expected_center[0]), int(expected_center[1])],
            "selected_radius": int(expected_radius),
            "expected_center": [int(expected_center[0]), int(expected_center[1])],
            "expected_radius": int(round(expected_radius)),
        }
        return expected_center, int(expected_radius)

    detect_center_hub.last_debug = {
        "version": VERSION,
        "hub_scoring_policy": "none",
        "needle_line_available": bool(needle_lines_for_hub),
        "needle_line_distance": None,
        "needle_line_allow": None,
        "needle_line": needle_lines_for_hub[0].get("line") if needle_lines_for_hub else None,
        "needle_line_count": len(needle_lines_for_hub),
        "needle_line_is_debug_only": True,
        "selected_source": "none",
        "selected_score": None,
        "hub_candidates": debug_candidates,
        "selected_center": None,
        "selected_radius": None,
        "expected_center": None,
        "expected_radius": None,
    }
    return None


def resolve_center(meter_img: np.ndarray, profile: Dict[str, Any], center_mode: str):
    """中心決定。v8.7 profile-guided hub。

    v8.7の方針:
      - 読取時に再検出したhub中心をそのまま信用しすぎない。
      - YAMLに保存されたhuman confirmed hubを、外周円スケールで現在画像へ写像した
        profile_guided_hub を基準中心として扱う。
      - 再検出hubは profile_guided_hub の近傍かつ半径が想定範囲内の場合だけ採用する。
      - 針検出用の中心除外半径は、検出半径に引っ張られすぎないよう上限を設ける。
    """
    h, w = meter_img.shape[:2]

    yaml_center = None
    if profile.get("center_point"):
        yaml_center = tuple(map(int, profile["center_point"]))
    elif profile.get("geometry", {}).get("center_xy_in_crop"):
        yaml_center = tuple(map(int, profile["geometry"]["center_xy_in_crop"]))

    fallback_center = yaml_center or (w // 2, h // 2)

    # v9.25:
    # assistant/read の YOLO crop サイズ比から、プロファイル外周円を現在cropへ写像しておく。
    # これを Hough 外周探索の expected_center として使い、さらに Hough が失敗/異常な場合は
    # 生YAML座標ではなく、必ずこの BBOX比スケール済み外周へフォールバックする。
    bbox_scaled_outer = _profile_scaled_outer_fallback(profile, w, h)
    expected_outer_center = None
    if bbox_scaled_outer is not None:
        expected_outer_center = bbox_scaled_outer.get("center")
    if expected_outer_center is None:
        expected_outer_center = yaml_center

    outer = detect_outer_circle(meter_img, expected_center=expected_outer_center, profile=profile)

    def _outer_geometry_is_invalid(candidate: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
        if candidate is None:
            return True, "outer_none"
        try:
            c = candidate.get("center")
            r = float(candidate.get("radius"))
            if c is None or len(c) < 2:
                return True, "missing_center"
            cx, cy = float(c[0]), float(c[1])
            min_dim = float(min(w, h))
            if cx < 0 or cx >= w or cy < 0 or cy >= h:
                return True, f"center_outside_crop=({cx:.1f},{cy:.1f}) size=({w},{h})"
            if r < max(8.0, min_dim * 0.18):
                return True, f"radius_too_small={r:.1f}<min_allowed={max(8.0, min_dim * 0.18):.1f}"
            if r > min_dim * 0.72:
                return True, f"radius_too_large={r:.1f}>{min_dim * 0.72:.1f}"
            return False, "ok"
        except Exception as e:
            return True, f"invalid_exception={type(e).__name__}"

    outer_invalid, outer_invalid_reason = _outer_geometry_is_invalid(outer)
    forced_bbox_scaled_outer = False
    if outer_invalid:
        if bbox_scaled_outer is not None:
            outer = dict(bbox_scaled_outer)
            outer["source"] = "forced_bbox_scaled_outer_fallback"
            outer["forced_bbox_scaled_outer_fallback"] = True
            outer["forced_bbox_scaled_outer_reason"] = outer_invalid_reason
            outer["valid"] = True
            forced_bbox_scaled_outer = True
        else:
            outer = None

    if outer is None:
        outer_center = (w // 2, h // 2)
        outer_radius = int(min(w, h) * 0.45)
        outer_source = "fallback_image_center_no_bbox_scale"
        outer_meta = {
            "valid": False,
            "candidate_count": 0,
            "reason": "no_valid_outer_candidate_and_no_bbox_scaled_profile_fallback",
            "outer_invalid_reason": outer_invalid_reason,
        }
    else:
        outer_center = outer["center"]
        outer_radius = int(outer["radius"])
        outer_source = outer.get("source", "outer_circle_guarded")
        outer_meta = {**outer, "valid": True}
        outer_meta["forced_bbox_scaled_outer_fallback"] = bool(forced_bbox_scaled_outer or outer.get("forced_bbox_scaled_outer_fallback", False))
        if outer_meta["forced_bbox_scaled_outer_fallback"]:
            outer_meta["forced_bbox_scaled_outer_reason"] = outer.get("forced_bbox_scaled_outer_reason", outer_invalid_reason)

    # 外周円を基準にYAML hubを現在画像の座標・半径へ写像する。
    hub_cfg = get_profile_hub(profile, current_outer_center=outer_center, current_outer_radius=outer_radius)
    profile_guided_center = hub_cfg.get("center")
    profile_hub_radius = hub_cfg.get("radius")

    # v9.11: 外周円が黒い外枠などに吸着して大きくなりすぎると、
    # 針長さ・hub半径スケールがすべて過大になる。後でhub実検出値と
    # YAMLのhub/outer比から外周半径を再評価するため、元値を保持する。
    raw_outer_center = outer_center
    raw_outer_radius = int(outer_radius)
    outer_scale_guard_debug = {
        "applied": False,
        "raw_outer_center": raw_outer_center,
        "raw_outer_radius": int(raw_outer_radius),
        "reason": "not_checked_yet",
    }

    # hub再検出は行うが、採用は後段のprofile-guided条件で厳しく判定する。
    hub = detect_center_hub(meter_img, outer_center, outer_radius, profile)
    hub_needle_line_debug = getattr(detect_center_hub, "last_debug", {})

    detected_hub_radius = hub[1] if hub is not None else None
    detected_hub_center = hub[0] if hub is not None else None

    # 再検出hubの妥当性判定。
    hub_candidate_accepted = False
    hub_candidate_reason = "no_hub_candidate"
    hub_dist_to_profile = None
    hub_radius_ratio = None

    if hub is not None:
        if profile_guided_center is not None:
            hub_dist_to_profile = math.hypot(
                detected_hub_center[0] - profile_guided_center[0],
                detected_hub_center[1] - profile_guided_center[1],
            )
        else:
            hub_dist_to_profile = math.hypot(
                detected_hub_center[0] - outer_center[0],
                detected_hub_center[1] - outer_center[1],
            )

        if profile_hub_radius:
            hub_radius_ratio = float(detected_hub_radius) / max(float(profile_hub_radius), 1.0)
            # 中心位置は近傍微調整だけ許可。半径は大きすぎる候補を棄却する。
            # 中心軸の反射や針根元を大きな円として拾う誤検出を防ぐため、
            # max_radius_scaleよりも採用条件は少し厳しくする。
            # v8.10: profile-guided中心から少し離れたhub候補でも、半径比率が妥当なら採用する。
            # 画角差・外周円中心差により、実ハブ中心がprofile写像中心から20～35px程度ずれることがあるため。
            dist_allow = _scaled_hub_adoption_distance(profile_hub_radius, outer_radius, w, h)
            radius_ok = 0.70 <= hub_radius_ratio <= 1.36
            dist_ok = hub_dist_to_profile <= dist_allow
            selected_source = hub_needle_line_debug.get("selected_source") if isinstance(hub_needle_line_debug, dict) else None
            selected_debug = None
            for cand_dbg in hub_needle_line_debug.get("hub_candidates", []) if isinstance(hub_needle_line_debug, dict) else []:
                if cand_dbg.get("selected"):
                    selected_debug = cand_dbg
                    break
            real_gates = (selected_debug or {}).get("real_candidate_gates") or {}
            real_candidate_selected = selected_source not in (None, "profile_guided_estimate", "profile_fallback", "none")
            # profile_guided_hub_center is the reference. A real candidate may replace it
            # only when the reader already verified color/radius/edge/needle-connection gates.
            natural_gate_ok = True
            if real_candidate_selected:
                natural_gate_ok = bool((selected_debug or {}).get("real_candidate_credible", False))

            # v9.33.7: detect_center_hub()で強い色・円エッジ・半径証拠により
            # 距離救済された実ハブ候補を、resolve_center()の旧距離条件で再棄却しない。
            # 通常のdist_okは維持し、一次判定が明示的にdistance_rescue_applied=Trueを
            # 記録した場合だけ、最終中心採用でも距離条件を満たしたものとして扱う。
            distance_rescue_applied = bool(real_gates.get("distance_rescue_applied", False))
            distance_rescue_allow = real_gates.get("distance_rescue_allow_px")
            effective_dist_ok = bool(dist_ok or distance_rescue_applied)

            hub_candidate_accepted = bool(radius_ok and effective_dist_ok and natural_gate_ok)
            if not radius_ok:
                hub_candidate_reason = f"rejected_radius_ratio={hub_radius_ratio:.3f} not_in_0.70_1.36"
            elif not effective_dist_ok:
                hub_candidate_reason = f"rejected_scale_adaptive_distance={hub_dist_to_profile:.1f}>{dist_allow:.1f}"
            elif not natural_gate_ok:
                hub_candidate_reason = f"rejected_real_candidate_natural_gates={real_gates}"
            elif distance_rescue_applied and real_candidate_selected:
                rescue_text = "unknown" if distance_rescue_allow is None else f"{float(distance_rescue_allow):.1f}"
                hub_candidate_reason = (
                    f"accepted_strong_visual_hub_distance_rescue="
                    f"{hub_dist_to_profile:.1f}>{dist_allow:.1f}_within_{rescue_text}"
                )
            else:
                hub_candidate_reason = "accepted_real_hub_color_radius_edge_needle_scale_adaptive" if real_candidate_selected else "accepted_profile_guided_hub_reference"
        else:
            # 古いYAMLなどでhub半径が無い場合は従来基準に近い判定。
            dist_outer = math.hypot(detected_hub_center[0] - outer_center[0], detected_hub_center[1] - outer_center[1])
            dist_allow = _scaled_hub_adoption_distance(None, outer_radius, w, h)
            hub_candidate_accepted = dist_outer <= dist_allow
            hub_candidate_reason = "accepted_legacy" if hub_candidate_accepted else f"rejected_legacy_distance={dist_outer:.1f}>{dist_allow:.1f}"

    # v9.13: hub半径から外周円を逆算する補正は廃止。
    # 小さいハブ半径の誤差を外周スケールへ伝播させると、針長さ・ハブ半径が大きく崩れるため、
    # 外周円はHough候補と「白い文字盤円らしさ」で選択した値をそのまま使う。
    outer_scale_guard_debug.update({
        "applied": False,
        "reason": "disabled_no_hub_based_outer_correction",
        "raw_outer_radius_kept": int(raw_outer_radius),
    })

    def profile_center_or_outer() -> Point:
        return profile_guided_center or outer_center

    def safe_display_radius() -> int:
        return int(profile_hub_radius or max(outer_radius * 0.04, 6))

    def safe_needle_radius(use_detected: bool = False) -> int:
        """針検出用の中心除外半径。

        表示半径とは別管理。YAML半径を基本に少しだけ大きくして針根元を除外する。
        再検出hubが妥当な場合でも、検出半径が大きすぎて針先検出を邪魔しないよう上限を設ける。
        """
        if profile_hub_radius:
            base = float(profile_hub_radius)
            if use_detected and detected_hub_radius is not None:
                # 検出値は使ってよいが、YAML基準の1.22倍までに制限する。
                return int(round(min(max(base * 1.08, float(detected_hub_radius)), base * 1.22)))
            return int(round(base * 1.12))
        return int(max(outer_radius * 0.04, 6))

    if center_mode == "yaml" and yaml_center is not None:
        center = yaml_center
        center_radius = safe_needle_radius(use_detected=False)
        source = "yaml"
    elif center_mode == "outer":
        center = outer_center
        center_radius = safe_needle_radius(use_detected=False)
        source = outer_source
    elif center_mode == "hub":
        if hub_candidate_accepted and detected_hub_center is not None:
            center = detected_hub_center
            center_radius = safe_needle_radius(use_detected=True)
            source = "hub_profile_refined"
        elif profile_guided_center is not None:
            center = profile_guided_center
            center_radius = safe_needle_radius(use_detected=False)
            source = "profile_guided_hub"
        elif hub is not None:
            center = detected_hub_center
            center_radius = safe_needle_radius(use_detected=True)
            source = "hub_legacy"
        else:
            center = outer_center
            center_radius = safe_needle_radius(use_detected=False)
            source = outer_source
    else:
        # auto: profile_guided_hubを基本とし、再検出hubは近傍微調整としてのみ採用する。
        if hub_candidate_accepted and detected_hub_center is not None:
            center = detected_hub_center
            center_radius = safe_needle_radius(use_detected=True)
            source = "hub_auto_profile_refined"
        elif profile_guided_center is not None:
            center = profile_guided_center
            center_radius = safe_needle_radius(use_detected=False)
            source = "profile_guided_hub_auto"
        elif hub is not None:
            # hub情報がない旧プロファイル向けフォールバック。
            center = detected_hub_center
            center_radius = safe_needle_radius(use_detected=True)
            source = "hub_auto_legacy"
        else:
            center = outer_center
            center_radius = safe_needle_radius(use_detected=False)
            source = outer_source

    info = {
        "center": center,
        "center_radius": center_radius,
        "outer_center": outer_center,
        "outer_radius": outer_radius,
        "outer_meta": outer_meta,
        "raw_outer_center": raw_outer_center,
        "raw_outer_radius": int(raw_outer_radius),
        "outer_scale_guard": outer_scale_guard_debug,
        "outer_valid": bool(outer_meta.get("valid", False)),
        "outer_candidate_count": int(outer_meta.get("candidate_count", 0)),
        "outer_candidate_details": outer_meta.get("candidate_details") or [],
        "hub": hub,
        "profile_hub": hub_cfg,
        "profile_guided_hub_center": profile_guided_center,
        "raw_profile_hub_radius": hub_cfg.get("raw_radius"),
        "scaled_profile_hub_radius": profile_hub_radius,
        "outer_radius_scale": hub_cfg.get("outer_radius_scale", 1.0),
        "profile_outer_radius": hub_cfg.get("profile_outer_radius"),
        "detected_hub_radius": detected_hub_radius,
        "hub_candidate_accepted": bool(hub_candidate_accepted),
        "hub_candidate_reason": hub_candidate_reason,
        "hub_distance_rescue_applied": bool(
            ((selected_debug or {}).get("real_candidate_gates") or {}).get("distance_rescue_applied", False)
        ) if 'selected_debug' in locals() else False,
        "hub_distance_rescue_allow_px": (
            ((selected_debug or {}).get("real_candidate_gates") or {}).get("distance_rescue_allow_px")
        ) if 'selected_debug' in locals() else None,
        "hub_candidate_distance_to_profile": hub_dist_to_profile,
        "hub_candidate_radius_ratio": hub_radius_ratio,
        "hub_adoption_distance_allow_px": dist_allow if 'dist_allow' in locals() else None,
        "hub_adoption_distance_policy": "scale_adaptive_no_fixed_24px",
        "hub_needle_line_debug": hub_needle_line_debug,
        "hub_candidate_details": hub_needle_line_debug.get("hub_candidates", []) if isinstance(hub_needle_line_debug, dict) else [],
        # 表示用の青丸は、ユーザーが円で囲んだYAML半径を現在画像へスケールしたものを優先。
        # center_radiusは針検出用であり、表示半径とは別管理にする。
        "display_hub_radius": safe_display_radius(),
        "profile_hub_debug": {
            "raw_center": hub_cfg.get("raw_center"),
            "mapped_center": profile_guided_center,
            "raw_radius": hub_cfg.get("raw_radius"),
            "scaled_radius": profile_hub_radius,
            "profile_outer_radius": hub_cfg.get("profile_outer_radius"),
            "current_outer_radius": hub_cfg.get("current_outer_radius"),
            "outer_radius_scale": hub_cfg.get("outer_radius_scale", 1.0),
            "source": hub_cfg.get("source"),
        },
        "source": source,
    }
    return info

# -----------------------------
# Mask creation
# -----------------------------

def make_color_mask(img: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """v2/v3/v4 profile形式に対応した色/濃淡マスク。"""
    mode = cfg.get("mask_mode", "hsv")

    if mode == "dark":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lower = int(cfg.get("gray_lower", 0))
        upper = int(cfg.get("gray_upper", 110))
        return cv2.inRange(gray, lower, upper)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = cfg.get("hsv_lower")
    upper = cfg.get("hsv_upper")

    if lower is None or upper is None:
        # dict形式に対応
        mean = cfg.get("hsv_mean", {})
        std = cfg.get("hsv_std", {})
        if isinstance(mean, dict):
            h = float(mean.get("h", 0))
            s = float(mean.get("s", 0))
            v = float(mean.get("v", 0))
            dh = max(float(std.get("h", 10)) * 3, 12)
            ds = max(float(std.get("s", 20)) * 3, 35)
            dv = max(float(std.get("v", 20)) * 3, 35)
            lower = [h - dh, s - ds, v - dv]
            upper = [h + dh, s + ds, v + dv]
        else:
            lower = [0, 0, 0]
            upper = [179, 255, 120]

    lower = np.array([
        max(0, min(179, int(lower[0]))),
        max(0, min(255, int(lower[1]))),
        max(0, min(255, int(lower[2]))),
    ], dtype=np.uint8)
    upper = np.array([
        max(0, min(179, int(upper[0]))),
        max(0, min(255, int(upper[1]))),
        max(0, min(255, int(upper[2]))),
    ], dtype=np.uint8)

    return cv2.inRange(hsv, lower, upper)


def create_needle_mask(meter_img: np.ndarray, center: Point, center_radius: int, outer_radius: int, profile: Dict[str, Any]):
    h, w = meter_img.shape[:2]
    needle_cfg = profile.get("needle", {})

    # 1) プロファイル色/濃淡
    profile_mask = make_color_mask(meter_img, needle_cfg)

    # 2) 汎用の暗色マスク: 黒針が照明変化で明るくなっても拾う
    gray = cv2.cvtColor(meter_img, cv2.COLOR_BGR2GRAY)
    dark_upper = int(needle_cfg.get("dark_gray_upper", max(130, needle_cfg.get("gray_upper", 110))))
    dark_lower = int(needle_cfg.get("dark_gray_lower", 0))
    dark_mask = cv2.inRange(gray, dark_lower, dark_upper)

    # 3) エッジマスク
    edges = cv2.Canny(gray, 60, 160)

    # 色マスクと暗色マスクを統合。ただし後で形状スコアで選別する。
    mask = cv2.bitwise_or(profile_mask, dark_mask)

    # 有効範囲: 外周内、中心ハブは除外
    valid = np.zeros((h, w), dtype=np.uint8)
    inner = int(max(center_radius * 1.1, outer_radius * 0.04))

    # v9.24 A/B検証のB:
    # 旧設定の needle.outer_radius_ratio=0.43 などが小さいと、縮小されたread画像で
    # 針先側が探索範囲から外れ、ray/Hough候補が空になる。
    # プロファイル作成時の NEEDLE_TIP_LONG 由来の tip_length_ratio があれば、
    # それに少し余裕を足した範囲まで自動拡張する。
    configured_outer_ratio = float(needle_cfg.get("outer_radius_ratio", 0.78))
    dynamic_outer_ratio = configured_outer_ratio
    try:
        ns = profile.get("needle_shape", {}) or {}
        tip_ratio = ns.get("tip_length_ratio")
        if tip_ratio is None:
            ref = get_profile_outer_reference(profile)
            ref_radius = ref.get("radius")
            if ns.get("tip_length_px") is not None and ref_radius:
                tip_ratio = float(ns.get("tip_length_px")) / max(float(ref_radius), 1.0)
        if tip_ratio is not None:
            dynamic_outer_ratio = max(dynamic_outer_ratio, min(0.95, max(0.78, float(tip_ratio) + 0.05)))
    except Exception:
        dynamic_outer_ratio = configured_outer_ratio

    outer = int(outer_radius * float(dynamic_outer_ratio))
    if outer <= inner:
        outer = int(outer_radius * max(0.78, float(dynamic_outer_ratio)))

    cv2.circle(valid, center, outer, 255, -1)
    cv2.circle(valid, center, inner, 0, -1)

    mask = cv2.bitwise_and(mask, valid)
    edges = cv2.bitwise_and(edges, valid)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask, profile_mask, dark_mask, edges, valid


def extract_hub_connected_needle_mask(mask: np.ndarray, center: Point, center_radius: int, outer_radius: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """ハブ近傍から連結している針候補領域だけを抽出する。

    文字や目盛りは黒色なので通常のdark maskに混ざる。そこで、中心ハブの外側
    近傍リングに接している連結成分だけを残し、PRESSURE GAUGE等の文字が
    針幅/テーパー判定へ混入するのを抑える。
    """
    h, w = mask.shape[:2]
    cx, cy = center
    base = (mask > 0).astype(np.uint8) * 255
    if int(np.count_nonzero(base)) == 0:
        return base, {"available": False, "reason": "empty_mask"}

    # 細い針の欠けを少しだけ連結する。膨張しすぎると文字と接続するため小さめにする。
    kernel = np.ones((3, 3), np.uint8)
    work = cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel, iterations=1)
    work = cv2.dilate(work, kernel, iterations=1)

    # ハブはvalid処理で除外されているため、中心そのものではなくハブ外周直後のリングをseedにする。
    r_inner = int(max(center_radius * 0.95, outer_radius * 0.035, 5))
    r_outer = int(max(center_radius * 2.8, outer_radius * 0.13, r_inner + 8))
    seed = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(seed, center, r_outer, 255, -1)
    cv2.circle(seed, center, r_inner, 0, -1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    selected_labels = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        comp = (labels == lab)
        if np.any(comp & (seed > 0)):
            selected_labels.append(lab)

    if not selected_labels:
        # フォールバック: ハブ周辺に近い画素を含む成分を距離で選ぶ。
        best_lab = None
        best_dist = 1e9
        ys, xs = np.where(work > 0)
        for lab in range(1, n):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < 12:
                continue
            x = float(stats[lab, cv2.CC_STAT_LEFT] + stats[lab, cv2.CC_STAT_WIDTH] * 0.5)
            y = float(stats[lab, cv2.CC_STAT_TOP] + stats[lab, cv2.CC_STAT_HEIGHT] * 0.5)
            d = math.hypot(x - cx, y - cy)
            if d < best_dist:
                best_dist = d
                best_lab = lab
        if best_lab is not None:
            selected_labels = [best_lab]

    out = np.zeros((h, w), dtype=np.uint8)
    for lab in selected_labels:
        out[labels == lab] = 255

    # 元マスクの輪郭を維持しつつ、走査用に軽く閉じる。
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    debug = {
        "available": bool(selected_labels),
        "selected_component_count": int(len(selected_labels)),
        "selected_labels": [int(v) for v in selected_labels[:10]],
        "seed_inner_radius": int(r_inner),
        "seed_outer_radius": int(r_outer),
        "mask_pixels_before": int(np.count_nonzero(base)),
        "mask_pixels_after": int(np.count_nonzero(out)),
    }
    return out, debug


# -----------------------------
# Needle detection
# -----------------------------

def sample_line_width(mask: np.ndarray, center: Point, angle_deg: float, distance_r: float, max_half_width: int = 28) -> float:
    """指定角度・距離の位置で、針に直交する方向の白画素幅を測る。小さいほど細い。"""
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
        return float(max_half_width * 2 + 1)

    center_hit = False
    width = 0
    for sign in (-1, 1):
        for i in range(0, max_half_width + 1):
            x = int(round(x0 + sign * i * px))
            y = int(round(y0 + sign * i * py))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            hit = mask[y, x] > 0
            if i == 0 and hit:
                center_hit = True
            if hit:
                width += 1
            elif i > 1:
                break

    if not center_hit:
        return float(max_half_width * 2 + 1)
    return float(width)


def ray_direction_profile(
    mask: np.ndarray,
    center: Point,
    angle_deg: float,
    center_radius: int,
    outer_radius: int,
    max_gap: int = 10,
) -> Dict[str, Any]:
    """中心から指定方向へ走査し、その方向の針らしさを評価する。

    太い短い側を針先と誤認識しないため、単純な太さだけではなく
    「中心からどれだけ長く伸びているか」を強く評価する。
    """
    h, w = mask.shape[:2]
    cx, cy = center
    rad = math.radians(angle_deg)
    ux = math.cos(rad)
    uy = -math.sin(rad)

    start_r = int(max(center_radius * 1.15, outer_radius * 0.04))
    end_r = int(outer_radius * 0.86)

    hit_count = 0
    first_hit_r = None
    last_hit_r = None
    continuous = 0
    max_continuous = 0
    gap = 0

    for r in range(start_r, end_r):
        x = int(round(cx + r * ux))
        y = int(round(cy + r * uy))
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            break

        # 太い針・ブレに対応するため近傍も見る
        patch = mask[y-1:y+2, x-1:x+2]
        hit = bool(np.any(patch > 0))

        if hit:
            hit_count += 1
            if first_hit_r is None:
                first_hit_r = r
            last_hit_r = r
            continuous += 1
            max_continuous = max(max_continuous, continuous)
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                continuous = 0

    if last_hit_r is None:
        tip = point_from_angle(center, angle_deg, start_r)
        return {
            "angle": normalize_360(angle_deg),
            "tip": tip,
            "end_r": 0.0,
            "length": 0.0,
            "hit_count": 0.0,
            "max_continuous": 0.0,
            "tip_width": 999.0,
            "mid_width": 999.0,
            "base_width": 999.0,
            "score": -1e9,
        }

    length = max(0.0, float(last_hit_r - start_r))
    tip_r = float(last_hit_r)
    mid_r = float(start_r + length * 0.65)
    base_r = float(start_r + length * 0.25)

    tip_width = sample_line_width(mask, center, angle_deg, tip_r, max_half_width=30)
    mid_width = sample_line_width(mask, center, angle_deg, mid_r, max_half_width=30)
    base_width = sample_line_width(mask, center, angle_deg, base_r, max_half_width=30)

    # 先端が細く、根元/中間が太いほどよい。
    # v8.8: 単純な「base-tip差」だけでは、太い根元側を両方向同じ太さとして
    # 評価してしまうケースがあるため、tip < mid < base の順序性も見る。
    taper_score = max(0.0, base_width - tip_width) + 0.5 * max(0.0, mid_width - tip_width)
    taper_sequence_score = max(0.0, base_width - mid_width) + max(0.0, mid_width - tip_width)
    taper_is_valid = bool(tip_width <= mid_width + 3.0 and mid_width <= base_width + 6.0)

    # tip_widthが異常に大きい場合は、針先ではなく中心金具・太い尾側・反射を
    # 拾っている可能性が高い。外周に応じた上限を設けて強く減点する。
    expected_tip_width_max = max(18.0, float(outer_radius) * 0.035)
    wide_tip_excess = max(0.0, tip_width - expected_tip_width_max)
    thin_tip_penalty = wide_tip_excess * 9.0
    bad_taper_penalty = 0.0 if taper_is_valid else max(20.0, expected_tip_width_max * 1.5)
    width_penalty = min(tip_width, 90.0) * 1.5 + thin_tip_penalty + bad_taper_penalty

    score = (
        tip_r * 2.0
        + length * 1.6
        + float(hit_count) * 0.55
        + float(max_continuous) * 0.65
        + taper_score * 3.5
        + taper_sequence_score * 4.0
        - width_penalty
    )

    tip = point_from_angle(center, angle_deg, tip_r)

    return {
        "angle": normalize_360(angle_deg),
        "tip": tip,
        "end_r": float(tip_r),
        "length": float(length),
        "hit_count": float(hit_count),
        "max_continuous": float(max_continuous),
        "tip_width": float(tip_width),
        "mid_width": float(mid_width),
        "base_width": float(base_width),
        "taper_score": float(taper_score),
        "taper_sequence_score": float(taper_sequence_score),
        "taper_is_valid": bool(taper_is_valid),
        "expected_tip_width_max": float(expected_tip_width_max),
        "thin_tip_penalty": float(thin_tip_penalty),
        "bad_taper_penalty": float(bad_taper_penalty),
        "score": float(score),
    }


def best_direction_profile_near(
    mask: np.ndarray,
    center: Point,
    base_angle: float,
    center_radius: int,
    outer_radius: int,
    search_deg: float = 28.0,
    step_deg: float = 0.5,
) -> Dict[str, Any]:
    """base_angle近傍で最も長く針らしい方向を探す。

    反対側をちょうど180度だけで見ると、実際の針角度が数度ずれた場合に
    針を見失うことがあるため、近傍探索する。
    """
    best = None
    a = base_angle - search_deg
    while a <= base_angle + search_deg:
        prof = ray_direction_profile(mask, center, a, center_radius, outer_radius)

        # v8.8: 長さだけでなく、先端が細く根元側が太いというテーパー性を強く評価する。
        # tip_widthが大きすぎる候補は、長く見えても太い尾側・中心金具・反射の可能性が高い。
        prof["direction_score"] = (
            prof["end_r"] * 2.6
            + prof["length"] * 1.8
            + prof["max_continuous"] * 0.8
            + prof["hit_count"] * 0.35
            - prof["tip_width"] * 1.2
            + prof.get("taper_score", 0.0) * 3.0
            + prof.get("taper_sequence_score", 0.0) * 5.0
            - prof.get("thin_tip_penalty", 0.0) * 1.1
            - prof.get("bad_taper_penalty", 0.0) * 1.0
        )

        if best is None or prof["direction_score"] > best["direction_score"]:
            best = prof

        a += step_deg

    return best


def farthest_mask_tip_near_angle(
    mask: np.ndarray,
    center: Point,
    base_angle: float,
    outer_radius: int,
    angle_window: float = 35.0,
) -> Optional[Dict[str, Any]]:
    """マスク中の最遠画素を、base_angle近傍から探す。

    細い針がレイ1本から少し外れていても、扇形内の最遠点として拾う。
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    cx, cy = center
    best = None

    for x, y in zip(xs, ys):
        dx = float(x - cx)
        dy = float(cy - y)
        r = math.hypot(dx, dy)
        if r < outer_radius * 0.12 or r > outer_radius * 0.90:
            continue

        angle = normalize_360(math.degrees(math.atan2(dy, dx)))
        if angle_delta_abs(angle, base_angle) > angle_window:
            continue

        if best is None or r > best["r"]:
            best = {
                "tip": (int(x), int(y)),
                "angle": angle,
                "r": float(r),
            }

    return best



def profile_guided_tip_extension(
    mask: np.ndarray,
    center: Point,
    base_angle: float,
    outer_radius: int,
    center_radius: int,
    needle_shape: Optional[Dict[str, Any]],
    current_tip: Optional[Point] = None,
    raw_line_tip: Optional[Point] = None,
    angle_window: float = 14.0,
) -> Dict[str, Any]:
    """YAMLのNEEDLE_TIP距離を使って、短く止まった針先を補正する。

    v9.9:
    方向が正しいのに連結マスク/黒マスクが途中で途切れると、先端が短く
    止まることがある。プロファイル作成時の CENTER->NEEDLE_TIP 距離
    (外周円スケール補正済み) を目安に、現在tip・line端点・扇形内の
    マスク最遠点を比較し、期待距離に近い候補を採用する。
    """
    cx, cy = center

    def pt_info(pt: Optional[Point], source: str) -> Optional[Dict[str, Any]]:
        if pt is None:
            return None
        x, y = int(pt[0]), int(pt[1])
        dx = float(x - cx)
        dy_img = float(y - cy)
        dy_math = float(cy - y)
        r = math.hypot(dx, dy_img)
        if r <= 1.0:
            return None
        ang = normalize_360(math.degrees(math.atan2(dy_math, dx)))
        return {"tip": (x, y), "angle": float(ang), "r": float(r), "source": source}

    expected = None
    if isinstance(needle_shape, dict) and needle_shape.get("available"):
        v = needle_shape.get("tip_length_px_scaled")
        if v is not None:
            try:
                expected = float(v)
            except Exception:
                expected = None

    candidates: List[Dict[str, Any]] = []
    for item in (pt_info(current_tip, "current_tip"), pt_info(raw_line_tip, "raw_line_tip")):
        if item is not None and angle_delta_abs(item["angle"], base_angle) <= angle_window * 1.8:
            candidates.append(item)

    # まず通常のマスク最遠点。ただし、短く止まる場合があるのでこれだけでは決めない。
    far = farthest_mask_tip_near_angle(mask, center, base_angle, outer_radius, angle_window=angle_window)
    if far is not None:
        far = dict(far)
        far["source"] = "mask_farthest"
        candidates.append(far)

    # 期待距離付近で、針方向近傍の黒/針マスクが見える点を探す。
    # 連結性が切れても、期待距離周辺に細い針線が残っていれば拾えるようにする。
    if expected is not None and expected > 1:
        ys, xs = np.where(mask > 0)
        best_window = None
        min_r = max(float(center_radius) * 2.0, expected * 0.62)
        max_r = min(float(outer_radius) * 0.93, expected * 1.45)
        for x, y in zip(xs, ys):
            dx = float(x - cx)
            dy = float(cy - y)
            r = math.hypot(dx, dy)
            if r < min_r or r > max_r:
                continue
            ang = normalize_360(math.degrees(math.atan2(dy, dx)))
            ad = angle_delta_abs(ang, base_angle)
            if ad > angle_window:
                continue
            # 期待距離に近く、かつ少し遠い点を優先する。
            dist_err = abs(r - expected)
            score = -dist_err * 2.0 - ad * 3.0 + r * 0.25
            if best_window is None or score > best_window["score"]:
                best_window = {
                    "tip": (int(x), int(y)),
                    "angle": float(ang),
                    "r": float(r),
                    "source": "profile_expected_window_mask",
                    "score": float(score),
                }
        if best_window is not None:
            candidates.append(best_window)

        # マスク候補が弱い場合でも、角度と期待距離から幾何的な候補点を作る。
        # これは最終手段で、raw_line_tip等より期待距離に近い場合だけ採用される。
        rad = math.radians(base_angle)
        px = int(round(cx + expected * math.cos(rad)))
        py = int(round(cy - expected * math.sin(rad)))
        h, w = mask.shape[:2]
        if 0 <= px < w and 0 <= py < h:
            candidates.append({
                "tip": (px, py),
                "angle": float(base_angle),
                "r": float(expected),
                "source": "profile_expected_geometry",
            })

    if not candidates:
        return {"used": False, "reason": "no_candidates"}

    def candidate_score(c: Dict[str, Any]) -> float:
        r = float(c.get("r", 0.0))
        ang = float(c.get("angle", base_angle))
        ad = angle_delta_abs(ang, base_angle)
        score = r * 0.30 - ad * 4.0
        if expected is not None and expected > 1:
            err = abs(r - expected)
            tol = max(float(outer_radius) * 0.10, expected * 0.22, 18.0)
            score += max(0.0, 220.0 - 220.0 * err / max(tol, 1.0))
            # 明らかに短い先端は強く減点する。
            if r < expected * 0.72:
                score -= 160.0 + (expected * 0.72 - r) * 1.2
            # 少し長いのは許容。長すぎだけ減点。
            if r > expected * 1.40:
                score -= (r - expected * 1.40) * 1.5
        src = c.get("source")
        if src == "raw_line_tip":
            score += 35.0
        elif src == "profile_expected_window_mask":
            score += 45.0
        elif src == "mask_farthest":
            score += 20.0
        elif src == "profile_expected_geometry":
            score -= 15.0
        return float(score)

    scored = []
    for c in candidates:
        cc = dict(c)
        cc["selection_score"] = candidate_score(cc)
        if expected is not None:
            cc["expected_tip_length"] = float(expected)
            cc["length_ratio"] = float(cc["r"] / max(expected, 1.0))
        scored.append(cc)
    best = max(scored, key=lambda x: x["selection_score"])
    current = pt_info(current_tip, "current_tip")

    used = True
    reason = "selected_profile_guided_tip"
    if current is not None:
        # 現在tipがすでに期待距離に近いなら無理に変更しない。
        if expected is not None and current["r"] >= expected * 0.86 and abs(current["r"] - expected) <= abs(best["r"] - expected) + max(8.0, expected * 0.05):
            best = current
            best["selection_score"] = candidate_score(best)
            if expected is not None:
                best["expected_tip_length"] = float(expected)
                best["length_ratio"] = float(best["r"] / max(expected, 1.0))
            used = False
            reason = "kept_current_tip_already_near_expected"

    return {
        "used": bool(used),
        "reason": reason,
        "selected": best,
        "candidates": scored,
        "expected_tip_length": None if expected is None else float(expected),
    }


def choose_thin_tip(mask: np.ndarray, center: Point, angle_deg: float, outer_radius: int, fallback_tip: Point, center_radius: int = 8, needle_shape: Optional[Dict[str, Any]] = None, direction_search_deg: float = 14.0):
    """針の両方向を比較し、長く伸びている側を針先として採用する。

    v6.6:
      - 180度反対方向を単一点ではなく±28度で探索
      - 扇形内の最遠マスク画素も併用
      - 短い尾側より、長い針先側を強く優先
    """
    angle_a = normalize_360(angle_deg)
    angle_b = normalize_360(angle_deg + 180.0)

    prof_a = best_direction_profile_near(mask, center, angle_a, center_radius, outer_radius, search_deg=direction_search_deg)
    prof_b = best_direction_profile_near(mask, center, angle_b, center_radius, outer_radius, search_deg=direction_search_deg)

    far_a = farthest_mask_tip_near_angle(mask, center, angle_a, outer_radius, angle_window=max(10.0, direction_search_deg + 4.0))
    far_b = farthest_mask_tip_near_angle(mask, center, angle_b, outer_radius, angle_window=max(10.0, direction_search_deg + 4.0))

    if far_a is not None:
        prof_a["far_tip"] = far_a
        # 最遠点がレイ結果よりかなり遠ければ、角度・tipを最遠点に寄せる
        if far_a["r"] > prof_a["end_r"] + outer_radius * 0.08:
            prof_a["tip"] = far_a["tip"]
            prof_a["angle"] = far_a["angle"]
            prof_a["end_r"] = far_a["r"]
            prof_a["direction_score"] += far_a["r"] * 1.2

    if far_b is not None:
        prof_b["far_tip"] = far_b
        if far_b["r"] > prof_b["end_r"] + outer_radius * 0.08:
            prof_b["tip"] = far_b["tip"]
            prof_b["angle"] = far_b["angle"]
            prof_b["end_r"] = far_b["r"]
            prof_b["direction_score"] += far_b["r"] * 1.2

    # v9.6: apply the user-entered tip/tail length template to direction choice.
    # Candidate A means angle_a is tip and angle_b is tail; candidate B is the opposite.
    shape_choice_debug = {}
    combined_a = float(prof_a.get("direction_score", -1e9))
    combined_b = float(prof_b.get("direction_score", -1e9))
    if isinstance(needle_shape, dict) and needle_shape.get("available"):
        exp_tip = needle_shape.get("tip_length_px_scaled")
        exp_tail = needle_shape.get("tail_length_px_scaled")
        exp_full = needle_shape.get("full_length_px_scaled")
        if exp_full is None and exp_tip is not None and exp_tail is not None:
            exp_full = float(exp_tip) + float(exp_tail)
        tip_tol = max(float(outer_radius) * 0.16, float(exp_tip or 0) * 0.28, 24.0)
        tail_tol = max(float(outer_radius) * 0.14, float(exp_tail or 0) * 0.35, 20.0)
        full_tol = max(float(outer_radius) * 0.20, float(exp_full or 0) * 0.22, 30.0)
        def _shape_score(tip_prof, tail_prof):
            sc = 0.0
            dbg = {}
            if exp_tip is not None:
                tip_err = abs(float(tip_prof.get("end_r", 0.0)) - float(exp_tip))
                sc += max(0.0, 170.0 - 170.0 * tip_err / max(tip_tol, 1.0))
                dbg["tip_err"] = float(tip_err)
            if exp_tail is not None:
                tail_err = abs(float(tail_prof.get("end_r", 0.0)) - float(exp_tail))
                sc += max(0.0, 115.0 - 115.0 * tail_err / max(tail_tol, 1.0))
                dbg["tail_err"] = float(tail_err)
            if exp_full is not None:
                full_err = abs((float(tip_prof.get("end_r", 0.0)) + float(tail_prof.get("end_r", 0.0))) - float(exp_full))
                sc += max(0.0, 125.0 - 125.0 * full_err / max(full_tol, 1.0))
                dbg["full_err"] = float(full_err)
            # The tip side should generally be longer and thinner than the tail side.
            if float(tip_prof.get("end_r", 0.0)) >= float(tail_prof.get("end_r", 0.0)):
                sc += 55.0
            if float(tip_prof.get("tip_width", 999.0)) + 4.0 <= float(tail_prof.get("tip_width", 999.0)):
                sc += 45.0
            sc += max(0.0, float(tip_prof.get("taper_sequence_score", 0.0)) - float(tail_prof.get("taper_sequence_score", 0.0))) * 1.5
            dbg["shape_score"] = float(sc)
            return sc, dbg
        score_a, dbg_a = _shape_score(prof_a, prof_b)
        score_b, dbg_b = _shape_score(prof_b, prof_a)
        combined_a += score_a
        combined_b += score_b
        prof_a["shape_direction_score"] = float(score_a)
        prof_b["shape_direction_score"] = float(score_b)
        prof_a["combined_direction_score"] = float(combined_a)
        prof_b["combined_direction_score"] = float(combined_b)
        shape_choice_debug = {
            "expected_tip_length": None if exp_tip is None else float(exp_tip),
            "expected_tail_length": None if exp_tail is None else float(exp_tail),
            "expected_full_length": None if exp_full is None else float(exp_full),
            "candidate_a_if_tip": dbg_a,
            "candidate_b_if_tip": dbg_b,
            "combined_a": float(combined_a),
            "combined_b": float(combined_b),
        }

    length_diff = abs(prof_a["end_r"] - prof_b["end_r"])
    tip_width_diff = abs(prof_a.get("tip_width", 999.0) - prof_b.get("tip_width", 999.0))
    score_diff = abs(combined_a - combined_b)

    # If the profile length+width template clearly prefers one side, use it before
    # the simpler length-only rule. Otherwise fall back to length and taper.
    if shape_choice_debug and abs(combined_a - combined_b) > 85.0:
        chosen = prof_a if combined_a >= combined_b else prof_b
        reason = "profile_tip_tail_length_taper_score"
    elif length_diff > outer_radius * 0.08:
        longer = prof_a if prof_a["end_r"] > prof_b["end_r"] else prof_b
        shorter = prof_b if longer is prof_a else prof_a
        # 長い側のtipが太すぎ、短い側のtipが明確に細い場合は長さ優先を取り消す。
        if longer.get("thin_tip_penalty", 0.0) > 0 and shorter.get("tip_width", 999.0) + 10.0 < longer.get("tip_width", 999.0):
            chosen = shorter
            reason = "thinner_tip_overrode_longer_direction"
        else:
            chosen = longer
            reason = "longer_direction_window"
    elif tip_width_diff >= 8.0:
        chosen = prof_a if prof_a.get("tip_width", 999.0) < prof_b.get("tip_width", 999.0) else prof_b
        reason = "thinner_tip_width_window"
    elif score_diff < 15.0:
        # スコアがほぼ同点の場合は、テーパー順序性がより自然な候補を採用する。
        ta = prof_a.get("taper_sequence_score", 0.0) - prof_a.get("bad_taper_penalty", 0.0)
        tb = prof_b.get("taper_sequence_score", 0.0) - prof_b.get("bad_taper_penalty", 0.0)
        chosen = prof_a if ta >= tb else prof_b
        reason = "taper_sequence_tie_break"
    else:
        chosen = prof_a if combined_a >= combined_b else prof_b
        reason = "combined_length_taper_score_window"

    return chosen["tip"], chosen["angle"], {
        "angle_a": prof_a["angle"],
        "angle_b": prof_b["angle"],
        "profile_a": prof_a,
        "profile_b": prof_b,
        "width_a_tip": prof_a["tip_width"],
        "width_b_tip": prof_b["tip_width"],
        "thin_score_a": prof_a["direction_score"],
        "thin_score_b": prof_b["direction_score"],
        "corrected_angle": chosen["angle"],
        "chosen_reason": reason,
        "length_diff": float(length_diff),
        "tip_width_diff": float(tip_width_diff),
        "direction_score_diff": float(score_diff),
        "shape_choice_debug": shape_choice_debug,
        "combined_score_a": float(combined_a),
        "combined_score_b": float(combined_b),
    }



def score_angle_by_ray(mask: np.ndarray, edges: np.ndarray, center: Point, center_radius: int, outer_radius: int, angle_deg: float):
    h, w = mask.shape[:2]
    cx, cy = center
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    start_r = int(max(center_radius * 1.2, outer_radius * 0.05))
    end_r = int(outer_radius * 0.78)

    hit_count = 0
    edge_count = 0
    max_continuous = 0
    continuous = 0
    last_hit_r = None
    gap = 0
    max_gap = 7

    for r in range(start_r, end_r):
        x = int(round(cx + r * cos_a))
        y = int(round(cy - r * sin_a))

        if x < 0 or x >= w or y < 0 or y >= h:
            break

        hit = mask[y, x] > 0
        # 太い針の中心線から少し外れても拾うよう近傍も見る
        if not hit:
            x1 = max(0, x - 1)
            x2 = min(w, x + 2)
            y1 = max(0, y - 1)
            y2 = min(h, y + 2)
            hit = np.any(mask[y1:y2, x1:x2] > 0)

        if hit:
            hit_count += 1
            continuous += 1
            max_continuous = max(max_continuous, continuous)
            last_hit_r = r
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                continuous = 0

        if edges[y, x] > 0:
            edge_count += 1

    if last_hit_r is None:
        return None

    length = last_hit_r - start_r
    if length < outer_radius * 0.15:
        return None

    score = (
        hit_count * 1.0
        + max_continuous * 2.2
        + length * 0.65
        + edge_count * 0.35
    )
    tip = (
        int(round(cx + last_hit_r * cos_a)),
        int(round(cy - last_hit_r * sin_a)),
    )
    return {
        "angle": float(angle_deg),
        "score": float(score),
        "tip": tip,
        "length": float(length),
        "method": "ray",
    }


def candidate_angles_from_hough(mask: np.ndarray, edges: np.ndarray, center: Point, center_radius: int, outer_radius: int):
    """HoughLinesPで線分候補を出し、中心に近く長い線の角度候補を返す。"""
    h, w = mask.shape[:2]
    combined = cv2.bitwise_or(mask, edges)

    lines = cv2.HoughLinesP(
        combined,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=max(20, int(outer_radius * 0.18)),
        maxLineGap=12,
    )

    if lines is None:
        return []

    cx, cy = center
    candidates = []

    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])
        length = math.hypot(x2 - x1, y2 - y1)
        if length < outer_radius * 0.12:
            continue

        # 線分から中心までの距離
        denom = max(length, 1e-6)
        dist_to_center = abs((x2 - x1) * (y1 - cy) - (x1 - cx) * (y2 - y1)) / denom

        # 中心から離れすぎている線は、目盛りや数字の可能性が高い
        if dist_to_center > max(center_radius * 2.5, outer_radius * 0.08):
            continue

        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)
        tip = (x1, y1) if d1 > d2 else (x2, y2)
        angle = calc_angle(center, tip)

        score = length * 1.1 - dist_to_center * 2.0 + max(d1, d2) * 0.25
        candidates.append({
            "angle": angle,
            "score": float(score),
            "tip": tip,
            "length": float(length),
            "dist_to_center": float(dist_to_center),
            "method": "hough",
        })

    return candidates



def detect_needle_by_line_segment(
    meter_img: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
    center: Point,
    center_radius: int,
    outer_radius: int,
    needle_shape: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """長い直線成分から針先を検出する。

    v8.13:
      HoughLinesPの線分は針の片側エッジを拾うことがあるため、検出線分を
      針マスク断面の中心線へ補正する。値変換で使うマゼンタ線も、針幅の
      中央に近い線になるようにする。
    """
    gray = cv2.cvtColor(meter_img, cv2.COLOR_BGR2GRAY)

    dark = cv2.inRange(gray, 0, 165)
    combined = cv2.bitwise_or(mask, dark)
    combined = cv2.bitwise_or(combined, edges)

    h, w = gray.shape[:2]
    valid = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(valid, center, int(outer_radius * 0.88), 255, -1)
    cv2.circle(valid, center, int(max(center_radius * 0.9, outer_radius * 0.025)), 0, -1)
    combined = cv2.bitwise_and(combined, valid)

    connected_mask, connected_debug = extract_hub_connected_needle_mask(combined, center, center_radius, outer_radius)
    edge_near_connected = cv2.bitwise_and(edges, cv2.dilate(connected_mask, np.ones((3, 3), np.uint8), iterations=1))
    line_input = cv2.bitwise_or(connected_mask, edge_near_connected)

    lines = cv2.HoughLinesP(
        line_input,
        rho=1,
        theta=np.pi / 180,
        threshold=28,
        minLineLength=max(35, int(outer_radius * 0.18)),
        maxLineGap=max(18, int(outer_radius * 0.06)),
    )

    if lines is None:
        return None

    cx, cy = center
    best = None

    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])
        length = math.hypot(x2 - x1, y2 - y1)
        if length < outer_radius * 0.15:
            continue

        denom = max(length, 1e-6)
        dist_to_center = abs((x2 - x1) * (y1 - cy) - (x1 - cx) * (y2 - y1)) / denom
        if dist_to_center > max(center_radius * 2.2, outer_radius * 0.055):
            continue

        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)
        max_d = max(d1, d2)
        min_d = min(d1, d2)
        if min_d > outer_radius * 0.30:
            continue

        # v9.6: The user-entered NEEDLE_TIP/NEEDLE_TAIL gives the expected full
        # needle span.  Prefer line axes whose two endpoints span approximately
        # tip+tail length around the hub, instead of only looking at the farther end.
        axis_span = float(d1 + d2)
        expected_full_len = None
        expected_tip_len = None
        expected_tail_len = None
        length_match_score = 0.0
        length_penalty = 0.0
        side_pair_penalty = 0.0
        if isinstance(needle_shape, dict) and needle_shape.get("available"):
            expected_full_len = needle_shape.get("full_length_px_scaled")
            expected_tip_len = needle_shape.get("tip_length_px_scaled")
            expected_tail_len = needle_shape.get("tail_length_px_scaled")
            if expected_full_len is None and expected_tip_len is not None and expected_tail_len is not None:
                expected_full_len = float(expected_tip_len) + float(expected_tail_len)
            if expected_full_len is not None and expected_full_len > 1:
                full_tol = max(float(outer_radius) * 0.22, float(expected_full_len) * 0.22, 32.0)
                full_err = abs(axis_span - float(expected_full_len))
                length_match_score = max(0.0, 220.0 - 220.0 * full_err / max(full_tol, 1.0))
                length_penalty = min(260.0, 180.0 * full_err / max(full_tol, 1.0))
                # Very short/long spans are likely ticks, text, rim lines, or a single edge.
                if axis_span < float(expected_full_len) * 0.58 or axis_span > float(expected_full_len) * 1.45:
                    length_penalty += 140.0
            if expected_tip_len is not None and expected_tail_len is not None:
                pair_tol = max(float(outer_radius) * 0.16, 28.0)
                pair_err_a = abs(d1 - float(expected_tip_len)) + abs(d2 - float(expected_tail_len))
                pair_err_b = abs(d2 - float(expected_tip_len)) + abs(d1 - float(expected_tail_len))
                side_pair_penalty = min(220.0, 80.0 * min(pair_err_a, pair_err_b) / max(pair_tol, 1.0))

        raw_tip = (x1, y1) if d1 > d2 else (x2, y2)
        raw_angle = normalize_360(calc_angle(center, raw_tip))

        # v8.13: 針の片側エッジではなく、マスク幅の中央線に補正する。
        centerline = _refine_needle_centerline_from_mask(
            mask=connected_mask,
            center=center,
            angle_deg=raw_angle,
            center_radius=center_radius,
            outer_radius=outer_radius,
            half_width=int(max(14, min(65, outer_radius * 0.075))),
            min_points=10,
        )
        if centerline is not None:
            tip = centerline["centerline_tip"]
            angle = float(centerline["centerline_angle"])
            max_d = float(centerline["centerline_tip_r"])
            line_xyxy = centerline["centerline_line"]
            dist_to_center = _point_to_line_distance(center, line_xyxy)
            tip_width = float(centerline["centerline_tip_width"])
            mid_width = float(centerline["centerline_mid_width"])
            base_width = float(centerline["centerline_base_width"])
            centerline_bonus = 120.0 + float(centerline["centerline_sample_count"]) * 1.5
        else:
            tip = raw_tip
            angle = raw_angle
            line_xyxy = [x1, y1, x2, y2]
            tip_width = sample_line_width(mask, center, angle, max_d, max_half_width=35)
            mid_width = sample_line_width(mask, center, angle, max_d * 0.65, max_half_width=35)
            base_width = sample_line_width(mask, center, angle, max_d * 0.25, max_half_width=35)
            centerline_bonus = 0.0

        # v9.11: tip幅が異常に太い候補は、針ではなく外周・目盛り・反射を拾った
        # 可能性が高い。中心金具半径と外周半径から上限を作り、スコアで強く減点する。
        critical_tip_width_limit = max(36.0, float(center_radius) * 1.25, float(outer_radius) * 0.075)
        critical_tip_width_penalty = max(0.0, float(tip_width) - critical_tip_width_limit) * 10.0

        score = (
            max_d * 3.2
            + length * 1.5
            - dist_to_center * 8.0
            - min(tip_width, 80.0) * 0.8
            + max(0.0, mid_width - tip_width) * 1.2
            + centerline_bonus
            + length_match_score
            - length_penalty
            - side_pair_penalty
            - critical_tip_width_penalty
        )

        cand = {
            "method": "line_segment_centerline" if centerline is not None else "line_segment",
            "tip": tip,
            "angle": angle,
            "length": float(length),
            "max_d": float(max_d),
            "min_d": float(min_d),
            "axis_span": float(axis_span),
            "expected_full_length": None if expected_full_len is None else float(expected_full_len),
            "expected_tip_length": None if expected_tip_len is None else float(expected_tip_len),
            "expected_tail_length": None if expected_tail_len is None else float(expected_tail_len),
            "length_match_score": float(length_match_score),
            "length_penalty": float(length_penalty),
            "side_pair_penalty": float(side_pair_penalty),
            "dist_to_center": float(dist_to_center),
            "tip_width": float(tip_width),
            "mid_width": float(mid_width),
            "base_width": float(base_width),
            "critical_tip_width_limit": float(critical_tip_width_limit),
            "critical_tip_width_penalty": float(critical_tip_width_penalty),
            "score": float(score),
            "final_score": float(score),
            "line": [int(v) for v in line_xyxy],
            "raw_line": [x1, y1, x2, y2],
            "raw_angle": float(raw_angle),
            "raw_tip": raw_tip,
            "centerline_used": bool(centerline is not None),
        }
        if centerline is not None:
            cand.update(centerline)

        if best is None or cand["score"] > best["score"]:
            best = cand

    return best


def _needle_template_similarity(actual: float, expected: Optional[float], tolerance: float, max_score: float) -> Tuple[float, float]:
    """Return bonus and absolute error for one expected needle length."""
    if expected is None or float(expected) <= 1.0:
        return 0.0, 0.0
    err = abs(float(actual) - float(expected))
    bonus = max(0.0, float(max_score) * (1.0 - err / max(float(tolerance), 1.0)))
    return float(bonus), float(err)


def _score_needle_axis_by_tip_tail_lengths(
    mask: np.ndarray,
    center: Point,
    base_angle: float,
    center_radius: int,
    outer_radius: int,
    needle_shape: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score one undirected needle axis using learned long/short needle lengths.

    The axis is evaluated in both assignments:
      A: base_angle side is long needle (tip), opposite side is short needle (tail)
      B: opposite side is long needle (tip), base_angle side is short needle (tail)

    This is applied before final candidate selection, so tick marks, text strokes and
    rim lines are disadvantaged even when their local ray/Hough score is high.
    """
    if not isinstance(needle_shape, dict) or not needle_shape.get("available"):
        return {"available": False, "reason": "needle_shape_unavailable", "score": 0.0}

    exp_tip = needle_shape.get("tip_length_px_scaled")
    exp_tail = needle_shape.get("tail_length_px_scaled")
    exp_full = needle_shape.get("full_length_px_scaled")
    try:
        exp_tip = None if exp_tip is None else float(exp_tip)
        exp_tail = None if exp_tail is None else float(exp_tail)
        exp_full = None if exp_full is None else float(exp_full)
    except Exception:
        return {"available": False, "reason": "invalid_needle_shape_values", "score": 0.0}
    if exp_full is None and exp_tip is not None and exp_tail is not None:
        exp_full = exp_tip + exp_tail
    if exp_tip is None and exp_tail is None and exp_full is None:
        return {"available": False, "reason": "length_values_missing", "score": 0.0}

    angle_a = normalize_360(float(base_angle))
    angle_b = normalize_360(float(base_angle) + 180.0)
    prof_a = ray_direction_profile(mask, center, angle_a, center_radius, outer_radius, max_gap=12)
    prof_b = ray_direction_profile(mask, center, angle_b, center_radius, outer_radius, max_gap=12)
    len_a = float(prof_a.get("end_r", 0.0))
    len_b = float(prof_b.get("end_r", 0.0))
    full_observed = len_a + len_b

    tip_tol = max(float(outer_radius) * 0.12, float(exp_tip or 0.0) * 0.22, 18.0)
    tail_tol = max(float(outer_radius) * 0.10, float(exp_tail or 0.0) * 0.30, 15.0)
    full_tol = max(float(outer_radius) * 0.15, float(exp_full or 0.0) * 0.18, 24.0)

    def assignment(tip_prof: Dict[str, Any], tail_prof: Dict[str, Any], tip_angle: float) -> Dict[str, Any]:
        observed_tip = float(tip_prof.get("end_r", 0.0))
        observed_tail = float(tail_prof.get("end_r", 0.0))
        tip_bonus, tip_err = _needle_template_similarity(observed_tip, exp_tip, tip_tol, 260.0)
        tail_bonus, tail_err = _needle_template_similarity(observed_tail, exp_tail, tail_tol, 155.0)
        full_bonus, full_err = _needle_template_similarity(observed_tip + observed_tail, exp_full, full_tol, 180.0)

        penalty = 0.0
        if exp_tip is not None:
            if observed_tip < exp_tip * 0.55:
                penalty += 190.0 + (exp_tip * 0.55 - observed_tip) * 1.5
            elif observed_tip > exp_tip * 1.45:
                penalty += (observed_tip - exp_tip * 1.45) * 1.2
        if exp_tail is not None:
            if observed_tail < exp_tail * 0.40:
                penalty += 95.0 + (exp_tail * 0.40 - observed_tail)
            elif observed_tail > exp_tail * 1.65:
                penalty += (observed_tail - exp_tail * 1.65) * 0.8
        if exp_tip is not None and exp_tail is not None and exp_tip > exp_tail:
            if observed_tip <= observed_tail:
                penalty += 150.0
            else:
                tip_bonus += 65.0

        tip_width = float(tip_prof.get("tip_width", 999.0))
        tail_width = float(tail_prof.get("tip_width", 999.0))
        width_bonus = 0.0
        if tip_width + 3.0 < tail_width:
            width_bonus = min(55.0, (tail_width - tip_width) * 3.0)

        total = tip_bonus + tail_bonus + full_bonus + width_bonus - penalty
        return {
            "tip_angle": float(tip_angle),
            "observed_tip_length": observed_tip,
            "observed_tail_length": observed_tail,
            "observed_full_length": observed_tip + observed_tail,
            "tip_error": tip_err,
            "tail_error": tail_err,
            "full_error": full_err,
            "tip_bonus": tip_bonus,
            "tail_bonus": tail_bonus,
            "full_bonus": full_bonus,
            "width_bonus": width_bonus,
            "penalty": penalty,
            "score": float(total),
            "tip_profile": tip_prof,
            "tail_profile": tail_prof,
        }

    assign_a = assignment(prof_a, prof_b, angle_a)
    assign_b = assignment(prof_b, prof_a, angle_b)
    selected = assign_a if assign_a["score"] >= assign_b["score"] else assign_b
    margin = abs(float(assign_a["score"]) - float(assign_b["score"]))

    return {
        "available": True,
        "score": float(selected["score"]),
        "preferred_tip_angle": float(selected["tip_angle"]),
        "orientation_margin": float(margin),
        "expected_tip_length": exp_tip,
        "expected_tail_length": exp_tail,
        "expected_full_length": exp_full,
        "observed_axis_full_length": float(full_observed),
        "assignment_a_base_as_tip": assign_a,
        "assignment_b_opposite_as_tip": assign_b,
        "selected_assignment": "base_as_tip" if selected is assign_a else "opposite_as_tip",
        "policy": "bidirectional_tip_tail_full_length_match_before_final_candidate_selection",
    }



def _sample_axis_expected_position_support(
    mask: np.ndarray,
    center: Point,
    angle_deg: float,
    expected_length: Optional[float],
    outer_radius: int,
    center_radius: int,
    role: str,
) -> Dict[str, Any]:
    """Evaluate needle evidence around a profile-expected endpoint on one axis side.

    This does not require one unbroken ray from the hub.  It samples a narrow band
    around the expected endpoint and a wider corridor from the hub to that endpoint.
    Therefore a thin long needle can still be recognized when the binary mask has
    small gaps, while a counterweight that ends near the learned tail length receives
    little support at the learned long-needle endpoint.
    """
    if expected_length is None or float(expected_length) <= 1.0:
        return {"available": False, "reason": "expected_length_unavailable"}

    h, w = mask.shape[:2]
    cx, cy = center
    expected = float(expected_length)
    ux, uy = _angle_unit_image(angle_deg)
    nx, ny = -uy, ux

    # Scale-adaptive sampling dimensions.  The endpoint window is deliberately
    # tighter for the long needle, because the tail/counterweight is broad and can
    # otherwise create false support far beyond its actual end.
    if role == "tip":
        longitudinal_half = max(5.0, expected * 0.10, outer_radius * 0.035)
        transverse_half = int(max(4, min(18, outer_radius * 0.035)))
        corridor_half = int(max(4, min(20, outer_radius * 0.040)))
    else:
        longitudinal_half = max(5.0, expected * 0.14, outer_radius * 0.040)
        transverse_half = int(max(5, min(22, outer_radius * 0.045)))
        corridor_half = int(max(5, min(24, outer_radius * 0.050)))

    endpoint_hits = 0
    endpoint_total = 0
    endpoint_rows_with_hit = 0
    endpoint_rows = 0
    for rr in np.linspace(expected - longitudinal_half, expected + longitudinal_half, 17):
        row_hit = False
        for t in range(-transverse_half, transverse_half + 1):
            x = int(round(cx + rr * ux + t * nx))
            y = int(round(cy + rr * uy + t * ny))
            if 0 <= x < w and 0 <= y < h:
                endpoint_total += 1
                if mask[y, x] > 0:
                    endpoint_hits += 1
                    row_hit = True
        endpoint_rows += 1
        if row_hit:
            endpoint_rows_with_hit += 1

    endpoint_density = endpoint_hits / max(endpoint_total, 1)
    endpoint_row_coverage = endpoint_rows_with_hit / max(endpoint_rows, 1)

    start_r = max(float(center_radius) * 1.15, float(outer_radius) * 0.045)
    end_r = max(start_r + 1.0, expected - longitudinal_half * 0.35)
    corridor_rows = 0
    corridor_rows_with_hit = 0
    max_gap_rows = 0
    current_gap = 0
    for rr in np.linspace(start_r, end_r, 48):
        row_hit = False
        for t in range(-corridor_half, corridor_half + 1):
            x = int(round(cx + rr * ux + t * nx))
            y = int(round(cy + rr * uy + t * ny))
            if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
                row_hit = True
                break
        corridor_rows += 1
        if row_hit:
            corridor_rows_with_hit += 1
            current_gap = 0
        else:
            current_gap += 1
            max_gap_rows = max(max_gap_rows, current_gap)

    corridor_coverage = corridor_rows_with_hit / max(corridor_rows, 1)

    # Check whether evidence continues beyond the expected endpoint.  For a tail
    # assignment, strong continuation toward the learned tip distance is suspicious.
    beyond_hits = 0
    beyond_total = 0
    beyond_start = expected + longitudinal_half
    beyond_end = min(float(outer_radius) * 0.92, expected + max(longitudinal_half * 2.0, outer_radius * 0.10))
    if beyond_end > beyond_start + 1.0:
        for rr in np.linspace(beyond_start, beyond_end, 12):
            for t in range(-transverse_half, transverse_half + 1):
                x = int(round(cx + rr * ux + t * nx))
                y = int(round(cy + rr * uy + t * ny))
                if 0 <= x < w and 0 <= y < h:
                    beyond_total += 1
                    if mask[y, x] > 0:
                        beyond_hits += 1
    beyond_density = beyond_hits / max(beyond_total, 1)

    score = endpoint_row_coverage * 210.0 + endpoint_density * 180.0 + corridor_coverage * 130.0
    score -= max(0, max_gap_rows - 7) * 5.0
    if role == "tail":
        score -= beyond_density * 120.0
    else:
        # A true long needle should normally have reasonable corridor support.
        if corridor_coverage < 0.35:
            score -= (0.35 - corridor_coverage) * 220.0

    return {
        "available": True,
        "role": role,
        "angle": float(normalize_360(angle_deg)),
        "expected_length": float(expected),
        "longitudinal_half_window": float(longitudinal_half),
        "transverse_half_width": int(transverse_half),
        "endpoint_density": float(endpoint_density),
        "endpoint_row_coverage": float(endpoint_row_coverage),
        "corridor_coverage": float(corridor_coverage),
        "max_gap_rows": int(max_gap_rows),
        "beyond_density": float(beyond_density),
        "score": float(score),
    }


def _score_axis_orientation_by_profile_positions(
    mask: np.ndarray,
    center: Point,
    line_angle: float,
    opposite_angle: float,
    expected_tip: Optional[float],
    expected_tail: Optional[float],
    outer_radius: int,
    center_radius: int,
) -> Dict[str, Any]:
    """Compare the two 180-degree assignments at learned tip/tail positions."""
    if expected_tip is None or expected_tail is None:
        return {"available": False, "reason": "tip_or_tail_length_unavailable"}

    a_tip = _sample_axis_expected_position_support(mask, center, line_angle, expected_tip, outer_radius, center_radius, "tip")
    a_tail = _sample_axis_expected_position_support(mask, center, opposite_angle, expected_tail, outer_radius, center_radius, "tail")
    b_tip = _sample_axis_expected_position_support(mask, center, opposite_angle, expected_tip, outer_radius, center_radius, "tip")
    b_tail = _sample_axis_expected_position_support(mask, center, line_angle, expected_tail, outer_radius, center_radius, "tail")

    score_a = float(a_tip.get("score", -1e9)) + float(a_tail.get("score", -1e9))
    score_b = float(b_tip.get("score", -1e9)) + float(b_tail.get("score", -1e9))
    margin = abs(score_a - score_b)
    selected = "line_as_tip" if score_a >= score_b else "opposite_as_tip"

    # Require actual evidence near the selected long-needle endpoint.
    selected_tip = a_tip if selected == "line_as_tip" else b_tip
    tip_endpoint_ok = bool(
        selected_tip.get("endpoint_row_coverage", 0.0) >= 0.24
        or selected_tip.get("endpoint_density", 0.0) >= 0.045
    )
    clear = bool(margin >= 45.0 and tip_endpoint_ok)
    return {
        "available": True,
        "assignment_a_line_as_tip": {"score": score_a, "tip": a_tip, "tail": a_tail},
        "assignment_b_opposite_as_tip": {"score": score_b, "tip": b_tip, "tail": b_tail},
        "selected": selected,
        "selected_tip_angle": float(line_angle if selected == "line_as_tip" else opposite_angle),
        "margin": float(margin),
        "selected_tip_endpoint_ok": bool(tip_endpoint_ok),
        "clear": bool(clear),
        "policy": "profile_expected_tip_tail_position_band_support_on_detected_axis",
    }

def detect_needle(meter_img: np.ndarray, center: Point, center_radius: int, outer_radius: int, profile: Dict[str, Any], outer_valid: bool = True):
    mask, profile_mask, dark_mask, edges, valid = create_needle_mask(
        meter_img, center, center_radius, outer_radius, profile
    )
    needle_connected_mask, needle_connected_debug = extract_hub_connected_needle_mask(mask, center, center_radius, outer_radius)
    if int(np.count_nonzero(needle_connected_mask)) > 0:
        mask_for_tip = needle_connected_mask
        edges_for_tip = cv2.bitwise_and(edges, cv2.dilate(needle_connected_mask, np.ones((3, 3), np.uint8), iterations=1))
    else:
        mask_for_tip = mask
        edges_for_tip = edges

    candidates = []
    needle_shape = get_scaled_needle_shape(profile, outer_radius)

    # まず長い直線成分から針候補を探す。
    # v9.6: YAMLの針先/針尾から得たfull_length/tip_length/tail_lengthも使って、
    #       針全体の長さに近い線分を優先する。
    line_best = detect_needle_by_line_segment(
        meter_img=meter_img,
        mask=mask_for_tip,
        edges=edges_for_tip,
        center=center,
        center_radius=center_radius,
        outer_radius=outer_radius,
        needle_shape=needle_shape,
    )

    # Hough候補
    candidates.extend(candidate_angles_from_hough(mask_for_tip, edges_for_tip, center, center_radius, outer_radius))

    # レイキャスト候補: 全角度を探索
    angle_step = float(profile.get("needle", {}).get("angle_step", 0.5))
    for angle in np.arange(-180.0, 180.0, angle_step):
        cand = score_angle_by_ray(mask_for_tip, edges_for_tip, center, center_radius, outer_radius, angle)
        if cand is not None:
            candidates.append(cand)

    if not candidates:
        # v9.24 A/B検証のA:
        # ray/Hough候補が空でも、先に検出済みの長い線分候補(line_best)があれば
        # それを候補として採用する。縮小画像ではray/Houghが空になりやすいため。
        if line_best is not None:
            lb = dict(line_best)
            lb["method"] = str(lb.get("method", "line_segment")) + "_fallback_when_ray_empty"
            lb["used_as_only_candidate"] = True
            lb["line_best_fallback_reason"] = "ray_and_hough_candidates_empty"
            candidates.append(lb)
        else:
            raise RuntimeError("針候補を検出できませんでした")

    # スケール範囲外の角度を少し不利にする
    min_angle = float(profile.get("min_angle", profile.get("angle", {}).get("min_angle_deg", 225.0)))
    max_angle = float(profile.get("max_angle", profile.get("angle", {}).get("max_angle_deg", -45.0)))
    direction = profile.get("scale", {}).get("direction", "clockwise")

    def within_scale_penalty(angle):
        # 目盛り範囲内なら0、範囲外なら軽くペナルティ
        a = normalize_360(angle)
        mn = normalize_360(min_angle)
        mx = normalize_360(max_angle)
        if direction == "clockwise":
            total = (mn - mx) % 360.0
            pos = (mn - a) % 360.0
        else:
            total = (mx - mn) % 360.0
            pos = (a - mn) % 360.0
        if 0 <= pos <= total:
            return 0.0
        return 60.0

    def scale_position(angle):
        """角度がMIN/MAXの有効目盛り範囲内かと、0..1の位置を返す。"""
        a = normalize_360(angle)
        mn = normalize_360(min_angle)
        mx = normalize_360(max_angle)
        if direction == "clockwise":
            total = (mn - mx) % 360.0
            pos = (mn - a) % 360.0
        else:
            total = (mx - mn) % 360.0
            pos = (a - mn) % 360.0
        ratio = pos / total if total != 0 else 0.0
        return (0.0 <= pos <= total), ratio

    # v9.33.5: Apply the learned long-needle/short-needle/full-span template to
    # every Hough/ray candidate before choosing ray_best.  Previously, length
    # information was used mainly after a line candidate had already been chosen.
    # This stage makes candidate extraction itself length-aware.
    for c in candidates:
        base_angle = float(c.get("angle", 0.0))
        template_eval = _score_needle_axis_by_tip_tail_lengths(
            mask=mask_for_tip,
            center=center,
            base_angle=base_angle,
            center_radius=center_radius,
            outer_radius=outer_radius,
            needle_shape=needle_shape,
        )
        c["tip_tail_length_template"] = template_eval
        template_score = float(template_eval.get("score", 0.0)) if template_eval.get("available") else 0.0
        c["tip_tail_length_template_score"] = template_score
        c["preferred_tip_angle_by_length"] = template_eval.get("preferred_tip_angle")
        c["final_score"] = c["score"] - within_scale_penalty(base_angle) + template_score

    ray_best = max(candidates, key=lambda x: x["final_score"])

    # line_segmentが十分長い場合でも、線分には表裏180度の曖昧性がある。
    # v8.5: line_segment候補を優先する場合も、必ず反対方向比較を行う。
    #       これにより、中心から太い尾側・カウンターウェイト側を針先として採用する誤検出を抑制する。
    line_priority_enabled = False
    line_priority_reject_reason = None
    line_opposite_override = None
    line_direction_debug = {}
    if line_best is not None and line_best.get("max_d", 0) > outer_radius * 0.34:
        # v8.10: tip_widthだけでline_segmentを「尾側」と決めつけない。
        # 今回の画像では、正しい細い針先方向でも中心ズレ/マスク膨張でtip_widthが大きく出る。
        line_angle = float(line_best.get("angle", 0.0))
        opp_angle = normalize_360(line_angle + 180.0)
        line_in_scale, line_ratio = scale_position(line_angle)
        opp_in_scale, opp_ratio = scale_position(opp_angle)
        line_tip_width = float(line_best.get("tip_width", 999.0))
        line_width_limit = max(24.0, float(outer_radius) * float(admin_value("needle_segment.tip_width.normal_outer_ratio", 0.045)))
        critical_line_width_limit = max(
            36.0,
            float(center_radius) * float(admin_value("needle_segment.tip_width.critical_center_ratio", 1.25)),
            float(outer_radius) * float(admin_value("needle_segment.tip_width.critical_outer_ratio", 0.075)),
        )
        critical_line_width_bad = bool(line_tip_width > critical_line_width_limit)
        shape_tip_len = needle_shape.get("tip_length_px_scaled") if needle_shape.get("available") else None
        shape_tail_len = needle_shape.get("tail_length_px_scaled") if needle_shape.get("available") else None
        shape_len_ok = False
        shape_len_score = 0.0
        if shape_tip_len is not None:
            max_d = float(line_best.get("max_d", 0.0))
            min_d = float(line_best.get("min_d", 0.0))
            tip_tol = max(float(outer_radius) * 0.18, float(shape_tip_len) * 0.35, 24.0)
            tip_err = abs(max_d - float(shape_tip_len))
            shape_len_score += max(0.0, 100.0 - 100.0 * tip_err / max(tip_tol, 1.0))
            if tip_err <= tip_tol:
                shape_len_ok = True
            if shape_tail_len is not None:
                tail_tol = max(float(outer_radius) * 0.12, float(shape_tail_len) * 0.50, 18.0)
                tail_err = abs(min_d - float(shape_tail_len))
                shape_len_score += max(0.0, 60.0 - 60.0 * tail_err / max(tail_tol, 1.0))
        line_direction_debug = {
            "line_angle": line_angle,
            "line_in_scale": bool(line_in_scale),
            "line_ratio": float(line_ratio),
            "opposite_angle": float(opp_angle),
            "opposite_in_scale": bool(opp_in_scale),
            "opposite_ratio": float(opp_ratio),
            "line_tip_width": float(line_tip_width),
            "line_width_limit": float(line_width_limit),
            "critical_line_width_limit": float(critical_line_width_limit),
            "critical_line_width_bad": bool(critical_line_width_bad),
            "needle_shape_available": bool(needle_shape.get("available")),
            "shape_tip_length_px_scaled": shape_tip_len,
            "shape_tail_length_px_scaled": shape_tail_len,
            "shape_length_match": bool(shape_len_ok),
            "shape_length_score": float(shape_len_score),
        }
        if critical_line_width_bad:
            # v10.0.6 safe rescue:
            # 先端幅だけで正しい線分を捨てると、目盛線・文字・影が先端近傍で
            # 接続した高解像度画像を誤棄却する。救済は次を全て満たす場合だけに限定する。
            #   - 線分方向だけが目盛範囲内
            #   - プロファイル長一致が強い
            #   - 中心線抽出済み
            #   - 線分が十分長い
            # それ以外は従来どおり棄却する。
            safe_width_rescue = bool(
                line_in_scale
                and (not opp_in_scale)
                and shape_len_ok
                and bool(admin_value("needle_segment.tip_width.rescue_enabled", True))
                and shape_len_score >= float(admin_value("needle_segment.tip_width.rescue_min_shape_score", 90.0))
                and (
                    (not bool(admin_value("needle_segment.tip_width.rescue_require_centerline", True)))
                    or bool(line_best.get("centerline_used"))
                )
                and float(line_best.get("max_d", 0.0)) >= float(outer_radius) * float(admin_value("needle_segment.tip_width.rescue_min_length_ratio", 0.72))
            )
            line_direction_debug["safe_width_rescue"] = bool(safe_width_rescue)
            line_direction_debug["safe_width_rescue_conditions"] = {
                "line_only_in_scale": bool(line_in_scale and not opp_in_scale),
                "shape_length_match": bool(shape_len_ok),
                "shape_length_score": float(shape_len_score),
                "centerline_used": bool(line_best.get("centerline_used")),
                "max_d_ratio": float(line_best.get("max_d", 0.0)) / max(float(outer_radius), 1.0),
            }
            if safe_width_rescue:
                line_priority_enabled = True
                line_priority_reject_reason = (
                    f"rescued_critical_tip_width_by_strong_axis_evidence="
                    f"{line_tip_width:.1f}>{critical_line_width_limit:.1f} "
                    f"shape_score={shape_len_score:.1f}"
                )
            else:
                line_priority_enabled = False
                line_priority_reject_reason = f"rejected_critical_tip_width={line_tip_width:.1f}>{critical_line_width_limit:.1f}"
        elif line_in_scale and shape_len_ok:
            # v9.1: YAMLの針形状テンプレート（針先/後方長さ比）と合う場合はline方向を強く優先する。
            line_priority_enabled = True
            line_priority_reject_reason = f"accepted_by_needle_shape_template width={line_tip_width:.1f} shape_score={shape_len_score:.1f}"
        elif line_in_scale and not opp_in_scale:
            # line方向だけが目盛り有効範囲に入るなら、太さより角度妥当性を優先する。
            line_priority_enabled = True
            line_priority_reject_reason = f"accepted_line_in_scale_even_if_wide={line_tip_width:.1f}>{line_width_limit:.1f}"
        elif line_in_scale and line_best.get("max_d", 0) > outer_radius * 0.40:
            # 両方向が範囲内に見えるケースでも、長い線分であればline方向を維持する。
            line_priority_enabled = True
            line_priority_reject_reason = f"accepted_long_line_width_penalty_only={line_tip_width:.1f}>{line_width_limit:.1f}"
        elif line_tip_width <= line_width_limit:
            line_priority_enabled = True
            line_priority_reject_reason = None
        else:
            # 反対側を採用するのは、line方向が目盛り範囲外で、反対側だけが範囲内の場合に限定する。
            line_priority_reject_reason = f"line_segment_tip_too_wide={line_tip_width:.1f}>{line_width_limit:.1f}"
            if (not line_in_scale) and opp_in_scale:
                try:
                    far_opp = farthest_mask_tip_near_angle(mask, center, opp_angle, outer_radius, angle_window=18.0)
                    if far_opp is not None and far_opp.get("r", 0.0) >= max(float(center_radius) * 2.2, float(outer_radius) * 0.20):
                        tip_r = float(far_opp["r"])
                        tw = sample_line_width(mask, center, far_opp["angle"], tip_r, max_half_width=35)
                        mw = sample_line_width(mask, center, far_opp["angle"], max(float(center_radius) * 2.2, tip_r * 0.65), max_half_width=35)
                        bw = sample_line_width(mask, center, far_opp["angle"], max(float(center_radius) * 1.3, tip_r * 0.35), max_half_width=35)
                        line_opposite_override = {
                            "method": "line_segment_opposite_probe",
                            "tip": far_opp["tip"],
                            "angle": float(far_opp["angle"]),
                            "score": float(line_best.get("score", 0.0)) * 0.72 + tip_r * 1.8,
                            "final_score": float(line_best.get("score", 0.0)) * 0.72 + tip_r * 1.8,
                            "length": float(tip_r),
                            "r": tip_r,
                            "tip_width": float(tw),
                            "mid_width": float(mw),
                            "base_width": float(bw),
                            "base_line_angle": line_angle,
                            "opposite_base_angle": float(opp_angle),
                        }
                except Exception as e:
                    line_opposite_override = {"method": "line_segment_opposite_probe_error", "error": str(e)}

    # v10.0.13: ray/line disagreement rescue.
    # A line candidate rejected only by a modest tip-width excess can still be the
    # true needle when text/ticks touch the tip. Limit the rescue to medium-scale,
    # valid-dial images and require strong geometry agreement so other cases remain unchanged.
    disagreement_rescue = {
        "enabled": bool(admin_value("ray_line_disagreement_rescue.enabled", True)),
        "attempted": False,
        "accepted": False,
    }
    if line_best is not None and disagreement_rescue["enabled"] and (not line_priority_enabled):
        cfg_t = admin_value("ray_line_disagreement_rescue.trigger", {}) or {}
        cfg_a = admin_value("ray_line_disagreement_rescue.acceptance", {}) or {}
        line_angle_dbg = float(line_direction_debug.get("line_angle", line_best.get("angle", 0.0)))
        ray_angle_dbg = float(ray_best.get("angle", 0.0))
        angle_diff = abs(((line_angle_dbg - ray_angle_dbg + 180.0) % 360.0) - 180.0)
        raw_centerline_angle = float(line_best.get("raw_angle", line_angle_dbg))
        raw_centerline_diff = abs(((raw_centerline_angle - line_angle_dbg + 180.0) % 360.0) - 180.0)
        width_limit = float(line_direction_debug.get("critical_line_width_limit", 1.0))
        width_ratio = float(line_best.get("tip_width", 999.0)) / max(width_limit, 1.0)
        shape_score = float(line_direction_debug.get("shape_length_score", 0.0))
        checks = {
            "angle_difference": angle_diff >= float(cfg_t.get("minimum_angle_difference_deg", 3.0)),
            "outer_valid": (bool(outer_valid) if bool(cfg_t.get("require_outer_valid", True)) else True),
            "outer_radius_min": float(outer_radius) >= float(cfg_t.get("outer_radius_min_px", 180)),
            "outer_radius_max": float(outer_radius) <= float(cfg_t.get("outer_radius_max_px", 400)),
            "line_in_scale": (bool(line_direction_debug.get("line_in_scale")) if bool(cfg_t.get("require_line_in_scale", True)) else True),
            "length_match": (bool(line_direction_debug.get("shape_length_match")) if bool(cfg_t.get("require_length_match", True)) else True),
            "shape_score": shape_score >= float(cfg_t.get("minimum_shape_score", 75.0)),
            "tip_width_excess": width_ratio <= float(cfg_t.get("maximum_tip_width_excess_ratio", 1.30)),
            "centerline_stable": raw_centerline_diff <= float(cfg_a.get("maximum_centerline_raw_difference_deg", 2.0)),
            "line_ray_difference_max": angle_diff <= float(cfg_a.get("maximum_line_ray_difference_deg", 6.0)),
            "profile_length_match": (bool(line_direction_debug.get("shape_length_match")) if bool(cfg_a.get("require_profile_tip_tail_length_match", True)) else True),
            "rejected_for_tip_width": bool(line_priority_reject_reason and "critical_tip_width" in str(line_priority_reject_reason)),
        }
        disagreement_rescue.update({
            "attempted": True,
            "line_angle": line_angle_dbg,
            "ray_angle": ray_angle_dbg,
            "angle_difference": angle_diff,
            "raw_centerline_angle": raw_centerline_angle,
            "raw_centerline_difference": raw_centerline_diff,
            "tip_width": float(line_best.get("tip_width", 999.0)),
            "critical_tip_width_limit": width_limit,
            "tip_width_excess_ratio": width_ratio,
            "shape_score": shape_score,
            "checks": checks,
        })
        if all(checks.values()):
            line_priority_enabled = True
            line_priority_reject_reason = (
                f"rescued_ray_line_disagreement line={line_angle_dbg:.2f} ray={ray_angle_dbg:.2f} "
                f"diff={angle_diff:.2f} width_ratio={width_ratio:.3f} shape_score={shape_score:.1f}"
            )
            disagreement_rescue["accepted"] = True
    line_direction_debug["ray_line_disagreement_rescue"] = disagreement_rescue

    if line_priority_enabled:
        best = line_best
        raw_angle = float(best["angle"])

        # v9.10:
        # The line segment axis is usually good, but the old connected-ray/taper
        # direction choice could flip to the tail side when both directions were
        # technically inside the MIN-MAX arc.  Decide the tip direction first by
        # higher-priority information:
        #   1) scale validity / scale ratio separation
        #   2) profile CENTER->NEEDLE_TIP and CENTER->NEEDLE_TAIL lengths
        #   3) center-side length (longer side is normally the tip)
        #   4) only then use taper as a fallback
        def _pt_from_center(angle_deg: float, r: float) -> Point:
            rad = math.radians(float(angle_deg))
            return (int(round(center[0] + float(r) * math.cos(rad))), int(round(center[1] - float(r) * math.sin(rad))))

        def _length_similarity(actual: float, expected: Optional[float], tol: float, max_score: float) -> float:
            if expected is None or expected <= 1:
                return 0.0
            err = abs(float(actual) - float(expected))
            return max(0.0, float(max_score) - float(max_score) * err / max(float(tol), 1.0))

        line_ang = float(line_direction_debug.get("line_angle", raw_angle))
        opp_ang = float(line_direction_debug.get("opposite_angle", normalize_360(raw_angle + 180.0)))
        line_in = bool(line_direction_debug.get("line_in_scale"))
        opp_in = bool(line_direction_debug.get("opposite_in_scale"))
        line_ratio = float(line_direction_debug.get("line_ratio", 0.0))
        opp_ratio = float(line_direction_debug.get("opposite_ratio", 0.0))
        max_d = float(best.get("max_d", best.get("length", 0.0)))
        min_d = float(best.get("min_d", 0.0))
        if min_d <= 0.0 and best.get("length") is not None:
            min_d = max(0.0, float(best.get("length", 0.0)) - max_d)

        exp_tip = needle_shape.get("tip_length_px_scaled") if isinstance(needle_shape, dict) and needle_shape.get("available") else None
        exp_tail = needle_shape.get("tail_length_px_scaled") if isinstance(needle_shape, dict) and needle_shape.get("available") else None
        tip_tol = max(float(outer_radius) * 0.16, float(exp_tip or 0.0) * 0.28, 24.0)
        tail_tol = max(float(outer_radius) * 0.14, float(exp_tail or 0.0) * 0.40, 20.0)

        # v10.0.3: Evaluate the learned long-needle and counterweight endpoint
        # positions directly on the detected undirected line axis.  This avoids
        # relying on Hough endpoint distances or one unbroken ray from the hub.
        axis_position_eval = _score_axis_orientation_by_profile_positions(
            mask=mask,
            center=center,
            line_angle=line_ang,
            opposite_angle=opp_ang,
            expected_tip=exp_tip,
            expected_tail=exp_tail,
            outer_radius=outer_radius,
            center_radius=center_radius,
        )

        # Candidate A: raw line direction is the tip, opposite side is the tail.
        # Candidate B: opposite direction is the tip, raw line direction is the tail.
        score_a = 0.0
        score_b = 0.0
        reasons_a = []
        reasons_b = []

        if axis_position_eval.get("available"):
            axis_a = float((axis_position_eval.get("assignment_a_line_as_tip") or {}).get("score", 0.0))
            axis_b = float((axis_position_eval.get("assignment_b_opposite_as_tip") or {}).get("score", 0.0))
            # Direct learned-position evidence is the primary polarity signal.
            score_a += axis_a * 1.35
            score_b += axis_b * 1.35
            reasons_a.append("profile_axis_position_score")
            reasons_b.append("profile_axis_position_score")

        if line_in:
            score_a += 170.0
            reasons_a.append("line_in_scale")
        if opp_in:
            score_b += 170.0
            reasons_b.append("opposite_in_scale")
        if line_in and not opp_in:
            score_a += 180.0
            reasons_a.append("only_line_in_scale")
        elif opp_in and not line_in:
            score_b += 180.0
            reasons_b.append("only_opposite_in_scale")
        elif line_in and opp_in:
            # When both are inside the arc, the side farther along the scale is
            # often the actual tip for this gauge family. Keep this below length
            # matching so low-reading cases can still be handled by tip/tail length.
            ratio_gap = abs(line_ratio - opp_ratio)
            if ratio_gap >= 0.18:
                if line_ratio > opp_ratio:
                    score_a += min(25.0, 35.0 * ratio_gap)
                    reasons_a.append("larger_scale_ratio")
                else:
                    score_b += min(25.0, 35.0 * ratio_gap)
                    reasons_b.append("larger_scale_ratio")

        score_a += _length_similarity(max_d, exp_tip, tip_tol, 230.0)
        score_a += _length_similarity(min_d, exp_tail, tail_tol, 130.0)
        score_b += _length_similarity(min_d, exp_tip, tip_tol, 230.0)
        score_b += _length_similarity(max_d, exp_tail, tail_tol, 130.0)
        if exp_tip is not None and exp_tail is not None and float(exp_tip) > float(exp_tail):
            if max_d >= min_d + max(12.0, float(outer_radius) * 0.025):
                score_a += 95.0
                reasons_a.append("longer_side_matches_tip")
            elif min_d >= max_d + max(12.0, float(outer_radius) * 0.025):
                score_b += 95.0
                reasons_b.append("longer_side_matches_tip")

        priority_selector = {
            "candidate_a_line_as_tip_score": float(score_a),
            "candidate_b_opposite_as_tip_score": float(score_b),
            "candidate_a_reasons": reasons_a,
            "candidate_b_reasons": reasons_b,
            "line_angle": float(line_ang),
            "opposite_angle": float(opp_ang),
            "line_in_scale": bool(line_in),
            "opposite_in_scale": bool(opp_in),
            "line_ratio": float(line_ratio),
            "opposite_ratio": float(opp_ratio),
            "line_side_length_max_d": float(max_d),
            "opposite_side_length_min_d": float(min_d),
            "expected_tip_length": None if exp_tip is None else float(exp_tip),
            "expected_tail_length": None if exp_tail is None else float(exp_tail),
            "tip_length_tolerance": float(tip_tol),
            "tail_length_tolerance": float(tail_tol),
            "profile_axis_position_eval": axis_position_eval,
        }

        # v10.0.18: When both 180-degree directions lie inside the valid scale arc,
        # lock polarity using the learned long-needle/short-needle lengths before
        # centerline refinement. This prevents a correctly detected axis from being
        # flipped to the counterweight side on very small meter crops.
        dual_guard = {
            "enabled": bool(admin_value("dual_scale_direction_guard.enabled", True)),
            "attempted": False,
            "accepted": False,
        }
        if dual_guard["enabled"] and line_in and opp_in:
            cfg_dt = admin_value("dual_scale_direction_guard.trigger", {}) or {}
            cfg_da = admin_value("dual_scale_direction_guard.acceptance", {}) or {}
            length_profile_available = bool(
                isinstance(needle_shape, dict)
                and needle_shape.get("available")
                and exp_tip is not None
                and exp_tail is not None
                and float(exp_tip) > 1.0
                and float(exp_tail) > 1.0
            )
            template_polarity = _score_needle_axis_by_tip_tail_lengths(
                mask=mask_for_tip,
                center=center,
                base_angle=line_ang,
                center_radius=center_radius,
                outer_radius=outer_radius,
                needle_shape=needle_shape,
            ) if length_profile_available else {"available": False, "reason": "length_profile_unavailable"}
            assign_a = template_polarity.get("assignment_a_base_as_tip") or {}
            assign_b = template_polarity.get("assignment_b_opposite_as_tip") or {}
            tpl_score_a = float(assign_a.get("score", -1e9))
            tpl_score_b = float(assign_b.get("score", -1e9))
            tpl_best = max(tpl_score_a, tpl_score_b)
            tpl_margin = abs(tpl_score_a - tpl_score_b)
            tpl_advantage = tpl_margin / max(abs(tpl_best), 1.0)
            expected_ratio = float(exp_tip) / max(float(exp_tail), 1e-6) if length_profile_available else 0.0
            preferred = template_polarity.get("selected_assignment")
            preferred_line = preferred == "base_as_tip"
            preferred_opp = preferred == "opposite_as_tip"
            observed_tip = float((assign_a if preferred_line else assign_b).get("observed_tip_length", 0.0)) if (preferred_line or preferred_opp) else 0.0
            observed_tail = float((assign_a if preferred_line else assign_b).get("observed_tail_length", 0.0)) if (preferred_line or preferred_opp) else 0.0
            observed_ratio = observed_tip / max(observed_tail, 1e-6) if observed_tail > 0.0 else 0.0
            selected_assignment = assign_a if preferred_line else assign_b if preferred_opp else {}
            selected_tip_profile = selected_assignment.get("tip_profile") or {}
            observed_tip_width = float(selected_tip_profile.get("tip_width", 999.0))
            observed_mid_width = float(selected_tip_profile.get("mid_width", 0.0))
            tip_length_ratio_to_expected = observed_tip / max(float(exp_tip or 0.0), 1e-6) if length_profile_available else 0.0
            tail_length_ratio_to_expected = observed_tail / max(float(exp_tail or 0.0), 1e-6) if length_profile_available else 0.0
            observed_ratio_relative_error = abs(observed_ratio - expected_ratio) / max(expected_ratio, 1e-6) if expected_ratio > 0.0 else 999.0
            tip_mid_width_ratio = observed_tip_width / max(observed_mid_width, 1e-6) if observed_mid_width > 0.0 else 999.0
            reject_widening = bool(cfg_da.get("reject_suspicious_tip_widening", True))
            checks = {
                "both_in_scale": bool(line_in and opp_in) if bool(cfg_dt.get("require_both_directions_in_scale", True)) else True,
                "length_profile_available": length_profile_available if bool(cfg_dt.get("require_length_profile_available", True)) else True,
                "expected_long_short_ratio": expected_ratio >= float(cfg_da.get("minimum_long_short_ratio", 1.35)),
                "orientation_margin": tpl_margin >= float(cfg_da.get("minimum_orientation_margin", 35.0)),
                "length_match_advantage": tpl_advantage >= float(cfg_da.get("minimum_length_match_advantage", 0.12)),
                "observed_tip_longer": (observed_tip > observed_tail) if bool(cfg_da.get("prefer_longer_endpoint", True)) else True,
                "tip_length_absolute_match": tip_length_ratio_to_expected >= float(cfg_da.get("minimum_tip_length_ratio_to_expected", 0.70)),
                "tail_length_absolute_match": tail_length_ratio_to_expected >= float(cfg_da.get("minimum_tail_length_ratio_to_expected", 0.55)),
                "observed_ratio_match": observed_ratio_relative_error <= float(cfg_da.get("maximum_observed_ratio_relative_error", 0.45)),
                "tip_not_suspiciously_wide": (tip_mid_width_ratio <= float(cfg_da.get("maximum_tip_mid_width_ratio", 1.50))) if reject_widening else True,
            }
            dual_guard.update({
                "attempted": True,
                "template": template_polarity,
                "line_assignment_score": tpl_score_a,
                "opposite_assignment_score": tpl_score_b,
                "orientation_margin": tpl_margin,
                "length_match_advantage": tpl_advantage,
                "expected_long_short_ratio": expected_ratio,
                "observed_tip_length": observed_tip,
                "observed_tail_length": observed_tail,
                "observed_long_short_ratio": observed_ratio,
                "tip_length_ratio_to_expected": tip_length_ratio_to_expected,
                "tail_length_ratio_to_expected": tail_length_ratio_to_expected,
                "observed_ratio_relative_error": observed_ratio_relative_error,
                "observed_tip_width": observed_tip_width,
                "observed_mid_width": observed_mid_width,
                "tip_mid_width_ratio": tip_mid_width_ratio,
                "preferred_assignment": preferred,
                "checks": checks,
            })
            if all(checks.values()) and (preferred_line or preferred_opp):
                # Give the selected assignment a decisive margin over all weaker
                # scale/taper evidence, but do not change the undirected axis itself.
                if preferred_line:
                    score_a = max(score_a, score_b + max(120.0, tpl_margin))
                    reasons_a.append("dual_scale_length_polarity_guard")
                else:
                    score_b = max(score_b, score_a + max(120.0, tpl_margin))
                    reasons_b.append("dual_scale_length_polarity_guard")
                dual_guard["accepted"] = True
            else:
                # v10.0.20: On very small crops, a counterweight/short-hand side can
                # merge with ticks or text and appear much longer than it is. When the
                # absolute length checks reject that assignment and its far end is
                # conspicuously wider than the middle, reverse polarity only if the
                # opposite side has a physically plausible thin end.
                rescue_cfg = cfg_da.get("low_resolution_widening_reverse_rescue", {}) or {}
                rescue_enabled = bool(rescue_cfg.get("enabled", True))
                require_outer_invalid = bool(rescue_cfg.get("require_outer_invalid", True))
                outer_invalid_ok = (not bool(outer_valid)) if require_outer_invalid else True
                small_outer_ok = float(outer_radius) <= float(rescue_cfg.get("outer_radius_max_px", 140))
                base_prof = assign_a.get("tip_profile") or {}
                opp_prof = assign_b.get("tip_profile") or {}
                base_tip_mid = float(base_prof.get("tip_width", 999.0)) / max(float(base_prof.get("mid_width", 0.0)), 1e-6)
                opp_tip_mid = float(opp_prof.get("tip_width", 999.0)) / max(float(opp_prof.get("mid_width", 0.0)), 1e-6)
                wrong_side_limit = float(rescue_cfg.get("minimum_wrong_side_tip_mid_width_ratio", 1.80))
                opposite_limit = float(rescue_cfg.get("maximum_opposite_tip_mid_width_ratio", 1.50))
                rescue_checks = {
                    "enabled": rescue_enabled,
                    "both_in_scale": bool(line_in and opp_in),
                    "outer_invalid": outer_invalid_ok,
                    "small_outer_radius": small_outer_ok,
                    "base_assignment_widened": base_tip_mid >= wrong_side_limit,
                    "opposite_assignment_thin": opp_tip_mid <= opposite_limit,
                    "absolute_guard_rejected": not all(checks.values()),
                }
                dual_guard["low_resolution_widening_reverse_rescue"] = {
                    "attempted": bool(rescue_enabled and line_in and opp_in),
                    "accepted": False,
                    "base_tip_mid_width_ratio": base_tip_mid,
                    "opposite_tip_mid_width_ratio": opp_tip_mid,
                    "checks": rescue_checks,
                }
                if all(rescue_checks.values()):
                    score_b = max(score_b, score_a + max(120.0, tpl_margin))
                    reasons_b.append("low_resolution_widening_reverse_polarity_rescue")
                    dual_guard["preferred_assignment"] = "opposite_as_tip"
                    dual_guard["accepted"] = True
                    dual_guard["accepted_by"] = "low_resolution_widening_reverse_rescue"
                    dual_guard["low_resolution_widening_reverse_rescue"]["accepted"] = True
        priority_selector["dual_scale_direction_guard"] = dual_guard

        # If direct profile-position evidence is clear, it overrides weaker scale
        # ratio and local taper cues. Otherwise use the combined selector margin.
        # tip along that direction using the profile tip length. Otherwise fall back
        # to the old taper routine.
        selector_margin = abs(score_a - score_b)
        axis_eval_clear = bool(axis_position_eval.get("available") and axis_position_eval.get("clear"))
        dual_guard_locked = bool(dual_guard.get("accepted"))
        dual_guard_preferred = dual_guard.get("preferred_assignment")

        # v10.0.18: once the dual-scale length guard has locked polarity,
        # later profile-axis evidence must not flip the direction by 180 degrees.
        # The previous implementation correctly diagnosed base_as_tip, but the
        # axis-position block below could overwrite it before centerline refinement.
        if dual_guard_locked and dual_guard_preferred in ("base_as_tip", "opposite_as_tip"):
            if dual_guard_preferred == "base_as_tip":
                score_a = max(score_a, score_b + max(120.0, selector_margin))
            else:
                score_b = max(score_b, score_a + max(120.0, selector_margin))
            priority_selector["axis_position_override_suppressed_by_dual_guard"] = bool(axis_eval_clear)
            priority_selector["locked_tip_angle"] = float(line_ang if dual_guard_preferred == "base_as_tip" else opp_ang)
            selector_margin = abs(score_a - score_b)
        elif axis_eval_clear:
            axis_selected = axis_position_eval.get("selected")
            if axis_selected == "line_as_tip":
                score_a = max(score_a, score_b + max(55.0, selector_margin))
            else:
                score_b = max(score_b, score_a + max(55.0, selector_margin))
            selector_margin = abs(score_a - score_b)

        selector_clear = bool(dual_guard_locked or axis_eval_clear or selector_margin >= 45.0)
        priority_selector["axis_position_clear"] = bool(axis_eval_clear)
        priority_selector["dual_scale_direction_guard_locked"] = bool(dual_guard_locked)
        priority_selector["selector_clear"] = bool(selector_clear)
        if selector_clear:
            if dual_guard_locked and dual_guard_preferred in ("base_as_tip", "opposite_as_tip"):
                selected_angle_raw = float(line_ang if dual_guard_preferred == "base_as_tip" else opp_ang)
                selected_r = float(max_d if dual_guard_preferred == "base_as_tip" else min_d)
                raw_tip_for_selected = best.get("tip") if dual_guard_preferred == "base_as_tip" else _pt_from_center(selected_angle_raw, selected_r)
            else:
                selected_angle_raw = line_ang if score_a >= score_b else opp_ang
                selected_r = max_d if score_a >= score_b else min_d
                raw_tip_for_selected = best.get("tip") if score_a >= score_b else _pt_from_center(selected_angle_raw, selected_r)
            centerline_refine = refine_tip_angle_by_mask_center_symmetry(
                mask=mask,
                center=center,
                base_angle=selected_angle_raw,
                outer_radius=outer_radius,
                center_radius=center_radius,
                search_deg=float(admin_value("centerline_refine.search_deg", 6.0)),
                step_deg=float(admin_value("centerline_refine.step_deg", 0.5)),
            )
            selected_angle = float(centerline_refine.get("refined_angle", selected_angle_raw)) if centerline_refine.get("available") else float(selected_angle_raw)
            # A local centerline refinement must never cross to the 180-degree
            # counterweight side after the polarity guard has locked the tip.
            if centerline_refine.get("available") and bool((priority_selector.get("dual_scale_direction_guard") or {}).get("accepted")):
                refine_flip = abs(((float(selected_angle) - float(selected_angle_raw) + 180.0) % 360.0) - 180.0)
                max_refine_flip = float(admin_value("dual_scale_direction_guard.acceptance.maximum_refine_flip_deg", 90.0))
                centerline_refine["dual_scale_direction_guard_locked"] = True
                centerline_refine["dual_scale_refine_difference_deg"] = float(refine_flip)
                centerline_refine["dual_scale_maximum_refine_flip_deg"] = float(max_refine_flip)
                if refine_flip > max_refine_flip:
                    selected_angle = float(selected_angle_raw)
                    centerline_refine["refined_angle"] = float(selected_angle_raw)
                    centerline_refine["delta_deg"] = 0.0
                    centerline_refine["dual_scale_refine_flip_rejected"] = True
                else:
                    centerline_refine["dual_scale_refine_flip_rejected"] = False

            # v10.0.19: When polarity is locked on a very small crop, reject a
            # centerline adjustment that moves far from the line axis while the
            # centerline score peak is flat. In that situation the sub-pixel mask
            # evidence is not strong enough to improve the stable line angle.
            if centerline_refine.get("available"):
                lr_cfg = admin_value("dual_scale_direction_guard.low_resolution_refine_guard", {}) or {}
                lr_enabled = bool(lr_cfg.get("enabled", True))
                lr_locked = bool((priority_selector.get("dual_scale_direction_guard") or {}).get("accepted"))
                lr_require_locked = bool(lr_cfg.get("require_direction_locked", True))
                lr_require_outer_invalid = bool(lr_cfg.get("require_outer_invalid", True))
                lr_outer_max = int(lr_cfg.get("outer_radius_max_px", 140))
                lr_coverage_max = float(lr_cfg.get("coverage_max", 0.85))
                lr_delta_max = float(lr_cfg.get("maximum_refine_delta_deg", 1.5))
                lr_score_margin_min = float(lr_cfg.get("minimum_score_margin", 5.0))
                lr_selected = centerline_refine.get("selected") or {}
                lr_coverage = float(lr_selected.get("coverage", 0.0))
                lr_delta = abs(float(centerline_refine.get("delta_deg", 0.0)))
                lr_candidates = list(centerline_refine.get("top_candidates") or [])
                lr_scores = sorted([float(c.get("score", 0.0)) for c in lr_candidates], reverse=True)
                lr_score_margin = (lr_scores[0] - lr_scores[1]) if len(lr_scores) >= 2 else 999999.0
                lr_checks = {
                    "enabled": lr_enabled,
                    "direction_locked": lr_locked if lr_require_locked else True,
                    "outer_invalid": (not bool(outer_valid)) if lr_require_outer_invalid else True,
                    "small_outer_radius": int(outer_radius) <= lr_outer_max,
                    "coverage_low": lr_coverage <= lr_coverage_max,
                    "refine_delta_large": lr_delta > lr_delta_max,
                    "score_peak_flat": lr_score_margin < lr_score_margin_min,
                }
                lr_reject = bool(all(lr_checks.values()))
                centerline_refine["low_resolution_refine_guard"] = {
                    "enabled": lr_enabled,
                    "attempted": bool(lr_enabled and lr_locked),
                    "rejected": lr_reject,
                    "checks": lr_checks,
                    "base_angle": float(selected_angle_raw),
                    "candidate_angle": float(selected_angle),
                    "refine_delta_deg": float(lr_delta),
                    "coverage": float(lr_coverage),
                    "top_score_margin": float(lr_score_margin),
                    "thresholds": {
                        "outer_radius_max_px": lr_outer_max,
                        "coverage_max": lr_coverage_max,
                        "maximum_refine_delta_deg": lr_delta_max,
                        "minimum_score_margin": lr_score_margin_min,
                    },
                }
                if lr_reject:
                    selected_angle = float(selected_angle_raw)
                    centerline_refine["refined_angle"] = float(selected_angle_raw)
                    centerline_refine["delta_deg"] = 0.0
                    centerline_refine["low_resolution_refine_rejected"] = True
                else:
                    centerline_refine["low_resolution_refine_rejected"] = False
            # v10.0.6: 低解像度・低信頼条件では中心線補正の過補正を抑え、診断情報を常時記録する。
            # 特に探索端に張り付く結果は、細い針ではなく文字・目盛り・反射を拾った可能性がある。
            if centerline_refine.get("available"):
                raw_delta = float(centerline_refine.get("delta_deg", 0.0))
                selected_refine = centerline_refine.get("selected") or {}
                coverage = float(selected_refine.get("coverage", 0.0))
                search_width = float(centerline_refine.get("search_deg", admin_value("centerline_refine.search_deg", 6.0)))
                search_edge_margin = float(admin_value("centerline_refine.search_edge_margin_deg", 0.10))
                small_outer_threshold = int(admin_value("centerline_refine.small_outer_radius_px", 140))
                low_coverage_threshold = float(admin_value("centerline_refine.low_coverage_threshold", 0.90))
                limit_deg = float(admin_value("centerline_refine.low_confidence_limit_deg", 3.0))
                at_search_edge = bool(abs(raw_delta) >= max(0.0, search_width - search_edge_margin))
                low_confidence_reasons = []
                if not bool(outer_valid):
                    low_confidence_reasons.append("outer_invalid")
                if int(outer_radius) < small_outer_threshold:
                    low_confidence_reasons.append("small_outer_radius")
                if coverage < low_coverage_threshold:
                    low_confidence_reasons.append("low_coverage")
                if at_search_edge:
                    low_confidence_reasons.append("search_edge_hit")

                # 診断情報は制限発動の有無にかかわらず常時保持する。
                centerline_refine["raw_refined_angle"] = float(centerline_refine.get("refined_angle", selected_angle_raw))
                centerline_refine["raw_delta_deg"] = float(raw_delta)
                centerline_refine["limit_deg"] = float(limit_deg)
                centerline_refine["limit_reasons"] = list(low_confidence_reasons)
                centerline_refine["diagnostics"] = {
                    "outer_valid": bool(outer_valid),
                    "outer_radius": int(outer_radius),
                    "small_outer_radius_threshold": int(small_outer_threshold),
                    "coverage": float(coverage),
                    "low_coverage_threshold": float(low_coverage_threshold),
                    "search_deg": float(search_width),
                    "search_edge_hit": bool(at_search_edge),
                    "raw_delta_deg": float(raw_delta),
                    "limit_deg": float(limit_deg),
                }

                if low_confidence_reasons and abs(raw_delta) > limit_deg:
                    limited_delta = max(-limit_deg, min(limit_deg, raw_delta))
                    selected_angle = normalize_360(float(selected_angle_raw) + limited_delta)
                    centerline_refine["limited"] = True
                    centerline_refine["refined_angle"] = float(selected_angle)
                    centerline_refine["delta_deg"] = float(limited_delta)
                else:
                    centerline_refine["limited"] = False

                # v10.0.6 safe broad-axis rescue:
                # 通常画像には適用しない。低解像度かつ局所探索が端に張り付き、
                # さらに針先側の幅が中間部より異常に太い場合だけ±18度を再探索する。
                # 元の合格画像を保護するため、新候補は局所候補を明確に上回り、
                # かつ6度を超えて離れた別軸である場合に限って採用する。
                broad_rescue = {
                    "attempted": False,
                    "accepted": False,
                    "trigger_reasons": [],
                }
                broad_enabled = bool(admin_value("broad_axis_rescue.enabled", True))
                broad_outer_max = int(admin_value("broad_axis_rescue.trigger.outer_radius_max_px", 140))
                broad_coverage_max = float(admin_value("broad_axis_rescue.trigger.coverage_max", 0.85))
                broad_range_deg = float(admin_value("broad_axis_rescue.search.range_deg", 18.0))
                broad_step_deg = float(admin_value("broad_axis_rescue.search.step_deg", 0.5))
                min_axis_sep = float(admin_value("broad_axis_rescue.search.minimum_axis_separation_deg", 6.5))
                max_axis_sep = float(admin_value("broad_axis_rescue.search.maximum_axis_separation_deg", 18.1))
                min_score_ratio = float(admin_value("broad_axis_rescue.acceptance.minimum_score_ratio", 1.20))
                min_score_difference = float(admin_value("broad_axis_rescue.acceptance.minimum_score_difference", 55.0))
                min_cov_improvement = float(admin_value("broad_axis_rescue.acceptance.minimum_coverage_improvement", 0.08))
                min_offset_improvement = float(admin_value("broad_axis_rescue.acceptance.minimum_offset_improvement_px", 1.5))
                base_w = float(best.get("base_width", 0.0))
                mid_w = float(best.get("mid_width", 0.0))
                tip_w = float(best.get("tip_width", 0.0))
                suspicious_tip_widening = bool(
                    mid_w > 0.0 and tip_w >= max(
                        mid_w * float(admin_value("broad_axis_rescue.trigger.tip_widening_ratio", 1.75)),
                        mid_w + float(admin_value("broad_axis_rescue.trigger.tip_widening_add_px", 8.0)),
                    )
                )
                strict_broad_trigger = bool(
                    broad_enabled
                    and ((not bool(admin_value("broad_axis_rescue.trigger.require_outer_invalid", True))) or (not bool(outer_valid)))
                    and int(outer_radius) < broad_outer_max
                    and coverage < broad_coverage_max
                    and ((not bool(admin_value("broad_axis_rescue.trigger.require_local_search_edge_hit", True))) or at_search_edge)
                    and ((not bool(admin_value("broad_axis_rescue.trigger.require_suspicious_tip_widening", True))) or suspicious_tip_widening)
                )
                lowres_override_cfg = admin_value("broad_axis_rescue.low_resolution_widening_override", {}) or {}
                lowres_override_enabled = bool(lowres_override_cfg.get("enabled", True))
                lowres_tip_mid_ratio = tip_w / max(mid_w, 1.0)
                lowres_override_trigger = bool(
                    broad_enabled
                    and lowres_override_enabled
                    and ((not bool(lowres_override_cfg.get("require_outer_invalid", True))) or (not bool(outer_valid)))
                    and int(outer_radius) < int(lowres_override_cfg.get("outer_radius_max_px", 140))
                    and coverage <= float(lowres_override_cfg.get("coverage_max", 0.95))
                    and ((not bool(lowres_override_cfg.get("require_suspicious_tip_widening", True))) or suspicious_tip_widening)
                    and lowres_tip_mid_ratio >= float(lowres_override_cfg.get("minimum_tip_mid_width_ratio", 2.0))
                    and ((not bool(lowres_override_cfg.get("require_local_search_edge_hit", True))) or at_search_edge)
                    and abs(float(raw_delta)) >= float(lowres_override_cfg.get("minimum_local_refine_delta_deg", 3.0))
                )
                broad_trigger = bool(strict_broad_trigger or lowres_override_trigger)
                broad_rescue["strict_trigger"] = bool(strict_broad_trigger)
                broad_rescue["low_resolution_widening_override_trigger"] = bool(lowres_override_trigger)
                broad_rescue["tip_mid_width_ratio"] = float(lowres_tip_mid_ratio)
                broad_rescue["coverage"] = float(coverage)
                broad_rescue["raw_local_refine_delta"] = float(raw_delta)
                if broad_trigger:
                    broad_rescue["attempted"] = True
                    broad_rescue["trigger_reasons"] = []
                    if strict_broad_trigger:
                        broad_rescue["trigger_reasons"].extend([
                            "outer_invalid",
                            "small_outer_radius",
                            f"coverage_below_{broad_coverage_max}",
                            "local_search_edge_hit",
                            "suspicious_tip_widening",
                        ])
                    if lowres_override_trigger:
                        broad_rescue["trigger_reasons"].extend([
                            "low_resolution_widening_override",
                            f"coverage_at_or_below_{float(lowres_override_cfg.get('coverage_max', 0.95))}",
                            "large_local_refine_delta",
                        ])
                    broad = refine_tip_angle_by_mask_center_symmetry(
                        mask=mask,
                        center=center,
                        base_angle=selected_angle_raw,
                        outer_radius=outer_radius,
                        center_radius=center_radius,
                        search_deg=broad_range_deg,
                        step_deg=broad_step_deg,
                    )
                    broad_rescue["result"] = broad
                    if broad.get("available"):
                        broad_sel = broad.get("selected") or {}
                        local_sel = centerline_refine.get("selected") or {}
                        broad_angle = float(broad.get("refined_angle", selected_angle_raw))
                        broad_delta = ((broad_angle - float(selected_angle_raw) + 180.0) % 360.0) - 180.0
                        broad_score = float(broad_sel.get("score", 0.0))
                        local_score = float(local_sel.get("score", 0.0))
                        broad_cov = float(broad_sel.get("coverage", 0.0))
                        local_cov = float(local_sel.get("coverage", 0.0))
                        broad_off = float(broad_sel.get("offset", 999.0))
                        local_off = float(local_sel.get("offset", 999.0))
                        broad_axis_eval = _score_axis_orientation_by_profile_positions(
                            mask=mask,
                            center=center,
                            line_angle=broad_angle,
                            opposite_angle=normalize_360(broad_angle + 180.0),
                            expected_tip=exp_tip,
                            expected_tail=exp_tail,
                            outer_radius=outer_radius,
                            center_radius=center_radius,
                        )
                        axis_supports_broad = bool(
                            broad_axis_eval.get("available")
                            and broad_axis_eval.get("clear")
                            and broad_axis_eval.get("selected") == "line_as_tip"
                        )
                        score_clear = bool(
                            broad_score >= local_score * min_score_ratio
                            or broad_score >= local_score + min_score_difference
                        )
                        geometry_clear = bool(
                            broad_cov >= local_cov + min_cov_improvement
                            or broad_off <= local_off - min_offset_improvement
                        )
                        distinct_axis = bool(abs(broad_delta) > min_axis_sep and abs(broad_delta) <= max_axis_sep)
                        broad_in_scale, broad_scale_ratio = scale_position(broad_angle)
                        override_enabled = bool(admin_value("broad_axis_rescue.extreme_low_resolution_override.enabled", True))
                        override_score_ratio = float(admin_value("broad_axis_rescue.extreme_low_resolution_override.minimum_score_ratio", 1.60))
                        override_coverage_max = float(admin_value("broad_axis_rescue.extreme_low_resolution_override.coverage_max", 0.80))
                        override_outer_max = int(admin_value("broad_axis_rescue.extreme_low_resolution_override.outer_radius_max_px", 140))
                        extreme_override = bool(
                            override_enabled
                            and ((not bool(admin_value("broad_axis_rescue.extreme_low_resolution_override.require_outer_invalid", True))) or (not bool(outer_valid)))
                            and int(outer_radius) < override_outer_max
                            and coverage < override_coverage_max
                            and ((not bool(admin_value("broad_axis_rescue.extreme_low_resolution_override.require_local_search_edge_hit", True))) or at_search_edge)
                            and broad_score >= local_score * override_score_ratio
                            and ((not bool(admin_value("broad_axis_rescue.extreme_low_resolution_override.require_geometry_clear", True))) or geometry_clear)
                            and ((not bool(admin_value("broad_axis_rescue.extreme_low_resolution_override.require_scale_range", True))) or broad_in_scale)
                        )
                        require_axis_support = bool(admin_value("broad_axis_rescue.acceptance.require_profile_axis_support", True))
                        normal_accept_broad = bool(
                            distinct_axis
                            and score_clear
                            and geometry_clear
                            and ((not require_axis_support) or axis_supports_broad or extreme_override)
                        )

                        # v10.0.24: 316相当の極低解像度ではcoverageが高く見えても、
                        # 先端肥大と探索端到達が同時に起きることがある。限定条件下で
                        # 広域候補を別軸として評価し、通常の3度制限では届かない軸を救済する。
                        lowres_acc = lowres_override_cfg.get("acceptance", {}) or {}
                        top_n = max(1, int(lowres_acc.get("top_candidate_count", 3)))
                        top_angles = [float(c.get("angle", broad_angle)) for c in (broad.get("top_candidates") or [])[:top_n]]
                        if len(top_angles) >= 2:
                            ref_angle = top_angles[0]
                            top_spread = max(abs(((a - ref_angle + 180.0) % 360.0) - 180.0) for a in top_angles[1:])
                        else:
                            top_spread = 0.0
                        lowres_score_clear = bool(
                            broad_score >= local_score * float(lowres_acc.get("minimum_score_ratio", 1.08))
                            or broad_score >= local_score + float(lowres_acc.get("minimum_score_difference", 20.0))
                        )
                        lowres_offset_limit_px = max(
                            float(lowres_acc.get("maximum_axis_offset_px", 6.0)),
                            float(outer_radius) * float(lowres_acc.get("maximum_axis_offset_outer_ratio", 0.06)),
                        )
                        lowres_offset_clear = bool(broad_off <= lowres_offset_limit_px)
                        lowres_spread_clear = bool(top_spread <= float(lowres_acc.get("maximum_top_angle_spread_deg", 2.0)))
                        lowres_scale_clear = bool((not bool(lowres_acc.get("require_scale_range", True))) or broad_in_scale)
                        lowres_profile_length_clear = bool((not bool(lowres_acc.get("require_profile_length_support", True))) or shape_len_ok)
                        bypass_normal_geometry = bool(lowres_acc.get("bypass_normal_geometry_clear", True))
                        bypass_normal_axis = bool(lowres_acc.get("bypass_normal_axis_support", True))
                        lowres_normal_geometry_clear = bool(bypass_normal_geometry or geometry_clear)
                        lowres_normal_axis_clear = bool(bypass_normal_axis or (not require_axis_support) or axis_supports_broad)
                        lowres_accept_broad = bool(
                            lowres_override_trigger
                            and distinct_axis
                            and lowres_score_clear
                            and lowres_offset_clear
                            and lowres_spread_clear
                            and lowres_scale_clear
                            and lowres_profile_length_clear
                            and lowres_normal_geometry_clear
                            and lowres_normal_axis_clear
                        )
                        accept_broad = bool(normal_accept_broad or lowres_accept_broad)
                        broad_rescue.update({
                            "broad_angle": float(broad_angle),
                            "broad_delta": float(broad_delta),
                            "broad_score": float(broad_score),
                            "local_score": float(local_score),
                            "broad_coverage": float(broad_cov),
                            "local_coverage": float(local_cov),
                            "broad_offset": float(broad_off),
                            "local_offset": float(local_off),
                            "axis_supports_broad": bool(axis_supports_broad),
                            "score_clear": bool(score_clear),
                            "geometry_clear": bool(geometry_clear),
                            "distinct_axis": bool(distinct_axis),
                            "profile_axis_position_eval": broad_axis_eval,
                            "broad_in_scale": bool(broad_in_scale),
                            "broad_scale_ratio": float(broad_scale_ratio),
                            "extreme_low_resolution_override": bool(extreme_override),
                            "normal_acceptance": bool(normal_accept_broad),
                            "low_resolution_widening_override_acceptance": bool(lowres_accept_broad),
                            "low_resolution_widening_override_checks": {
                                "score_clear": bool(lowres_score_clear),
                                "offset_clear": bool(lowres_offset_clear),
                                "offset_value_px": float(broad_off),
                                "offset_limit_px": float(lowres_offset_limit_px),
                                "bypass_normal_geometry_clear": bool(bypass_normal_geometry),
                                "bypass_normal_axis_support": bool(bypass_normal_axis),
                                "normal_geometry_clear": bool(lowres_normal_geometry_clear),
                                "normal_axis_clear": bool(lowres_normal_axis_clear),
                                "top_spread_clear": bool(lowres_spread_clear),
                                "scale_clear": bool(lowres_scale_clear),
                                "profile_length_clear": bool(lowres_profile_length_clear),
                            },
                            "top_angle_spread_deg": float(top_spread),
                            "accepted": bool(accept_broad),
                        })
                        if accept_broad:
                            selected_angle = normalize_360(broad_angle)
                            centerline_refine["broad_axis_rescue_applied"] = True
                            centerline_refine["refined_angle"] = float(selected_angle)
                            centerline_refine["delta_deg"] = float(broad_delta)
                            centerline_refine["limited"] = False
                        else:
                            centerline_refine["broad_axis_rescue_applied"] = False
                centerline_refine["broad_axis_rescue"] = broad_rescue

                # v10.0.12 full-scale axis rescue with low-resolution fallback:
                # 初期軸が大きく外れ、±18度探索でも正解に届かない極低信頼画像だけを対象に、
                # プロファイルの有効目盛範囲全体を再探索する。通常画像には発動しない。
                full_rescue = {
                    "enabled": False,
                    "trigger": False,
                    "attempted": False,
                    "accepted": False,
                    "trigger_reasons": [],
                    "trigger_checks": {},
                    "candidate_count": 0,
                    "top_candidates": [],
                }
                full_enabled = bool(admin_value("full_scale_axis_rescue.enabled", True))
                full_outer_max = int(admin_value("full_scale_axis_rescue.trigger.outer_radius_max_px", 140))
                full_cov_max = float(admin_value("full_scale_axis_rescue.trigger.coverage_max", 0.50))
                full_shape_max = float(admin_value("full_scale_axis_rescue.trigger.shape_score_max", 85.0))
                full_require_edge = bool(admin_value("full_scale_axis_rescue.trigger.require_local_search_edge_hit", False))
                full_require_widening = bool(admin_value("full_scale_axis_rescue.trigger.require_suspicious_tip_widening", True))
                full_require_outer_invalid = bool(admin_value("full_scale_axis_rescue.trigger.require_outer_invalid", True))
                # v10.0.15: 通常の低coverage条件に加え、極低解像度でcoverageだけが
                # 中程度に見える誤軸を、中心線オフセットと先端/中間幅比で限定検出する。
                local_selected_for_trigger = centerline_refine.get("selected") or {}
                local_centerline_offset = float(local_selected_for_trigger.get("offset", 999.0))
                tip_mid_width_ratio = float(tip_w) / max(float(mid_w), 1.0)
                geom_enabled = bool(admin_value("full_scale_axis_rescue.trigger.geometry_fallback.enabled", True))
                geom_cov_max = float(admin_value("full_scale_axis_rescue.trigger.geometry_fallback.coverage_max", 0.80))
                geom_min_offset = float(admin_value("full_scale_axis_rescue.trigger.geometry_fallback.minimum_centerline_offset_px", 5.0))
                geom_min_width_ratio = float(admin_value("full_scale_axis_rescue.trigger.geometry_fallback.minimum_tip_mid_width_ratio", 2.0))
                geom_require_outer_invalid = bool(admin_value("full_scale_axis_rescue.trigger.geometry_fallback.require_outer_invalid", True))
                geom_outer_max = int(admin_value("full_scale_axis_rescue.trigger.geometry_fallback.outer_radius_max_px", 140))

                common_trigger_checks = {
                    "enabled": bool(full_enabled),
                    "shape_score_clear": bool(float(shape_len_score) < full_shape_max),
                    "tip_widening_clear": bool((not full_require_widening) or suspicious_tip_widening),
                    "local_search_edge_clear": bool((not full_require_edge) or at_search_edge),
                    "broad_rescue_not_applied": bool(not bool(centerline_refine.get("broad_axis_rescue_applied"))),
                }
                normal_trigger_checks = {
                    "outer_invalid_clear": bool((not full_require_outer_invalid) or (not bool(outer_valid))),
                    "outer_radius_clear": bool(int(outer_radius) < full_outer_max),
                    "coverage_clear": bool(coverage < full_cov_max),
                }
                geometry_trigger_checks = {
                    "geometry_enabled": bool(geom_enabled),
                    "outer_invalid_clear": bool((not geom_require_outer_invalid) or (not bool(outer_valid))),
                    "outer_radius_clear": bool(int(outer_radius) < geom_outer_max),
                    "coverage_clear": bool(coverage < geom_cov_max),
                    "centerline_offset_clear": bool(local_centerline_offset >= geom_min_offset),
                    "tip_mid_width_ratio_clear": bool(tip_mid_width_ratio >= geom_min_width_ratio),
                }
                normal_trigger = bool(all(common_trigger_checks.values()) and all(normal_trigger_checks.values()))
                geometry_trigger = bool(all(common_trigger_checks.values()) and all(geometry_trigger_checks.values()))
                full_trigger = bool(normal_trigger or geometry_trigger)
                full_trigger_checks = dict(common_trigger_checks)
                full_trigger_checks.update({
                    "normal_trigger": bool(normal_trigger),
                    "geometry_trigger": bool(geometry_trigger),
                    "normal": normal_trigger_checks,
                    "geometry_fallback": geometry_trigger_checks,
                })
                full_rescue["enabled"] = bool(full_enabled)
                full_rescue["trigger"] = bool(full_trigger)
                full_rescue["trigger_checks"] = full_trigger_checks
                full_rescue["trigger_values"] = {
                    "outer_valid": bool(outer_valid),
                    "outer_radius": int(outer_radius),
                    "outer_radius_max_px": int(full_outer_max),
                    "coverage": float(coverage),
                    "coverage_max": float(full_cov_max),
                    "shape_score": float(shape_len_score),
                    "shape_score_max": float(full_shape_max),
                    "tip_width": float(tip_w),
                    "mid_width": float(mid_w),
                    "suspicious_tip_widening": bool(suspicious_tip_widening),
                    "search_edge_hit": bool(at_search_edge),
                    "require_local_search_edge_hit": bool(full_require_edge),
                    "normal_trigger": bool(normal_trigger),
                    "geometry_trigger": bool(geometry_trigger),
                    "centerline_offset": float(local_centerline_offset),
                    "geometry_minimum_centerline_offset_px": float(geom_min_offset),
                    "tip_mid_width_ratio": float(tip_mid_width_ratio),
                    "geometry_minimum_tip_mid_width_ratio": float(geom_min_width_ratio),
                    "geometry_coverage_max": float(geom_cov_max),
                    "geometry_outer_radius_max_px": int(geom_outer_max),
                }
                if full_trigger:
                    full_rescue["attempted"] = True
                    full_rescue["trigger_reasons"] = [
                        "geometry_fallback" if geometry_trigger and not normal_trigger else "normal_low_coverage",
                        "outer_invalid" if not bool(outer_valid) else "outer_condition_disabled",
                        "small_outer_radius",
                        (f"coverage_below_{geom_cov_max}" if geometry_trigger and not normal_trigger else f"coverage_below_{full_cov_max}"),
                        f"shape_score_below_{full_shape_max}",
                        "suspicious_tip_widening" if suspicious_tip_widening else "tip_widening_condition_disabled",
                    ]
                    full_step = float(admin_value("full_scale_axis_rescue.search.step_deg", 0.5))
                    use_scale_range = bool(admin_value("full_scale_axis_rescue.search.use_effective_scale_range", True))
                    mn_n = normalize_360(min_angle)
                    mx_n = normalize_360(max_angle)
                    if direction == "clockwise":
                        total_span = (mn_n - mx_n) % 360.0
                        sample_angles = [normalize_360(mn_n - i * full_step) for i in range(int(total_span / full_step) + 1)]
                    else:
                        total_span = (mx_n - mn_n) % 360.0
                        sample_angles = [normalize_360(mn_n + i * full_step) for i in range(int(total_span / full_step) + 1)]
                    if not use_scale_range:
                        sample_angles = [normalize_360(i * full_step) for i in range(int(360.0 / full_step))]

                    ranked = []
                    for test_angle in sample_angles:
                        test = refine_tip_angle_by_mask_center_symmetry(
                            mask=mask,
                            center=center,
                            base_angle=float(test_angle),
                            outer_radius=outer_radius,
                            center_radius=center_radius,
                            search_deg=0.0,
                            step_deg=full_step,
                        )
                        if not test.get("available"):
                            continue
                        sel = test.get("selected") or {}
                        test_score = float(sel.get("score", 0.0))
                        test_cov = float(sel.get("coverage", 0.0))
                        test_off = float(sel.get("offset", 999.0))
                        test_angle_refined = float(test.get("refined_angle", test_angle))
                        in_scale, scale_ratio_value = scale_position(test_angle_refined)
                        axis_eval = _score_axis_orientation_by_profile_positions(
                            mask=mask,
                            center=center,
                            line_angle=test_angle_refined,
                            opposite_angle=normalize_360(test_angle_refined + 180.0),
                            expected_tip=exp_tip,
                            expected_tail=exp_tail,
                            outer_radius=outer_radius,
                            center_radius=center_radius,
                        )
                        axis_support = bool(
                            axis_eval.get("available")
                            and axis_eval.get("clear")
                            and axis_eval.get("selected") == "line_as_tip"
                        )
                        ranked.append({
                            "angle": test_angle_refined,
                            "score": test_score,
                            "coverage": test_cov,
                            "offset": test_off,
                            "in_scale": bool(in_scale),
                            "scale_ratio": float(scale_ratio_value),
                            "axis_support": bool(axis_support),
                            "profile_axis_position_eval": axis_eval,
                        })
                    ranked.sort(key=lambda x: (x["score"], x["coverage"], -x["offset"]), reverse=True)
                    full_rescue["candidate_count"] = len(ranked)
                    full_rescue["top_candidates"] = ranked[:5]
                    if ranked:
                        cand = ranked[0]
                        local_sel = centerline_refine.get("selected") or {}
                        local_score = float(local_sel.get("score", 0.0))
                        local_cov = float(local_sel.get("coverage", 0.0))
                        min_ratio = float(admin_value("full_scale_axis_rescue.acceptance.minimum_score_ratio", 1.80))
                        min_diff = float(admin_value("full_scale_axis_rescue.acceptance.minimum_score_difference", 120.0))
                        min_cov = float(admin_value("full_scale_axis_rescue.acceptance.minimum_coverage", 0.70))
                        min_cov_imp = float(admin_value("full_scale_axis_rescue.acceptance.minimum_coverage_improvement", 0.25))
                        max_off = float(admin_value("full_scale_axis_rescue.acceptance.maximum_axis_offset_px", 3.0))
                        require_axis = bool(admin_value("full_scale_axis_rescue.acceptance.require_profile_axis_support", True))
                        require_scale = bool(admin_value("full_scale_axis_rescue.acceptance.require_scale_range", True))
                        score_clear = bool(cand["score"] >= local_score * min_ratio and cand["score"] >= local_score + min_diff)
                        coverage_clear = bool(cand["coverage"] >= min_cov and cand["coverage"] >= local_cov + min_cov_imp)

                        # v10.0.12: full-scale候補でoffsetが未計算のとき、999などの
                        # センチネル値を実測オフセットとして失格にしない。極低解像度かつ
                        # 上位候補が同一角度へ集中している場合のみ、限定的な代替判定を許可する。
                        low_cfg_enabled = bool(admin_value("full_scale_axis_rescue.low_resolution_fallback.enabled", True))
                        offset_unavailable_threshold = float(admin_value("full_scale_axis_rescue.low_resolution_fallback.offset_unavailable_threshold", 900.0))
                        low_min_ratio = float(admin_value("full_scale_axis_rescue.low_resolution_fallback.minimum_score_ratio", 2.50))
                        low_min_cov = float(admin_value("full_scale_axis_rescue.low_resolution_fallback.minimum_coverage", 0.90))
                        low_top_n = max(2, int(admin_value("full_scale_axis_rescue.low_resolution_fallback.top_candidate_count", 3)))
                        low_max_spread = float(admin_value("full_scale_axis_rescue.low_resolution_fallback.maximum_top_angle_spread_deg", 2.0))
                        low_min_scale_ratio = float(admin_value("full_scale_axis_rescue.low_resolution_fallback.minimum_scale_ratio", 0.35))
                        low_max_scale_ratio = float(admin_value("full_scale_axis_rescue.low_resolution_fallback.maximum_scale_ratio", 0.75))
                        low_require_outer_invalid = bool(admin_value("full_scale_axis_rescue.low_resolution_fallback.require_outer_invalid", True))
                        low_outer_max = int(admin_value("full_scale_axis_rescue.low_resolution_fallback.outer_radius_max_px", 140))

                        top_for_cluster = ranked[:low_top_n]
                        top_angles = [float(x["angle"]) for x in top_for_cluster]
                        if len(top_angles) >= 2:
                            # 候補は同一目盛範囲内に並ぶため、先頭角度を基準に循環差へ変換する。
                            base_cluster_angle = top_angles[0]
                            unwrapped = [base_cluster_angle + (((a - base_cluster_angle + 180.0) % 360.0) - 180.0) for a in top_angles]
                            top_angle_spread = float(max(unwrapped) - min(unwrapped))
                        else:
                            top_angle_spread = 999.0

                        offset_available = bool(float(cand["offset"]) < offset_unavailable_threshold)
                        lowres_geometry_clear = bool(
                            low_cfg_enabled
                            and ((not low_require_outer_invalid) or (not bool(outer_valid)))
                            and int(outer_radius) < low_outer_max
                            and float(cand["score"]) >= local_score * low_min_ratio
                            and float(cand["coverage"]) >= low_min_cov
                            and len(top_for_cluster) >= low_top_n
                            and top_angle_spread <= low_max_spread
                            and bool(cand["in_scale"])
                            and low_min_scale_ratio <= float(cand["scale_ratio"]) <= low_max_scale_ratio
                        )

                        offset_clear = bool((offset_available and cand["offset"] <= max_off) or ((not offset_available) and lowres_geometry_clear))
                        axis_clear = bool((not require_axis) or cand["axis_support"] or lowres_geometry_clear)
                        scale_clear = bool((not require_scale) or cand["in_scale"])
                        normal_accept_full = bool(score_clear and coverage_clear and offset_clear and axis_clear and scale_clear)

                        # v10.0.16: geometry_fallbackで探索したケースだけに専用の採用条件を適用する。
                        # 通常の全範囲救済条件は一切緩めず、低解像度・外周円無効・候補集中時に限定する。
                        geom_acc_enabled = bool(admin_value("full_scale_axis_rescue.geometry_fallback_acceptance.enabled", True))
                        geom_acc_min_ratio = float(admin_value("full_scale_axis_rescue.geometry_fallback_acceptance.minimum_score_ratio", 1.50))
                        geom_acc_min_diff = float(admin_value("full_scale_axis_rescue.geometry_fallback_acceptance.minimum_score_difference", 150.0))
                        geom_acc_min_cov = float(admin_value("full_scale_axis_rescue.geometry_fallback_acceptance.minimum_coverage", 0.88))
                        geom_acc_min_cov_imp = float(admin_value("full_scale_axis_rescue.geometry_fallback_acceptance.minimum_coverage_improvement", 0.20))
                        geom_acc_max_spread = float(admin_value("full_scale_axis_rescue.geometry_fallback_acceptance.maximum_top_angle_spread_deg", 2.0))
                        geom_acc_min_scale = float(admin_value("full_scale_axis_rescue.geometry_fallback_acceptance.minimum_scale_ratio", 0.60))
                        geom_acc_max_scale = float(admin_value("full_scale_axis_rescue.geometry_fallback_acceptance.maximum_scale_ratio", 0.80))
                        geom_acc_require_outer_invalid = bool(admin_value("full_scale_axis_rescue.geometry_fallback_acceptance.require_outer_invalid", True))
                        geom_acc_outer_max = int(admin_value("full_scale_axis_rescue.geometry_fallback_acceptance.outer_radius_max_px", 140))
                        geom_acc_allow_unavailable_offset = bool(admin_value("full_scale_axis_rescue.geometry_fallback_acceptance.allow_unavailable_offset", True))
                        geom_acc_allow_axis_substitution = bool(admin_value("full_scale_axis_rescue.geometry_fallback_acceptance.allow_lowres_axis_support_substitution", True))

                        geom_score_clear = bool(
                            float(cand["score"]) >= local_score * geom_acc_min_ratio
                            and float(cand["score"]) >= local_score + geom_acc_min_diff
                        )
                        geom_coverage_clear = bool(
                            float(cand["coverage"]) >= geom_acc_min_cov
                            and float(cand["coverage"]) >= local_cov + geom_acc_min_cov_imp
                        )
                        geom_cluster_clear = bool(
                            len(top_for_cluster) >= low_top_n
                            and top_angle_spread <= geom_acc_max_spread
                        )
                        geom_scale_clear = bool(
                            bool(cand["in_scale"])
                            and geom_acc_min_scale <= float(cand["scale_ratio"]) <= geom_acc_max_scale
                        )
                        geom_outer_clear = bool(
                            ((not geom_acc_require_outer_invalid) or (not bool(outer_valid)))
                            and int(outer_radius) < geom_acc_outer_max
                        )
                        geom_offset_clear = bool(
                            (offset_available and float(cand["offset"]) <= max_off)
                            or ((not offset_available) and geom_acc_allow_unavailable_offset)
                        )
                        geom_axis_clear = bool(
                            (not require_axis)
                            or bool(cand["axis_support"])
                            or geom_acc_allow_axis_substitution
                        )
                        geometry_accept_full = bool(
                            geometry_trigger
                            and geom_acc_enabled
                            and geom_score_clear
                            and geom_coverage_clear
                            and geom_cluster_clear
                            and geom_scale_clear
                            and geom_outer_clear
                            and geom_offset_clear
                            and geom_axis_clear
                        )
                        accept_full = bool(normal_accept_full or geometry_accept_full)
                        full_delta = ((float(cand["angle"]) - float(selected_angle_raw) + 180.0) % 360.0) - 180.0
                        full_rescue.update({
                            "original_angle": float(selected_angle_raw),
                            "candidate_angle": float(cand["angle"]),
                            "delta_deg": float(full_delta),
                            "original_score": float(local_score),
                            "candidate_score": float(cand["score"]),
                            "score_ratio": float(cand["score"] / max(local_score, 1e-6)),
                            "original_coverage": float(local_cov),
                            "candidate_coverage": float(cand["coverage"]),
                            "candidate_offset": float(cand["offset"]),
                            "score_clear": bool(score_clear),
                            "coverage_clear": bool(coverage_clear),
                            "offset_clear": bool(offset_clear),
                            "offset_available": bool(offset_available),
                            "axis_support_clear": bool(axis_clear),
                            "scale_range_clear": bool(scale_clear),
                            "low_resolution_fallback_clear": bool(lowres_geometry_clear),
                            "top_candidate_angle_spread_deg": float(top_angle_spread),
                            "normal_acceptance_clear": bool(normal_accept_full),
                            "geometry_acceptance_clear": bool(geometry_accept_full),
                            "geometry_acceptance_checks": {
                                "enabled": bool(geom_acc_enabled),
                                "triggered_by_geometry": bool(geometry_trigger),
                                "score_clear": bool(geom_score_clear),
                                "coverage_clear": bool(geom_coverage_clear),
                                "candidate_cluster_clear": bool(geom_cluster_clear),
                                "scale_ratio_clear": bool(geom_scale_clear),
                                "outer_geometry_clear": bool(geom_outer_clear),
                                "offset_clear": bool(geom_offset_clear),
                                "axis_support_clear": bool(geom_axis_clear),
                            },
                            "accepted": bool(accept_full),
                            "acceptance_thresholds": {
                                "minimum_score_ratio": float(min_ratio),
                                "minimum_score_difference": float(min_diff),
                                "minimum_coverage": float(min_cov),
                                "minimum_coverage_improvement": float(min_cov_imp),
                                "maximum_axis_offset_px": float(max_off),
                                "require_profile_axis_support": bool(require_axis),
                                "require_scale_range": bool(require_scale),
                                "geometry_fallback_acceptance": {
                                    "enabled": bool(geom_acc_enabled),
                                    "minimum_score_ratio": float(geom_acc_min_ratio),
                                    "minimum_score_difference": float(geom_acc_min_diff),
                                    "minimum_coverage": float(geom_acc_min_cov),
                                    "minimum_coverage_improvement": float(geom_acc_min_cov_imp),
                                    "maximum_top_angle_spread_deg": float(geom_acc_max_spread),
                                    "minimum_scale_ratio": float(geom_acc_min_scale),
                                    "maximum_scale_ratio": float(geom_acc_max_scale),
                                    "require_outer_invalid": bool(geom_acc_require_outer_invalid),
                                    "outer_radius_max_px": int(geom_acc_outer_max),
                                    "allow_unavailable_offset": bool(geom_acc_allow_unavailable_offset),
                                    "allow_lowres_axis_support_substitution": bool(geom_acc_allow_axis_substitution),
                                },
                                "low_resolution_fallback": {
                                    "enabled": bool(low_cfg_enabled),
                                    "offset_unavailable_threshold": float(offset_unavailable_threshold),
                                    "minimum_score_ratio": float(low_min_ratio),
                                    "minimum_coverage": float(low_min_cov),
                                    "top_candidate_count": int(low_top_n),
                                    "maximum_top_angle_spread_deg": float(low_max_spread),
                                    "minimum_scale_ratio": float(low_min_scale_ratio),
                                    "maximum_scale_ratio": float(low_max_scale_ratio),
                                    "require_outer_invalid": bool(low_require_outer_invalid),
                                    "outer_radius_max_px": int(low_outer_max),
                                },
                            },
                        })
                        if accept_full:
                            selected_angle = normalize_360(float(cand["angle"]))
                            centerline_refine["full_scale_axis_rescue_applied"] = True
                            centerline_refine["refined_angle"] = float(selected_angle)
                            centerline_refine["delta_deg"] = float(full_delta)
                            centerline_refine["limited"] = False
                        else:
                            centerline_refine["full_scale_axis_rescue_applied"] = False
                centerline_refine["full_scale_axis_rescue"] = full_rescue

                # 異常に大きい補正は従来どおり不採用。
                # ただし、厳格条件を通過した広域救済は別軸への意図的な再選択なので除外する。
                if (not bool(centerline_refine.get("broad_axis_rescue_applied"))) and (not bool(centerline_refine.get("full_scale_axis_rescue_applied"))) and abs(float(centerline_refine.get("delta_deg", 0.0))) > float(admin_value("centerline_refine.reject_delta_deg", 7.0)):
                    centerline_refine["rejected"] = True
                    centerline_refine["reject_reason"] = "delta_too_large"
                    selected_angle = float(selected_angle_raw)
            raw_tip_for_selected = _pt_from_center(selected_angle, selected_r)
            ext = profile_guided_tip_extension(
                mask=mask,
                center=center,
                base_angle=selected_angle,
                outer_radius=outer_radius,
                center_radius=center_radius,
                needle_shape=needle_shape,
                current_tip=raw_tip_for_selected,
                raw_line_tip=raw_tip_for_selected,
                angle_window=10.0,
            )
            selected = ext.get("selected") if isinstance(ext, dict) else None
            if isinstance(selected, dict) and selected.get("tip") is not None:
                tip = selected["tip"]
                corrected_angle = float(selected.get("angle", selected_angle))
                ext_used = bool(ext.get("used"))
            else:
                tip = raw_tip_for_selected
                corrected_angle = selected_angle
                ext_used = False
            priority_selector["selected"] = "line_angle" if score_a >= score_b else "opposite_angle"
            priority_selector["selector_margin"] = float(selector_margin)
            priority_selector["profile_guided_extension_used"] = bool(ext_used)
            thin_debug = {
                "chosen_reason": "line_segment_priority_scale_ratio_tip_tail_length_profile_guided_tip_extension",
                "corrected_angle": float(corrected_angle),
                "angle_a": line_ang,
                "angle_b": opp_ang,
                "profile_a": {},
                "profile_b": {},
                "profile_guided_tip_extension": ext,
                "needle_centerline_refine": centerline_refine,
                "priority_tip_selector": priority_selector,
                "connected_mask_debug": needle_connected_debug,
                "line_segment_priority_before_tip_fix": True,
                "line_segment_raw_angle": raw_angle,
                "line_segment_raw_tip": best.get("tip"),
                "line_segment_reject_reason": line_priority_reject_reason,
                "line_direction_debug": line_direction_debug,
            }
        else:
            tip, corrected_angle, thin_debug = choose_thin_tip(
                mask=mask_for_tip,
                center=center,
                angle_deg=raw_angle,
                outer_radius=outer_radius,
                fallback_tip=best.get("tip"),
                center_radius=center_radius,
                needle_shape=needle_shape,
                direction_search_deg=14.0,
            )
            ch_in_scale, _ = scale_position(corrected_angle)
            if (not ch_in_scale) and bool(line_direction_debug.get("line_in_scale")):
                ext = profile_guided_tip_extension(
                    mask=mask, center=center, base_angle=raw_angle, outer_radius=outer_radius,
                    center_radius=center_radius, needle_shape=needle_shape, current_tip=best.get("tip"), raw_line_tip=best.get("tip"), angle_window=14.0
                )
                selected = ext.get("selected") if isinstance(ext, dict) else None
                if isinstance(selected, dict) and selected.get("tip") is not None:
                    tip = selected["tip"]
                    corrected_angle = float(selected.get("angle", raw_angle))
                    thin_debug["profile_guided_tip_extension"] = ext
                else:
                    tip = best.get("tip")
                    corrected_angle = raw_angle
                thin_debug["chosen_reason"] = "scale_guard_restored_line_direction_from_" + str(thin_debug.get("chosen_reason"))
            thin_debug["connected_mask_debug"] = needle_connected_debug
            thin_debug["chosen_reason"] = "line_segment_taper_fallback_after_unclear_priority_" + str(thin_debug.get("chosen_reason"))
            thin_debug["priority_tip_selector"] = priority_selector
            thin_debug["line_segment_priority_before_tip_fix"] = True
            thin_debug["line_segment_raw_angle"] = raw_angle
            thin_debug["line_segment_raw_tip"] = best.get("tip")
            thin_debug["line_segment_reject_reason"] = line_priority_reject_reason
            thin_debug["line_direction_debug"] = line_direction_debug
    elif isinstance(line_opposite_override, dict) and line_opposite_override.get("method") == "line_segment_opposite_probe":
        best = line_opposite_override
        raw_angle = float(best["angle"])
        # line軸の反対側を狭い窓で再評価しているため、ここでは180度比較で再び尾側へ戻さない。
        tip = best["tip"]
        corrected_angle = float(best["angle"])
        thin_debug = {
            "chosen_reason": "line_segment_opposite_probe_after_wide_tail_reject",
            "corrected_angle": corrected_angle,
            "line_segment_priority_before_tip_fix": False,
            "line_segment_reject_reason": line_priority_reject_reason,
            "line_direction_debug": line_direction_debug,
            "line_opposite_override": best,
            "angle_a": corrected_angle,
            "angle_b": normalize_360(corrected_angle + 180.0),
            "profile_a": best,
            "profile_b": {},
        }
    else:
        best = ray_best
        # v9.33.5: the pre-selection template may already have identified which
        # side of this axis matches the learned long needle. Use that orientation
        # as the starting angle, then keep choose_thin_tip as the final verifier.
        template_pref_angle = (best.get("tip_tail_length_template") or {}).get("preferred_tip_angle")
        template_margin = float((best.get("tip_tail_length_template") or {}).get("orientation_margin", 0.0) or 0.0)
        raw_angle = float(template_pref_angle) if template_pref_angle is not None and template_margin >= 35.0 else calc_angle(center, best["tip"])
        tip, corrected_angle, thin_debug = choose_thin_tip(
            mask=mask_for_tip,
            center=center,
            angle_deg=raw_angle,
            outer_radius=outer_radius,
            fallback_tip=best["tip"],
            center_radius=center_radius,
            needle_shape=needle_shape,
        )
        thin_debug["line_segment_priority_before_tip_fix"] = False
        thin_debug["preselection_tip_tail_length_template"] = best.get("tip_tail_length_template")
        thin_debug["preselection_used_preferred_tip_angle"] = bool(template_pref_angle is not None and template_margin >= 35.0)
        thin_debug["line_segment_reject_reason"] = line_priority_reject_reason
        thin_debug["line_direction_debug"] = line_direction_debug
        thin_debug["line_opposite_override"] = line_opposite_override

    # v10.0.14: post-selection ray/line disagreement rescue.
    # Evaluate after the ray path has produced its final corrected angle.  This is
    # essential because the provisional ray angle can be close to the line angle,
    # while choose_thin_tip may later move the final ray result several degrees.
    post_rescue = {
        "enabled": bool(admin_value("ray_line_disagreement_rescue.enabled", True)),
        "attempted": False,
        "accepted": False,
    }
    if (
        post_rescue["enabled"]
        and line_best is not None
        and not bool(line_priority_enabled)
        and isinstance(best, dict)
        and best is ray_best
    ):
        cfg_t = admin_value("ray_line_disagreement_rescue.trigger", {}) or {}
        cfg_a = admin_value("ray_line_disagreement_rescue.acceptance", {}) or {}
        line_angle_dbg = float(line_direction_debug.get("line_angle", line_best.get("angle", 0.0)))
        final_ray_angle_dbg = float(corrected_angle)
        angle_diff = abs(((line_angle_dbg - final_ray_angle_dbg + 180.0) % 360.0) - 180.0)
        raw_centerline_angle = float(line_best.get("raw_angle", line_angle_dbg))
        raw_centerline_diff = abs(((raw_centerline_angle - line_angle_dbg + 180.0) % 360.0) - 180.0)
        width_limit = float(line_direction_debug.get("critical_line_width_limit", 1.0))
        width_ratio = float(line_best.get("tip_width", 999.0)) / max(width_limit, 1.0)
        shape_score = float(line_direction_debug.get("shape_length_score", 0.0))
        checks = {
            "angle_difference": angle_diff >= float(cfg_t.get("minimum_angle_difference_deg", 3.0)),
            "outer_valid": (bool(outer_valid) if bool(cfg_t.get("require_outer_valid", True)) else True),
            "outer_radius_min": float(outer_radius) >= float(cfg_t.get("outer_radius_min_px", 180)),
            "outer_radius_max": float(outer_radius) <= float(cfg_t.get("outer_radius_max_px", 400)),
            "line_in_scale": (bool(line_direction_debug.get("line_in_scale")) if bool(cfg_t.get("require_line_in_scale", True)) else True),
            "length_match": (bool(line_direction_debug.get("shape_length_match")) if bool(cfg_t.get("require_length_match", True)) else True),
            "shape_score": shape_score >= float(cfg_t.get("minimum_shape_score", 75.0)),
            "tip_width_excess": width_ratio <= float(cfg_t.get("maximum_tip_width_excess_ratio", 1.30)),
            "centerline_stable": raw_centerline_diff <= float(cfg_a.get("maximum_centerline_raw_difference_deg", 2.0)),
            "line_ray_difference_max": angle_diff <= float(cfg_a.get("maximum_line_ray_difference_deg", 6.0)),
            "profile_length_match": (bool(line_direction_debug.get("shape_length_match")) if bool(cfg_a.get("require_profile_tip_tail_length_match", True)) else True),
            "rejected_for_tip_width": bool(line_priority_reject_reason and "critical_tip_width" in str(line_priority_reject_reason)),
        }
        post_rescue.update({
            "attempted": True,
            "line_angle": line_angle_dbg,
            "final_ray_angle": final_ray_angle_dbg,
            "angle_difference": angle_diff,
            "raw_centerline_angle": raw_centerline_angle,
            "raw_centerline_difference": raw_centerline_diff,
            "tip_width": float(line_best.get("tip_width", 999.0)),
            "critical_tip_width_limit": width_limit,
            "tip_width_excess_ratio": width_ratio,
            "shape_score": shape_score,
            "reject_reason": line_priority_reject_reason,
            "checks": checks,
        })
        if all(checks.values()):
            corrected_angle = normalize_360(line_angle_dbg)
            line_r = float(line_best.get("max_d", line_best.get("length", outer_radius * 0.8)))
            line_tip = line_best.get("tip")
            if line_tip is None:
                rad = math.radians(corrected_angle)
                line_tip = (
                    int(round(center[0] + line_r * math.cos(rad))),
                    int(round(center[1] - line_r * math.sin(rad))),
                )
            tip = line_tip
            best = line_best
            raw_angle = line_angle_dbg
            post_rescue["accepted"] = True
            thin_debug["chosen_reason"] = "post_selection_ray_line_disagreement_rescue"
            thin_debug["corrected_angle"] = float(corrected_angle)
            thin_debug["line_segment_reject_reason"] = line_priority_reject_reason
    if isinstance(thin_debug, dict):
        thin_debug["ray_line_disagreement_rescue_post_selection"] = post_rescue

    # v10.0.24: 298相当。正しい線分が文字・目盛との軽微な接触で
    # tip-width超過だけを理由に棄却され、rayが大きく外れた場合に線分を復活する。
    width_rescue = {
        "enabled": bool(admin_value("line_width_rejection_rescue.enabled", True)),
        "attempted": False,
        "accepted": False,
    }
    if (
        width_rescue["enabled"]
        and line_best is not None
        and not bool(line_priority_enabled)
        and isinstance(best, dict)
        and best is ray_best
        and not bool(post_rescue.get("accepted"))
    ):
        cfg_t = admin_value("line_width_rejection_rescue.trigger", {}) or {}
        cfg_a = admin_value("line_width_rejection_rescue.acceptance", {}) or {}
        line_angle_dbg = float(line_direction_debug.get("line_angle", line_best.get("angle", 0.0)))
        final_ray_angle_dbg = float(corrected_angle)
        angle_diff = abs(((line_angle_dbg - final_ray_angle_dbg + 180.0) % 360.0) - 180.0)
        raw_centerline_angle = float(line_best.get("raw_angle", line_angle_dbg))
        centerline_line_diff = abs(((raw_centerline_angle - line_angle_dbg + 180.0) % 360.0) - 180.0)
        width_limit = float(line_direction_debug.get("critical_line_width_limit", 1.0))
        width_ratio = float(line_best.get("tip_width", 999.0)) / max(width_limit, 1.0)
        shape_score = float(line_direction_debug.get("shape_length_score", 0.0))
        reject_text = str(line_priority_reject_reason or "")
        only_tip_width_reject = bool(reject_text.startswith("rejected_critical_tip_width="))
        checks = {
            "outer_valid": (bool(outer_valid) if bool(cfg_t.get("require_outer_valid", True)) else True),
            "outer_radius_min": float(outer_radius) >= float(cfg_t.get("outer_radius_min_px", 180)),
            "outer_radius_max": float(outer_radius) <= float(cfg_t.get("outer_radius_max_px", 400)),
            "line_in_scale": (bool(line_direction_debug.get("line_in_scale")) if bool(cfg_t.get("require_line_in_scale", True)) else True),
            "opposite_out_of_scale": ((not bool(line_direction_debug.get("opposite_in_scale"))) if bool(cfg_t.get("require_opposite_out_of_scale", True)) else True),
            "profile_length_match": (bool(line_direction_debug.get("shape_length_match")) if bool(cfg_t.get("require_profile_length_match", True)) else True),
            "shape_score": shape_score >= float(cfg_t.get("minimum_shape_score", 75.0)),
            "rejection_reason_only_tip_width": (only_tip_width_reject if bool(cfg_t.get("rejection_reason_only_tip_width", True)) else True),
            "angle_difference_min": angle_diff >= float(cfg_t.get("minimum_line_ray_difference_deg", 8.0)),
            "angle_difference_max": angle_diff <= float(cfg_t.get("maximum_line_ray_difference_deg", 25.0)),
            "tip_width_excess": width_ratio <= float(cfg_t.get("maximum_tip_width_excess_ratio", 1.30)),
            "centerline_stable": centerline_line_diff <= float(cfg_a.get("maximum_centerline_line_difference_deg", 2.0)),
        }
        width_rescue.update({
            "attempted": True,
            "line_angle": float(line_angle_dbg),
            "ray_angle": float(final_ray_angle_dbg),
            "angle_difference": float(angle_diff),
            "raw_centerline_angle": float(raw_centerline_angle),
            "centerline_line_difference": float(centerline_line_diff),
            "tip_width": float(line_best.get("tip_width", 999.0)),
            "critical_tip_width_limit": float(width_limit),
            "tip_width_excess_ratio": float(width_ratio),
            "shape_score": float(shape_score),
            "reject_reason": line_priority_reject_reason,
            "checks": checks,
        })
        if all(checks.values()):
            corrected_angle = normalize_360(line_angle_dbg)
            line_r = float(line_best.get("max_d", line_best.get("length", outer_radius * 0.8)))
            line_tip = line_best.get("tip")
            if line_tip is None:
                rad = math.radians(corrected_angle)
                line_tip = (
                    int(round(center[0] + line_r * math.cos(rad))),
                    int(round(center[1] - line_r * math.sin(rad))),
                )
            tip = line_tip
            best = line_best
            raw_angle = line_angle_dbg
            width_rescue["accepted"] = True
            thin_debug["chosen_reason"] = "line_width_rejection_rescue"
            thin_debug["corrected_angle"] = float(corrected_angle)
            thin_debug["line_segment_reject_reason"] = line_priority_reject_reason
    if isinstance(thin_debug, dict):
        thin_debug["line_width_rejection_rescue"] = width_rescue

    debug = {
        "mask": mask,
        "profile_mask": profile_mask,
        "dark_mask": dark_mask,
        "edges": edges,
        "valid": valid,
        "candidates": candidates,
        "best": best,
        "line_best": line_best,
        "ray_best": ray_best,
        "line_opposite_override": line_opposite_override,
        "raw_angle": raw_angle,
        "needle_shape_scaled": needle_shape,
        "tip_tail_length_scoring_enabled": bool(needle_shape.get("available")),
        "tip_tail_length_scoring_policy": "all_hough_and_ray_candidates_bidirectional_tip_tail_full_span",
        "thin_debug": thin_debug,
    }

    return tip, corrected_angle, debug



def draw_meter_id(img: np.ndarray, meter_id: str, anchor: Point, scale: float = 0.85) -> None:
    """メーターIDを紫で描画する。"""
    x, y = anchor
    color = (255, 0, 255)
    cv2.putText(
        img,
        meter_id,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def make_meter_contact_sheet(items: List[Dict[str, Any]], max_width: int = 1400) -> np.ndarray:
    """複数のannotated_meterを横/グリッドに並べたmeter画面用画像を作る。"""
    if not items:
        return np.zeros((200, 400, 3), dtype=np.uint8)

    # 各cropを見やすい最大高さに揃える
    max_cell_h = 520
    prepared = []
    for item in items:
        img = item["annotated_meter"]
        h, w = img.shape[:2]
        scale = min(1.0, max_cell_h / max(1, h))
        disp = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else img.copy()
        prepared.append((item, disp))

    # 簡易グリッド
    pad = 18
    cell_ws = [d.shape[1] for _, d in prepared]
    cell_hs = [d.shape[0] for _, d in prepared]
    cell_w = max(cell_ws) + pad * 2
    cell_h = max(cell_hs) + pad * 2 + 32

    cols = max(1, min(len(prepared), max_width // max(1, cell_w)))
    rows = int(math.ceil(len(prepared) / cols))

    canvas = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)

    for idx, (item, disp) in enumerate(prepared):
        r = idx // cols
        c = idx % cols
        x0 = c * cell_w + pad
        y0 = r * cell_h + pad + 32

        # IDを紫で表示
        draw_meter_id(canvas, item["meter_id"], (c * cell_w + pad, r * cell_h + pad + 22), scale=0.8)

        h, w = disp.shape[:2]
        canvas[y0:y0 + h, x0:x0 + w] = disp

    return canvas


# -----------------------------
# Visualization
# -----------------------------

def draw_angle_line(img: np.ndarray, center: Point, angle: float, color: Tuple[int, int, int], label: str, radius: Optional[int] = None):
    if radius is None:
        radius = int(min(img.shape[:2]) * 0.42)

    p = point_from_angle(center, angle, radius)
    cv2.line(img, center, p, color, 1)
    cv2.putText(img, label, p, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def make_ray_score_map(shape, candidates, center: Point, outer_radius: int):
    h, w = shape[:2]
    score_map = np.zeros((h, w), dtype=np.uint8)
    if not candidates:
        return score_map

    max_score = max(float(c.get("final_score", c.get("score", 0))) for c in candidates)
    if max_score <= 0:
        max_score = 1.0

    for c in candidates:
        angle = c["angle"]
        val = int(max(0, min(255, 255 * float(c.get("final_score", c.get("score", 0))) / max_score)))
        p = point_from_angle(center, angle, outer_radius * 0.78)
        cv2.line(score_map, center, p, val, 1)

    return score_map



def score_radial_tick_angle(gray: np.ndarray, edges: np.ndarray, center: Point, outer_radius: int, angle_deg: float) -> float:
    """指定角度方向にある目盛り線らしさをスコア化する。"""
    h, w = gray.shape[:2]
    cx, cy = center
    rad = math.radians(angle_deg)
    ux = math.cos(rad)
    uy = -math.sin(rad)

    # 目盛りが存在する外周寄りだけを見る
    r1 = int(outer_radius * 0.62)
    r2 = int(outer_radius * 0.92)

    samples = []
    edge_hits = 0

    for r in range(r1, r2):
        x = int(round(cx + r * ux))
        y = int(round(cy + r * uy))
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            continue

        # 周辺3x3で暗さとエッジを見る
        patch = gray[y-1:y+2, x-1:x+2]
        samples.append(float(np.mean(patch)))
        if np.any(edges[y-1:y+2, x-1:x+2] > 0):
            edge_hits += 1

    if not samples:
        return -1e9

    # 暗いほど良い、エッジが多いほど良い
    darkness = 255.0 - float(np.mean(samples))
    contrast = float(np.std(samples))
    return darkness * 1.2 + edge_hits * 2.5 + contrast * 0.7


def refine_scale_angle(
    meter_img: np.ndarray,
    center: Point,
    outer_radius: int,
    approx_angle: float,
    search_deg: float = 10.0,
    step_deg: float = 0.25,
) -> Tuple[float, Dict[str, Any]]:
    """MIN/MAX角度を近傍の目盛り線へスナップする。

    キャリブレーション時と撮影時で中心が微妙に変わった場合、
    保存済み角度のままだとMIN/MAX線がズレるため、画像上の暗い放射状目盛りに寄せる。
    """
    gray = cv2.cvtColor(meter_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 60, 160)

    best_angle = approx_angle
    best_score = -1e9

    start = approx_angle - search_deg
    end = approx_angle + search_deg
    a = start
    while a <= end:
        score = score_radial_tick_angle(gray, edges, center, outer_radius, a)
        # 元角度から離れすぎるほど軽くペナルティ
        score -= abs(a - approx_angle) * 1.5
        if score > best_score:
            best_score = score
            best_angle = a
        a += step_deg

    return float(best_angle), {
        "approx_angle": float(approx_angle),
        "refined_angle": float(best_angle),
        "score": float(best_score),
        "search_deg": float(search_deg),
    }


def make_refined_profile_for_image(
    meter_img: np.ndarray,
    profile: Dict[str, Any],
    center: Point,
    outer_radius: int,
    enabled: bool,
    search_deg: float,
    accept_deg: float = 6.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """画像ごとのMIN/MAX角度補正を行った一時プロファイルを作る。"""
    refined = dict(profile)

    min_angle = float(profile.get("min_angle", profile.get("angle", {}).get("min_angle_deg", 225.0)))
    max_angle = float(profile.get("max_angle", profile.get("angle", {}).get("max_angle_deg", -45.0)))

    info = {
        "enabled": enabled,
        "min": {"approx_angle": min_angle, "refined_angle": min_angle},
        "max": {"approx_angle": max_angle, "refined_angle": max_angle},
    }

    if not enabled:
        return refined, info

    refined_min, min_info = refine_scale_angle(
        meter_img, center, outer_radius, min_angle, search_deg=search_deg
    )
    refined_max, max_info = refine_scale_angle(
        meter_img, center, outer_radius, max_angle, search_deg=search_deg
    )

    # 大きすぎる補正は、針や隣の目盛りへ吸着した可能性が高いので破棄する
    min_shift = angle_delta_abs(refined_min, min_angle)
    max_shift = angle_delta_abs(refined_max, max_angle)

    min_info["accepted"] = min_shift <= accept_deg
    max_info["accepted"] = max_shift <= accept_deg
    min_info["shift_deg"] = float(min_shift)
    max_info["shift_deg"] = float(max_shift)

    if min_info["accepted"]:
        refined["min_angle"] = refined_min
    else:
        refined["min_angle"] = min_angle
        min_info["refined_angle"] = min_angle

    if max_info["accepted"]:
        refined["max_angle"] = refined_max
    else:
        refined["max_angle"] = max_angle
        max_info["refined_angle"] = max_angle

    info["min"] = min_info
    info["max"] = max_info

    return refined, info



def process_meter(meter_img: np.ndarray, profile: Dict[str, Any], center_mode: str, meter_id: str = "M01_01", refine_scale_angles: bool = False, scale_refine_search_deg: float = 8.0, scale_refine_accept_deg: float = 6.0, center_offset_x: int = 0, center_offset_y: int = 0, min_angle_offset: float = 0.0, max_angle_offset: float = 0.0):
    center_info = resolve_center(meter_img, profile, center_mode=center_mode)
    center = center_info["center"]
    if center_offset_x != 0 or center_offset_y != 0:
        center = (int(center[0] + center_offset_x), int(center[1] + center_offset_y))
        center_info["center"] = center
        center_info["source"] = center_info["source"] + "_offset"
    center_radius = center_info["center_radius"]
    outer_radius = center_info["outer_radius"]

    refined_profile, scale_refine_info = make_refined_profile_for_image(
        meter_img=meter_img,
        profile=profile,
        center=center,
        outer_radius=outer_radius,
        enabled=refine_scale_angles,
        search_deg=scale_refine_search_deg,
        accept_deg=scale_refine_accept_deg,
    )

    # YAML/CLI fine_adjustを、この画像用の一時プロファイルに反映する。
    # 以後の針検出・値変換・MIN/MAX描画はすべて refined_profile を使う。
    base_min_angle = float(refined_profile.get("min_angle", refined_profile.get("angle", {}).get("min_angle_deg", 225.0)))
    base_max_angle = float(refined_profile.get("max_angle", refined_profile.get("angle", {}).get("max_angle_deg", -45.0)))
    refined_profile["min_angle"] = base_min_angle + float(min_angle_offset)
    refined_profile["max_angle"] = base_max_angle + float(max_angle_offset)

    tip, angle, debug = detect_needle(
        meter_img,
        center=center,
        center_radius=center_radius,
        outer_radius=outer_radius,
        profile=refined_profile,
        outer_valid=bool(center_info.get("outer_valid", False)),
    )

    value = angle_to_value(angle, refined_profile)
    unit = refined_profile.get("unit", refined_profile.get("scale", {}).get("unit", ""))

    annotated = meter_img.copy()

    # v9.12 debug: 外周円候補をすべて表示する。
    # 灰色: 候補 / 橙色太線: 採用候補
    outer_candidate_details = center_info.get("outer_candidate_details") or []
    for oc in outer_candidate_details:
        try:
            cxy = oc.get("center")
            rr = int(oc.get("radius") or 0)
            if not cxy or rr <= 0:
                continue
            pt = (int(cxy[0]), int(cxy[1]))
            selected = bool(oc.get("selected"))
            color = (0, 128, 255) if selected else (170, 170, 170)
            thickness = 3 if selected else 1
            cv2.circle(annotated, pt, rr, color, thickness)
            label = f"O{int(oc.get('index', 0))}"
            cv2.putText(annotated, label, (pt[0] + 8, max(18, pt[1] - rr - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2 if selected else 1, cv2.LINE_AA)
        except Exception:
            pass

    # 外周円・中心（候補一覧がない場合の保険表示）
    if not outer_candidate_details:
        cv2.circle(annotated, center_info["outer_center"], center_info["outer_radius"], (0, 128, 255), 2)

    # v9.3 debug: ハブ候補をすべて表示する。
    # 赤   : 外周円からYAMLハブを写像した推定ハブ
    # 緑   : HoughCirclesで検出した円ハブ候補
    # 水色 : 中心色マスク由来のハブ候補
    # 黄色 : 採用候補
    hub_candidate_details = center_info.get("hub_candidate_details") or []
    for cand in hub_candidate_details:
        try:
            cxy = cand.get("center")
            rr = int(cand.get("radius") or 0)
            if not cxy or rr <= 0:
                continue
            pt = (int(cxy[0]), int(cxy[1]))
            source = cand.get("source", "")
            if source == "profile_guided_estimate":
                color = (0, 0, 255)      # red
            elif source == "hough_hub":
                color = (0, 255, 0)      # green
            elif source == "center_color_mask":
                color = (255, 255, 0)    # cyan
            else:
                color = (180, 180, 180)
            thickness = 3 if cand.get("selected") else 1
            cv2.circle(annotated, pt, rr, color, thickness)
            cv2.drawMarker(annotated, pt, color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
            label = "P" if source == "profile_guided_estimate" else ("H" if source == "hough_hub" else ("C" if source == "center_color_mask" else "?"))
            cv2.putText(annotated, label, (pt[0] + 4, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        except Exception:
            pass

    # v8.1: 青丸は「ユーザーがYAMLで指定したハブ円」を優先して表示する。
    # 中心座標は読取画像で再検出した値を使ってよいが、表示半径まで
    # 再検出値にすると、反射・針根元・白領域を拾って青丸が大きくなるため。
    display_center_radius = int(center_info.get("display_hub_radius") or center_radius)
    display_center_radius = int(clamp(display_center_radius, 3, max(8, outer_radius * 0.18)))
    cv2.circle(annotated, center, display_center_radius, (255, 0, 0), 2)
    cv2.circle(annotated, center, 3, (255, 0, 0), -1)

    # hub候補: 検出位置の確認用。半径はYAML値を優先して描画する。
    if center_info.get("hub") is not None:
        hub_center, hub_r = center_info["hub"]
        hub_draw_r = int(center_info.get("display_hub_radius") or hub_r)
        hub_draw_r = int(clamp(hub_draw_r, 3, max(8, outer_radius * 0.18)))
        cv2.circle(annotated, hub_center, hub_draw_r, (255, 255, 0), 1)

    # ハブ候補評価に使った仮針中央線を参考表示（薄い水色）する。
    hub_line_dbg = center_info.get("hub_needle_line_debug", {})
    hub_line = hub_line_dbg.get("needle_line") if isinstance(hub_line_dbg, dict) else None
    if hub_line is not None and len(hub_line) == 4:
        x1, y1, x2, y2 = [int(v) for v in hub_line]
        cv2.line(annotated, (x1, y1), (x2, y2), (255, 220, 120), 1)

    # min/max角度
    min_angle = float(refined_profile.get("min_angle", refined_profile.get("angle", {}).get("min_angle_deg", 225.0)))
    max_angle = float(refined_profile.get("max_angle", refined_profile.get("angle", {}).get("max_angle_deg", -45.0)))
    draw_angle_line(annotated, center, min_angle, (0, 255, 0), "MIN", int(outer_radius * 0.75))
    draw_angle_line(annotated, center, max_angle, (0, 0, 255), "MAX", int(outer_radius * 0.75))

    # 針
    cv2.line(annotated, center, tip, (255, 0, 255), 2)
    cv2.circle(annotated, tip, 5, (0, 255, 255), -1)

    cv2.putText(
        annotated,
        f"{value:.3f} {unit}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        annotated,
        f"center={center_info['source']} angle={angle:.1f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # meter画面用に、認識IDをメーター近くへ紫で表示
    draw_meter_id(annotated, meter_id, (20, 110), scale=0.9)

    info = {
        "meter_id": meter_id,
        "center": center,
        "center_radius": center_radius,
        "display_hub_radius": center_info.get("display_hub_radius"),
        "raw_profile_hub_radius": center_info.get("raw_profile_hub_radius"),
        "scaled_profile_hub_radius": center_info.get("scaled_profile_hub_radius"),
        "outer_radius_scale": center_info.get("outer_radius_scale"),
        "profile_outer_radius": center_info.get("profile_outer_radius"),
        "detected_hub_radius": center_info.get("detected_hub_radius"),
        "profile_guided_hub_center": center_info.get("profile_guided_hub_center"),
        "hub_candidate_accepted": center_info.get("hub_candidate_accepted"),
        "hub_candidate_reason": center_info.get("hub_candidate_reason"),
        "hub_candidate_distance_to_profile": center_info.get("hub_candidate_distance_to_profile"),
        "hub_candidate_radius_ratio": center_info.get("hub_candidate_radius_ratio"),
        "hub_needle_line_debug": center_info.get("hub_needle_line_debug"),
        "hub_candidate_details": center_info.get("hub_candidate_details"),
        "profile_hub_debug": center_info.get("profile_hub_debug"),
        "center_source": center_info["source"],
        "outer_center": center_info["outer_center"],
        "outer_radius": outer_radius,
        "raw_outer_center": center_info.get("raw_outer_center"),
        "raw_outer_radius": center_info.get("raw_outer_radius"),
        "outer_scale_guard": center_info.get("outer_scale_guard"),
        "outer_valid": center_info.get("outer_valid"),
        "outer_candidate_count": center_info.get("outer_candidate_count"),
        "outer_candidate_details": center_info.get("outer_candidate_details"),
        "outer_meta": center_info.get("outer_meta"),
        "scale_refine": scale_refine_info,
        "applied_fine_adjust": {
            "center_offset_x": int(center_offset_x),
            "center_offset_y": int(center_offset_y),
            "min_angle_offset": float(min_angle_offset),
            "max_angle_offset": float(max_angle_offset),
            "effective_min_angle": float(refined_profile.get("min_angle")),
            "effective_max_angle": float(refined_profile.get("max_angle")),
        },
        "tip": tip,
        "angle": angle,
        "value": value,
        "unit": unit,
        "debug": debug,
        "annotated": annotated,
    }
    return annotated, info




def resize_image_to_max_size(image: np.ndarray, max_width: int, max_height: int) -> Tuple[np.ndarray, float]:
    """GUI表示だけを最大サイズ内へ縮小する。保存画像と読取計算には影響しない。"""
    if image is None or image.size == 0:
        return image, 1.0
    h, w = image.shape[:2]
    max_w = max(1, int(max_width))
    max_h = max(1, int(max_height))
    scale = min(float(max_w) / max(float(w), 1.0), float(max_h) / max(float(h), 1.0), 1.0)
    if scale >= 0.999999:
        return image, 1.0
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, float(scale)


def draw_center_adjust_overlay(img: np.ndarray, fine_adjust: Dict[str, Any]) -> np.ndarray:
    """中心補正GUIの操作ヘルプを画像上に重ねる。"""
    out = img.copy()
    help_lines = [
        f"CENTER ADJUST  x={int(fine_adjust.get('center_offset_x', 0)):+d}px  y={int(fine_adjust.get('center_offset_y', 0)):+d}px",
        "Keys: arrow/hjkl=1px, Shift-arrow/HJKL=5px, r=reset, u=save YAML, q/ESC=quit",
    ]
    x0, y0 = 12, max(24, out.shape[0] - 58)
    cv2.rectangle(out, (8, y0 - 22), (min(out.shape[1] - 8, 1040), y0 + 34), (0, 0, 0), -1)
    for i, line in enumerate(help_lines):
        cv2.putText(
            out,
            line,
            (x0, y0 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def center_adjust_delta_from_key(key: int) -> Optional[Tuple[int, int]]:
    """OpenCV waitKeyExのキーコードから中心補正量を返す。"""
    # Qt/Linuxでよく出る矢印キーコード、Windows系の拡張キーコードの両方を扱う。
    if key in (81, 2424832):      # Left
        return (-1, 0)
    if key in (83, 2555904):      # Right
        return (1, 0)
    if key in (82, 2490368):      # Up
        return (0, -1)
    if key in (84, 2621440):      # Down
        return (0, 1)
    if key in (ord('h'),):
        return (-1, 0)
    if key in (ord('l'),):
        return (1, 0)
    if key in (ord('k'),):
        return (0, -1)
    if key in (ord('j'),):
        return (0, 1)
    if key in (ord('H'),):
        return (-5, 0)
    if key in (ord('L'),):
        return (5, 0)
    if key in (ord('K'),):
        return (0, -5)
    if key in (ord('J'),):
        return (0, 5)
    # Shift + arrow on some platforms
    if key in (2162688,):         # Shift Left
        return (-5, 0)
    if key in (2228224,):         # Shift Right
        return (5, 0)
    if key in (2097152,):         # Shift Up
        return (0, -5)
    if key in (2359296,):         # Shift Down
        return (0, 5)
    return None



def make_polar_dial_view(meter_img: np.ndarray, center: Point, outer_radius: int) -> np.ndarray:
    """文字盤を極座標に展開したデバッグ画像。目盛り/針/中心ズレ確認用。"""
    max_radius = int(outer_radius * 0.92)
    if max_radius <= 0:
        return meter_img.copy()

    # 横=角度、縦=半径。OpenCVのwarpPolarで円を矩形化。
    dsize = (720, max_radius)
    polar = cv2.warpPolar(
        meter_img,
        dsize,
        center,
        max_radius,
        cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )

    # 見やすいよう半径方向を反転して、外周を上にする
    polar = cv2.flip(polar, 0)
    cv2.putText(
        polar,
        "polar dial view: x=angle, y=radius",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return polar


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Robust analog meter reader v8.11 center-adjust GUI")
    parser.add_argument("--model", default="yolo26x.pt")
    parser.add_argument("--image", type=Path, default=Path("InputImages/FixedCamera.jpg"))
    parser.add_argument("--output", type=Path, default=Path("OutputImages/result.jpg"))
    parser.add_argument("--profiles", type=Path, default=Path("meter_profiles.yaml"))
    parser.add_argument("--admin-config", type=Path, default=Path("meter_reader_admin.yaml"), help="管理者専用アルゴリズム設定YAML。通常ユーザーは編集しない。")
    parser.add_argument("--validate-admin-config", action="store_true", help="管理者設定を検証して終了する。")
    parser.add_argument("--profile", default="default_pressure_meter")
    parser.add_argument("--class-name", default="clock")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--crop-only", action="store_true")
    parser.add_argument("--crop-padding", type=int, default=10, help="YOLO検出bboxから上下左右に広げる余白[pixel]。ベゼル外周を含めたい場合に使用。")
    parser.add_argument("--debug", action="store_true", help="詳細ログを表示")
    parser.add_argument("--show-masks", action="store_true", help="needle_mask/profile_mask/dark_mask/edges/ray_score_mapを表示")
    parser.add_argument("--show-normalized", action="store_true", help="極座標展開した文字盤ビューを表示する。中心/MAXズレ確認用。")
    parser.add_argument("--no-gui", action="store_true", help="OpenCVウィンドウを表示しない")
    parser.add_argument("--center-mode", choices=["auto", "outer", "hub", "yaml"], default="auto")
    parser.add_argument("--refine-scale-angles", action="store_true", help="MIN/MAX角度を画像内の目盛り線へ自動補正する。検証用。通常は無効推奨。")
    parser.add_argument("--scale-refine-search-deg", type=float, default=8.0, help="MIN/MAX角度補正の探索範囲。広げすぎると針や別目盛りへ吸着するため注意。")
    parser.add_argument("--scale-refine-accept-deg", type=float, default=6.0, help="MIN/MAX角度補正を採用する最大角度差。これを超える補正は破棄。")
    parser.add_argument("--center-offset-x", type=int, default=None, help="検出中心の手動補正X[pixel]。右がプラス。未指定ならYAML fine_adjustを使用。")
    parser.add_argument("--center-offset-y", type=int, default=None, help="検出中心の手動補正Y[pixel]。下がプラス。未指定ならYAML fine_adjustを使用。")
    parser.add_argument("--min-angle-offset", type=float, default=None, help="MIN角度の手動補正[deg]。未指定ならYAML fine_adjustを使用。")
    parser.add_argument("--max-angle-offset", type=float, default=None, help="MAX角度の手動補正[deg]。未指定ならYAML fine_adjustを使用。")
    parser.add_argument("--save-fine-adjust", action="store_true", help="今回使った補正値をYAMLのfine_adjustへ保存する。")
    parser.add_argument("--meter-index", type=int, default=0, help="重複統合後、左優先・上優先で並べたメーターの読取番号。1始まり。0なら全件。")
    parser.add_argument("--max-meters", type=int, default=0, help="読取対象の最大件数。0なら全件。--meter-index指定時は通常使用しない。")
    parser.add_argument("--log-dir", type=Path, default=Path("Logs"), help="ログファイルの保存先ディレクトリ。")
    parser.add_argument("--no-log-file", action="store_true", help="ログファイルを作成せず、コンソール出力のみ行う。")
    args = parser.parse_args()

    global ADMIN_CONFIG, ADMIN_CONFIG_META
    ADMIN_CONFIG, ADMIN_CONFIG_META = load_admin_config(args.admin_config)
    meter_display_cfg = admin_value("display_defaults.meter_window", {}) or {}
    meter_force_initial_size = bool(meter_display_cfg.get("force_initial_size", True))
    meter_initial_width = int(meter_display_cfg.get("initial_width_px", 600))
    meter_initial_height = int(meter_display_cfg.get("initial_height_px", 600))
    result_display_cfg = admin_value("display_defaults.result_window", {}) or {}
    result_fit_to_max_size = bool(result_display_cfg.get("fit_to_max_size", True))
    result_max_width = int(result_display_cfg.get("max_width_px", 1280))
    result_max_height = int(result_display_cfg.get("max_height_px", 800))
    result_force_initial_size = bool(result_display_cfg.get("force_initial_size", True))
    result_initial_width = int(result_display_cfg.get("initial_width_px", 1200))
    result_initial_height = int(result_display_cfg.get("initial_height_px", 800))
    if args.validate_admin_config:
        print(f"admin config valid: {ADMIN_CONFIG_META['path']}")
        print(f"schema_version={ADMIN_CONFIG_META['schema_version']} sha256={ADMIN_CONFIG_META['sha256']}")
        return

    run_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = None
    log_file = None
    if not args.no_log_file:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        image_stem = args.image.stem if hasattr(args.image, "stem") else "image"
        safe_profile = str(args.profile).replace("/", "_").replace("\\", "_")
        log_path = args.log_dir / f"meter_read_{VERSION}_{safe_profile}_{image_stem}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        sys.stdout = TeeStream(sys.__stdout__, log_file)
        sys.stderr = TeeStream(sys.__stderr__, log_file)

    print(f"run_info: version={VERSION} started_at={run_started_at} image={args.image} profile={args.profile}")
    if log_path is not None:
        print(f"log_file: {log_path}")
    print("admin_config:")
    print(f"  file={ADMIN_CONFIG_META['path']}")
    print(f"  source={ADMIN_CONFIG_META['source']}")
    print(f"  schema_version={ADMIN_CONFIG_META['schema_version']}")
    print(f"  sha256={ADMIN_CONFIG_META['sha256']}")
    print(
        "display_defaults: "
        f"meter_window={{'force_initial_size': {meter_force_initial_size}, "
        f"'initial_width_px': {meter_initial_width}, 'initial_height_px': {meter_initial_height}}} "
        f"result_window={{'fit_to_max_size': {result_fit_to_max_size}, "
        f"'max_width_px': {result_max_width}, 'max_height_px': {result_max_height}, "
        f"'force_initial_size': {result_force_initial_size}, "
        f"'initial_width_px': {result_initial_width}, 'initial_height_px': {result_initial_height}}}"
    )
    print("admin_effective_parameters:")
    print(f"  hub_profile_guard={ADMIN_CONFIG['hub_selection']['profile_guard']}")
    print(f"  needle_tip_width={ADMIN_CONFIG['needle_segment']['tip_width']}")
    print(f"  centerline_refine={ADMIN_CONFIG['centerline_refine']}")
    print(f"  ray_line_disagreement_rescue={ADMIN_CONFIG['ray_line_disagreement_rescue']}")
    print(f"  broad_axis_rescue={ADMIN_CONFIG['broad_axis_rescue']}")
    print(f"  full_scale_axis_rescue={ADMIN_CONFIG['full_scale_axis_rescue']}")
    print(f"  yolo_duplicate_merge={ADMIN_CONFIG['yolo_duplicate_merge']}")

    profiles = load_profiles(args.profiles)
    if args.profile not in profiles:
        raise KeyError(f"指定プロファイルがありません: {args.profile}")
    profile = profiles[args.profile]
    fine_adjust = resolve_fine_adjust(args, profile)

    print("fine_adjust:")
    print(f"  center_offset_x={fine_adjust['center_offset_x']} px")
    print(f"  center_offset_y={fine_adjust['center_offset_y']} px")
    print(f"  min_angle_offset={fine_adjust['min_angle_offset']} deg")
    print(f"  max_angle_offset={fine_adjust['max_angle_offset']} deg")

    if args.save_fine_adjust:
        save_fine_adjust_to_profile(profiles, args.profile, fine_adjust, args.profiles)
        print(f"saved fine_adjust to YAML: {args.profiles} profile={args.profile}")

    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {args.image}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.crop_only:
        detections = [
            {
                "meter_id": "M01_01",
                "crop": img.copy(),
                "bbox": [0, 0, img.shape[1], img.shape[0]],
                "conf": 1.0,
                "center": (img.shape[1] / 2.0, img.shape[0] / 2.0),
            }
        ]
    else:
        detections = detect_meters(img, args.model, args.class_name, args.conf, args.device, args.crop_padding)

    if args.meter_index < 0:
        raise ValueError("--meter-index は0以上で指定してください。0は全件、1以上は指定番号のみを読み取ります。")
    if args.max_meters < 0:
        raise ValueError("--max-meters は0以上で指定してください。")

    def render_with_fine_adjust(current_fine_adjust: Dict[str, Any]):
        """現在のfine_adjustで全メーターを再処理し、表示画像とログ情報を返す。"""
        result_img = img.copy()
        meter_items: List[Dict[str, Any]] = []
        infos: List[Dict[str, Any]] = []
        processed: List[Dict[str, Any]] = []

        for source_index, d in enumerate(detections):
            meter_id = d["meter_id"]
            bbox = d["bbox"]
            conf = d["conf"]
            meter_img = d["crop"]

            annotated_meter, info = process_meter(
                meter_img,
                profile,
                center_mode=args.center_mode,
                meter_id=meter_id,
                refine_scale_angles=args.refine_scale_angles,
                scale_refine_search_deg=args.scale_refine_search_deg,
                scale_refine_accept_deg=args.scale_refine_accept_deg,
                center_offset_x=current_fine_adjust["center_offset_x"],
                center_offset_y=current_fine_adjust["center_offset_y"],
                min_angle_offset=current_fine_adjust["min_angle_offset"],
                max_angle_offset=current_fine_adjust["max_angle_offset"],
            )
            info["bbox"] = bbox
            info["confidence"] = conf
            processed.append({
                "source_index": source_index,
                "source_detection": d,
                "meter_id": meter_id,
                "annotated_meter": annotated_meter,
                "info": info,
            })

        kept_processed, duplicate_decisions = consolidate_duplicate_meter_results(detections, processed, img.shape)

        # 重複統合後の最終候補に対して読取番号を適用する。
        # 番号は左を第一優先、上を第二優先とした1始まり。
        available_count = len(kept_processed)
        selected_meter_index = int(args.meter_index or 0)
        if selected_meter_index > 0:
            if selected_meter_index > available_count:
                raise IndexError(
                    f"--meter-index={selected_meter_index} は範囲外です。"
                    f" 重複統合後の検出数は {available_count} 件です。"
                )
            kept_processed = [kept_processed[selected_meter_index - 1]]
        elif args.max_meters and args.max_meters > 0:
            kept_processed = kept_processed[: int(args.max_meters)]

        # 出力対象だけで番号を振り直す。元の順位も診断用に保存する。
        output_total = len(kept_processed)
        for output_i, item in enumerate(kept_processed, start=1):
            original_rank = int(item.get("selection_rank", 0) or 0)
            if original_rank <= 0:
                # consolidate_duplicate_meter_resultsで付与されたIDから元順位を取得。
                try:
                    original_rank = int(str(item.get("meter_id", "M00_00"))[1:3])
                except Exception:
                    original_rank = output_i
            item["selection_rank"] = original_rank
            item["meter_id"] = f"M{output_i:02d}_{output_total:02d}"
            item["info"]["meter_id"] = item["meter_id"]
            item["info"]["selection_rank"] = original_rank

        print("meter_selection:")
        print("  ordering=left_first_then_top")
        print(f"  available_after_duplicate_merge={available_count}")
        print(f"  requested_meter_index={selected_meter_index}")
        print(f"  max_meters={int(args.max_meters or 0)}")
        print(f"  selected_count={len(kept_processed)}")
        print(f"  selected_original_ranks={[item.get('selection_rank') for item in kept_processed]}")

        for item in kept_processed:
            meter_id = item["meter_id"]
            annotated_meter = item["annotated_meter"]
            info = item["info"]
            bbox = info["bbox"]
            infos.append(info)
            if args.crop_only:
                result_img = annotated_meter.copy()
            else:
                x1, y1, x2, y2 = bbox
                result_img[y1:y2, x1:x2] = annotated_meter
                label_x = max(0, x1)
                label_y = max(25, y1 - 8)
                draw_meter_id(result_img, meter_id, (label_x, label_y), scale=0.9)
                cx_crop, cy_crop = info["center"]
                cv2.circle(result_img, (int(x1 + cx_crop), int(y1 + cy_crop)), 4, (255, 0, 255), -1)
            meter_items.append({"meter_id": meter_id, "annotated_meter": annotated_meter, "info": info})

        if duplicate_decisions:
            print("yolo_duplicate_merge:")
            for dec in duplicate_decisions:
                print(f"  group members={dec['members']} chosen={dec['chosen']} suppressed={dec['suppressed']} ambiguous={dec['ambiguous']} quality_margin={dec['quality_margin']:.2f}")
                for mid, sc in dec["scores"].items():
                    print(f"    {mid}: score={sc['score']:.2f} confidence={sc['confidence']:.2f} dial_edge={sc['dial_edge']:.2f} pair_score={sc['pair_score']:.2f} radius_ratio_penalty={sc['radius_ratio_penalty']:.2f} edge_cut_penalty={sc['image_edge_cut_penalty']:.2f} oversized_penalty={sc['oversized_crop_penalty']:.2f} dial_fill_ratio={sc['dial_fill_ratio']:.3f}")

        meter_view = make_meter_contact_sheet(meter_items)
        cv2.imwrite(str(args.output), result_img)
        return result_img, meter_view, meter_items, infos

    result_img, meter_view, meter_items, infos = render_with_fine_adjust(fine_adjust)

    print(f"detected_meters={len(infos)}")
    for info in infos:
        print(f"[{info['meter_id']}]")
        print(f"  confidence={info['confidence']:.2f}")
        print(f"  center={info['center']} source={info['center_source']}")
        print(f"  center_radius_for_needle={info['center_radius']} display_hub_radius={info.get('display_hub_radius')} detected_hub_radius={info.get('detected_hub_radius')}")
        print(f"  profile_guided_hub_center={info.get('profile_guided_hub_center')} hub_candidate_accepted={info.get('hub_candidate_accepted')} reason={info.get('hub_candidate_reason')}")
        phdbg = info.get('profile_hub_debug') or {}
        print(f"  profile_hub_debug raw_center={phdbg.get('raw_center')} mapped_center={phdbg.get('mapped_center')} raw_radius={phdbg.get('raw_radius')} scaled_radius={phdbg.get('scaled_radius')} source={phdbg.get('source')}")
        hldbg = info.get('hub_needle_line_debug') or {}
        print(f"  hub_needle_line available={hldbg.get('needle_line_available')} distance={hldbg.get('needle_line_distance')} allow={hldbg.get('needle_line_allow')} candidates={hldbg.get('needle_line_count')} selected={hldbg.get('selected_source')}")
        print(f"  hub_selection reason={hldbg.get('hub_selection_reason')} profile_role={hldbg.get('hub_profile_candidate_role')} assistant_role={hldbg.get('assistant_hub_selection_role')} credible_real={hldbg.get('credible_real_candidate_count')}")
        hub_cands = info.get('hub_candidate_details') or []
        print(f"  hub_debug_candidates={len(hub_cands)}  P=profile/red H=hough/green C=center_color/cyan selected=yellow-thick")
        for idx, hc in enumerate(hub_cands[:12], start=1):
            print(f"    hub_cand[{idx}] source={hc.get('source')} center={hc.get('center')} r={hc.get('radius')} score={hc.get('score')} selected={hc.get('selected')} rr={hc.get('radius_ratio')} distP={hc.get('profile_distance')} colorP={hc.get('hub_color_penalty')} edge={hc.get('circle_edge_support')} selP={hc.get('selection_model_penalty')} note={hc.get('note')}")
        if info.get('hub_candidate_distance_to_profile') is not None:
            print(f"  hub_candidate_distance_to_profile={info.get('hub_candidate_distance_to_profile'):.1f} radius_ratio={info.get('hub_candidate_radius_ratio') if info.get('hub_candidate_radius_ratio') is not None else 'None'}")
        print(f"  hub_radius_scale={info.get('outer_radius_scale'):.3f} profile_outer_radius={info.get('profile_outer_radius')} raw_profile_hub_radius={info.get('raw_profile_hub_radius')} scaled_profile_hub_radius={info.get('scaled_profile_hub_radius')}")
        print(f"  outer_center={info['outer_center']} outer_radius={info['outer_radius']} outer_valid={info.get('outer_valid')} outer_candidates={info.get('outer_candidate_count')}")
        om = info.get("outer_meta") or {}
        if om.get("valid"):
            print(f"  outer_score={om.get('score', 0):.2f} edge_support={om.get('edge_support', 0):.3f} valid_fraction={om.get('valid_fraction', 0):.3f} range={om.get('range_label')}")
            print(f"  outer_search_scale={om.get('outer_search_scale', 1.0):.3f} search_size={om.get('outer_search_size')} elapsed_ms={om.get('outer_search_elapsed_ms')} early_exit={om.get('outer_search_early_exit')}")
            if om.get('forced_bbox_scaled_outer_fallback'):
                print(f"  outer_forced_bbox_scaled_fallback=True reason={om.get('forced_bbox_scaled_outer_reason')} crop_scale_info={om.get('crop_scale_info')}")
            pair_top3 = om.get('outer_pair_top3_by_role') or info.get('outer_pair_top3_by_role') or {}
            pair_selected = om.get('outer_pair_selected_by_ratio_model') or info.get('outer_pair_selected_by_ratio_model')
            if om.get('outer_pair_selected_by_ratio_model'):
                print(f"  outer_pair_selection selected_by_ratio_model=True preferred={om.get('outer_pair_preferred')} ratio={om.get('outer_pair_ratio')} expected={om.get('outer_pair_expected_ratio')} center_dist={om.get('outer_pair_center_distance_px')} offset_err={om.get('outer_pair_center_offset_error_px')} pair_score={om.get('outer_pair_score')} model={om.get('outer_pair_selection_model')}")
            if pair_top3:
                print(f"  outer_pair_top3 dial={pair_top3.get('dial_face_circle')} bezel={pair_top3.get('bezel_case_circle')}")
            outer_details = info.get('outer_candidate_details') or []
            if outer_details:
                print(f"  outer_debug_candidates={len(outer_details)}  selected=orange-thick others=gray")
                for oc in outer_details[:12]:
                    ratio = oc.get('radius_ratio_to_profile')
                    ratio_s = f"{float(ratio):.3f}" if ratio is not None else "None"
                    print(f"    outer_cand[{oc.get('index')}] center={oc.get('center')} r={oc.get('radius')} score={float(oc.get('score', 0.0)):.2f} edge={float(oc.get('edge_support', 0.0)):.3f} ratio={ratio_s} class={oc.get('outer_pair_class')} pref={oc.get('outer_pair_preferred')} pair_bonus={float(oc.get('outer_pair_class_bonus', 0.0)):.1f} dial_white={float(oc.get('dial_white_score', 0.0)):.3f} color_score={float(oc.get('outer_color_score', 0.0)):.3f} selected={oc.get('selected')} range={oc.get('range_label')}")
        osg = info.get("outer_scale_guard") or {}
        if osg:
            print(f"  outer_scale_guard applied={osg.get('applied')} raw_outer_radius={osg.get('raw_outer_radius')} corrected_outer_radius={osg.get('corrected_outer_radius')} hub_based_outer_radius={osg.get('hub_based_outer_radius')} ratio={osg.get('outer_to_hub_scale_ratio')} reason={osg.get('reason')}")
        if info.get("scale_refine", {}).get("enabled"):
            sr = info["scale_refine"]
            print(f"  min_angle: {sr['min']['approx_angle']:.2f} -> {sr['min']['refined_angle']:.2f} accepted={sr['min'].get('accepted')}")
            print(f"  max_angle: {sr['max']['approx_angle']:.2f} -> {sr['max']['refined_angle']:.2f} accepted={sr['max'].get('accepted')}")
        fa = info.get("applied_fine_adjust", {})
        print(f"  effective_min_angle={fa.get('effective_min_angle'):.2f}")
        print(f"  effective_max_angle={fa.get('effective_max_angle'):.2f}")
        print(f"  applied_max_angle_offset={fa.get('max_angle_offset'):.2f} deg")
        print(f"  tip={info['tip']}")
        print(f"  angle={info['angle']:.2f} deg")
        print(f"  value={info['value']:.3f} {info['unit']}")
    print(f"saved: {args.output}")

    # 以降のdebug表示は先頭メーターを代表として表示
    info = infos[0]
    annotated_meter = meter_items[0]["annotated_meter"]
    if args.debug:
        dbg = info["debug"]
        best = dbg["best"]
        thin = dbg.get("thin_debug", {})
        print("best_candidate:")
        print(f"  method={best.get('method')}")
        print(f"  score={best.get('score'):.2f}")
        print(f"  final_score={best.get('final_score'):.2f}")
        print(f"  length={best.get('length'):.2f}")
        print(f"  raw_angle={dbg.get('raw_angle'):.2f}")
        if dbg.get("line_best") is not None:
            lb = dbg["line_best"]
            print("line_segment_candidate:")
            print(f"  angle={lb.get('angle'):.2f} max_d={lb.get('max_d'):.1f} length={lb.get('length'):.1f} tip_width={lb.get('tip_width'):.1f} score={lb.get('score'):.2f}")
            if lb.get('centerline_used'):
                print(f"  centerline_used=True raw_angle={lb.get('raw_angle', 0):.2f} centerline_samples={lb.get('centerline_sample_count')} base/mid/tip_width={lb.get('base_width',0):.1f}/{lb.get('mid_width',0):.1f}/{lb.get('tip_width',0):.1f}")
        print("tip_direction_check:")
        pa = thin.get("profile_a", {})
        pb = thin.get("profile_b", {})
        print(f"  angle_a={thin.get('angle_a'):.2f} end_r={pa.get('end_r', 0):.1f} tip_width={pa.get('tip_width', 0):.1f} mid_width={pa.get('mid_width', 0):.1f} base_width={pa.get('base_width', 0):.1f} taper={pa.get('taper_score', 0):.1f} seq={pa.get('taper_sequence_score', 0):.1f} thin_penalty={pa.get('thin_tip_penalty', 0):.1f} score={pa.get('score', 0):.2f} dir_score={pa.get('direction_score', 0):.2f}")
        print(f"  angle_b={thin.get('angle_b'):.2f} end_r={pb.get('end_r', 0):.1f} tip_width={pb.get('tip_width', 0):.1f} mid_width={pb.get('mid_width', 0):.1f} base_width={pb.get('base_width', 0):.1f} taper={pb.get('taper_score', 0):.1f} seq={pb.get('taper_sequence_score', 0):.1f} thin_penalty={pb.get('thin_tip_penalty', 0):.1f} score={pb.get('score', 0):.2f} dir_score={pb.get('direction_score', 0):.2f}")
        print(f"  line_segment_reject_reason={thin.get('line_segment_reject_reason')}")
        ldd = thin.get("line_direction_debug", {})
        if ldd:
            print(f"  line_direction: angle={ldd.get('line_angle'):.2f} in_scale={ldd.get('line_in_scale')} ratio={ldd.get('line_ratio'):.3f} opposite={ldd.get('opposite_angle'):.2f} opp_in_scale={ldd.get('opposite_in_scale')} opp_ratio={ldd.get('opposite_ratio'):.3f}")
            dsg = (thin.get("priority_tip_selector") or {}).get("dual_scale_direction_guard") or {}
            if dsg:
                print(f"  dual_scale_direction_guard: enabled={dsg.get('enabled')} attempted={dsg.get('attempted')} accepted={dsg.get('accepted')} preferred={dsg.get('preferred_assignment')} margin={dsg.get('orientation_margin')} advantage={dsg.get('length_match_advantage')}")
                print(f"    expected_ratio={dsg.get('expected_long_short_ratio')} observed_tip/tail={dsg.get('observed_tip_length')}/{dsg.get('observed_tail_length')} observed_ratio={dsg.get('observed_long_short_ratio')} checks={dsg.get('checks')}")
                print(f"    absolute_consistency: tip_ratio={dsg.get('tip_length_ratio_to_expected')} tail_ratio={dsg.get('tail_length_ratio_to_expected')} ratio_error={dsg.get('observed_ratio_relative_error')} tip/mid_width={dsg.get('observed_tip_width')}/{dsg.get('observed_mid_width')} width_ratio={dsg.get('tip_mid_width_ratio')}")
                wr = dsg.get('low_resolution_widening_reverse_rescue') or {}
                if wr:
                    print(f"    widening_reverse_rescue: attempted={wr.get('attempted')} accepted={wr.get('accepted')} base_width_ratio={wr.get('base_tip_mid_width_ratio')} opposite_width_ratio={wr.get('opposite_tip_mid_width_ratio')} checks={wr.get('checks')}")
            print(f"  needle_shape_template: available={ldd.get('needle_shape_available')} tip_len={ldd.get('shape_tip_length_px_scaled')} tail_len={ldd.get('shape_tail_length_px_scaled')} length_match={ldd.get('shape_length_match')} shape_score={ldd.get('shape_length_score')}")
        post = thin.get("ray_line_disagreement_rescue_post_selection") or {}
        if post:
            print(
                "  ray_line_disagreement_rescue_post_selection: "
                f"attempted={post.get('attempted', False)} "
                f"accepted={post.get('accepted', False)} "
                f"line_angle={post.get('line_angle')} "
                f"final_ray_angle={post.get('final_ray_angle')} "
                f"angle_difference={post.get('angle_difference')} "
                f"tip_width_excess_ratio={post.get('tip_width_excess_ratio')}"
            )
            print(f"    checks={post.get('checks', {})} reject_reason={post.get('reject_reason')}")
        width_rescue = thin.get("line_width_rejection_rescue") or {}
        if width_rescue:
            print(
                "  line_width_rejection_rescue: "
                f"attempted={width_rescue.get('attempted', False)} "
                f"accepted={width_rescue.get('accepted', False)} "
                f"line_angle={width_rescue.get('line_angle')} "
                f"ray_angle={width_rescue.get('ray_angle')} "
                f"angle_difference={width_rescue.get('angle_difference')} "
                f"width_excess_ratio={width_rescue.get('tip_width_excess_ratio')}"
            )
            print(f"    checks={width_rescue.get('checks', {})} reject_reason={width_rescue.get('reject_reason')}")
        ncr = thin.get("needle_centerline_refine") or {}
        if ncr:
            print(f"  needle_centerline_refine: available={ncr.get('available')} base={ncr.get('base_angle')} refined={ncr.get('refined_angle')} delta={ncr.get('delta_deg')} rejected={ncr.get('rejected', False)}")
            lrg = ncr.get("low_resolution_refine_guard") or {}
            if lrg:
                print(f"    low_resolution_refine_guard: attempted={lrg.get('attempted')} rejected={lrg.get('rejected')} base={lrg.get('base_angle')} candidate={lrg.get('candidate_angle')} delta={lrg.get('refine_delta_deg')} coverage={lrg.get('coverage')} score_margin={lrg.get('top_score_margin')}")
                print(f"    low_resolution_refine_guard_checks: {lrg.get('checks')}")
            if ncr.get("available"):
                reasons = ncr.get("limit_reasons") or []
                diag = ncr.get("diagnostics") or {}
                print(
                    "    refine_guard: "
                    f"limited={ncr.get('limited', False)} "
                    f"raw_refined={ncr.get('raw_refined_angle')} "
                    f"raw_delta={ncr.get('raw_delta_deg')} "
                    f"applied_delta={ncr.get('delta_deg')} "
                    f"limit_deg={ncr.get('limit_deg')}"
                )
                print(
                    "    refine_guard_conditions: "
                    f"outer_valid={diag.get('outer_valid')} "
                    f"outer_radius={diag.get('outer_radius')} "
                    f"small_outer_threshold={diag.get('small_outer_radius_threshold')} "
                    f"coverage={diag.get('coverage')} "
                    f"low_coverage_threshold={diag.get('low_coverage_threshold')} "
                    f"search_deg={diag.get('search_deg')} "
                    f"search_edge_hit={diag.get('search_edge_hit')}"
                )
                print(f"    refine_guard_reasons={reasons if reasons else ['none']}")
            broad = ncr.get('broad_axis_rescue') or {}
            if broad:
                print(
                    "    broad_axis_rescue: "
                    f"attempted={broad.get('attempted', False)} "
                    f"accepted={broad.get('accepted', False)} "
                    f"broad_angle={broad.get('broad_angle')} "
                    f"broad_delta={broad.get('broad_delta')} "
                    f"local_score={broad.get('local_score')} "
                    f"broad_score={broad.get('broad_score')}"
                )
                print(
                    "    broad_axis_rescue_checks: "
                    f"distinct_axis={broad.get('distinct_axis')} "
                    f"score_clear={broad.get('score_clear')} "
                    f"geometry_clear={broad.get('geometry_clear')} "
                    f"axis_supports_broad={broad.get('axis_supports_broad')} "
                    f"lowres_override_trigger={broad.get('low_resolution_widening_override_trigger')} "
                    f"lowres_override_accept={broad.get('low_resolution_widening_override_acceptance')} "
                    f"top_spread={broad.get('top_angle_spread_deg')} "
                    f"reasons={broad.get('trigger_reasons', [])}"
                )
            full = ncr.get('full_scale_axis_rescue') or {}
            if full:
                print(
                    "    full_scale_axis_rescue: "
                    f"enabled={full.get('enabled', False)} "
                    f"trigger={full.get('trigger', False)} "
                    f"attempted={full.get('attempted', False)} "
                    f"accepted={full.get('accepted', False)} "
                    f"candidate_count={full.get('candidate_count', 0)}"
                )
                checks = full.get('trigger_checks') or {}
                values = full.get('trigger_values') or {}
                normal_checks = checks.get('normal') or {}
                geometry_checks = checks.get('geometry_fallback') or {}
                print(
                    "    full_scale_trigger_checks: "
                    f"normal_trigger={checks.get('normal_trigger')} "
                    f"geometry_trigger={checks.get('geometry_trigger')} "
                    f"shape_score={checks.get('shape_score_clear')} "
                    f"tip_widening={checks.get('tip_widening_clear')} "
                    f"search_edge={checks.get('local_search_edge_clear')} "
                    f"broad_not_applied={checks.get('broad_rescue_not_applied')}"
                )
                print(
                    "    full_scale_normal_trigger_checks: "
                    f"outer_invalid={normal_checks.get('outer_invalid_clear')} "
                    f"outer_radius={normal_checks.get('outer_radius_clear')} "
                    f"coverage={normal_checks.get('coverage_clear')}"
                )
                print(
                    "    full_scale_geometry_trigger_checks: "
                    f"enabled={geometry_checks.get('geometry_enabled')} "
                    f"outer_invalid={geometry_checks.get('outer_invalid_clear')} "
                    f"outer_radius={geometry_checks.get('outer_radius_clear')} "
                    f"coverage={geometry_checks.get('coverage_clear')} "
                    f"centerline_offset={geometry_checks.get('centerline_offset_clear')} "
                    f"tip_mid_width_ratio={geometry_checks.get('tip_mid_width_ratio_clear')}"
                )
                print(
                    "    full_scale_trigger_values: "
                    f"outer_valid={values.get('outer_valid')} "
                    f"outer_radius={values.get('outer_radius')}/{values.get('outer_radius_max_px')} "
                    f"coverage={values.get('coverage')}/{values.get('coverage_max')} "
                    f"shape_score={values.get('shape_score')}/{values.get('shape_score_max')} "
                    f"tip_width={values.get('tip_width')} mid_width={values.get('mid_width')} "
                    f"suspicious_tip_widening={values.get('suspicious_tip_widening')} "
                    f"search_edge_hit={values.get('search_edge_hit')} "
                    f"normal_trigger={values.get('normal_trigger')} "
                    f"geometry_trigger={values.get('geometry_trigger')} "
                    f"centerline_offset={values.get('centerline_offset')}/{values.get('geometry_minimum_centerline_offset_px')} "
                    f"tip_mid_width_ratio={values.get('tip_mid_width_ratio')}/{values.get('geometry_minimum_tip_mid_width_ratio')} "
                    f"geometry_coverage_max={values.get('geometry_coverage_max')}"
                )
                if full.get('attempted'):
                    print(
                        "    full_scale_result: "
                        f"original_angle={full.get('original_angle')} "
                        f"candidate_angle={full.get('candidate_angle')} "
                        f"delta={full.get('delta_deg')} "
                        f"original_score={full.get('original_score')} "
                        f"candidate_score={full.get('candidate_score')} "
                        f"score_ratio={full.get('score_ratio')} "
                        f"original_coverage={full.get('original_coverage')} "
                        f"candidate_coverage={full.get('candidate_coverage')} "
                        f"candidate_offset={full.get('candidate_offset')}"
                    )
                    print(
                        "    full_scale_acceptance_checks: "
                        f"score={full.get('score_clear')} "
                        f"coverage={full.get('coverage_clear')} "
                        f"offset={full.get('offset_clear')} "
                        f"offset_available={full.get('offset_available')} "
                        f"axis_support={full.get('axis_support_clear')} "
                        f"scale_range={full.get('scale_range_clear')} "
                        f"lowres_fallback={full.get('low_resolution_fallback_clear')} "
                        f"normal_acceptance={full.get('normal_acceptance_clear')} "
                        f"geometry_acceptance={full.get('geometry_acceptance_clear')} "
                        f"top_angle_spread={full.get('top_candidate_angle_spread_deg')}"
                    )
                    geom_accept_checks = full.get('geometry_acceptance_checks') or {}
                    print(
                        "    full_scale_geometry_acceptance_checks: "
                        f"enabled={geom_accept_checks.get('enabled')} "
                        f"geometry_trigger={geom_accept_checks.get('triggered_by_geometry')} "
                        f"score={geom_accept_checks.get('score_clear')} "
                        f"coverage={geom_accept_checks.get('coverage_clear')} "
                        f"cluster={geom_accept_checks.get('candidate_cluster_clear')} "
                        f"scale_ratio={geom_accept_checks.get('scale_ratio_clear')} "
                        f"outer_geometry={geom_accept_checks.get('outer_geometry_clear')} "
                        f"offset={geom_accept_checks.get('offset_clear')} "
                        f"axis_support={geom_accept_checks.get('axis_support_clear')}"
                    )
                    thresholds = full.get('acceptance_thresholds') or {}
                    print(f"    full_scale_acceptance_thresholds={thresholds}")
                    for i, c in enumerate((full.get('top_candidates') or [])[:5], start=1):
                        print(
                            f"    full_scale_cand[{i}] "
                            f"angle={c.get('angle'):.2f} score={c.get('score'):.2f} "
                            f"coverage={c.get('coverage'):.2f} offset={c.get('offset'):.2f} "
                            f"axis_support={c.get('axis_support')} in_scale={c.get('in_scale')} "
                            f"scale_ratio={c.get('scale_ratio'):.3f}"
                        )
            top = ncr.get('top_candidates') or []
            for i, c in enumerate(top[:3], start=1):
                print(f"    centerline_refine_cand[{i}] angle={c.get('angle'):.2f} score={c.get('score'):.2f} coverage={c.get('coverage'):.2f} offset={c.get('mean_abs_center_offset_px'):.2f} far_r={c.get('far_r'):.1f}")
        print(f"  chosen_reason={thin.get('chosen_reason')} corrected_angle={thin.get('corrected_angle'):.2f}")

    if not args.no_gui:
        print("\n中心補正GUIを有効にしました。")
        print("  arrow/hjkl: 1px移動 / Shift-arrowまたはHJKL: 5px移動")
        print("  u: 現在のcenter_offsetをYAMLへ保存 / r: center_offsetを0へ戻す / q or ESC: 終了")

        cv2.namedWindow("meter", cv2.WINDOW_NORMAL)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        result_display_scale_logged = False
        window_size_applied = False
        while True:
            meter_disp = draw_center_adjust_overlay(meter_view, fine_adjust)
            result_disp = draw_center_adjust_overlay(result_img, fine_adjust)
            result_display_scale = 1.0
            if result_fit_to_max_size:
                result_disp, result_display_scale = resize_image_to_max_size(
                    result_disp, result_max_width, result_max_height
                )
            if not result_display_scale_logged:
                print(
                    f"result_window: fit_to_max_size={result_fit_to_max_size} "
                    f"max_size=({result_max_width}x{result_max_height}) "
                    f"display_size=({result_disp.shape[1]}x{result_disp.shape[0]}) "
                    f"scale={result_display_scale:.4f}"
                )
                result_display_scale_logged = True
            cv2.imshow("meter", meter_disp)
            cv2.imshow("result", result_disp)
            # WSL/VcXsrv/OpenCV(Qt)では WINDOW_NORMAL の初期サイズが小さくなることがあるため、
            # 最初のimshow後に明示的なウィンドウサイズを適用する。
            if not window_size_applied:
                cv2.waitKey(1)
                if meter_force_initial_size:
                    cv2.resizeWindow("meter", meter_initial_width, meter_initial_height)
                if result_force_initial_size:
                    cv2.resizeWindow("result", result_initial_width, result_initial_height)
                print(
                    f"opencv_window_size: meter=({meter_initial_width}x{meter_initial_height}) "
                    f"result=({result_initial_width}x{result_initial_height})"
                )
                window_size_applied = True

            if args.show_normalized:
                cv2.imshow("polar_dial_view", make_polar_dial_view(annotated_meter, info["center"], info["outer_radius"]))

            if args.show_masks:
                dbg = info["debug"]
                cv2.imshow("needle_mask", dbg["mask"])
                cv2.imshow("profile_mask", dbg["profile_mask"])
                cv2.imshow("dark_mask", dbg["dark_mask"])
                cv2.imshow("edges", dbg["edges"])
                first_meter_img = detections[0]["crop"]
                score_map = make_ray_score_map(first_meter_img.shape, dbg["candidates"], info["center"], info["outer_radius"])
                cv2.imshow("ray_score_map", score_map)

            key = cv2.waitKeyEx(0)
            if key in (27, ord('q'), ord('Q')):
                break

            delta = center_adjust_delta_from_key(key)
            if delta is not None:
                dx, dy = delta
                fine_adjust["center_offset_x"] = int(fine_adjust["center_offset_x"] + dx)
                fine_adjust["center_offset_y"] = int(fine_adjust["center_offset_y"] + dy)
                print(
                    f"center_adjust: x={fine_adjust['center_offset_x']:+d}px "
                    f"y={fine_adjust['center_offset_y']:+d}px"
                )
                result_img, meter_view, meter_items, infos = render_with_fine_adjust(fine_adjust)
                info = infos[0]
                annotated_meter = meter_items[0]["annotated_meter"]
                continue

            if key in (ord('r'), ord('R')):
                fine_adjust["center_offset_x"] = 0
                fine_adjust["center_offset_y"] = 0
                print("center_adjust reset: x=+0px y=+0px")
                result_img, meter_view, meter_items, infos = render_with_fine_adjust(fine_adjust)
                info = infos[0]
                annotated_meter = meter_items[0]["annotated_meter"]
                continue

            if key in (ord('u'), ord('U')):
                save_fine_adjust_to_profile(profiles, args.profile, fine_adjust, args.profiles)
                print(
                    f"saved center fine_adjust to YAML: {args.profiles} profile={args.profile} "
                    f"x={fine_adjust['center_offset_x']:+d}px y={fine_adjust['center_offset_y']:+d}px"
                )
                continue

            print(f"ignored key={key}")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
