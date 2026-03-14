# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot multi-backend benchmark results.

Reads the directory tree produced by ``run_multi_backend_benchmarks.sh`` and
generates publication-quality figures:

  1.  Effective-FPS scaling curves  (per task, one line per config)
  2.  GPU-memory scaling curves     (per task, one line per config)
  3.  CPU-memory (RSS) scaling      (per task)
  4.  Render-type comparison        (bar chart, Shadow Benchmark env)
  5.  Renderer comparison           (RTX vs Newton renderer)
  6.  Physics-backend comparison    (PhysX vs Newton)
  7.  Resolution impact             (64 / 128 / 256 bar chart)
  8.  Step-time breakdown           (physics + render frametimes, if available)

Usage::

    python scripts/benchmarks/plot_multi_backend_benchmarks.py \\
        --results_dir benchmark_results/multi_backend \\
        --output_dir benchmark_results/multi_backend/plots
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless rendering

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "0.8",
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

# Colour palette for bar charts (sequential, colour-blind safe)
_PALETTE = [
    "#0077BB",  # blue
    "#EE7733",  # orange
    "#009988",  # teal
    "#CC3311",  # red
    "#33BBEE",  # cyan
    "#EE3377",  # magenta
    "#BBBBBB",  # grey
    "#AA3377",  # purple
]

# Marker cycle (for bar charts)
_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

# ── Deterministic visual encoding for scaling plots ───────────────────────────
# Every line gets a unique color; no two lines share the same color.
#   Hue    = render type × renderer  (6 distinct hue families)
#   Shade  = resolution              (light=64, medium=128, dark=256)
#   Marker = physics backend         (○=PhysX, □=Newton physics)
#   Line   = physics backend         (solid=PhysX, dashed=Newton — double-encoded)

_RENDER_RENDERER_SHADES: dict[tuple[str, str], list[str]] = {
    # key: (render_type, renderer)    64px (light) → 128px (mid) → 256px (dark)
    ("rgb",   "isaacsim_rtx_renderer"):                             ["#AED6F1", "#2980B9", "#1A5276"],  # blues
    ("rgb",   "newton_renderer"):                                   ["#D2B4DE", "#8E44AD", "#512E5F"],  # purples
    ("depth", "isaacsim_rtx_renderer"):                             ["#FAD7A0", "#E67E22", "#784212"],  # oranges
    ("depth", "newton_renderer"):                                   ["#F1948A", "#C0392B", "#641E16"],  # reds
    ("simple_shading_constant_diffuse", "isaacsim_rtx_renderer"):   ["#A9DFBF", "#27AE60", "#1E8449"],  # greens
    ("simple_shading_constant_diffuse", "newton_renderer"):         ["#A2D9CE", "#17A589", "#0E6655"],  # teals
}
_PHYSICS_COLORS_NO_CAM: dict[str, str] = {"physx": "#2874A6", "newton": "#B7950B"}
_RESOLUTION_IDX: dict[str, int] = {"64": 0, "128": 1, "256": 2}

_PHYSICS_MARKER: dict[str, str] = {"physx": "o", "newton": "s"}
_PHYSICS_LINESTYLE: dict[str, str] = {"physx": "-", "newton": "--"}


def _line_style(physics: str, renderer: str, render_type: str, resolution: str) -> dict:
    """Return plot() kwargs for a scaling line based on its config."""
    if render_type in ("none", ""):
        # No-camera task: colour by physics, no renderer distinction needed
        color = _PHYSICS_COLORS_NO_CAM.get(physics, "#888888")
    else:
        shades = _RENDER_RENDERER_SHADES.get(
            (render_type, renderer), ["#AAAAAA", "#777777", "#333333"]
        )
        color = shades[_RESOLUTION_IDX.get(resolution, 1)]
    return {
        "color":     color,
        "marker":    _PHYSICS_MARKER.get(physics, "D"),
        "linestyle": _PHYSICS_LINESTYLE.get(physics, "-"),
        "linewidth": 2,
        "markersize": 7,
    }

# ── Human-readable labels ─────────────────────────────────────────────────────
TASK_LABELS: dict[str, str] = {
    "shadow_vision_direct": "Shadow Hand Vision (Direct, RL)",
    "shadow_vision_benchmark": "Shadow Hand Vision (Benchmark, No CNN)",
    "g1_flat": "G1 Locomotion (Flat)",
    "dexsuite_kuka_allegro": "Dexsuite Kuka Allegro Lift",
}

PHYSICS_LABELS: dict[str, str] = {
    "physx": "PhysX",
    "newton": "Newton",
}

RENDERER_LABELS: dict[str, str] = {
    "isaacsim_rtx_renderer": "Isaac RTX",
    "newton_renderer": "Newton Renderer",
    "none": "—",
}

RENDER_TYPE_LABELS: dict[str, str] = {
    "rgb": "RGB",
    "depth": "Depth",
    "albedo": "Albedo",
    "simple_shading_constant_diffuse": "SS-Const",
    "simple_shading_diffuse_mdl": "SS-Diff",
    "simple_shading_full_mdl": "SS-Full",
    "none": "—",
}

# Render types excluded from all plots (too noisy / not of primary interest)
EXCLUDED_RENDER_TYPES: set[str] = {"simple_shading_diffuse_mdl", "simple_shading_full_mdl", "albedo"}

# Task directories excluded from all plots
EXCLUDED_TASK_DIRS: set[str] = {"shadow_vision_direct"}


# ── JSON loading ──────────────────────────────────────────────────────────────

def _safe_float(value: Any) -> float | None:
    """Return float or None for missing / non-numeric values."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _phase_key(name: str, phase_name: str) -> str:
    """Extract the measurement/metadata key from a full name like '{workflow} {phase} {key}'."""
    sep = f" {phase_name} "
    idx = name.find(sep)
    if idx >= 0:
        return name[idx + len(sep):]
    return name


def _normalise_list_format(phases: list) -> dict:
    """Convert the new list-of-phases JSON format into the flat dict format.

    New format: ``[{"phase_name": "runtime", "measurements": [...], "metadata": [...]}, ...]``
    Old format: ``{"benchmark_info": {...}, "runtime": {...}, ...}``
    """
    result: dict[str, dict] = {}
    for phase in phases:
        phase_name = phase.get("phase_name", "")
        phase_dict: dict[str, Any] = {"phase": phase_name}
        # Metadata entries carry config/identity data
        for meta in phase.get("metadata", []):
            key = _phase_key(meta.get("name", ""), phase_name)
            phase_dict[key] = meta.get("data")
        # Measurement entries carry numeric/aggregate values
        for meas in phase.get("measurements", []):
            if meas.get("type") in ("single", "dict", "list"):
                key = _phase_key(meas.get("name", ""), phase_name)
                phase_dict[key] = meas.get("value")
        result[phase_name] = phase_dict
    return result


def _load_json(path: Path) -> dict:
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return _normalise_list_format(raw)
    return raw


def _extract_metrics(data: dict) -> dict[str, Any]:
    """Extract the metrics we care about from one benchmark JSON file."""
    info = data.get("benchmark_info", {})
    runtime = data.get("runtime", {})

    # ── Identity ──
    task = info.get("task", "")
    num_envs = info.get("num_envs")
    presets_str = info.get("presets", "default")

    # ── FPS ──
    # Non-RL produces "Mean Environment step effective FPS".
    # RSL-RL produces "Mean Collection FPS" (already includes num_envs and num_steps_per_env).
    fps = _safe_float(runtime.get("Mean Environment step effective FPS")) or _safe_float(
        runtime.get("Mean Collection FPS")
    )
    fps_key = (
        "Env Step Effective FPS"
        if runtime.get("Mean Environment step effective FPS") is not None
        else "Collection FPS"
    )

    # ── Memory ──
    gpu_mem_gb = _safe_float(runtime.get("GPU Memory Used"))
    cpu_mem_rss_gb = _safe_float(runtime.get("System Memory RSS"))

    # ── Step-time breakdown (from StepTimeRecorder) ──
    # "Mean Physics Step Time", "Mean Render Step Time", "Mean Scene Update Time"
    physics_ft_ms = _safe_float(runtime.get("Mean Physics Step Time"))
    render_ft_ms = _safe_float(runtime.get("Mean Render Step Time"))
    # Whole-step env step time as fallback
    env_step_ms = _safe_float(runtime.get("Mean Environment step times"))

    return {
        "task_raw": task,
        "num_envs": num_envs,
        "presets_str": presets_str,
        "fps": fps,
        "fps_key": fps_key,
        "gpu_mem_gb": gpu_mem_gb,
        "cpu_mem_rss_gb": cpu_mem_rss_gb,
        "physics_ft_ms": physics_ft_ms,
        "render_ft_ms": render_ft_ms,
        "env_step_ms": env_step_ms,
    }


# ── Directory scanner ─────────────────────────────────────────────────────────

# Ordered longest-first so "128" doesn't accidentally match the "8" tail of "256".
_KNOWN_RESOLUTIONS = ("256", "128", "64")


def _split_render_type_res(combined: str) -> tuple[str, str]:
    """Split a combined render_type+resolution string (e.g. ``"rgb64"``) into parts.

    Returns ``(combined, "none")`` when no known resolution suffix is found.
    """
    for res in _KNOWN_RESOLUTIONS:
        if combined.endswith(res) and len(combined) > len(res):
            return combined[: -len(res)], res
    return combined, "none"


def _parse_config_from_path(json_path: Path, results_root: Path) -> dict[str, str]:
    """Infer benchmark config from the directory structure relative to results_root.

    Handles two layouts:

    * **5-level** (shadow_hand / g1 benchmarks)::

        <task_dir> / <physics> / <renderer> / <render_type> / <resolution> / *.json

    * **4-level** (dexsuite benchmarks, combined render_type+resolution)::

        <task_dir> / <physics> / <renderer> / <render_type><resolution> / *.json

      e.g. ``dexsuite_kuka_allegro/physx/isaacsim_rtx_renderer/rgb64/``
    """
    rel = json_path.parent.relative_to(results_root)
    parts = list(rel.parts)  # e.g. ["shadow_vision_benchmark", "physx", "isaacsim_rtx_renderer", "rgb", "64"]

    task_dir = parts[0] if len(parts) > 0 else "unknown"
    physics = parts[1] if len(parts) > 1 else "unknown"
    renderer = parts[2] if len(parts) > 2 else "none"
    render_type = parts[3] if len(parts) > 3 else "none"
    resolution_str = parts[4] if len(parts) > 4 else "none"

    # Dexsuite uses a combined ``rgb64`` / ``depth128`` component at level 3 with no
    # separate resolution level (parts[4] is absent).  Detect and split it.
    if resolution_str == "none" and render_type not in ("none", ""):
        render_type, resolution_str = _split_render_type_res(render_type)

    return {
        "task_dir": task_dir,
        "physics": physics,
        "renderer": renderer,
        "render_type": render_type,
        "resolution": resolution_str,
    }


def scan_results(results_root: Path) -> pd.DataFrame:
    """Walk the results tree, load every JSON, return a tidy DataFrame."""
    rows = []

    for json_path in sorted(results_root.rglob("benchmark_*.json")):
        try:
            data = _load_json(json_path)
        except Exception as exc:
            print(f"[WARN] Could not load {json_path}: {exc}")
            continue

        cfg = _parse_config_from_path(json_path, results_root)
        if EXCLUDED_TASK_DIRS and cfg["task_dir"] in EXCLUDED_TASK_DIRS:
            continue
        metrics = _extract_metrics(data)

        rows.append({**cfg, **metrics, "json_path": str(json_path)})

    if not rows:
        print("[ERROR] No benchmark JSON files found. Is the results_dir correct?")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Coerce numeric types
    for col in ("num_envs", "fps", "gpu_mem_gb", "cpu_mem_rss_gb",
                "physics_ft_ms", "render_ft_ms", "env_step_ms"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    resolution_map = {
        str(r): int(r) for r in ["64", "128", "256"] if r.isdigit()
    }
    df["resolution_int"] = df["resolution"].map(resolution_map).fillna(-1).astype(int)

    # Drop excluded render types globally
    if EXCLUDED_RENDER_TYPES:
        df = df[~df["render_type"].isin(EXCLUDED_RENDER_TYPES)].copy()

    # Aggregate duplicate runs (same config + num_envs) by median to suppress outliers
    config_cols = ["task_dir", "physics", "renderer", "render_type", "resolution",
                   "resolution_int", "num_envs"]
    metric_cols = ["fps", "gpu_mem_gb", "cpu_mem_rss_gb",
                   "physics_ft_ms", "render_ft_ms", "env_step_ms"]
    str_cols = ["task_raw", "presets_str", "fps_key"]
    # Take minimum of numeric metrics (best/stable run, suppresses high outliers)
    agg = {c: "min" for c in metric_cols}
    agg.update({c: "first" for c in str_cols})
    df = df.groupby(config_cols, dropna=False).agg(agg).reset_index()

    return df


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fpath = out_dir / f"{name}.{ext}"
        fig.savefig(fpath)
    plt.close(fig)
    print(f"  Saved: {name}")


def _config_label(physics: str, renderer: str, render_type: str, resolution: str) -> str:
    """Short, human-readable string for a legend entry."""
    parts = [PHYSICS_LABELS.get(physics, physics)]
    if renderer not in ("none", ""):
        parts.append(RENDERER_LABELS.get(renderer, renderer))
    if render_type not in ("none", ""):
        parts.append(RENDER_TYPE_LABELS.get(render_type, render_type))
    if resolution not in ("none", "") and resolution != "-1":
        parts.append(f"{resolution}px")
    return " / ".join(parts)


def _scaling_plot(
    ax: plt.Axes,
    sub: pd.DataFrame,
    y_col: str,
    y_label: str,
    group_cols: list[str],
) -> None:
    """Draw one scaling line per unique config in ``sub``.

    Visual encoding (deterministic, not index-dependent):
      color   = render_type × resolution  (same type/res always same hue/shade)
      marker  = physics backend            (○ PhysX, □ Newton physics)
      linestyle = renderer                 (solid = Isaac RTX, dashed = Newton renderer)
    """
    groups = sub.groupby(group_cols, dropna=False)
    for key, grp in groups:
        if not isinstance(key, tuple):
            key = (key,)
        # Pad to 4 elements: physics, renderer, render_type, resolution
        key4 = tuple(str(k) for k in key) + ("none",) * (4 - len(key))
        physics, renderer, render_type, resolution = key4[:4]

        label = _config_label(physics, renderer, render_type, resolution)
        style = _line_style(physics, renderer, render_type, resolution)

        grp_sorted = grp.sort_values("num_envs")
        valid = grp_sorted.dropna(subset=["num_envs", y_col])
        if valid.empty:
            continue
        ax.plot(valid["num_envs"], valid[y_col], label=label, **style)

    ax.set_xlabel("Number of Environments")
    ax.set_ylabel(y_label)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))


# ── Figure generators ─────────────────────────────────────────────────────────

def plot_fps_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    """Effective FPS vs num_envs, one figure per task."""
    for task_dir in df["task_dir"].unique():
        sub = df[df["task_dir"] == task_dir].copy()
        valid = sub.dropna(subset=["fps"])
        if valid.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 5.5))
        _scaling_plot(
            ax, valid, "fps", "Environment FPS (mean × num_envs)",
            ["physics", "renderer", "render_type", "resolution"],
        )
        ax.set_title(f"Effective FPS Scaling – {TASK_LABELS.get(task_dir, task_dir)}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=7.5, ncol=1)
        fig.tight_layout()
        _save(fig, out_dir, f"fps_scaling_{task_dir}")


def plot_gpu_memory_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    """GPU memory vs num_envs, one figure per task."""
    for task_dir in df["task_dir"].unique():
        sub = df[df["task_dir"] == task_dir].copy()
        valid = sub.dropna(subset=["gpu_mem_gb"])
        if valid.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 5.5))
        _scaling_plot(
            ax, valid, "gpu_mem_gb", "GPU Memory Used [GB]",
            ["physics", "renderer", "render_type", "resolution"],
        )
        ax.set_title(f"GPU Memory Scaling – {TASK_LABELS.get(task_dir, task_dir)}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=7.5, ncol=1)
        fig.tight_layout()
        _save(fig, out_dir, f"gpu_mem_scaling_{task_dir}")


def plot_cpu_memory_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    """CPU RSS vs num_envs, one figure per task."""
    for task_dir in df["task_dir"].unique():
        sub = df[df["task_dir"] == task_dir].copy()
        valid = sub.dropna(subset=["cpu_mem_rss_gb"])
        if valid.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 5.5))
        _scaling_plot(
            ax, valid, "cpu_mem_rss_gb", "CPU Memory RSS [GB]",
            ["physics", "renderer", "render_type", "resolution"],
        )
        ax.set_title(f"CPU Memory Scaling – {TASK_LABELS.get(task_dir, task_dir)}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=7.5, ncol=1)
        fig.tight_layout()
        _save(fig, out_dir, f"cpu_mem_scaling_{task_dir}")


def _bar_comparison(
    ax: plt.Axes,
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    x_labels: dict[str, str],
    hue_labels: dict[str, str],
    y_label: str,
    num_envs_fixed: int,
) -> None:
    """Grouped bar chart: x_col categories, hue_col groups."""
    sub = sub[sub["num_envs"] == num_envs_fixed].copy()
    if sub.empty:
        return

    x_vals = sorted(sub[x_col].dropna().unique(), key=lambda v: list(x_labels.keys()).index(v) if v in x_labels else 99)
    hue_vals = sorted(sub[hue_col].dropna().unique())

    n_x = len(x_vals)
    n_hue = len(hue_vals)
    width = 0.8 / n_hue
    x_pos = np.arange(n_x)

    for hi, hv in enumerate(hue_vals):
        ys = []
        for xv in x_vals:
            sel = sub[(sub[x_col] == xv) & (sub[hue_col] == hv)][y_col]
            ys.append(sel.mean() if not sel.empty else np.nan)
        offset = (hi - n_hue / 2 + 0.5) * width
        ax.bar(
            x_pos + offset,
            ys,
            width=width * 0.9,
            color=_PALETTE[hi % len(_PALETTE)],
            label=hue_labels.get(hv, hv),
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([x_labels.get(v, v) for v in x_vals], rotation=25, ha="right")
    ax.set_ylabel(y_label)
    ax.legend()


def plot_render_type_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: FPS and GPU memory across render types for the benchmark env."""
    task_dir = "shadow_vision_benchmark"
    sub = df[df["task_dir"] == task_dir].copy()
    if sub.empty:
        return

    # Use the largest successfully completed num_envs per config
    max_envs = int(sub.groupby(["physics", "renderer", "render_type", "resolution"])["num_envs"].max().min())
    if max_envs <= 0:
        max_envs = int(sub["num_envs"].max())

    for physics in sub["physics"].unique():
        ps = sub[sub["physics"] == physics]
        for res in sorted(ps["resolution_int"].unique()):
            if res < 0:
                continue
            rs = ps[ps["resolution_int"] == res]
            if rs.empty:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(
                f"Render Type Comparison – {PHYSICS_LABELS.get(physics, physics)}, {res}×{res} px\n"
                f"(Shadow Vision Benchmark, num_envs≈{max_envs:,})",
                fontsize=12,
                fontweight="bold",
            )

            _bar_comparison(
                axes[0], rs, "render_type", "fps", "renderer",
                RENDER_TYPE_LABELS, RENDERER_LABELS,
                "Effective FPS", max_envs,
            )
            axes[0].set_title("Effective FPS")

            _bar_comparison(
                axes[1], rs, "render_type", "gpu_mem_gb", "renderer",
                RENDER_TYPE_LABELS, RENDERER_LABELS,
                "GPU Memory [GB]", max_envs,
            )
            axes[1].set_title("GPU Memory")

            fig.tight_layout()
            _save(fig, out_dir, f"render_type_comparison_{physics}_{res}px")


def plot_renderer_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: RTX vs Newton renderer for rgb render type, physics backends."""
    for task_dir in ["shadow_vision_benchmark", "shadow_vision_direct"]:
        sub = df[(df["task_dir"] == task_dir) & (df["render_type"] == "rgb")].copy()
        if sub.empty:
            continue

        max_envs = int(sub.groupby(["physics", "renderer", "resolution"])["num_envs"].max().min())
        if max_envs <= 0:
            max_envs = int(sub["num_envs"].max())

        for res in sorted(sub["resolution_int"].unique()):
            if res < 0:
                continue
            rs = sub[sub["resolution_int"] == res]
            if rs.empty:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(
                f"Renderer Comparison (RGB) – {TASK_LABELS.get(task_dir, task_dir)}\n"
                f"{res}×{res} px, num_envs≈{max_envs:,}",
                fontsize=11,
                fontweight="bold",
            )

            _bar_comparison(
                axes[0], rs, "physics", "fps", "renderer",
                PHYSICS_LABELS, RENDERER_LABELS,
                "Effective FPS", max_envs,
            )
            axes[0].set_title("Effective FPS")

            _bar_comparison(
                axes[1], rs, "physics", "gpu_mem_gb", "renderer",
                PHYSICS_LABELS, RENDERER_LABELS,
                "GPU Memory [GB]", max_envs,
            )
            axes[1].set_title("GPU Memory")

            fig.tight_layout()
            _save(fig, out_dir, f"renderer_comparison_{task_dir}_{res}px")


def plot_physics_backend_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: PhysX vs Newton physics across all tasks."""
    for task_dir in df["task_dir"].unique():
        sub = df[df["task_dir"] == task_dir].copy()
        if sub.empty:
            continue

        max_envs = int(sub.groupby(["physics"])["num_envs"].max().min())
        if max_envs <= 0:
            max_envs = int(sub["num_envs"].max())

        is_vision = task_dir != "g1_flat"

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle(
            f"Physics Backend Comparison – {TASK_LABELS.get(task_dir, task_dir)}\n"
            f"num_envs≈{max_envs:,}"
            + (" (all renderer/type/res configs)" if is_vision else ""),
            fontsize=11,
            fontweight="bold",
        )

        if is_vision:
            # For vision envs: aggregate over renderer + render_type, group by resolution
            x_col, hue_col = "renderer", "physics"
            xl, hl = RENDERER_LABELS, PHYSICS_LABELS
        else:
            # G1: single bar per physics backend
            x_col, hue_col = "physics", "physics"
            xl, hl = PHYSICS_LABELS, PHYSICS_LABELS

        _bar_comparison(axes[0], sub, x_col, "fps", hue_col, xl, hl, "Effective FPS", max_envs)
        axes[0].set_title("Effective FPS")

        _bar_comparison(axes[1], sub, x_col, "gpu_mem_gb", hue_col, xl, hl, "GPU Memory [GB]", max_envs)
        axes[1].set_title("GPU Memory")

        fig.tight_layout()
        _save(fig, out_dir, f"physics_comparison_{task_dir}")


def plot_resolution_impact(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart: FPS and memory for 64 / 128 / 256 resolutions."""
    for task_dir in ["shadow_vision_benchmark", "shadow_vision_direct"]:
        sub = df[(df["task_dir"] == task_dir) & (df["resolution_int"] > 0)].copy()
        if sub.empty:
            continue

        max_envs = int(sub.groupby(["physics", "renderer", "render_type", "resolution"])["num_envs"].max().min())
        if max_envs <= 0:
            max_envs = int(sub["num_envs"].max())

        # One figure per physics + renderer + render_type combo
        for physics in sub["physics"].unique():
            for renderer in sub[sub["physics"] == physics]["renderer"].unique():
                for rtype in (
                    sub[(sub["physics"] == physics) & (sub["renderer"] == renderer)]["render_type"].unique()
                ):
                    chunk = sub[
                        (sub["physics"] == physics)
                        & (sub["renderer"] == renderer)
                        & (sub["render_type"] == rtype)
                    ]
                    if chunk.empty:
                        continue

                    # Convert resolution_int to str for axis
                    chunk = chunk.copy()
                    chunk["res_label"] = chunk["resolution_int"].apply(lambda r: f"{r}×{r}")
                    res_order = {str(r): f"{r}×{r}" for r in [64, 128, 256]}

                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    title = (
                        f"Resolution Impact – {PHYSICS_LABELS.get(physics, physics)} / "
                        f"{RENDERER_LABELS.get(renderer, renderer)} / {RENDER_TYPE_LABELS.get(rtype, rtype)}\n"
                        f"{TASK_LABELS.get(task_dir, task_dir)}, num_envs≈{max_envs:,}"
                    )
                    fig.suptitle(title, fontsize=10, fontweight="bold")

                    sel = chunk[chunk["num_envs"] == max_envs]
                    if sel.empty:
                        plt.close(fig)
                        continue

                    x_vals = [64, 128, 256]
                    fps_vals = [sel[sel["resolution_int"] == r]["fps"].mean() for r in x_vals]
                    mem_vals = [sel[sel["resolution_int"] == r]["gpu_mem_gb"].mean() for r in x_vals]
                    x_labels = [f"{r}×{r}" for r in x_vals]
                    x_pos = np.arange(len(x_vals))

                    for ax, vals, ylabel, title_suffix, color in [
                        (axes[0], fps_vals, "Effective FPS", "FPS", _PALETTE[0]),
                        (axes[1], mem_vals, "GPU Memory [GB]", "GPU Memory", _PALETTE[1]),
                    ]:
                        valid_mask = [not np.isnan(v) if v is not None else True for v in vals]
                        bars = ax.bar(
                            x_pos,
                            [v if v is not None and not np.isnan(v) else 0 for v in vals],
                            color=color,
                            alpha=0.85,
                            edgecolor="white",
                            linewidth=0.5,
                        )
                        ax.set_xticks(x_pos)
                        ax.set_xticklabels(x_labels)
                        ax.set_ylabel(ylabel)
                        ax.set_xlabel("Camera Resolution")
                        ax.set_title(title_suffix)

                    fig.tight_layout()
                    safe_rtype = re.sub(r"[^a-zA-Z0-9_]", "_", rtype)
                    _save(
                        fig,
                        out_dir,
                        f"resolution_impact_{task_dir}_{physics}_{renderer}_{safe_rtype}",
                    )


def plot_steptime_breakdown(df: pd.DataFrame, out_dir: Path) -> None:
    """Physics vs render step-time breakdown (when frametime data is available)."""
    cols_needed = ["physics_ft_ms", "render_ft_ms"]
    sub = df.dropna(subset=cols_needed, how="all").copy()

    # Fall back to env_step_ms if frametimes are entirely absent
    if sub.empty:
        sub = df.dropna(subset=["env_step_ms"]).copy()
        if sub.empty:
            print("  [INFO] No frametime data available; skipping step-time breakdown.")
            return
        # Single bar = total step time
        for task_dir in sub["task_dir"].unique():
            ts = sub[sub["task_dir"] == task_dir]
            max_envs = int(ts["num_envs"].max())
            ts_sel = ts[ts["num_envs"] == max_envs]
            if ts_sel.empty:
                continue
            _plot_single_steptime(ts_sel, task_dir, max_envs, out_dir, use_total=True)
        return

    for task_dir in sub["task_dir"].unique():
        ts = sub[sub["task_dir"] == task_dir]
        max_envs = int(ts["num_envs"].max())
        ts_sel = ts[ts["num_envs"] == max_envs]
        if ts_sel.empty:
            continue
        _plot_single_steptime(ts_sel, task_dir, max_envs, out_dir, use_total=False)


def _plot_single_steptime(
    sub: pd.DataFrame,
    task_dir: str,
    num_envs: int,
    out_dir: Path,
    use_total: bool,
) -> None:
    """Stacked bar of physics + render step times for one task."""
    if use_total:
        groups = sub.groupby(["physics", "renderer", "render_type", "resolution"])["env_step_ms"].mean().reset_index()
        groups["label"] = groups.apply(
            lambda r: _config_label(r["physics"], r["renderer"], r["render_type"], r["resolution"]), axis=1
        )
        fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.6 + 2), 5))
        ax.bar(range(len(groups)), groups["env_step_ms"], color=_PALETTE[0], alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups["label"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Env Step Time [ms]")
        ax.set_title(
            f"Environment Step Time – {TASK_LABELS.get(task_dir, task_dir)}\nnum_envs={num_envs:,}"
        )
        fig.tight_layout()
        _save(fig, out_dir, f"steptime_{task_dir}")
        return

    groups = (
        sub.groupby(["physics", "renderer", "render_type", "resolution"])[["physics_ft_ms", "render_ft_ms"]]
        .mean()
        .reset_index()
    )
    groups["label"] = groups.apply(
        lambda r: _config_label(r["physics"], r["renderer"], r["render_type"], r["resolution"]), axis=1
    )

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.6 + 2), 5))
    x = np.arange(len(groups))
    physics_vals = groups["physics_ft_ms"].fillna(0).values
    render_vals = groups["render_ft_ms"].fillna(0).values

    ax.bar(x, physics_vals, color=_PALETTE[0], alpha=0.85, label="Physics", edgecolor="white")
    ax.bar(x, render_vals, bottom=physics_vals, color=_PALETTE[1], alpha=0.85, label="Render", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(groups["label"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Step Time [ms]")
    ax.set_title(
        f"Physics + Render Step Time – {TASK_LABELS.get(task_dir, task_dir)}\nnum_envs={num_envs:,}"
    )
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, f"steptime_{task_dir}")


def plot_combined_overview(df: pd.DataFrame, out_dir: Path) -> None:
    """2×3 overview figure: FPS and GPU memory vs num_envs for all three tasks."""
    task_dirs = ["g1_flat", "shadow_vision_benchmark", "shadow_vision_direct"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("Multi-Backend Performance Overview", fontsize=14, fontweight="bold")

    for col, task_dir in enumerate(task_dirs):
        sub = df[df["task_dir"] == task_dir].copy()
        # FPS row
        ax = axes[0, col]
        valid = sub.dropna(subset=["fps"])
        _scaling_plot(ax, valid, "fps", "Eff. FPS", ["physics", "renderer", "render_type", "resolution"])
        ax.set_title(TASK_LABELS.get(task_dir, task_dir), fontsize=10)
        if col == 0:
            ax.set_ylabel("Effective FPS (mean × N envs)", fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1),
                      borderaxespad=0, fontsize=6.5, ncol=1)

        # GPU memory row
        ax = axes[1, col]
        valid = sub.dropna(subset=["gpu_mem_gb"])
        _scaling_plot(ax, valid, "gpu_mem_gb", "GPU Mem [GB]", ["physics", "renderer", "render_type", "resolution"])
        if col == 0:
            ax.set_ylabel("GPU Memory [GB]", fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1),
                      borderaxespad=0, fontsize=6.5, ncol=1)

    fig.tight_layout()
    _save(fig, out_dir, "overview")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot multi-backend IsaacLab benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("benchmark_results/multi_backend"),
        help="Root directory produced by run_multi_backend_benchmarks.sh",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write plot files (default: <results_dir>/plots)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Restrict to specific task dirs (e.g. g1_flat shadow_vision_benchmark)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        sys.exit(1)

    out_dir = args.output_dir.resolve() if args.output_dir else results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir : {results_dir}")
    print(f"Output dir  : {out_dir}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nScanning results…")
    df = scan_results(results_dir)

    if args.tasks:
        df = df[df["task_dir"].isin(args.tasks)]

    print(f"Loaded {len(df)} benchmark records across tasks: {df['task_dir'].unique().tolist()}")
    print(f"  num_envs values: {sorted(df['num_envs'].dropna().unique().astype(int).tolist())}")

    # Export raw data table for inspection
    df.to_csv(out_dir / "raw_data.csv", index=False)
    print(f"\nRaw data → {out_dir / 'raw_data.csv'}")

    # ── Generate plots ────────────────────────────────────────────────────────
    print("\nGenerating plots…")

    print("  [1/8] FPS scaling curves")
    plot_fps_scaling(df, out_dir)

    print("  [2/8] GPU memory scaling curves")
    plot_gpu_memory_scaling(df, out_dir)

    print("  [3/8] CPU memory scaling curves")
    plot_cpu_memory_scaling(df, out_dir)

    print("  [4/8] Render type comparison")
    plot_render_type_comparison(df, out_dir)

    print("  [5/8] Renderer comparison (RTX vs Newton)")
    plot_renderer_comparison(df, out_dir)

    print("  [6/8] Physics backend comparison")
    plot_physics_backend_comparison(df, out_dir)

    print("  [7/8] Resolution impact")
    plot_resolution_impact(df, out_dir)

    print("  [8/8] Step-time breakdown (physics + render)")
    plot_steptime_breakdown(df, out_dir)

    print("\n  [bonus] Combined overview figure")
    plot_combined_overview(df, out_dir)

    print(f"\nDone. All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
