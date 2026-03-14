#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Multi-backend performance benchmarks for IsaacLab.
#
# Sweeps physics backends (PhysX/Newton), rendering backends (Isaac RTX / Newton
# renderer), render data types (rgb/depth/albedo/…), camera resolutions (64/128/256)
# and environment counts (2048/4096/8192/16384).  Each combination is run until an
# OOM or other failure is detected, after which larger env counts are skipped for
# that configuration.
#
# Directory layout produced:
#   <OUTPUT_DIR>/
#     shadow_vision_direct/           Isaac-Repose-Cube-Shadow-Vision-Direct-v0
#       <physics>/<renderer>/<render_type>/<resolution>/  benchmark_rsl_rl_train_*.json
#     shadow_vision_benchmark/        Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0
#       <physics>/<renderer>/<render_type>/<resolution>/  benchmark_non_rl_*.json
#     g1_flat/                        Isaac-Velocity-Flat-G1-v0
#       <physics>/                    benchmark_non_rl_*.json
#
# Usage:
#   bash scripts/benchmarks/run_multi_backend_benchmarks.sh [OUTPUT_DIR]

set -uo pipefail

# Ctrl+C / SIGTERM: kill the whole process group and exit cleanly.
trap 'log "Interrupted – stopping benchmark suite."; kill -- -$$ 2>/dev/null; exit 130' INT TERM

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUTPUT_DIR="${1:-${ROOT_DIR}/benchmark_results/multi_backend}"

# ── Benchmark settings ────────────────────────────────────────────────────────
NUM_FRAMES=500          # env steps for benchmark_non_rl.py
MAX_ITERATIONS=50       # RL iterations for benchmark_rsl_rl.py
RUN_TIMEOUT=1800        # seconds before a stuck run is force-killed (30 minutes)

# Environment counts to try (ascending); a failed run stops the sequence for
# that config.
NUM_ENVS=(2048 4096 8192 16384)

# Camera resolutions for vision environments
RESOLUTIONS=(64 128 256)

# ── Task names ────────────────────────────────────────────────────────────────
SHADOW_DIRECT_TASK="Isaac-Repose-Cube-Shadow-Vision-Direct-v0"
SHADOW_BENCH_TASK="Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0"
G1_TASK="Isaac-Velocity-Flat-G1-v0"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE=""

init_logging() {
    mkdir -p "${OUTPUT_DIR}"
    LOG_FILE="${OUTPUT_DIR}/run_log.txt"
    log "Multi-backend benchmarks started: $(date)"
    log "Output directory: ${OUTPUT_DIR}"
    log "NUM_ENVS:    ${NUM_ENVS[*]}"
    log "RESOLUTIONS: ${RESOLUTIONS[*]}"
    log "NUM_FRAMES:  ${NUM_FRAMES}  MAX_ITERATIONS: ${MAX_ITERATIONS}"
}

log() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}"
    [[ -n "${LOG_FILE}" ]] && echo "${msg}" >> "${LOG_FILE}"
}

# ── Helpers ───────────────────────────────────────────────────────────────────

# _has_matching_result OUT_DIR TASK NUM_ENVS
# Returns 0 if OUT_DIR contains a benchmark JSON whose filename includes TASK
# and whose content records NUM_ENVS in the benchmark_info metadata.
_has_matching_result() {
    local out_dir="$1" task="$2" num_envs="$3"
    # Task name is embedded in the filename; num_envs appears as
    # "data": <int>, in the JSON metadata (no other metadata field uses
    # values in the range 2048/4096/8192/16384).
    for f in "${out_dir}"/benchmark_*"${task}"*.json; do
        [[ -f "${f}" ]] || continue
        if grep -qE "\"data\": ${num_envs}," "${f}"; then
            return 0
        fi
    done
    return 1
}

# ── Core runner ───────────────────────────────────────────────────────────────
# run_benchmark OUT_DIR SCRIPT TASK NUM_ENVS [EXTRA_ARGS…]
# Returns 0 on success, non-zero on failure (OOM or crash).
run_benchmark() {
    local out_dir="$1"
    local script="$2"
    local task="$3"
    local num_envs="$4"
    shift 4
    local extra=("$@")

    # Skip if a completed benchmark JSON matching this exact config already exists.
    if _has_matching_result "${out_dir}" "${task}" "${num_envs}"; then
        log "  SKIP (already complete): task=${task} num_envs=${num_envs} dir=${out_dir}"
        return 0
    fi

    mkdir -p "${out_dir}"

    local cmd=(
        "${ROOT_DIR}/isaaclab.sh" -p "${script}"
        --task "${task}"
        --num_envs "${num_envs}"
        --headless
        --benchmark_backend json
        --output_path "${out_dir}"
    )

    if [[ "${script}" == *"non_rl"* ]]; then
        cmd+=(--num_frames "${NUM_FRAMES}")
    else
        # benchmark_rsl_rl.py
        cmd+=(--max_iterations "${MAX_ITERATIONS}" --seed 42)
    fi

    # Append extra flags (--enable_cameras, hydra overrides, etc.)
    if [[ ${#extra[@]} -gt 0 ]]; then
        cmd+=("${extra[@]}")
    fi

    log "  CMD: ${cmd[*]}"
    # Pipe through tee so output is visible live AND captured in the log.
    # timeout kills the process if it hangs (e.g. stuck after CUDA OOM).
    # pipefail propagates the benchmark exit code through the tee pipe.
    if timeout --kill-after=60 "${RUN_TIMEOUT}" "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"; then
        log "  OK  (task=${task} num_envs=${num_envs})"
        return 0
    else
        local rc=${PIPESTATUS[0]}
        if [[ "${rc}" -eq 124 ]]; then
            log "  TIMEOUT after ${RUN_TIMEOUT}s (task=${task} num_envs=${num_envs}) – skipping larger env counts"
        else
            log "  FAIL rc=${rc} (task=${task} num_envs=${num_envs}) – skipping larger env counts"
        fi
        return "${rc}"
    fi
}

# ── Per-task sweep helpers ────────────────────────────────────────────────────

# Vision env with cameras.
# run_vision TASK SCRIPT TASK_DIR PHYSICS RENDERER RENDER_TYPE RESOLUTION
run_vision() {
    local task="$1"
    local script="$2"
    local task_dir="$3"
    local physics="$4"
    local renderer="$5"
    local render_type="$6"
    local res="$7"

    local presets="${physics},${renderer},${render_type}"
    local out_dir="${OUTPUT_DIR}/${task_dir}/${physics}/${renderer}/${render_type}/${res}"

    log "── ${task_dir} | presets=${presets} res=${res}x${res} ──"

    local oom=false
    for n in "${NUM_ENVS[@]}"; do
        if $oom; then
            log "    skip num_envs=${n} (OOM)"
            continue
        fi
        if ! run_benchmark "${out_dir}" \
            "${script}" "${task}" "${n}" \
            --enable_cameras \
            "presets=${presets}" \
            "env.tiled_camera.width=${res}" \
            "env.tiled_camera.height=${res}"; then
            oom=true
        fi
    done
}

# Physics-only env (no cameras).
# run_physics TASK SCRIPT TASK_DIR PHYSICS [PRESETS_STR]
# PRESETS_STR is empty for the PhysX default, "newton" for Newton.
run_physics_only() {
    local task="$1"
    local script="$2"
    local task_dir="$3"
    local physics="$4"
    local presets_str="${5:-}"

    local out_dir="${OUTPUT_DIR}/${task_dir}/${physics}"

    log "── ${task_dir} | physics=${physics} ──"

    local extra=()
    [[ -n "${presets_str}" ]] && extra+=("presets=${presets_str}")

    local oom=false
    for n in "${NUM_ENVS[@]}"; do
        if $oom; then
            log "    skip num_envs=${n} (OOM)"
            continue
        fi
        if ! run_benchmark "${out_dir}" \
            "${script}" "${task}" "${n}" \
            "${extra[@]}"; then
            oom=true
        fi
    done
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    init_logging

    # ── G1 Flat: physics backends only ──────────────────────────────────────
    log "════════ G1 Flat ════════"
    # PhysX: no explicit preset (uses "default")
    run_physics_only "${G1_TASK}" \
        "scripts/benchmarks/benchmark_non_rl.py" \
        "g1_flat" "physx" ""
    # Newton: explicit preset
    run_physics_only "${G1_TASK}" \
        "scripts/benchmarks/benchmark_non_rl.py" \
        "g1_flat" "newton" "newton"

    # ── Shadow Vision Benchmark ──────────────────────────────────────────────
    # Uses benchmark_non_rl.py (CNN disabled – pure rendering throughput).
    log "════════ Shadow Vision Benchmark ════════"
    for physics in physx newton; do

        # Isaac RTX renderer: all render types (rgb + depth + material modes)
        for render_type in rgb depth \
                           simple_shading_constant_diffuse; do
            for res in "${RESOLUTIONS[@]}"; do
                run_vision "${SHADOW_BENCH_TASK}" \
                    "scripts/benchmarks/benchmark_non_rl.py" \
                    "shadow_vision_benchmark" \
                    "${physics}" "isaacsim_rtx_renderer" "${render_type}" "${res}"
            done
        done

        # Newton renderer: only rgb and depth
        for render_type in rgb depth; do
            for res in "${RESOLUTIONS[@]}"; do
                run_vision "${SHADOW_BENCH_TASK}" \
                    "scripts/benchmarks/benchmark_non_rl.py" \
                    "shadow_vision_benchmark" \
                    "${physics}" "newton_renderer" "${render_type}" "${res}"
            done
        done

    done

    # ── Shadow Vision Direct ─────────────────────────────────────────────────
    # Uses benchmark_rsl_rl.py; depth excluded (not supported by this env).
    log "════════ Shadow Vision Direct ════════"
    for physics in physx newton; do

        # Isaac RTX renderer: all render types EXCEPT depth
        for render_type in rgb albedo \
                           simple_shading_constant_diffuse \
                           simple_shading_diffuse_mdl \
                           simple_shading_full_mdl; do
            for res in "${RESOLUTIONS[@]}"; do
                run_vision "${SHADOW_DIRECT_TASK}" \
                    "scripts/benchmarks/benchmark_rsl_rl.py" \
                    "shadow_vision_direct" \
                    "${physics}" "isaacsim_rtx_renderer" "${render_type}" "${res}"
            done
        done

        # Newton renderer: only rgb (no depth for this env)
        for res in "${RESOLUTIONS[@]}"; do
            run_vision "${SHADOW_DIRECT_TASK}" \
                "scripts/benchmarks/benchmark_rsl_rl.py" \
                "shadow_vision_direct" \
                "${physics}" "newton_renderer" "rgb" "${res}"
        done

    done

    log "════════ All benchmarks complete ════════"
    log "Results stored in: ${OUTPUT_DIR}"
}

main "$@"
