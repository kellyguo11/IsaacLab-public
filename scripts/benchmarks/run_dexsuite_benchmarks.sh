#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Multi-backend performance benchmarks for Isaac-Dexsuite-Kuka-Allegro-Lift-v0.
#
# Sweeps physics backends (PhysX/Newton), rendering backends (Isaac RTX /
# Newton renderer), and combined render-type+resolution presets (rgb64,
# depth128, …).  Environment counts are swept in ascending order; a failed
# run (OOM or timeout) stops the sequence for that configuration.
#
# Directory layout produced:
#   <OUTPUT_DIR>/
#     dexsuite_kuka_allegro/
#       <physics>/<renderer>/<render_type_res>/  benchmark_rsl_rl_train_*.json
#
# Usage:
#   bash scripts/benchmarks/run_dexsuite_benchmarks.sh [OUTPUT_DIR]

set -uo pipefail

# Ctrl+C / SIGTERM: kill the whole process group and exit cleanly.
trap 'log "Interrupted – stopping benchmark suite."; kill -- -$$ 2>/dev/null; exit 130' INT TERM

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUTPUT_DIR="${1:-${ROOT_DIR}/benchmark_results/dexsuite}"

# ── Benchmark settings ────────────────────────────────────────────────────────
MAX_ITERATIONS=100       # frames for benchmark_non_rl.py
RUN_TIMEOUT=900         # seconds before a stuck run is force-killed (15 minutes)

# Environment counts to try (ascending); a failed run stops the sequence for
# that config.
NUM_ENVS=(2048 4096 8192 16384)

# ── Task / preset constants ───────────────────────────────────────────────────
TASK="Isaac-Dexsuite-Kuka-Allegro-Lift-v0"
TASK_DIR="dexsuite_kuka_allegro"
OBJECT="cube"
CAMERA="single_camera"
SCRIPT="scripts/benchmarks/benchmark_non_rl.py"

# Render-type+resolution presets for each renderer.
# Isaac RTX renderer supports all types; Newton renderer supports rgb and depth.
RTX_RENDER_TYPES=(
    rgb64 rgb128 rgb256
    depth64 depth128 depth256
    # albedo64 albedo128 albedo256
    simple_shading_constant_diffuse64 simple_shading_constant_diffuse128 simple_shading_constant_diffuse256
    # simple_shading_diffuse_mdl64 simple_shading_diffuse_mdl128 simple_shading_diffuse_mdl256
    # simple_shading_full_mdl64 simple_shading_full_mdl128 simple_shading_full_mdl256
)
NEWTON_RENDER_TYPES=(rgb64 rgb128 rgb256 depth64 depth128 depth256)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE=""

init_logging() {
    mkdir -p "${OUTPUT_DIR}"
    LOG_FILE="${OUTPUT_DIR}/run_log.txt"
    log "Dexsuite benchmarks started: $(date)"
    log "Output directory: ${OUTPUT_DIR}"
    log "NUM_ENVS:       ${NUM_ENVS[*]}"
    log "MAX_ITERATIONS: ${MAX_ITERATIONS}"
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
    for f in "${out_dir}"/benchmark_*"${task}"*.json; do
        [[ -f "${f}" ]] || continue
        if grep -qE "\"data\": ${num_envs}," "${f}"; then
            return 0
        fi
    done
    return 1
}

# ── Core runner ───────────────────────────────────────────────────────────────
# run_benchmark OUT_DIR NUM_ENVS PRESETS_STR
# Returns 0 on success, non-zero on failure (OOM or crash).
run_benchmark() {
    local out_dir="$1"
    local num_envs="$2"
    local presets="$3"

    # Skip if a completed benchmark JSON matching this exact config already exists.
    if _has_matching_result "${out_dir}" "${TASK}" "${num_envs}"; then
        log "  SKIP (already complete): task=${TASK} num_envs=${num_envs} dir=${out_dir}"
        return 0
    fi

    mkdir -p "${out_dir}"

    local cmd=(
        "${ROOT_DIR}/isaaclab.sh" -p "${SCRIPT}"
        --task "${TASK}"
        --num_envs "${num_envs}"
        --headless
        --benchmark_backend json
        --output_path "${out_dir}"
        --num_frames "${MAX_ITERATIONS}"
        --seed 42
        --enable_cameras
        "presets=${presets}"
    )

    log "  CMD: ${cmd[*]}"
    # Pipe through tee so output is visible live AND captured in the log.
    # --kill-after=60: send SIGKILL 60s after SIGTERM if the process ignores it
    # (CUDA driver hangs don't respond to SIGTERM).
    if timeout --kill-after=60 "${RUN_TIMEOUT}" "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"; then
        log "  OK  (task=${TASK} num_envs=${num_envs})"
        return 0
    else
        local rc=${PIPESTATUS[0]}
        if [[ "${rc}" -eq 124 ]]; then
            log "  TIMEOUT after ${RUN_TIMEOUT}s (task=${TASK} num_envs=${num_envs}) – skipping larger env counts"
        else
            log "  FAIL rc=${rc} (task=${TASK} num_envs=${num_envs}) – skipping larger env counts"
        fi
        return "${rc}"
    fi
}

# run_sweep OUT_DIR PRESETS_STR
# Iterates over NUM_ENVS, stopping on first OOM/failure.
run_sweep() {
    local out_dir="$1"
    local presets="$2"

    log "── presets=${presets} ──"

    local oom=false
    for n in "${NUM_ENVS[@]}"; do
        if $oom; then
            log "    skip num_envs=${n} (OOM)"
            continue
        fi
        if ! run_benchmark "${out_dir}" "${n}" "${presets}"; then
            oom=true
        fi
    done
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    init_logging

    for physics in physx newton; do
        # ── Isaac RTX renderer ──────────────────────────────────────────────
        log "════════ ${physics} / isaacsim_rtx_renderer ════════"
        for rt in "${RTX_RENDER_TYPES[@]}"; do
            run_sweep \
                "${OUTPUT_DIR}/${TASK_DIR}/${physics}/isaacsim_rtx_renderer/${rt}" \
                "${OBJECT},${physics},${CAMERA},isaacsim_rtx_renderer,${rt}"
        done

        # ── Newton renderer ─────────────────────────────────────────────────
        log "════════ ${physics} / newton_renderer ════════"
        for rt in "${NEWTON_RENDER_TYPES[@]}"; do
            run_sweep \
                "${OUTPUT_DIR}/${TASK_DIR}/${physics}/newton_renderer/${rt}" \
                "${OBJECT},${physics},${CAMERA},newton_renderer,${rt}"
        done
    done

    log "════════ All benchmarks complete ════════"
    log "Results stored in: ${OUTPUT_DIR}"
}

main "$@"
