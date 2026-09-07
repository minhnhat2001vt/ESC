#!/usr/bin/env bash
# ============================================================================
# Wall-clock measurement for ESC overhead reporting.
#
# Runs baseline inference and ESC inference on the SAME GPU, SAME batch size,
# SAME sample subset, and reports per-sample wall-clock difference.
#
# Designed to be run as a one-shot during T4 (LLaVA-OneVision) where you're
# doing fresh inference anyway.
#
# Usage:
#   # Set env vars first (model, baseline path, etc.)
#   bash measure_overhead_wallclock.sh \
#       --model llava-ov \
#       --benchmark vlsafe \
#       --baseline_json /path/to/llava_onevision_7b/vlsafe_baseline.json \
#       --max_samples 100
#
# Output: a small JSON report at $LOG_DIR/wallclock_<model>_<benchmark>.json
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOGS_ROOT="${ESC_LOGS_ROOT:-${REPO_ROOT}/logs}"

# Defaults
PIPELINE_BASELINE="${PIPELINE_BASELINE:-${REPO_ROOT}/scripts/inference_baseline.py}"
PIPELINE_SAFETY="${PIPELINE_SAFETY:-${REPO_ROOT}/scripts/method/run_esc_safety.py}"
VERIFIER="${VERIFIER:-gemma3-12b}"
BATCH_SIZE="${BATCH_SIZE:-6}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"   # use a subset for fast wall-clock measurement
LOG_DIR="${LOG_DIR:-${LOGS_ROOT}/wallclock}"
mkdir -p "$LOG_DIR"

MODEL=""
BENCHMARK=""
BASELINE_JSON=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)         MODEL="$2"; shift 2 ;;
    --benchmark)     BENCHMARK="$2"; shift 2 ;;
    --baseline_json) BASELINE_JSON="$2"; shift 2 ;;
    --max_samples)   MAX_SAMPLES="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -z "$MODEL" || -z "$BENCHMARK" || -z "$BASELINE_JSON" ]] && {
  echo "Required: --model --benchmark --baseline_json"
  exit 1
}

if [[ ! -f "$BASELINE_JSON" ]]; then
  echo "Baseline JSON not found: $BASELINE_JSON"
  exit 1
fi

LOG_BASE="$LOG_DIR/baseline_${MODEL}_${BENCHMARK}.log"
LOG_ESC="$LOG_DIR/esc_${MODEL}_${BENCHMARK}.log"
RESULT_JSON="$LOG_DIR/wallclock_${MODEL}_${BENCHMARK}.json"

# ─── Run baseline (1 forward pass per sample, no verifier) ─────────────────
echo ""
echo "[$(date '+%H:%M:%S')] BASELINE: $MODEL on $BENCHMARK ($MAX_SAMPLES samples)"

# inference_baseline.py CLI: --model --benchmark --batch_size --max_samples
# (adapt to your script's actual flags if different)
T_START_BASE=$(date +%s)
{ time python3 "$PIPELINE_BASELINE" \
    --model "$MODEL" \
    --benchmark "$BENCHMARK" \
    --batch_size "$BATCH_SIZE" \
    --max_samples "$MAX_SAMPLES" 2>&1 ; } 2> "${LOG_BASE}.time"
cat "${LOG_BASE}.time"
T_END_BASE=$(date +%s)
WALL_BASE=$((T_END_BASE - T_START_BASE))

# ─── Run ESC (Step 1 + Step 2 + maybe Step 4 + Step 5) ─────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] ESC: $MODEL + $VERIFIER on $BENCHMARK ($MAX_SAMPLES samples)"

T_START_ESC=$(date +%s)
{ time python3 "$PIPELINE_SAFETY" \
    --model_a_results "$BASELINE_JSON" \
    --model_a "$MODEL" \
    --model_b "$VERIFIER" \
    --benchmark "$BENCHMARK" \
    --prompt_source emotion \
    --selection_type fixed \
    --quadrant negative_low \
    --location start \
    --multiple_emotion 2 \
    --batch_size "$BATCH_SIZE" \
    --num_loops 1 \
    --max_samples "$MAX_SAMPLES" 2>&1 ; } 2> "${LOG_ESC}.time"
cat "${LOG_ESC}.time"
T_END_ESC=$(date +%s)
WALL_ESC=$((T_END_ESC - T_START_ESC))

# ─── Summarize ─────────────────────────────────────────────────────────────
PER_SAMPLE_BASE=$(python3 -c "print($WALL_BASE / $MAX_SAMPLES)")
PER_SAMPLE_ESC=$(python3 -c "print($WALL_ESC / $MAX_SAMPLES)")
MULTIPLIER=$(python3 -c "print($WALL_ESC / max($WALL_BASE, 1))")

cat > "$RESULT_JSON" << EOF
{
  "model": "$MODEL",
  "benchmark": "$BENCHMARK",
  "verifier": "$VERIFIER",
  "n_samples": $MAX_SAMPLES,
  "batch_size": $BATCH_SIZE,
  "wall_baseline_s": $WALL_BASE,
  "wall_esc_s": $WALL_ESC,
  "per_sample_baseline_s": $PER_SAMPLE_BASE,
  "per_sample_esc_s": $PER_SAMPLE_ESC,
  "multiplier": $MULTIPLIER,
  "note": "Includes model load time. For pure inference cost, see compute_overhead.py."
}
EOF

echo ""
echo "=========================================================================="
echo "WALL-CLOCK SUMMARY ($MODEL on $BENCHMARK, $MAX_SAMPLES samples, batch $BATCH_SIZE)"
echo "=========================================================================="
echo "Baseline total wall: ${WALL_BASE}s  →  ${PER_SAMPLE_BASE}s/sample"
echo "ESC total wall:      ${WALL_ESC}s  →  ${PER_SAMPLE_ESC}s/sample"
echo "Multiplier:          ${MULTIPLIER}× baseline"
echo ""
echo "Saved: $RESULT_JSON"