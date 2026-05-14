#!/usr/bin/env bash
# ============================================================================
# Rebuttal evaluation runner
# ============================================================================
# Runs the impartial-judge evaluator (eval_vlsafe.py / eval_pope.py / etc.)
# on inference outputs produced by run_rebuttal_matrix.sh.
#
# Inference produces:    method1_results_*.json  (raw model responses)
# Eval produces:         loop_K_evaluated.json   (per-sample safe/unsafe labels)
#                        + summary metrics
#
# Skip-on-existing: each output dir is checked for any *_evaluated.json before
# running eval. Re-run an experiment by deleting its evaluated.json file.
#
# Usage:
#   bash run_rebuttal_eval.sh --benchmark vlsafe --tier 1
#   bash run_rebuttal_eval.sh --benchmark vlsafe --tier 1 --filter LLaVA
#   bash run_rebuttal_eval.sh --benchmark vlsafe --tier 1 --dry-run
# ============================================================================

set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────
EVAL_VLSAFE="${EVAL_VLSAFE:-/workspace/scripts/eval_vlsafe.py}"
EVAL_POPE="${EVAL_POPE:-/workspace/scripts/eval_pope_v2.py}"
EVAL_MATHVISTA="${EVAL_MATHVISTA:-/workspace/scripts/eval/eval_mathvista.py}"

OUTPUT_BASE="${OUTPUT_BASE:-/workspace/results/method1}"
LOG_DIR="${LOG_DIR:-/workspace/logs/rebuttal_eval}"
mkdir -p "$LOG_DIR"

EVALUATOR_MODEL="${EVALUATOR_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
LOAD_4BIT="${LOAD_4BIT:---load_in_4bit}"   # set to empty string to disable

# ─── Arg parsing ──────────────────────────────────────────────────────────
BENCHMARK=""
TIER="all"
FILTER=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --benchmark) BENCHMARK="$2"; shift 2 ;;
    --tier)      TIER="$2"; shift 2 ;;
    --filter)    FILTER="$2"; shift 2 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    -h|--help)
      echo "Usage: $0 --benchmark {vlsafe|pope|mathvista} [--tier 1..4] [--filter REGEX] [--dry-run]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -z "$BENCHMARK" ]] && { echo "ERROR: --benchmark is required"; exit 1; }

# ─── Map (tier, model, source) -> outdir ──────────────────────────────────
# Mirrors the path logic in run_rebuttal_matrix.sh / run_pipeline().
GEMMA_SHORT="gemma_3_12b_it"

expected_outdir() {
  local model_short=$1
  local benchmark=$2
  local prompt_source=$3
  local verifier_short=${4:-$GEMMA_SHORT}    # default to Gemma3-12B if not specified
  local multi=${5:-1}                         # multiple_emotion (default 1)

  local psource_tag=""
  [[ "$prompt_source" != "emotion" ]] && psource_tag="_${prompt_source}"

  if [[ "$prompt_source" == "emotion" ]]; then
    # ESC main paper setting: fixed/negative_low/start/multiN
    echo "$OUTPUT_BASE/${model_short}__${verifier_short}${psource_tag}/${benchmark}/fixed/negative_low/start/multi${multi}"
  else
    echo "$OUTPUT_BASE/${model_short}__${verifier_short}${psource_tag}/${benchmark}/random"
  fi
}

# ─── Define which experiments to evaluate per tier ────────────────────────
# Standard row format: LABEL | MODEL_SHORT | PROMPT_SOURCE
# T5 row format extends this with an optional 4th column: VERIFIER_SHORT
# (matches the verifier-tagged outdir produced by the matrix runner).

T1_EVAL=(
  "T1_LLaVA_VLSafe_neutral|llava_1_5_7b|neutral"
  "T1_LLaVA_VLSafe_psych  |llava_1_5_7b|psychological"
  "T1_LLaVA_VLSafe_none   |llava_1_5_7b|none"
  "T1_Qwen2_VLSafe_neutral|qwen2_vl_7b |neutral"
  "T1_Qwen2_VLSafe_psych  |qwen2_vl_7b |psychological"
  "T1_Qwen2_VLSafe_none   |qwen2_vl_7b |none"
)
T2_EVAL=(
  "T2_LLaVA_POPE_neutral|llava_1_5_7b|neutral"
  "T2_LLaVA_POPE_psych  |llava_1_5_7b|psychological"
  "T2_LLaVA_POPE_none   |llava_1_5_7b|none"
  "T2_Qwen2_POPE_neutral|qwen2_vl_7b |neutral"
  "T2_Qwen2_POPE_psych  |qwen2_vl_7b |psychological"
  "T2_Qwen2_POPE_none   |qwen2_vl_7b |none"
)
T3_EVAL=(
  "T3_LLaVA_MV_neutral|llava_1_5_7b|neutral"
  "T3_LLaVA_MV_psych  |llava_1_5_7b|psychological"
  "T3_LLaVA_MV_none   |llava_1_5_7b|none"
  "T3_Qwen2_MV_neutral|qwen2_vl_7b |neutral"
  "T3_Qwen2_MV_psych  |qwen2_vl_7b |psychological"
  "T3_Qwen2_MV_none   |qwen2_vl_7b |none"
)
# T4: Newer backbones × VLSafe × ESC (baseline evaluated separately)
T4_EVAL=(
  "T4_LLaVANeXT_VLSafe_emotion|llava_1_6_vicuna_7b|emotion|gemma_3_12b_it|2"
  "T4_Qwen3VL_VLSafe_emotion|qwen3_vl_8b_instruct|emotion|gemma_3_12b_it|2"
)
# T5: Small-verifier ablation. ESC (emotion, multi2) on LLaVA-1.5 × VLSafe × 3 small verifiers.
# Format: label|model_short|prompt_source|verifier_short|multi
T5_EVAL=(
  "T5_LLaVA_VLSafe_v_gemma3_4b   |llava_1_5_7b|emotion|gemma_3_4b_it|2"
  "T5_LLaVA_VLSafe_v_qwen25vl_3b |llava_1_5_7b|emotion|qwen2_5_vl_3b_instruct|2"
  "T5_LLaVA_VLSafe_v_internvl_2b |llava_1_5_7b|emotion|internvl2_5_2b|2"
)

# ─── Build the eval command for a given benchmark ─────────────────────────
build_eval_cmd() {
  local benchmark=$1
  local results_path=$2
  case "$benchmark" in
    vlsafe)
      echo "python3 $EVAL_VLSAFE --method --results_path $results_path $LOAD_4BIT --batch_size $EVAL_BATCH_SIZE --evaluator_model $EVALUATOR_MODEL"
      ;;
    pope)
      # eval_pope_v2.py uses --result_file (single file), not a directory.
      # We auto-discover the method1_results_*.json inside the inference outdir.
      local result_file
      result_file=$(find "$results_path" -maxdepth 2 -name "method1_results_*.json" \
                    ! -name "*summary*" ! -name "*evaluated*" 2>/dev/null | head -1)
      if [[ -z "$result_file" ]]; then
        echo "echo '  ⚠ No method1_results_*.json found under $results_path'; false"
      else
        local outdir="$results_path"
        echo "python3 $EVAL_POPE --result_file $result_file --output_dir $outdir"
      fi
      ;;
    mathvista)
      local result_file
      result_file=$(find "$results_path" -maxdepth 2 -name "method1_results_*.json" \
                    ! -name "*summary*" ! -name "*evaluated*" 2>/dev/null | head -1)
      if [[ -z "$result_file" ]]; then
        echo "echo '  ⚠ No method1_results_*.json found under $results_path'; false"
      else
        echo "python3 $EVAL_MATHVISTA --result_file $result_file --method --output_dir $results_path"
      fi
      ;;
    *)
      echo "ERROR: unknown benchmark: $benchmark" >&2
      return 1
      ;;
  esac
}

# ─── Has this experiment already been evaluated? ──────────────────────────
already_evaluated() {
  local outdir=$1
  # eval_vlsafe.py writes loop_K_evaluated.json for each loop subdir, OR a single
  # *_evaluated.json at the root of the results dir if num_loops=1.
  # Either presence counts as "done".
  if find "$outdir" -name "*_evaluated.json" 2>/dev/null | grep -q .; then
    return 0
  fi
  return 1
}

# ─── Run one eval ──────────────────────────────────────────────────────────
run_one_eval() {
  local label=$1
  local outdir=$2
  local cmd=$3

  echo ""
  echo "=========================================================================="
  echo "[$(date '+%H:%M:%S')]  ▶ EVAL $label"
  echo "  results_path: $outdir"

  if [[ ! -d "$outdir" ]]; then
    echo "  ⚠ SKIP (results dir not found — was inference run for this experiment?)"
    return 0
  fi

  if already_evaluated "$outdir"; then
    echo "  ✓ SKIP (eval already exists; delete *_evaluated.json files to re-run)"
    return 0
  fi

  echo "  cmd: $cmd"

  if [[ $DRY_RUN -eq 1 ]]; then
    return 0
  fi

  local logf="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_eval_${label// /_}.log"
  echo "  log: $logf"

  eval "$cmd" 2>&1 | tee "$logf"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -eq 0 ]]; then
    echo "[$(date '+%H:%M:%S')]  ✓ DONE: $label"
  else
    echo "[$(date '+%H:%M:%S')]  ✗ FAILED: $label  (rc=$rc, see $logf)"
  fi
}

# ─── Dispatch ─────────────────────────────────────────────────────────────
run_tier_eval() {
  local tier_name=$1
  shift
  local experiments=("$@")
  echo ""
  echo "##########################################################################"
  echo "## $tier_name (eval on: $BENCHMARK)"
  echo "##########################################################################"

  for row in "${experiments[@]}"; do
    # Standard format:    label|model_short|prompt_source
    # Extended (T5):      label|model_short|prompt_source|verifier_short[|multi]
    IFS='|' read -r label model_short prompt_source verifier_short multi <<< "$row"
    label=$(echo "$label" | xargs)
    model_short=$(echo "$model_short" | xargs)
    prompt_source=$(echo "$prompt_source" | xargs)
    verifier_short=$(echo "${verifier_short:-}" | xargs)
    multi=$(echo "${multi:-1}" | xargs)

    if [[ -n "$FILTER" ]] && ! [[ "$label" =~ $FILTER ]]; then
      continue
    fi

    # Tier rows are named for their inference benchmark; only eval matching ones.
    # T5 labels don't carry a benchmark suffix (they're all VLSafe), so skip the gate.
    if [[ ! "$label" =~ ^T5_ ]]; then
      case "$BENCHMARK" in
        vlsafe)    [[ "$label" =~ VLSafe ]] || continue ;;
        pope)      [[ "$label" =~ POPE   ]] || continue ;;
        mathvista) [[ "$label" =~ MV     ]] || continue ;;
      esac
    fi

    if [[ -n "$verifier_short" ]]; then
      outdir=$(expected_outdir "$model_short" "$BENCHMARK" "$prompt_source" "$verifier_short" "$multi")
    else
      outdir=$(expected_outdir "$model_short" "$BENCHMARK" "$prompt_source")
    fi
    cmd=$(build_eval_cmd "$BENCHMARK" "$outdir")
    run_one_eval "$label" "$outdir" "$cmd"
  done
}

case "$TIER" in
  1)   run_tier_eval "TIER 1" "${T1_EVAL[@]}" ;;
  2)   run_tier_eval "TIER 2" "${T2_EVAL[@]}" ;;
  3)   run_tier_eval "TIER 3" "${T3_EVAL[@]}" ;;
  4)   run_tier_eval "TIER 4" "${T4_EVAL[@]}" ;;
  5)   run_tier_eval "TIER 5" "${T5_EVAL[@]}" ;;
  all)
    run_tier_eval "TIER 1" "${T1_EVAL[@]}"
    run_tier_eval "TIER 2" "${T2_EVAL[@]}"
    run_tier_eval "TIER 3" "${T3_EVAL[@]}"
    run_tier_eval "TIER 4" "${T4_EVAL[@]}"
    run_tier_eval "TIER 5" "${T5_EVAL[@]}"
    ;;
  *) echo "Unknown tier: $TIER (1, 2, 3, 4, 5, all)"; exit 1 ;;
esac

echo ""
echo "=========================================================================="
echo "[$(date '+%H:%M:%S')]  ALL EVAL DONE"
echo "  Logs: $LOG_DIR"
echo "=========================================================================="