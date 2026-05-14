#!/usr/bin/env bash
# ============================================================================
# Rebuttal experiment runner
# ============================================================================
# Runs the missing baselines requested by ECCV reviewers:
#
#   For each (model, benchmark) pair, run THREE additional conditions
#   on top of the existing ESC results:
#     - prompt_source=neutral        ("Please reconsider...")
#     - prompt_source=psychological  (Li et al. EmotionPrompt)
#     - prompt_source=none           (verifier-loop only, no insertion)
#
#   Tier-priority experiments (for the rebuttal table):
#
#   T1: VLSafe         × {LLaVA-1.5, Qwen2-VL} × {neutral, psych, none}  (~15h)
#   T2: POPE           × {LLaVA-1.5, Qwen2-VL} × {neutral, psych, none}  (~45h)
#   T3: MathVista      × {LLaVA-1.5, Qwen2-VL} × {neutral, psych}        (~12h)
#   T4: Qwen2.5-VL warm-up × VLSafe + POPE × {baseline, ESC, neutral}    (~25h)
#
# Resumability:
#   Each experiment writes to a unique output dir tagged with prompt_source.
#   If the final method1_results_*.json exists, the experiment is skipped.
#   Inside an experiment, the pipeline already checkpoints at step 2/4/5.
#
# Usage:
#   # Run everything (Tier 1+2+3, in order):
#   bash run_rebuttal_matrix.sh
#
#   # Run a specific tier only:
#   bash run_rebuttal_matrix.sh --tier 1
#
#   # Dry run (print commands without executing):
#   bash run_rebuttal_matrix.sh --dry-run
#
#   # Override paths (defaults match your codebase):
#   VLSAFE_LLAVA_BASELINE=/foo/bar.json bash run_rebuttal_matrix.sh
# ============================================================================

set -euo pipefail

# ─── Paths (override via env vars) ─────────────────────────────────────────
PIPELINE_SAFETY="${PIPELINE_SAFETY:-/workspace/scripts/method/inference_method1_ver3.py}"
PIPELINE_VQA="${PIPELINE_VQA:-/workspace/scripts/method/vqa_inference_method1_ver3.py}"

# The baseline (Step 1 neutral responses) JSON files for each (model, benchmark).
# These are the outputs of inference_baseline.py — change paths if yours differ.
QWEN25_BASELINE_VLSAFE="${QWEN25_BASELINE_VLSAFE:-/workspace/results/baseline/qwen2_5_vl_7b/vlsafe/results.json}"
QWEN25_BASELINE_POPE="${QWEN25_BASELINE_POPE:-/workspace/results/baseline/qwen2_5_vl_7b/pope/results.json}"

VLSAFE_LLAVA_BASELINE="${VLSAFE_LLAVA_BASELINE:-/workspace/results/baseline/llava_1_5_7b/vlsafe/results.json}"
VLSAFE_QWEN2_BASELINE="${VLSAFE_QWEN2_BASELINE:-/workspace/results/baseline/qwen2_vl_7b/vlsafe/results.json}"

POPE_LLAVA_BASELINE="${POPE_LLAVA_BASELINE:-/workspace/results/baseline/llava_1_5_7b/pope/results.json}"
POPE_QWEN2_BASELINE="${POPE_QWEN2_BASELINE:-/workspace/results/baseline/qwen2_vl_7b/pope/results.json}"

MATHVISTA_LLAVA_BASELINE="${MATHVISTA_LLAVA_BASELINE:-/workspace/results/baseline/llava_1_5_7b/mathvista/results.json}"
MATHVISTA_QWEN2_BASELINE="${MATHVISTA_QWEN2_BASELINE:-/workspace/results/baseline/qwen2_vl_7b/mathvista/results.json}"

OUTPUT_BASE="${OUTPUT_BASE:-/workspace/results/method1}"
LOG_DIR="${LOG_DIR:-/workspace/logs/rebuttal}"
mkdir -p "$LOG_DIR"

# Common pipeline params
BATCH_SIZE="${BATCH_SIZE:-16}"
VERIFIER="${VERIFIER:-gemma3-12b}"

# ─── Arg parsing ──────────────────────────────────────────────────────────
TIER="all"
DRY_RUN=0
FILTER=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --tier)    TIER="$2"; shift 2 ;;
    --filter)  FILTER="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '1,/^# =\+$/p' "$0" | grep '^#' | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ─── Helper: compute the expected output dir for an experiment ─────────────
# Mirrors the path logic in run_pipeline() in the inference scripts.
expected_outdir() {
  local model_a_short=$1
  local model_b_short=$2
  local benchmark=$3
  local selection_type=$4
  local quadrant=$5            # may be empty
  local location=$6            # only used if quadrant non-empty
  local multi=$7               # only used if quadrant non-empty
  local prompt_source=$8

  local psource_tag=""
  [[ "$prompt_source" != "emotion" ]] && psource_tag="_${prompt_source}"

  if [[ -n "$quadrant" ]]; then
    echo "$OUTPUT_BASE/${model_a_short}__${model_b_short}${psource_tag}/${benchmark}/${selection_type}/${quadrant}/${location}/multi${multi}"
  else
    echo "$OUTPUT_BASE/${model_a_short}__${model_b_short}${psource_tag}/${benchmark}/${selection_type}"
  fi
}

# Returns 0 (success) if a final result file exists in the dir
already_done() {
  local d=$1
  if [[ -d "$d" ]] && ls "$d"/method1_results_*.json &>/dev/null; then
    return 0
  fi
  return 1
}

# ─── Helper: run one experiment ───────────────────────────────────────────
run_one() {
  local label=$1
  local cmd=$2
  local outdir=$3

  echo ""
  echo "=========================================================================="
  echo "[$(date '+%H:%M:%S')]  ▶ $label"
  echo "  outdir: $outdir"
  if already_done "$outdir"; then
    echo "  ✓ SKIP (output already exists)"
    return 0
  fi
  echo "  cmd:    $cmd"
  echo "=========================================================================="

  if [[ $DRY_RUN -eq 1 ]]; then
    return 0
  fi

  local logf="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_${label// /_}.log"
  echo "  log:    $logf"

  # Run the command, tee to log, but capture the COMMAND's exit code, not tee's.
  # Without this, tee always returns 0 and we'd report success on Python crashes.
  eval "$cmd" 2>&1 | tee "$logf"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -eq 0 ]]; then
    echo "[$(date '+%H:%M:%S')]  ✓ DONE: $label"
  else
    echo "[$(date '+%H:%M:%S')]  ✗ FAILED: $label  (rc=$rc, see $logf)"
    # Don't abort the matrix on a single failure — continue to next experiment.
  fi
}

# ─── Define experiments ────────────────────────────────────────────────────
# Each row: TIER LABEL MODEL_A MODEL_A_SHORT BASELINE_JSON BENCHMARK PIPELINE PROMPT_SOURCE [extra_args]
# extra_args are appended verbatim.

GEMMA_SHORT="gemma_3_12b_it"

# For consistency with existing ESC runs, all rebuttal runs use:
#   --num_loops 1  (one regen pass — fair comparison; ESC paper headline is also 1-loop)
#   --selection_type random  (not used by neutral/none, but specified for psych)
#   --location start
#   --multiple_emotion 1

# ── Tier 1: VLSafe rebuttal baselines ──────────────────────────────────────
T1_EXPERIMENTS=(
  # label | model_a | model_a_short | baseline | benchmark | pipeline | prompt_source
  "T1_LLaVA_VLSafe_neutral|llava_1.5|llava_1_5_7b|$VLSAFE_LLAVA_BASELINE|vlsafe|$PIPELINE_SAFETY|neutral"
  "T1_LLaVA_VLSafe_psych  |llava_1.5|llava_1_5_7b|$VLSAFE_LLAVA_BASELINE|vlsafe|$PIPELINE_SAFETY|psychological"
  "T1_LLaVA_VLSafe_none   |llava_1.5|llava_1_5_7b|$VLSAFE_LLAVA_BASELINE|vlsafe|$PIPELINE_SAFETY|none"
  "T1_Qwen2_VLSafe_neutral|qwen2-vl |qwen2_vl_7b |$VLSAFE_QWEN2_BASELINE|vlsafe|$PIPELINE_SAFETY|neutral"
  "T1_Qwen2_VLSafe_psych  |qwen2-vl |qwen2_vl_7b |$VLSAFE_QWEN2_BASELINE|vlsafe|$PIPELINE_SAFETY|psychological"
  "T1_Qwen2_VLSafe_none   |qwen2-vl |qwen2_vl_7b |$VLSAFE_QWEN2_BASELINE|vlsafe|$PIPELINE_SAFETY|none"
)

# ── Tier 2: POPE rebuttal baselines ────────────────────────────────────────
# POPE has 3 splits (Adversarial, Popular, Random). The pipeline handles them
# inside the same baseline JSON, so this is a single run per condition.
T2_EXPERIMENTS=(
  "T2_LLaVA_POPE_neutral|llava_1.5|llava_1_5_7b|$POPE_LLAVA_BASELINE|pope|$PIPELINE_VQA|neutral"
  "T2_LLaVA_POPE_psych  |llava_1.5|llava_1_5_7b|$POPE_LLAVA_BASELINE|pope|$PIPELINE_VQA|psychological"
  "T2_LLaVA_POPE_none   |llava_1.5|llava_1_5_7b|$POPE_LLAVA_BASELINE|pope|$PIPELINE_VQA|none"
  "T2_Qwen2_POPE_neutral|qwen2-vl |qwen2_vl_7b |$POPE_QWEN2_BASELINE|pope|$PIPELINE_VQA|neutral"
  "T2_Qwen2_POPE_psych  |qwen2-vl |qwen2_vl_7b |$POPE_QWEN2_BASELINE|pope|$PIPELINE_VQA|psychological"
  "T2_Qwen2_POPE_none   |qwen2-vl |qwen2_vl_7b |$POPE_QWEN2_BASELINE|pope|$PIPELINE_VQA|none"
)

# ── Tier 3: MathVista rebuttal baselines ───────────────────────────────────
# Per user decision: run all 3 conditions for thoroughness (matching VLSafe / POPE).
T3_EXPERIMENTS=(
  "T3_LLaVA_MV_neutral|llava_1.5|llava_1_5_7b|$MATHVISTA_LLAVA_BASELINE|mathvista|$PIPELINE_VQA|neutral"
  "T3_LLaVA_MV_psych  |llava_1.5|llava_1_5_7b|$MATHVISTA_LLAVA_BASELINE|mathvista|$PIPELINE_VQA|psychological"
  "T3_LLaVA_MV_none   |llava_1.5|llava_1_5_7b|$MATHVISTA_LLAVA_BASELINE|mathvista|$PIPELINE_VQA|none"
  "T3_Qwen2_MV_neutral|qwen2-vl |qwen2_vl_7b |$MATHVISTA_QWEN2_BASELINE|mathvista|$PIPELINE_VQA|neutral"
  "T3_Qwen2_MV_psych  |qwen2-vl |qwen2_vl_7b |$MATHVISTA_QWEN2_BASELINE|mathvista|$PIPELINE_VQA|psychological"
  "T3_Qwen2_MV_none   |qwen2-vl |qwen2_vl_7b |$MATHVISTA_QWEN2_BASELINE|mathvista|$PIPELINE_VQA|none"
)

# ── Tier 4: Qwen2.5-VL-7B (newer backbone) ─────────────────────────────────
# REQUIREMENT: Step 1 (baseline neutral inference) must have been run separately
# via inference_baseline.py. This script only runs ESC + rebuttal baselines.
# If $QWEN25_BASELINE_VLSAFE / $QWEN25_BASELINE_POPE don't exist, the whole tier
# is skipped (with a clear message).
#
# We use the EXISTING Qwen2-VL-7B as the verifier rather than Gemma3-12B so we
# don't introduce a confounding verifier change — but you can override by
# setting --model_b. We keep it as is for the rebuttal: still gemma3-12b verifier.
# ── Tier 4: LLaVA-OneVision (newer backbone) ─────────────────────────────
# REQUIREMENT: Step 1 baseline neutral inference must have been run separately
# via inference_baseline.py for LLaVA-OneVision on VLSafe.
# ── Tier 4: Newer backbones (LLaVA-NeXT-Vicuna-7B + Qwen3-VL) ─────────────
# REQUIREMENT: Step 1 baseline neutral inference must have been run separately
# via inference_baseline.py for each new backbone on VLSafe.
LLAVA_NEXT_BASELINE_VLSAFE="${LLAVA_NEXT_BASELINE_VLSAFE:-/workspace/results/infer/llava_1_6_vicuna_7b/vlsafe_finding3/results_vlsafe_finding3_NEUTRAL.json}"
QWEN3_VL_BASELINE_VLSAFE="${QWEN3_VL_BASELINE_VLSAFE:-/workspace/results/infer/qwen3_vl_8b_instruct/vlsafe_finding3/results_vlsafe_finding3_NEUTRAL.json}"

T4_EXPERIMENTS=(
  # LLaVA-NeXT-Vicuna-7B × VLSafe × ESC only (baseline produced separately by inference_baseline.py)
  "T4_LLaVANeXT_VLSafe_emotion|llava|llava_1_6_vicuna_7b|$LLAVA_NEXT_BASELINE_VLSAFE|vlsafe|$PIPELINE_SAFETY|emotion||2"

  # Qwen3-VL-8B-Instruct × VLSafe × ESC only
  "T4_Qwen3VL_VLSafe_emotion|qwen3-vl-8b|qwen3_vl_8b_instruct|$QWEN3_VL_BASELINE_VLSAFE|vlsafe|$PIPELINE_SAFETY|emotion||2"
)

# ── Tier 5: Small-verifier ablation (R1-W3 distillation hypothesis) ────────
# LLaVA-1.5 + VLSafe + ESC (emotion, neg_low, multi2, 1 loop), swapping the verifier.
# Reuses the existing LLaVA VLSafe baseline (baseline doesn't use a verifier).
# Row format: label|model_a|model_a_short|baseline|benchmark|pipeline|prompt_source|verifier_override|multi
# Using multi=2 to match the paper's headline ESC setting (2 emotions concatenated).
T5_EXPERIMENTS=(
  "T5_LLaVA_VLSafe_v_gemma3_4b   |llava_1.5|llava_1_5_7b|$VLSAFE_LLAVA_BASELINE|vlsafe|$PIPELINE_SAFETY|emotion|gemma3-4b|2"
  "T5_LLaVA_VLSafe_v_qwen25vl_3b |llava_1.5|llava_1_5_7b|$VLSAFE_LLAVA_BASELINE|vlsafe|$PIPELINE_SAFETY|emotion|qwen2.5-vl-3b|2"
  "T5_LLaVA_VLSafe_v_internvl_2b |llava_1.5|llava_1_5_7b|$VLSAFE_LLAVA_BASELINE|vlsafe|$PIPELINE_SAFETY|emotion|internvl2.5-2b|2"
)

# ─── Build the command for one experiment ──────────────────────────────────
build_cmd() {
  local model_a=$1
  local baseline=$2
  local benchmark=$3
  local pipeline=$4
  local prompt_source=$5
  local verifier_override=${6:-}
  local multi_override=${7:-}

  # Verifier: per-experiment override beats global $VERIFIER
  local verifier="${verifier_override:-$VERIFIER}"

  # multiple_emotion: per-experiment override beats default 1
  local multi="${multi_override:-1}"

  # For prompt_source=emotion, default to fixed/negative_low to match the paper's
  # main ESC result. For all other sources, selection_type=random suffices.
  local sel_type="random"
  local extra=""
  if [[ "$prompt_source" == "emotion" ]]; then
    sel_type="fixed"
    extra="--quadrant negative_low"
  fi

  # --num_loops is only on the safety pipeline; VQA pipeline doesn't accept it.
  local loops_arg=""
  if [[ "$pipeline" == *"vqa_"* ]]; then
    loops_arg=""
  else
    loops_arg="--num_loops 1"
  fi

  echo "python3 $pipeline" \
       "--model_a_results $baseline" \
       "--model_a $model_a" \
       "--model_b $verifier" \
       "--benchmark $benchmark" \
       "--prompt_source $prompt_source" \
       "--selection_type $sel_type" \
       "$extra" \
       "--location start" \
       "--multiple_emotion $multi" \
       "--batch_size $BATCH_SIZE" \
       "$loops_arg"
}

# ─── Dispatch ─────────────────────────────────────────────────────────────
run_tier() {
  local tier_name=$1
  shift
  local experiments=("$@")
  echo ""
  echo "##########################################################################"
  echo "## $tier_name"
  echo "##########################################################################"
  for row in "${experiments[@]}"; do
    # Row format: label|model_a|model_a_short|baseline|benchmark|pipeline|prompt_source[|verifier_override[|multi]]
    IFS='|' read -r label model_a model_a_short baseline benchmark pipeline prompt_source verifier_override multi_override <<< "$row"
    label=$(echo "$label" | xargs)
    model_a=$(echo "$model_a" | xargs)
    model_a_short=$(echo "$model_a_short" | xargs)
    verifier_override=$(echo "${verifier_override:-}" | xargs)
    multi_override=$(echo "${multi_override:-}" | xargs)

    # ── Apply --filter regex if set ──
    if [[ -n "$FILTER" ]] && ! [[ "$label" =~ $FILTER ]]; then
      continue
    fi

    if [[ ! -f "$baseline" ]]; then
      echo "  ⚠ SKIP $label: baseline JSON not found at $baseline"
      continue
    fi

    cmd=$(build_cmd "$model_a" "$baseline" "$benchmark" "$pipeline" \
                    "$prompt_source" "$verifier_override" "$multi_override")

    # Resolve verifier short name for outdir (matches model_short_name() output)
    local verifier_short="$GEMMA_SHORT"
    if [[ -n "$verifier_override" ]]; then
      case "$verifier_override" in
        gemma3-4b)        verifier_short="gemma_3_4b_it" ;;
        gemma3-1b)        verifier_short="gemma_3_1b_it" ;;
        qwen2.5-vl-3b)    verifier_short="qwen2_5_vl_3b_instruct" ;;
        internvl2.5-2b)   verifier_short="internvl2_5_2b" ;;
        *)                verifier_short="$verifier_override" ;;
      esac
    fi

    # multi: defaults to 1 if not specified
    local multi="${multi_override:-1}"

    # Outdir construction depends on prompt_source:
    #   emotion -> fixed/negative_low/start/multiN
    #   others  -> random (no quadrant subpath)
    if [[ "$prompt_source" == "emotion" ]]; then
      outdir=$(expected_outdir "$model_a_short" "$verifier_short" "$benchmark" \
               "fixed" "negative_low" "start" "$multi" "$prompt_source")
    else
      outdir=$(expected_outdir "$model_a_short" "$verifier_short" "$benchmark" \
               "random" "" "" "" "$prompt_source")
    fi
    run_one "$label" "$cmd" "$outdir"
  done
}

case "$TIER" in
  1)   run_tier "TIER 1 — VLSafe baselines" "${T1_EXPERIMENTS[@]}" ;;
  2)   run_tier "TIER 2 — POPE baselines"   "${T2_EXPERIMENTS[@]}" ;;
  3)   run_tier "TIER 3 — MathVista"        "${T3_EXPERIMENTS[@]}" ;;
  4)   run_tier "TIER 4 — LLaVA-OneVision"  "${T4_EXPERIMENTS[@]}" ;;
  5)   run_tier "TIER 5 — Small verifiers"  "${T5_EXPERIMENTS[@]}" ;;
  all)
    run_tier "TIER 1 — VLSafe baselines" "${T1_EXPERIMENTS[@]}"
    run_tier "TIER 2 — POPE baselines"   "${T2_EXPERIMENTS[@]}"
    run_tier "TIER 3 — MathVista"        "${T3_EXPERIMENTS[@]}"
    run_tier "TIER 4 — LLaVA-OneVision"  "${T4_EXPERIMENTS[@]}"
    run_tier "TIER 5 — Small verifiers"  "${T5_EXPERIMENTS[@]}"
    ;;
  *) echo "Unknown tier: $TIER (choose: 1, 2, 3, 4, 5, all)"; exit 1 ;;
esac

echo ""
echo "=========================================================================="
echo "[$(date '+%H:%M:%S')]  ALL DONE"
echo "  Logs: $LOG_DIR"
echo "=========================================================================="