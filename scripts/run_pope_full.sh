#!/bin/bash
# ==============================================================================
# POPE — 4 conditions × 2 models + auto-evaluation (FULL RERUN)
# ==============================================================================

SCRIPT="python3 /workspace/scripts/method/vqa_inference_method1_rebut.py"
EVAL_SCRIPT="/workspace/scripts/eval/eval_pope.py"

LLAVA_BASELINE="/workspace/results/infer/llava_1_5_7b/pope_baseline/results_pope_full_baseline.json"
QWEN_BASELINE="/workspace/results/infer/qwen2_vl_7b/pope_baseline/results_pope_full_baseline.json"

RESULTS_BASE="/workspace/results/method1"
BS=38
LOG_DIR="/workspace/logs/pope_t2_3"
mkdir -p $LOG_DIR

EMO_FLAGS="--selection_type fixed --quadrant negative_low --location start --multiple_emotion 1"
OTHER_FLAGS="--selection_type random --location start --multiple_emotion 1"

LLAVA_SHORT="llava_1_5_7b"
QWEN_SHORT="qwen2_vl_7b_instruct"
VERIFIER_SHORT="gemma_3_12b_it"

find_result_file() {
    local dir=$1
    if [ -d "$dir" ]; then
        ls -t "$dir"/method1_results_*.json 2>/dev/null | head -1
    fi
}

run_and_eval() {
    local label=$1
    local model_short=$2
    local psource_tag=$3
    local is_fixed=$4
    shift 4

    local logfile="$LOG_DIR/${label}.log"

    echo ""
    echo "============================================================"
    echo ">>> $label"
    echo ">>> Start: $(date)"
    echo ">>> Log:   $logfile"
    echo "============================================================"

    $SCRIPT "$@" 2>&1 | tee "$logfile"
    local status=${PIPESTATUS[0]}

    if [ $status -ne 0 ]; then
        echo ">>> $label INFERENCE FAILED with exit code $status ($(date))"
        echo ""
        return
    fi

    local result_dir
    if [ "$is_fixed" = "fixed" ]; then
        result_dir="$RESULTS_BASE/${model_short}__${VERIFIER_SHORT}${psource_tag}/pope/fixed/negative_low/start/multi1"
    else
        result_dir="$RESULTS_BASE/${model_short}__${VERIFIER_SHORT}${psource_tag}/pope/random"
    fi

    local result_file=$(find_result_file "$result_dir")

    if [ -z "$result_file" ]; then
        echo ">>> $label: Could not find result file in $result_dir"
    else
        echo ">>> Evaluating: $result_file"
        $EVAL_SCRIPT --result_file "$result_file" 2>&1 | tee -a "$logfile"
    fi
    echo ">>> $label DONE ($(date))"
    echo ""
}

echo "============================================================"
echo "POPE — 4 conditions × 2 models + eval (FULL RERUN)"
echo "Start time: $(date)"
echo "Logs: $LOG_DIR"
echo "============================================================"

# ── LLaVA-1.5-7B ──
run_and_eval "1of8_llava_emotion"       "$LLAVA_SHORT" ""                "fixed"  --model_a_results $LLAVA_BASELINE --model_a llava_1.5 --model_b gemma3-12b --benchmark pope --batch_size $BS --prompt_source emotion $EMO_FLAGS
run_and_eval "2of8_llava_neutral"       "$LLAVA_SHORT" "_neutral"        "random" --model_a_results $LLAVA_BASELINE --model_a llava_1.5 --model_b gemma3-12b --benchmark pope --batch_size $BS --prompt_source neutral $OTHER_FLAGS
run_and_eval "3of8_llava_psychological" "$LLAVA_SHORT" "_psychological"  "random" --model_a_results $LLAVA_BASELINE --model_a llava_1.5 --model_b gemma3-12b --benchmark pope --batch_size $BS --prompt_source psychological $OTHER_FLAGS
run_and_eval "4of8_llava_none"          "$LLAVA_SHORT" "_none"           "random" --model_a_results $LLAVA_BASELINE --model_a llava_1.5 --model_b gemma3-12b --benchmark pope --batch_size $BS --prompt_source none $OTHER_FLAGS

# ── Qwen2-VL-7B ──
run_and_eval "5of8_qwen_emotion"        "$QWEN_SHORT" ""                "fixed"  --model_a_results $QWEN_BASELINE --model_a qwen2-vl --model_b gemma3-12b --benchmark pope --batch_size $BS --prompt_source emotion $EMO_FLAGS
run_and_eval "6of8_qwen_neutral"        "$QWEN_SHORT" "_neutral"        "random" --model_a_results $QWEN_BASELINE --model_a qwen2-vl --model_b gemma3-12b --benchmark pope --batch_size $BS --prompt_source neutral $OTHER_FLAGS
run_and_eval "7of8_qwen_psychological"  "$QWEN_SHORT" "_psychological"  "random" --model_a_results $QWEN_BASELINE --model_a qwen2-vl --model_b gemma3-12b --benchmark pope --batch_size $BS --prompt_source psychological $OTHER_FLAGS
run_and_eval "8of8_qwen_none"           "$QWEN_SHORT" "_none"           "random" --model_a_results $QWEN_BASELINE --model_a qwen2-vl --model_b gemma3-12b --benchmark pope --batch_size $BS --prompt_source none $OTHER_FLAGS

echo ""
echo "============================================================"
echo "POPE — ALL 8 RUNS + EVALS COMPLETE"
echo "End time: $(date)"
echo "============================================================"
echo ""
echo "Logs: $LOG_DIR"
ls -lh $LOG_DIR