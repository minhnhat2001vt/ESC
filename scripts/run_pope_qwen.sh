#!/bin/bash
# ==============================================================================
# POPE Qwen2-VL — 4 remaining conditions + auto-evaluation
# ==============================================================================

SCRIPT="python3 /workspace/scripts/method/vqa_inference_method1_rebut.py"
EVAL_SCRIPT="python3 /workspace/scripts/eval/eval_pope.py"
QWEN_BASELINE="/workspace/results/infer/qwen2_vl_7b/vlsafe_finding3/results_vlsafe_finding3_NEUTRAL.json"
RESULTS_BASE="/workspace/results/method1"

BS=24
LOG_DIR="/workspace/logs/pope_t2"
mkdir -p $LOG_DIR

EMO_FLAGS="--selection_type fixed --quadrant negative_low --location start --multiple_emotion 1"
OTHER_FLAGS="--selection_type random --location start --multiple_emotion 1"

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
    local psource_tag=$2
    local is_fixed=$3
    shift 3

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
        result_dir="$RESULTS_BASE/${QWEN_SHORT}__${VERIFIER_SHORT}${psource_tag}/pope/fixed/negative_low/start/multi1"
    else
        result_dir="$RESULTS_BASE/${QWEN_SHORT}__${VERIFIER_SHORT}${psource_tag}/pope/random"
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
echo "POPE Qwen2-VL — 4 conditions + eval"
echo "Start time: $(date)"
echo "============================================================"

run_and_eval "5of8_qwen_emotion"       ""                "fixed"  --model_a_results $QWEN_BASELINE --model_a qwen2-vl --model_b gemma3-12b --benchmark pope --batch_size $BS --prompt_source emotion $EMO_FLAGS
run_and_eval "6of8_qwen_neutral"       "_neutral"        "random" --model_a_results $QWEN_BASELINE --model_a qwen2-vl --model_b gemma3-12b --benchmark pope --batch_size $BS --prompt_source neutral $OTHER_FLAGS
run_and_eval "7of8_qwen_psychological" "_psychological"  "random" --model_a_results $QWEN_BASELINE --model_a qwen2-vl --model_b gemma3-12b --benchmark pope --batch_size $BS --prompt_source psychological $OTHER_FLAGS
run_and_eval "8of8_qwen_none"          "_none"           "random" --model_a_results $QWEN_BASELINE --model_a qwen2-vl --model_b gemma3-12b --benchmark pope --batch_size $BS --prompt_source none $OTHER_FLAGS

echo ""
echo "============================================================"
echo "ALL 4 QWEN RUNS + EVALS COMPLETE"
echo "End time: $(date)"
echo "============================================================"