"""
debug_blink.py — Diagnose BLINK baseline vs method score discrepancy

Usage:
    # Diagnose baseline results only
    python debug_blink.py --baseline path/to/results_blink_full_baseline.json

    # Diagnose both baseline and method results together
    python debug_blink.py \
        --baseline path/to/results_blink_full_baseline.json \
        --method   path/to/method1_results_*.json

What this script checks:
    [A] Answer extraction audit          — how each extraction level fires
    [B] Baseline score inflation check   — Level 5 (\b word boundary) false-positive rate
    [C] Method pipeline routing audit    — judge/decide decisions and their correctness
    [D] Response identity check          — are final_response == response for LLaVA?
    [E] Correctness flip analysis        — was_correct→wrong and wrong→correct rates
    [F] GT answer sanity                 — distribution of ground truth labels
    [G] Sample-level diff report         — 20 examples where method diverges from baseline
"""

import json
import re
import argparse
from collections import defaultdict, Counter
from pathlib import Path


# ============================================================================
# Copy of eval_blink.py helpers (standalone, no dependency on project code)
# ============================================================================

ALL_SUBTASKS = [
    "Art_Style", "Counting", "Forensic_Detection", "Functional_Correspondence",
    "IQ_Test", "Jigsaw", "Multi-view_Reasoning", "Object_Localization",
    "Relative_Depth", "Relative_Reflectance", "Semantic_Correspondence",
    "Spatial_Relation", "Visual_Correspondence", "Visual_Similarity",
]

DEFAULT_NUM_CHOICES = {
    "Art_Style": 2, "Counting": 4, "Forensic_Detection": 4,
    "Functional_Correspondence": 4, "IQ_Test": 4, "Jigsaw": 2,
    "Multi-view_Reasoning": 2, "Object_Localization": 2,
    "Relative_Depth": 2, "Relative_Reflectance": 3,
    "Semantic_Correspondence": 4, "Spatial_Relation": 2,
    "Visual_Correspondence": 4, "Visual_Similarity": 2,
}


def _get_subtask(r):
    sample_id = r.get("id", "")
    for st in ALL_SUBTASKS:
        if st in sample_id:
            return st
    for field in ["subtask", "sub_task"]:
        val = r.get(field, "")
        if val and val in ALL_SUBTASKS:
            return val
    return "unknown"


def _parse_valid_choices(r):
    fq = r.get("full_question", "") or r.get("original_question", "")
    if fq:
        letters_found = sorted(set(re.findall(r'\(([A-Z])\)', fq)))
        if letters_found:
            return letters_found
    subtask = _get_subtask(r)
    n = DEFAULT_NUM_CHOICES.get(subtask, 4)
    return [chr(ord("A") + i) for i in range(n)]


def extract_answer_with_trace(response: str, valid_choices: list):
    """
    Same logic as eval_blink.py extract_answer(), but returns
    (answer, level) so we can audit which level fired.

    Levels:
        1a = exact strip match
        1b = regex ^(X)$
        2  = "answer is X" pattern
        3  = parenthesized (X)
        4  = first capital at start
        5  = word-boundary scan  ← most likely source of inflation
        F  = FAILED
    """
    if not response or not response.strip():
        return "FAILED", "F"

    valid_set = set(valid_choices)
    response = response.strip()

    # Level 1a
    cleaned = response.strip().upper().strip("().- \t\n")
    if cleaned in valid_set:
        return cleaned, "1a"

    # Level 1b
    match = re.match(r'^\(?([A-Z])\)?[\.\):\s]*$', response.strip(), re.IGNORECASE)
    if match and match.group(1).upper() in valid_set:
        return match.group(1).upper(), "1b"

    response_upper = response.upper()

    # Level 2
    patterns = [
        r'(?:the\s+)?answer\s+is\s*[:\s]*\(?([A-Z])\)?',
        r'(?:I\s+)?(?:choose|select|pick)\s*[:\s]*\(?([A-Z])\)?',
        r'(?:correct\s+)?(?:answer|option|choice)\s*[:\s]*\(?([A-Z])\)?',
        r'(?:it\s+is|it\'s|should\s+be)\s+\(?([A-Z])\)?',
        r'\b([A-Z])\s+is\s+(?:the\s+)?(?:correct|right|best)\s+(?:answer|option|choice)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response_upper)
        if match and match.group(1) in valid_set:
            return match.group(1), "2"

    # Level 3
    paren_matches = re.findall(r'\(([A-Z])\)', response_upper)
    for m in paren_matches:
        if m in valid_set:
            return m, "3"

    # Level 4
    match = re.match(r'^([A-Z])[\.\)\s]', response.strip())
    if match and match.group(1).upper() in valid_set:
        return match.group(1).upper(), "4"

    # Level 5 — word boundary scan (most likely inflator)
    for letter in valid_choices:
        if re.search(r'\b' + letter + r'\b', response_upper):
            return letter, "5"

    return "FAILED", "F"


# ============================================================================
# SECTION A+B: Extraction audit on baseline results
# ============================================================================

def audit_extraction(results: list, response_field: str = "response", label: str = ""):
    print(f"\n{'='*72}")
    print(f"[A+B] EXTRACTION AUDIT — {label or response_field}")
    print(f"{'='*72}")

    level_counts = Counter()
    level_correct = Counter()
    level_incorrect = Counter()
    level_5_examples = []

    total = len(results)
    correct = 0
    failed = 0

    for r in results:
        resp = r.get(response_field, r.get("response", ""))
        gt = str(r.get("gt_answer", "")).upper().strip()
        gt_match = re.search(r'\(?([A-Z])\)?', gt)
        if gt_match:
            gt = gt_match.group(1)

        valid_choices = _parse_valid_choices(r)
        pred, level = extract_answer_with_trace(resp, valid_choices)

        level_counts[level] += 1
        is_correct = (pred == gt) if pred != "FAILED" else False
        if is_correct:
            correct += 1
            level_correct[level] += 1
        else:
            level_incorrect[level] += 1

        if pred == "FAILED":
            failed += 1

        # Collect Level 5 examples for manual review
        if level == "5" and len(level_5_examples) < 20:
            level_5_examples.append({
                "id": r.get("id", "?"),
                "subtask": _get_subtask(r),
                "gt": gt,
                "pred": pred,
                "correct": is_correct,
                "valid_choices": valid_choices,
                "response_snippet": resp[:300].replace("\n", " "),
            })

    micro_acc = correct / total * 100 if total else 0
    print(f"  Total: {total}  |  Correct: {correct}  |  Failed: {failed}")
    print(f"  Micro accuracy: {micro_acc:.2f}%")

    print(f"\n  {'Level':<6} {'Count':>7} {'Correct':>8} {'Incorrect':>10} {'Accuracy':>10}  Description")
    print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*10} {'-'*10}  {'-'*30}")

    level_labels = {
        "1a": "Exact strip match",
        "1b": "Regex ^(X)$ match",
        "2":  '"answer is X" pattern',
        "3":  "Parenthesized (X)",
        "4":  "First capital at start",
        "5":  "⚠️  Word-boundary scan  ← INFLATION RISK",
        "F":  "FAILED (no extraction)",
    }
    for lvl in ["1a", "1b", "2", "3", "4", "5", "F"]:
        cnt = level_counts[lvl]
        if cnt == 0:
            continue
        c = level_correct[lvl]
        ic = level_incorrect[lvl]
        acc = c / cnt * 100 if cnt > 0 else 0
        print(f"  {lvl:<6} {cnt:>7,} {c:>8,} {ic:>10,} {acc:>9.1f}%  {level_labels[lvl]}")

    if level_5_examples:
        print(f"\n  [B] Level-5 examples (word-boundary scan, first {len(level_5_examples)}):")
        print(f"  {'ID':<35} {'Sub':<22} {'GT':>3} {'Pred':>5} {'OK':>4}  Response snippet")
        print(f"  {'-'*35} {'-'*22} {'-'*3} {'-'*5} {'-'*4}  {'-'*40}")
        for ex in level_5_examples:
            ok = "✓" if ex["correct"] else "✗"
            print(f"  {ex['id']:<35} {ex['subtask']:<22} {ex['gt']:>3} {ex['pred']:>5} {ok:>4}  {ex['response_snippet'][:60]}")

    return level_counts, level_correct


# ============================================================================
# SECTION C: Method pipeline routing audit
# ============================================================================

def audit_method_routing(method_results: list, baseline_map: dict = None):
    print(f"\n{'='*72}")
    print(f"[C] METHOD PIPELINE ROUTING AUDIT")
    print(f"{'='*72}")

    total = len(method_results)
    judged_correct = sum(1 for r in method_results if r.get("judge_is_correct", False))
    judged_incorrect = total - judged_correct
    was_regen = sum(1 for r in method_results if r.get("was_regenerated", False))

    print(f"  Total samples:             {total:,}")
    print(f"  Step 2 — judged correct:   {judged_correct:,}  ({judged_correct/total*100:.1f}%)")
    print(f"  Step 2 — judged incorrect: {judged_incorrect:,}  ({judged_incorrect/total*100:.1f}%)")
    print(f"  Step 5 — chose regen (B):  {was_regen:,}  ({was_regen/total*100:.1f}%)")
    print(f"  Step 5 — kept original (A):{total - was_regen:,}  ({(total-was_regen)/total*100:.1f}%)")

    # Safe samples: judged correct → final_response = response (no regen)
    safe = [r for r in method_results if not r.get("was_regenerated") and r.get("judge_is_correct")]
    # Unsafe but kept original at step 5
    kept_orig = [r for r in method_results if not r.get("was_regenerated") and not r.get("judge_is_correct", True)]
    regen_chosen = [r for r in method_results if r.get("was_regenerated")]

    print(f"\n  Routing breakdown:")
    print(f"    Safe (judge=correct, kept)       : {len(safe):,}")
    print(f"    Unsafe but kept original (step5) : {len(kept_orig):,}")
    print(f"    Unsafe + regen chosen (step5=B)  : {len(regen_chosen):,}")

    if baseline_map:
        print(f"\n  [C2] Routing accuracy cross-analysis:")
        print(f"  (Does the judge correctly identify truly correct baseline answers?)")

        # Among samples judged correct by step2
        judged_correct_list = [r for r in method_results if r.get("judge_is_correct", False)]
        judged_incorrect_list = [r for r in method_results if not r.get("judge_is_correct", False)]

        def baseline_acc(sample_list):
            correct = 0
            total = len(sample_list)
            for r in sample_list:
                sid = r.get("id")
                base = baseline_map.get(sid)
                if base is None:
                    continue
                valid = _parse_valid_choices(r)
                pred, _ = extract_answer_with_trace(base.get("response", ""), valid)
                gt = str(r.get("gt_answer", "")).upper().strip()
                gt_m = re.search(r'\(?([A-Z])\)?', gt)
                if gt_m:
                    gt = gt_m.group(1)
                if pred == gt:
                    correct += 1
            return correct, total

        c, t = baseline_acc(judged_correct_list)
        print(f"    Judged 'correct' by step2  → actually correct in baseline: {c}/{t} = {c/t*100:.1f}%")

        c2, t2 = baseline_acc(judged_incorrect_list)
        print(f"    Judged 'incorrect' by step2 → actually correct in baseline: {c2}/{t2} = {c2/t2*100:.1f}%")
        print()
        if t2 > 0 and c2 / t2 > 0.3:
            print(f"  ⚠️  WARNING: Judge is routing {c2/t2*100:.1f}% of CORRECT answers to regeneration!")
            print(f"      This is a primary cause of the score drop.")

    # decide_choice distribution
    decide_choices = Counter(r.get("decide_choice") for r in method_results if r.get("decide_choice") is not None)
    if decide_choices:
        print(f"\n  Step 5 decide_choice distribution: {dict(decide_choices)}")
        default_b = decide_choices.get("B", 0)
        if default_b / total > 0.5:
            print(f"  ⚠️  WARNING: {default_b/total*100:.1f}% of decisions went to B.")
            print(f"      parse_ab_choice() defaults to 'B' on ambiguous output — check judge responses.")


# ============================================================================
# SECTION D: Response identity check
# ============================================================================

def audit_response_identity(method_results: list, baseline_map: dict):
    print(f"\n{'='*72}")
    print(f"[D] RESPONSE IDENTITY CHECK (LLaVA: should be near-identical)")
    print(f"{'='*72}")

    total = len(method_results)
    identical_to_baseline = 0
    final_eq_original = 0
    final_eq_regen = 0
    final_missing = 0
    response_missing = 0
    mismatched_examples = []

    for r in method_results:
        sid = r.get("id")
        base = baseline_map.get(sid)

        resp_orig = r.get("response_original") or r.get("response", "")
        resp_final = r.get("final_response", "")
        resp_regen = r.get("response_regen", "")

        if not resp_final:
            final_missing += 1
        if not resp_orig:
            response_missing += 1

        if resp_final == resp_orig:
            final_eq_original += 1
        if resp_regen and resp_final == resp_regen:
            final_eq_regen += 1

        if base and resp_orig.strip() == base.get("response", "").strip():
            identical_to_baseline += 1
        elif base and len(mismatched_examples) < 10:
            mismatched_examples.append({
                "id": sid,
                "baseline_response": base.get("response", "")[:150],
                "method_response_original": resp_orig[:150],
            })

    print(f"  Total method samples: {total:,}")
    print(f"  response_original == baseline response: {identical_to_baseline:,} / {total} ({identical_to_baseline/total*100:.1f}%)")
    print(f"  final_response == response_original:    {final_eq_original:,} / {total} ({final_eq_original/total*100:.1f}%)")
    print(f"  final_response == response_regen:       {final_eq_regen:,} / {total}")
    print(f"  final_response missing:                 {final_missing:,}")
    print(f"  response_original missing:              {response_missing:,}")

    if identical_to_baseline < total * 0.95:
        print(f"\n  ⚠️  WARNING: Only {identical_to_baseline/total*100:.1f}% of method samples have the same")
        print(f"      original response as the baseline. This suggests loading/ID mismatch.")
        if mismatched_examples:
            print(f"\n  Mismatch examples (first {len(mismatched_examples)}):")
            for ex in mismatched_examples:
                print(f"    ID: {ex['id']}")
                print(f"      Baseline:  {ex['baseline_response']}")
                print(f"      Method:    {ex['method_response_original']}")
                print()


# ============================================================================
# SECTION E: Correctness flip analysis
# ============================================================================

def audit_flips(method_results: list, baseline_map: dict):
    print(f"\n{'='*72}")
    print(f"[E] CORRECTNESS FLIP ANALYSIS")
    print(f"{'='*72}")

    correct_to_wrong = []
    wrong_to_correct = []
    correct_stayed = 0
    wrong_stayed = 0

    for r in method_results:
        sid = r.get("id")
        base = baseline_map.get(sid)
        if base is None:
            continue

        gt = str(r.get("gt_answer", "")).upper().strip()
        gt_m = re.search(r'\(?([A-Z])\)?', gt)
        if gt_m:
            gt = gt_m.group(1)

        valid = _parse_valid_choices(r)

        base_pred, _ = extract_answer_with_trace(base.get("response", ""), valid)
        method_pred, _ = extract_answer_with_trace(r.get("final_response", ""), valid)

        was_correct = (base_pred == gt)
        now_correct = (method_pred == gt)

        if was_correct and not now_correct:
            correct_to_wrong.append(r)
        elif not was_correct and now_correct:
            wrong_to_correct.append(r)
        elif was_correct and now_correct:
            correct_stayed += 1
        else:
            wrong_stayed += 1

    total_matched = correct_to_wrong + wrong_to_correct
    n = correct_stayed + len(correct_to_wrong) + len(wrong_to_correct) + wrong_stayed
    print(f"  Matched samples: {n:,}")
    print(f"  ✅→✅ Stayed correct:   {correct_stayed:,}")
    print(f"  ✅→❌ CORRECT→WRONG:    {len(correct_to_wrong):,}  ← PRIMARY BUG SIGNAL")
    print(f"  ❌→✅ WRONG→CORRECT:    {len(wrong_to_correct):,}")
    print(f"  ❌→❌ Stayed wrong:     {wrong_stayed:,}")

    net = len(wrong_to_correct) - len(correct_to_wrong)
    print(f"\n  Net change: {net:+d}  ({'improvement' if net >= 0 else 'REGRESSION'})")

    if correct_to_wrong:
        print(f"\n  [E1] Samples flipped CORRECT→WRONG (first 15):")
        print(f"  {'ID':<35} {'GT':>3} {'Base':>5} {'Final':>6}  {'Regen':>6}  {'Judge':>6}  {'Decide':>7}")
        print(f"  {'-'*35} {'-'*3} {'-'*5} {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}")
        for r in correct_to_wrong[:15]:
            valid = _parse_valid_choices(r)
            gt = str(r.get("gt_answer", "")).upper().strip()
            gt_m = re.search(r'\(?([A-Z])\)?', gt)
            if gt_m:
                gt = gt_m.group(1)
            base = baseline_map.get(r.get("id"))
            base_pred, _ = extract_answer_with_trace(base.get("response", ""), valid) if base else ("?", "?")
            final_pred, _ = extract_answer_with_trace(r.get("final_response", ""), valid)
            regen_pred, _ = extract_answer_with_trace(r.get("response_regen", "") or "", valid)
            judge = "correct" if r.get("judge_is_correct") else "wrong"
            decide = r.get("decide_choice", "N/A")
            print(f"  {r.get('id','?'):<35} {gt:>3} {base_pred:>5} {final_pred:>6}  {regen_pred:>6}  {judge:>6}  {decide:>7}")

    if wrong_to_correct:
        print(f"\n  [E2] Samples flipped WRONG→CORRECT (first 10):")
        for r in wrong_to_correct[:10]:
            valid = _parse_valid_choices(r)
            gt = str(r.get("gt_answer", "")).upper().strip()
            gt_m = re.search(r'\(?([A-Z])\)?', gt)
            if gt_m:
                gt = gt_m.group(1)
            final_pred, _ = extract_answer_with_trace(r.get("final_response", ""), valid)
            print(f"    {r.get('id','?'):<40}  GT={gt}  Final={final_pred}")


# ============================================================================
# SECTION F: GT answer distribution
# ============================================================================

def audit_gt_distribution(results: list, label: str = ""):
    print(f"\n{'='*72}")
    print(f"[F] GT ANSWER DISTRIBUTION — {label}")
    print(f"{'='*72}")
    gt_dist = Counter()
    for r in results:
        gt = str(r.get("gt_answer", "")).upper().strip()
        gt_m = re.search(r'\(?([A-Z])\)?', gt)
        if gt_m:
            gt = gt_m.group(1)
        gt_dist[gt] += 1

    total = sum(gt_dist.values())
    for k, v in sorted(gt_dist.items()):
        pct = v / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {k}: {v:>5,}  ({pct:5.1f}%)  {bar}")

    if len(gt_dist) > 0:
        expected = total / len(gt_dist)
        max_v = max(gt_dist.values())
        if max_v / total > 0.4:
            dominant = max(gt_dist, key=gt_dist.get)
            print(f"\n  ⚠️  WARNING: Label '{dominant}' appears in {max_v/total*100:.1f}% of samples — possible label bias.")


# ============================================================================
# SECTION G: Sample-level diff report
# ============================================================================

def audit_sample_diffs(method_results: list, baseline_map: dict, n: int = 20):
    print(f"\n{'='*72}")
    print(f"[G] SAMPLE-LEVEL DIFF REPORT (first {n} where final_response ≠ baseline response)")
    print(f"{'='*72}")

    shown = 0
    for r in method_results:
        if shown >= n:
            break
        sid = r.get("id")
        base = baseline_map.get(sid)
        if base is None:
            continue

        base_resp = base.get("response", "")
        final_resp = r.get("final_response", "")
        if base_resp.strip() == final_resp.strip():
            continue

        gt = str(r.get("gt_answer", "")).upper().strip()
        gt_m = re.search(r'\(?([A-Z])\)?', gt)
        if gt_m:
            gt = gt_m.group(1)
        valid = _parse_valid_choices(r)

        base_pred, base_lvl = extract_answer_with_trace(base_resp, valid)
        regen_pred, regen_lvl = extract_answer_with_trace(r.get("response_regen", "") or "", valid)
        final_pred, final_lvl = extract_answer_with_trace(final_resp, valid)

        print(f"\n  ── {sid}  [GT={gt}] ──")
        print(f"    Judge verdict:   is_correct={r.get('judge_is_correct','?')}  raw='{str(r.get('judge_verdict_raw',''))[:60]}'")
        print(f"    Decide choice:   {r.get('decide_choice','?')}  raw='{str(r.get('decide_verdict_raw',''))[:60]}'")
        print(f"    Baseline resp:   [{base_lvl}→{base_pred}] {base_resp[:120].replace(chr(10),' ')}")
        print(f"    Regen resp:      [{regen_lvl}→{regen_pred}] {str(r.get('response_regen',''))[:120].replace(chr(10),' ')}")
        print(f"    Final resp:      [{final_lvl}→{final_pred}] {final_resp[:120].replace(chr(10),' ')}")
        print(f"    Emotion:         {str(r.get('emotion_prompt_concat',''))[:80]}")
        shown += 1

    if shown == 0:
        print("  ✅ No diverging samples found — final_response matches baseline for all matched IDs.")


# ============================================================================
# MAIN
# ============================================================================

def load_results(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    print(f"  Loaded {len(data):,} samples from {Path(path).name}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Debug BLINK baseline vs method score discrepancy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--baseline", required=True,
                        help="Path to baseline inference results JSON")
    parser.add_argument("--method", default=None,
                        help="Path to method1 results JSON (optional)")
    parser.add_argument("--no_level5_examples", action="store_true",
                        help="Skip printing Level-5 extraction examples")
    args = parser.parse_args()

    print("\n" + "="*72)
    print("BLINK DEBUG REPORT")
    print("="*72)

    # Load baseline
    print("\nLoading baseline results...")
    baseline = load_results(args.baseline)
    baseline_map = {r.get("id"): r for r in baseline}

    # Section F — GT distribution
    audit_gt_distribution(baseline, label="baseline")

    # Section A+B — Extraction audit on baseline
    audit_extraction(baseline, response_field="response", label="Baseline (response field)")

    if args.method:
        print(f"\nLoading method results...")
        method = load_results(args.method)
        method_map = {r.get("id"): r for r in method}

        # Section A+B — Extraction audit on method final_response
        audit_extraction(method, response_field="final_response", label="Method (final_response field)")

        # Section C — Pipeline routing
        audit_method_routing(method, baseline_map=baseline_map)

        # Section D — Response identity
        audit_response_identity(method, baseline_map)

        # Section E — Flips
        audit_flips(method, baseline_map)

        # Section G — Sample diffs
        audit_sample_diffs(method, baseline_map, n=20)

    print(f"\n{'='*72}")
    print("DEBUG COMPLETE")
    print("="*72)


if __name__ == "__main__":
    main()