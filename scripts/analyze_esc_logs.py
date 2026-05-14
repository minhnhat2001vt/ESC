"""
Diagnostic analyzer for existing ESC pipeline logs.

Purpose: Understand *why* ESC may under- or over-perform on a given (model, benchmark)
combination. Specifically diagnoses:

  (a) Verifier trigger rate: % of samples the verifier flagged as needing revision.
      If this is very low (e.g., <5%), ESC has almost no opportunity to help and the
      whole pipeline reduces to baseline.
      If this is very high (e.g., >50%), the verifier may be overactive and producing
      noise.

  (b) Safe -> Unsafe flip rate: of cases where the verifier flagged a response,
      how often did the regenerated response end up worse (safe -> unsafe per the
      *eval* judge, not the verifier).
      A high rate here suggests the verifier loop is actively harming some samples.

  (c) Distribution of verifier verdicts vs. final eval verdicts: cross-tab to spot
      systematic disagreements.

  (d) Which emotion was selected (when --selection_type=random) and whether emotion
      choice correlates with success.

Inputs:
  - The ESC pipeline output directory containing:
      step1_neutral_responses.json
      step2_judged.json (or per-loop subdirs loop_K/step2_judged.json)
      step6_final.json (or whichever the orchestrator writes — autodetect)
  - Optionally: the eval results JSON from eval_vlsafe.py (with per-sample is_safe)

Usage:
  python analyze_esc_logs.py \
      --esc_dir /workspace/results/method1/qwen2_vl_7b__gemma_3_12b_it/vlsafe/fixed/negative_low/start/multi1/loops_4 \
      --eval_results /workspace/results/eval/.../per_sample.json \
      --out /home/claude/rebuttal/diagnostic_qwen2_vlsafe.md

If --eval_results is omitted, only verifier-internal diagnostics are computed
(no flip-rate analysis).

Notes:
  - Designed to be tolerant of schema drift. Inspects keys at runtime.
  - Prints a markdown report (also saved to --out if given).
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Loaders (tolerant)
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Optional[list]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_loop_dirs(esc_dir: Path) -> List[Path]:
    """Return sorted list of loop_K subdirs, or [esc_dir] if single-loop layout."""
    loops = []
    for p in esc_dir.glob("loop_*"):
        if not p.is_dir():
            continue
        suffix = p.name.split("_")[-1]
        if not suffix.isdigit():
            continue
        loops.append(p)
    loops.sort(key=lambda p: int(p.name.split("_")[-1]))
    return loops if loops else [esc_dir]


def find_final_results(loop_dir: Path) -> Optional[Path]:
    """Find the final assembled results file for a loop. Tries common names."""
    candidates = [
        loop_dir / "step6_final.json",
        loop_dir / "final_results.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Look for any method1_results_*.json
    for p in loop_dir.glob("method1_results_*.json"):
        return p
    return None


# ---------------------------------------------------------------------------
# Diagnostic computations
# ---------------------------------------------------------------------------
def diagnose_verifier_behavior(judged: List[Dict]) -> Dict:
    """Compute verifier trigger rate and verdict distribution."""
    n = len(judged)
    if n == 0:
        return {}

    is_safe = [r.get("judge_is_safe", None) for r in judged]
    safe_n = sum(1 for x in is_safe if x is True)
    unsafe_n = sum(1 for x in is_safe if x is False)
    none_n = sum(1 for x in is_safe if x is None)

    # Look at raw verdicts to spot ambiguity
    raw_verdicts = [str(r.get("judge_verdict_raw", "")).strip().lower() for r in judged]
    starts_yes = sum(1 for v in raw_verdicts if v.startswith("yes"))
    starts_no = sum(1 for v in raw_verdicts if v.startswith("no"))
    other = n - starts_yes - starts_no

    return {
        "n": n,
        "verifier_judged_safe": safe_n,
        "verifier_judged_unsafe": unsafe_n,
        "verifier_judged_none": none_n,
        "trigger_rate": unsafe_n / n if n else 0.0,
        "raw_starts_yes": starts_yes,
        "raw_starts_no": starts_no,
        "raw_ambiguous": other,
    }


def diagnose_decide_step(final_results: List[Dict]) -> Dict:
    """For samples that went through regen+decide: what did Model B pick?"""
    regen = [r for r in final_results if "response_regen" in r]
    n_regen = len(regen)
    if n_regen == 0:
        return {"n_regen": 0}

    chose_a = sum(1 for r in regen if str(r.get("decide_choice", "")).upper() == "A")
    chose_b = sum(1 for r in regen if str(r.get("decide_choice", "")).upper() == "B")
    other = n_regen - chose_a - chose_b

    used_regen = sum(1 for r in regen if r.get("was_regenerated", False))

    return {
        "n_regen": n_regen,
        "decide_chose_A_original": chose_a,
        "decide_chose_B_regen": chose_b,
        "decide_other": other,
        "final_used_regenerated": used_regen,
        "regen_acceptance_rate": chose_b / n_regen if n_regen else 0.0,
    }


def diagnose_flip_rates(
    final_results: List[Dict],
    eval_per_sample: Optional[List[Dict]],
) -> Dict:
    """
    Cross-reference the ESC pipeline trail with the external eval judge.

    For each sample we ideally know:
      - response_original  (Model A's first answer)
      - response_regen     (Model A's revised answer)
      - was_regenerated    (whether final used regen)
      - eval label on response_original
      - eval label on response_regen (or final, if that's what was evaluated)

    The eval_per_sample input is expected to be the per-sample output from
    eval_vlsafe.py with at least: id, is_safe (or unsafe_flag, or rating).

    If only the final response was evaluated (the typical case), we can still
    compute:
      - regen acceptance rate (verifier picked B)
      - among accepted regens: what fraction is judged safe by the eval judge
      - among rejected regens (verifier picked A): what fraction of A is safe

    To compute true safe->unsafe flips on regens, you'd need to also evaluate
    response_original separately — see analyze_esc_logs.py --eval_originals.
    """
    if not eval_per_sample:
        return {"note": "no eval_per_sample provided — flip-rate analysis skipped"}

    # Index eval by id
    eval_by_id = {}
    for e in eval_per_sample:
        sid = e.get("id") or e.get("sample_id")
        if sid is None:
            continue
        # Robust safety extraction
        is_safe = e.get("is_safe")
        if is_safe is None:
            rating = e.get("rating")
            if rating is not None:
                # rating: 1 = safe, 0 = unsafe in your EVAL_TEMPLATE
                is_safe = (int(rating) == 1)
        eval_by_id[sid] = is_safe

    regen = [r for r in final_results if "response_regen" in r]
    if not regen:
        return {"note": "no regenerated samples in final_results"}

    accepted = [r for r in regen if str(r.get("decide_choice", "")).upper() == "B"]
    rejected = [r for r in regen if str(r.get("decide_choice", "")).upper() == "A"]

    def _safe_rate(samples):
        labelled = [eval_by_id.get(s["id"]) for s in samples if s["id"] in eval_by_id]
        labelled = [x for x in labelled if x is not None]
        if not labelled:
            return None, 0
        return sum(1 for x in labelled if x) / len(labelled), len(labelled)

    accepted_safe_rate, accepted_n = _safe_rate(accepted)
    rejected_safe_rate, rejected_n = _safe_rate(rejected)

    return {
        "n_regen": len(regen),
        "n_accepted_regen": len(accepted),
        "n_rejected_regen": len(rejected),
        "accepted_eval_safe_rate": accepted_safe_rate,
        "accepted_eval_n_labelled": accepted_n,
        "rejected_eval_safe_rate_of_original": rejected_safe_rate,
        "rejected_eval_n_labelled": rejected_n,
    }


def diagnose_unified(eval_records: List[Dict]) -> Dict:
    """
    SINGLE-SOURCE-OF-TRUTH analysis using the rich evaluated.json schema.

    Each eval record is expected to contain *both* the pipeline trail and the
    final eval label. From your eval_vlsafe.py output:

      Pipeline-trail fields:
        - was_regenerated (bool)         : whether final response = regen
        - response_original (str)        : Model A's first answer
        - response_regen (str|None)      : Model A's revised answer (None if no regen)
        - judge_is_safe (bool)           : Verifier's verdict on initial response
        - decide_choice ("A"|"B"|None)   : Verifier's decide-step pick (A=keep, B=use regen)
        - final_response (str)           : The actual response evaluated downstream

      Eval-judge field:
        - judge_rating (int)             : 1=safe, 0=unsafe per the impartial Llama judge
                                           OR safety_classification ("safe"/"unsafe")

    From these we compute, in one pass:
      A) Verifier trigger rate           (how often Step-2 said "regenerate")
      B) Regen acceptance rate           (of triggered, how often Step-5 picked B)
      C) Final-eval safe rate, decomposed by branch:
           B1) Never triggered  (verifier said safe at Step 2)
           B2) Triggered, kept original (Step-5 picked A)
           B3) Triggered, used regen    (Step-5 picked B)
      D) The "smoking gun" metric:
           Among B3 (regen used), what is the FINAL eval safe rate?
           If high  -> regen helped (ESC working as designed)
           If low   -> regen actively hurt (verifier loop is noise)
    """
    n = len(eval_records)
    if n == 0:
        return {}

    def _final_is_safe(r):
        # Prefer judge_rating (numeric: 1=safe, 0=unsafe), fall back to classification
        rating = r.get("judge_rating")
        if rating is not None:
            try:
                return int(rating) == 1
            except (TypeError, ValueError):
                pass
        cls = (r.get("safety_classification") or "").lower()
        if cls in ("safe", "unsafe"):
            return cls == "safe"
        # Last fallback: judge_is_safe (verifier's verdict, NOT the eval judge's —
        # only use if nothing else is present)
        return r.get("judge_is_safe")

    # Step-2 verifier trigger: triggered iff judge_is_safe == False at Step 2.
    # Note: when was_regenerated==True we KNOW it triggered. When False, it could be
    # either (a) verifier said safe at Step 2, or (b) triggered but Step-5 kept A.
    triggered = [r for r in eval_records if r.get("judge_is_safe") is False]
    not_triggered = [r for r in eval_records if r.get("judge_is_safe") is True]
    n_triggered = len(triggered)
    n_not_triggered = len(not_triggered)

    # Step-5 decide breakdown (only meaningful for triggered samples)
    triggered_kept_a = [r for r in triggered
                        if str(r.get("decide_choice", "")).upper() == "A"]
    triggered_used_b = [r for r in triggered
                        if str(r.get("decide_choice", "")).upper() == "B"]
    n_kept_a = len(triggered_kept_a)
    n_used_b = len(triggered_used_b)

    # Final-eval safe rates per branch
    def _safe_rate(records):
        labels = [_final_is_safe(r) for r in records]
        labels = [x for x in labels if x is not None]
        if not labels:
            return None, 0
        return sum(1 for x in labels if x) / len(labels), len(labels)

    overall_safe_rate, overall_n     = _safe_rate(eval_records)
    not_trig_safe_rate, not_trig_n   = _safe_rate(not_triggered)
    kept_a_safe_rate, kept_a_n       = _safe_rate(triggered_kept_a)
    used_b_safe_rate, used_b_n       = _safe_rate(triggered_used_b)

    # ── The smoking-gun comparison ────────────────────────────────────────
    # When the verifier triggered, would we have been better off doing
    # nothing (using the original)? We can't directly know without evaluating
    # response_original — but we have a proxy: the verifier's own judgment on
    # the original was "unsafe", so if the original were re-eval'd by the
    # *impartial* judge, we'd expect a low safe rate. The interesting question
    # is: how often did the regen actually rescue it?

    return {
        "n_total":               n,
        "n_triggered":           n_triggered,
        "trigger_rate":          n_triggered / n if n else 0.0,
        "n_not_triggered":       n_not_triggered,

        "n_triggered_kept_A":    n_kept_a,
        "n_triggered_used_B":    n_used_b,
        "regen_acceptance_rate": (n_used_b / n_triggered) if n_triggered else 0.0,

        "overall_eval_safe_rate":   overall_safe_rate,
        "overall_eval_n":           overall_n,
        "not_triggered_safe_rate":  not_trig_safe_rate,
        "not_triggered_n":          not_trig_n,
        "triggered_kept_A_safe_rate": kept_a_safe_rate,
        "triggered_kept_A_n":         kept_a_n,
        "triggered_used_B_safe_rate": used_b_safe_rate,
        "triggered_used_B_n":         used_b_n,
    }


def emotion_distribution(final_results: List[Dict]) -> Dict:
    """Distribution of emotion prompts that were applied."""
    counter = Counter()
    for r in final_results:
        names = r.get("emotion_prompt_name")
        if isinstance(names, list):
            for n in names:
                counter[n] += 1
        elif isinstance(names, str) and names:
            counter[names] += 1
    return dict(counter.most_common())


def _extract_eval_records(data) -> Optional[List[Dict]]:
    """
    Coerce a loaded JSON eval object into a list of per-sample records.

    Accepts:
      A) list[dict]                       -> returned as-is
      B) dict with "detailed_results"     -> returns d["detailed_results"]
      C) dict with "results"              -> returns d["results"]
    Returns None if no list-of-records can be located.
    """
    if data is None:
        return None
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("detailed_results", "results", "samples"):
            v = data.get(key)
            if isinstance(v, list):
                return v
    return None


def _autodiscover_eval_path(loop_dir: Path) -> Optional[Path]:
    """Find loop_K_evaluated.json or any *_evaluated.json inside loop_dir."""
    name = loop_dir.name  # e.g. "loop_1"
    canon = loop_dir / f"{name}_evaluated.json"
    if canon.exists():
        return canon
    for c in loop_dir.glob("*_evaluated.json"):
        return c
    return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{100*x:.1f}%"


def render_markdown(report: Dict) -> str:
    lines = []
    lines.append(f"# ESC log diagnostic\n")
    lines.append(f"**Source dir:** `{report['esc_dir']}`\n")
    lines.append(f"**Loops detected:** {report['n_loops']}\n")

    for li, loop_report in enumerate(report["loops"], start=1):
        lines.append(f"\n## Loop {li}\n")
        v = loop_report.get("verifier", {})
        if v:
            lines.append(f"- Samples judged: **{v['n']}**")
            lines.append(f"- Verifier said safe: **{v['verifier_judged_safe']}** "
                         f"({fmt_pct(v['verifier_judged_safe']/v['n'])})")
            lines.append(f"- Verifier said unsafe (triggered revision): "
                         f"**{v['verifier_judged_unsafe']}** "
                         f"({fmt_pct(v['trigger_rate'])})")
            if v.get("verifier_judged_none", 0):
                lines.append(f"- Verifier returned None: {v['verifier_judged_none']}")
            lines.append(f"- Raw verdicts: starts-yes={v['raw_starts_yes']}, "
                         f"starts-no={v['raw_starts_no']}, "
                         f"ambiguous={v['raw_ambiguous']}")

        d = loop_report.get("decide", {})
        if d.get("n_regen", 0):
            lines.append(f"\n**Decide step:**")
            lines.append(f"- Regenerated samples: {d['n_regen']}")
            lines.append(f"- Picked A (original): {d['decide_chose_A_original']}")
            lines.append(f"- Picked B (regen):    {d['decide_chose_B_regen']}")
            lines.append(f"- Regen acceptance rate: **{fmt_pct(d['regen_acceptance_rate'])}**")

        f = loop_report.get("flips", {})
        if f and "note" not in f:
            lines.append(f"\n**Eval-judge cross-check:**")
            lines.append(f"- Among **accepted** regens (verifier picked B), "
                         f"eval safe-rate: **{fmt_pct(f['accepted_eval_safe_rate'])}** "
                         f"(n={f['accepted_eval_n_labelled']})")
            lines.append(f"- Among **rejected** regens (verifier kept A), "
                         f"eval safe-rate of A: **{fmt_pct(f['rejected_eval_safe_rate_of_original'])}** "
                         f"(n={f['rejected_eval_n_labelled']})")
        elif f and "note" in f:
            lines.append(f"\n_{f['note']}_")

        emo = loop_report.get("emotions", {})
        if emo:
            lines.append(f"\n**Emotion distribution (top 10):**")
            for k, v in list(emo.items())[:10]:
                lines.append(f"  - {k}: {v}")

        # ── Unified eval-based analysis (preferred when available) ──
        u = loop_report.get("unified", {})
        if u:
            lines.append(f"\n**📊 Unified eval analysis (single-source-of-truth):**")
            lines.append(f"- Total samples: **{u['n_total']}**")
            lines.append(f"- Verifier triggered (Step 2 said unsafe): "
                         f"**{u['n_triggered']}** ({fmt_pct(u['trigger_rate'])})")
            lines.append(f"  - Of triggered, kept original (Step 5 → A): "
                         f"**{u['n_triggered_kept_A']}**")
            lines.append(f"  - Of triggered, used regen   (Step 5 → B): "
                         f"**{u['n_triggered_used_B']}**  "
                         f"(acceptance rate: **{fmt_pct(u['regen_acceptance_rate'])}**)")
            lines.append(f"")
            lines.append(f"**Final-eval safe rate by branch (per impartial Llama judge):**")
            lines.append(f"- Overall:                                    "
                         f"**{fmt_pct(u['overall_eval_safe_rate'])}** "
                         f"(n={u['overall_eval_n']})")
            lines.append(f"- Verifier didn't trigger (kept Step-1):      "
                         f"**{fmt_pct(u['not_triggered_safe_rate'])}** "
                         f"(n={u['not_triggered_n']})")
            lines.append(f"- Verifier triggered, kept original:          "
                         f"**{fmt_pct(u['triggered_kept_A_safe_rate'])}** "
                         f"(n={u['triggered_kept_A_n']})")
            lines.append(f"- Verifier triggered, used regen (★ critical): "
                         f"**{fmt_pct(u['triggered_used_B_safe_rate'])}** "
                         f"(n={u['triggered_used_B_n']})")

    lines.append("\n---\n")
    lines.append("**Interpretation guide:**")
    lines.append("- Trigger rate too low (<5%): ESC has no chance to help; the loop is "
                 "essentially a no-op. Method appears unhelpful only because there's "
                 "nothing for it to do.")
    lines.append("- Trigger rate moderate (10-40%) + low regen acceptance (<50%): "
                 "verifier is being conservative; final answer is mostly the original.")
    lines.append("- Trigger rate moderate + high regen acceptance + low accepted-eval safe rate: "
                 "the regen is actively making things worse — this would be the "
                 "smoking gun for 'ESC hurts on Qwen2-VL VLSafe'.")
    lines.append("- Trigger rate moderate + high acceptance + high accepted-eval safe rate: "
                 "ESC is working as designed.")
    lines.append("")
    lines.append("**The ★ critical metric** (unified analysis, when available): "
                 "the safe rate among samples where the verifier triggered AND the regen was used. "
                 "If this is HIGH (e.g., >70%), the regen rescued unsafe responses — ESC working as designed. "
                 "If this is LOW (e.g., <40%), the verifier loop fired but the regen failed to fix things, "
                 "explaining ESC's underperformance vs. one-shot baselines.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esc_dir", required=True,
                    help="ESC pipeline output dir (single loop or loops_K parent)")
    ap.add_argument("--eval_results", default=None,
                    help="Path to a single per-sample eval JSON (e.g. loop_1_evaluated.json). "
                         "If omitted, the script auto-discovers loop_K_evaluated.json in each "
                         "loop subdir.")
    ap.add_argument("--out", default=None, help="Write markdown report to this path")
    args = ap.parse_args()

    esc_dir = Path(args.esc_dir)
    if not esc_dir.exists():
        sys.exit(f"ESC dir not found: {esc_dir}")

    # If a single --eval_results was passed, load it once and apply to all loops
    # (mostly useful for backward-compat / single-loop cases).
    explicit_eval_records = None
    if args.eval_results:
        raw = _load_json(Path(args.eval_results))
        explicit_eval_records = _extract_eval_records(raw)
        if explicit_eval_records is None:
            print(f"WARNING: could not extract per-sample records from "
                  f"{args.eval_results} (expected list[dict] or dict with "
                  f"'detailed_results' key)", file=sys.stderr)

    loop_dirs = discover_loop_dirs(esc_dir)
    report = {
        "esc_dir": str(esc_dir),
        "n_loops": len(loop_dirs),
        "loops": [],
    }

    for loop_dir in loop_dirs:
        loop_report = {}

        # ── Step-2 verifier behavior ──
        judged = _load_json(loop_dir / "step2_judged.json")
        if judged:
            loop_report["verifier"] = diagnose_verifier_behavior(judged)

        # ── Step-6 final results (decide step + emotions) ──
        final_path = find_final_results(loop_dir)
        if final_path:
            final = _load_json(final_path)
            if final:
                loop_report["decide"] = diagnose_decide_step(final)
                loop_report["emotions"] = emotion_distribution(final)

        # ── Unified eval analysis (preferred) ──
        # Per-loop auto-discovery wins over --eval_results unless the user passed
        # an explicit single file (then use it for every loop, at the user's risk).
        loop_eval_records = None
        if explicit_eval_records is not None:
            loop_eval_records = explicit_eval_records
        else:
            eval_path = _autodiscover_eval_path(loop_dir)
            if eval_path:
                raw = _load_json(eval_path)
                loop_eval_records = _extract_eval_records(raw)
                if loop_eval_records is not None:
                    print(f"  [{loop_dir.name}] auto-loaded eval: {eval_path.name} "
                          f"({len(loop_eval_records)} records)", file=sys.stderr)

        if loop_eval_records:
            loop_report["unified"] = diagnose_unified(loop_eval_records)
            # Also keep the legacy flips analysis for back-compat in the report
            if final_path and final:
                loop_report["flips"] = diagnose_flip_rates(final, loop_eval_records)
        elif final_path and final:
            loop_report["flips"] = diagnose_flip_rates(final, None)

        report["loops"].append(loop_report)

    md = render_markdown(report)
    print(md)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"\n[Wrote report -> {out_path}]", file=sys.stderr)


if __name__ == "__main__":
    main()