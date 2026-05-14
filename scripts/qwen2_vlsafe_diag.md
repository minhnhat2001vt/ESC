# ESC log diagnostic

**Source dir:** `/workspace/results/method1/qwen2_vl_7b__gemma_3_12b_it/vlsafe/fixed/negative_low/start/multi2/4_loops`

**Loops detected:** 4


## Loop 1

- Samples judged: **1110**
- Verifier said safe: **745** (67.1%)
- Verifier said unsafe (triggered revision): **365** (32.9%)
- Raw verdicts: starts-yes=745, starts-no=365, ambiguous=0

**Decide step:**
- Regenerated samples: 1110
- Picked A (original): 41
- Picked B (regen):    324
- Regen acceptance rate: **29.2%**

**Eval-judge cross-check:**
- Among **accepted** regens (verifier picked B), eval safe-rate: **n/a** (n=0)
- Among **rejected** regens (verifier kept A), eval safe-rate of A: **n/a** (n=0)

**Emotion distribution (top 10):**
  - NEGATIVE_LOW_06: 130
  - NEGATIVE_LOW_02: 129
  - NEGATIVE_LOW_01: 122
  - NEGATIVE_LOW_05: 121
  - NEGATIVE_LOW_04: 116
  - NEGATIVE_LOW_03: 112

**📊 Unified eval analysis (single-source-of-truth):**
- Total samples: **1110**
- Verifier triggered (Step 2 said unsafe): **365** (32.9%)
  - Of triggered, kept original (Step 5 → A): **41**
  - Of triggered, used regen   (Step 5 → B): **324**  (acceptance rate: **88.8%**)

**Final-eval safe rate by branch (per impartial Llama judge):**
- Overall:                                    **88.4%** (n=1110)
- Verifier didn't trigger (kept Step-1):      **90.9%** (n=745)
- Verifier triggered, kept original:          **73.2%** (n=41)
- Verifier triggered, used regen (★ critical): **84.6%** (n=324)

## Loop 2

- Samples judged: **1110**
- Verifier said safe: **1019** (91.8%)
- Verifier said unsafe (triggered revision): **91** (8.2%)
- Raw verdicts: starts-yes=1019, starts-no=91, ambiguous=0

**Decide step:**
- Regenerated samples: 1110
- Picked A (original): 41
- Picked B (regen):    50
- Regen acceptance rate: **4.5%**

**Eval-judge cross-check:**
- Among **accepted** regens (verifier picked B), eval safe-rate: **n/a** (n=0)
- Among **rejected** regens (verifier kept A), eval safe-rate of A: **n/a** (n=0)

**Emotion distribution (top 10):**
  - NEGATIVE_LOW_06: 37
  - NEGATIVE_LOW_01: 32
  - NEGATIVE_LOW_02: 31
  - NEGATIVE_LOW_03: 30
  - NEGATIVE_LOW_05: 28
  - NEGATIVE_LOW_04: 24

**📊 Unified eval analysis (single-source-of-truth):**
- Total samples: **1110**
- Verifier triggered (Step 2 said unsafe): **91** (8.2%)
  - Of triggered, kept original (Step 5 → A): **41**
  - Of triggered, used regen   (Step 5 → B): **50**  (acceptance rate: **54.9%**)

**Final-eval safe rate by branch (per impartial Llama judge):**
- Overall:                                    **89.0%** (n=1110)
- Verifier didn't trigger (kept Step-1):      **89.6%** (n=1019)
- Verifier triggered, kept original:          **75.6%** (n=41)
- Verifier triggered, used regen (★ critical): **88.0%** (n=50)

## Loop 3

- Samples judged: **1110**
- Verifier said safe: **1056** (95.1%)
- Verifier said unsafe (triggered revision): **54** (4.9%)
- Raw verdicts: starts-yes=1056, starts-no=54, ambiguous=0

**Decide step:**
- Regenerated samples: 1110
- Picked A (original): 31
- Picked B (regen):    23
- Regen acceptance rate: **2.1%**

**Eval-judge cross-check:**
- Among **accepted** regens (verifier picked B), eval safe-rate: **n/a** (n=0)
- Among **rejected** regens (verifier kept A), eval safe-rate of A: **n/a** (n=0)

**Emotion distribution (top 10):**
  - NEGATIVE_LOW_06: 23
  - NEGATIVE_LOW_01: 21
  - NEGATIVE_LOW_02: 20
  - NEGATIVE_LOW_03: 18
  - NEGATIVE_LOW_05: 15
  - NEGATIVE_LOW_04: 11

**📊 Unified eval analysis (single-source-of-truth):**
- Total samples: **1110**
- Verifier triggered (Step 2 said unsafe): **54** (4.9%)
  - Of triggered, kept original (Step 5 → A): **31**
  - Of triggered, used regen   (Step 5 → B): **23**  (acceptance rate: **42.6%**)

**Final-eval safe rate by branch (per impartial Llama judge):**
- Overall:                                    **89.2%** (n=1110)
- Verifier didn't trigger (kept Step-1):      **89.9%** (n=1056)
- Verifier triggered, kept original:          **74.2%** (n=31)
- Verifier triggered, used regen (★ critical): **78.3%** (n=23)

## Loop 4

- Samples judged: **1110**
- Verifier said safe: **1073** (96.7%)
- Verifier said unsafe (triggered revision): **37** (3.3%)
- Raw verdicts: starts-yes=1073, starts-no=37, ambiguous=0

**Decide step:**
- Regenerated samples: 1110
- Picked A (original): 30
- Picked B (regen):    7
- Regen acceptance rate: **0.6%**

**Eval-judge cross-check:**
- Among **accepted** regens (verifier picked B), eval safe-rate: **n/a** (n=0)
- Among **rejected** regens (verifier kept A), eval safe-rate of A: **n/a** (n=0)

**Emotion distribution (top 10):**
  - NEGATIVE_LOW_01: 18
  - NEGATIVE_LOW_06: 13
  - NEGATIVE_LOW_02: 13
  - NEGATIVE_LOW_05: 12
  - NEGATIVE_LOW_03: 11
  - NEGATIVE_LOW_04: 7

**📊 Unified eval analysis (single-source-of-truth):**
- Total samples: **1110**
- Verifier triggered (Step 2 said unsafe): **37** (3.3%)
  - Of triggered, kept original (Step 5 → A): **30**
  - Of triggered, used regen   (Step 5 → B): **7**  (acceptance rate: **18.9%**)

**Final-eval safe rate by branch (per impartial Llama judge):**
- Overall:                                    **89.5%** (n=1110)
- Verifier didn't trigger (kept Step-1):      **89.7%** (n=1073)
- Verifier triggered, kept original:          **76.7%** (n=30)
- Verifier triggered, used regen (★ critical): **100.0%** (n=7)

---

**Interpretation guide:**
- Trigger rate too low (<5%): ESC has no chance to help; the loop is essentially a no-op. Method appears unhelpful only because there's nothing for it to do.
- Trigger rate moderate (10-40%) + low regen acceptance (<50%): verifier is being conservative; final answer is mostly the original.
- Trigger rate moderate + high regen acceptance + low accepted-eval safe rate: the regen is actively making things worse — this would be the smoking gun for 'ESC hurts on Qwen2-VL VLSafe'.
- Trigger rate moderate + high acceptance + high accepted-eval safe rate: ESC is working as designed.

**The ★ critical metric** (unified analysis, when available): the safe rate among samples where the verifier triggered AND the regen was used. If this is HIGH (e.g., >70%), the regen rescued unsafe responses — ESC working as designed. If this is LOW (e.g., <40%), the verifier loop fired but the regen failed to fix things, explaining ESC's underperformance vs. one-shot baselines.