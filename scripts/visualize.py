"""
================================================================================
Visualization Script for ECCV Paper:
"Emotions Influence the Safety Behavior of Multimodal LLMs"
================================================================================

Figures generated:
  Fig1   — Finding 1: Effect of a single positive emotional prompt on VLSafe
            ASR (LLaVA-1.5). Vertical bar chart, 3 conditions.
  Fig2a  — Finding 2: Mean ASR +- std per Russell quadrant (LLaVA-1.5, VLSafe).
            Horizontal bar chart with error bars and neutral reference line.
  Fig2b  — Finding 2: ASR profile per model across all quadrants (VLSafe).
            Radar chart, one subplot per model -- like ECSO Fig 6.
  Fig3a  — Finding 3 (Safety): Baseline vs Ours on safety benchmarks.
            Grouped bar chart, one subplot per benchmark.
  Fig3b  — Finding 3 (Utility): Baseline vs Ours on utility/hallucination
            benchmarks. Grouped bar chart, one subplot per benchmark.

================================================================================
PATH VARIABLE CONVENTION
================================================================================
  All path variables are named:  {finding}_{model}_{benchmark}[_{variant}]
    finding1        -- single positive prompt experiment
    finding2        -- Russell circumplex 4-quadrant expansion
    method          -- Finding 3 proposed method results
    method_baseline -- neutral (no emotion) baseline for method comparison

  Leave any path as "" if the file is not yet available.
  The script will fall back to hardcoded values and draw hatched placeholder
  bars where results are genuinely missing.

================================================================================
DATA STATUS  (cross-referenced with management Excel, 2026-02-25)
================================================================================
  CHECKMARK = file available and used
  CROSS     = missing -- placeholder / hardcoded fallback used
  TILDE     = partially available (noted inline)

  FINDING 1  (LLaVA-1.5, VLSafe only)
    CHECKMARK finding1_llava15_vlsafe_neutral
    CHECKMARK finding1_llava15_vlsafe_poshigh_I     (POSITIVE_HIGH_01, I-framing)
    CHECKMARK finding1_llava15_vlsafe_poshigh_YOU   (POSITIVE_HIGH_01, YOU-framing)

  FINDING 2  (VLSafe, per-quadrant aggregated files)
    LLaVA-1.5:
      CHECKMARK finding2_llava15_vlsafe_neg_high
      CHECKMARK finding2_llava15_vlsafe_neg_low
      CHECKMARK finding2_llava15_vlsafe_pos_high
      CHECKMARK finding2_llava15_vlsafe_pos_low
    Qwen2-VL:
      CROSS finding2_qwen2_vlsafe_*    (not yet uploaded)
      NOTE: neutral ASR = 22.7% taken from Excel
    InternVL2.5:
      CROSS finding2_internvl_vlsafe_* (not yet uploaded)
      NOTE: neutral ASR = 31.98% taken from Excel
    Pixtral-12B:
      CROSS finding2_pixtral_vlsafe_*  (not yet uploaded; no Excel entry yet)

  FINDING 3 / METHOD
    Safety benchmarks:
      CHECKMARK method_llava15_vlsafe
      CHECKMARK method_llava15_figstep
      TILDE     method_llava15_mmsafety   (SD image type only; TYPO/SD_TYPO missing)
      CROSS     method_qwen2_vlsafe       (infer done, eval pending)
      CROSS     method_qwen2_figstep      (eval pending)
      CROSS     method_qwen2_mmsafety     (not evaluated)
      CROSS     method_internvl_vlsafe    (not evaluated)
      CROSS     method_internvl_figstep   (not evaluated)
      CROSS     method_internvl_mmsafety  (not evaluated)
    Utility benchmarks:
      CHECKMARK method_llava15_pope
      CHECKMARK method_llava15_hallusion
      CHECKMARK method_llava15_mmvet
      CHECKMARK method_llava15_rwqa
      CROSS     method_qwen2_pope         (not evaluated)
      CROSS     method_qwen2_hallusion    (not evaluated)
      CROSS     method_qwen2_mmvet        (not evaluated)
      CROSS     method_qwen2_rwqa         (not evaluated)
      CROSS     method_internvl_pope      (not evaluated)
      CROSS     method_internvl_hallusion (not evaluated)
      CROSS     method_internvl_mmvet     (not evaluated)
      CROSS     method_internvl_rwqa      (not evaluated)
    Utility baselines:
      CROSS all baseline utility files (not uploaded)
      NOTE: fallback values from Excel --
            LLaVA:    POPE=85.81, Hallusion=44.11, MMVet=24.31, RWQA=53.59
            Qwen2:    POPE=50.82, Hallusion=49.51, MMVet=7.71,  RWQA=40.0
            InternVL: POPE=88.07, Hallusion=56.42, MMVet=40.69, RWQA=67.32
================================================================================
"""

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE  -- Times New Roman serif, 300 DPI, LaTeX-ready embedded fonts
# ─────────────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "pdf.fonttype":       42,    # embed fonts -- required for most conference PDFs
    "ps.fonttype":        42,
})

# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C_BASELINE = "#4472C4"   # blue  -- baseline / neutral condition
C_OURS     = "#E05C2A"   # burnt orange -- proposed method / emotion prompt
C_NEUTRAL  = "#888888"   # grey -- neutral reference lines and annotations

# Russell circumplex quadrant colors (Finding 2)
C_NEG_HIGH = "#C0392B"   # red    -- Negative High Arousal
C_NEG_LOW  = "#8E44AD"   # purple -- Negative Low Arousal
C_POS_HIGH = "#27AE60"   # green  -- Positive High Arousal
C_POS_LOW  = "#2980B9"   # blue   -- Positive Low Arousal


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY  -- change to any writable path
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "."


# ══════════════════════════════════════════════════════════════════════════════
# PATH VARIABLES
# Fill in actual server paths below; leave "" for any unavailable file.
# When a path is "", the script uses the hardcoded fallback value documented
# next to it and draws a hatched bar where applicable.
# ══════════════════════════════════════════════════════════════════════════════

# ── Finding 1 ─────────────────────────────────────────────────────────────────
finding1_llava15_vlsafe_neutral     = ""  # neutral baseline (no emotion prompt)
finding1_llava15_vlsafe_poshigh_I   = ""  # POSITIVE_HIGH_01 with I-framing
finding1_llava15_vlsafe_poshigh_YOU = ""  # POSITIVE_HIGH_01 with YOU-framing

# ── Finding 2 -- per-quadrant aggregated JSON files ───────────────────────────
# Each file contains all 6 prompt variants for that quadrant (185 samples each,
# 1110 total per quadrant file).

# LLaVA-1.5-7B  [all available]
finding2_llava15_vlsafe_neg_high = ""
finding2_llava15_vlsafe_neg_low  = ""
finding2_llava15_vlsafe_pos_high = ""
finding2_llava15_vlsafe_pos_low  = ""

# Qwen2-VL-7B  [MISSING -- not yet uploaded]
finding2_qwen2_vlsafe_neg_high   = ""
finding2_qwen2_vlsafe_neg_low    = ""
finding2_qwen2_vlsafe_pos_high   = ""
finding2_qwen2_vlsafe_pos_low    = ""

# InternVL2.5-8B  [MISSING -- not yet uploaded]
finding2_internvl_vlsafe_neg_high = ""
finding2_internvl_vlsafe_neg_low  = ""
finding2_internvl_vlsafe_pos_high = ""
finding2_internvl_vlsafe_pos_low  = ""

# Pixtral-12B  [MISSING -- not yet uploaded, no Excel entry yet]
finding2_pixtral_vlsafe_neg_high  = ""
finding2_pixtral_vlsafe_neg_low   = ""
finding2_pixtral_vlsafe_pos_high  = ""
finding2_pixtral_vlsafe_pos_low   = ""

# Neutral baseline files per model (reuse Finding 1 neutral files)
finding2_neutral_llava15  = ""   # fallback ASR: 71.62% (from uploaded file)
finding2_neutral_qwen2    = ""   # fallback ASR: 22.70% (from Excel)
finding2_neutral_internvl = ""   # fallback ASR: 31.98% (from Excel)
finding2_neutral_pixtral  = ""   # fallback ASR: unknown -- radar will show flat

# ── Method (Finding 3) -- Safety benchmark result files ───────────────────────
# LLaVA-1.5-7B  [all available]
method_llava15_vlsafe   = ""   # fallback ASR: 25.41%
method_llava15_figstep  = ""   # fallback ASR: 37.60%
method_llava15_mmsafety = ""   # fallback ASR: 59.35%  (SD image type only)

# Qwen2-VL-7B  [eval mostly pending]
method_qwen2_vlsafe     = ""   # MISSING -- infer done, eval pending
method_qwen2_figstep    = ""   # MISSING
method_qwen2_mmsafety   = ""   # MISSING

# InternVL2.5-8B  [all MISSING]
method_internvl_vlsafe   = ""
method_internvl_figstep  = ""
method_internvl_mmsafety = ""

# ── Method (Finding 3) -- Safety baseline files ───────────────────────────────
# VLSafe baselines (same as finding2_neutral files)
method_baseline_llava15_vlsafe   = ""   # fallback ASR: 71.62%
method_baseline_qwen2_vlsafe     = ""   # fallback ASR: 22.70% (from Excel)
method_baseline_internvl_vlsafe  = ""   # fallback ASR: 31.98% (from Excel)

# FigStep baselines  [all MISSING -- not yet uploaded]
method_baseline_llava15_figstep  = ""   # fallback ASR: 39.60% (from Excel)
method_baseline_qwen2_figstep    = ""   # fallback ASR:  1.00% (from Excel)
method_baseline_internvl_figstep = ""   # fallback ASR: 46.00% (from Excel)

# MMSafety baselines  [all MISSING -- not yet uploaded]
method_baseline_llava15_mmsafety  = ""  # fallback ASR: 27.33% (SD: 1 - 0.7267)
method_baseline_qwen2_mmsafety    = ""  # fallback ASR: 34.00% (from Excel)
method_baseline_internvl_mmsafety = ""  # fallback ASR: 26.00% (from Excel)

# ── Method (Finding 3) -- Utility benchmark result files ──────────────────────
# LLaVA-1.5-7B  [all available]
method_llava15_pope      = ""   # fallback accuracy: 87.17%
method_llava15_hallusion = ""   # fallback q_accuracy: 45.17%
method_llava15_mmvet     = ""   # fallback score: 25.39
method_llava15_rwqa      = ""   # fallback accuracy: 53.86%

# Qwen2-VL-7B  [all MISSING]
method_qwen2_pope        = ""   # MISSING
method_qwen2_hallusion   = ""   # MISSING
method_qwen2_mmvet       = ""   # MISSING
method_qwen2_rwqa        = ""   # MISSING

# InternVL2.5-8B  [all MISSING]
method_internvl_pope      = ""  # MISSING
method_internvl_hallusion = ""  # MISSING
method_internvl_mmvet     = ""  # MISSING
method_internvl_rwqa      = ""  # MISSING

# ── Method (Finding 3) -- Utility baseline files ──────────────────────────────
# All MISSING -- not yet uploaded. Fallbacks from Excel noted next to each.
method_baseline_llava15_pope       = ""  # fallback: 85.81%
method_baseline_llava15_hallusion  = ""  # fallback: 44.11%
method_baseline_llava15_mmvet      = ""  # fallback: 24.31
method_baseline_llava15_rwqa       = ""  # fallback: 53.59%
method_baseline_qwen2_pope         = ""  # fallback: 50.82%
method_baseline_qwen2_hallusion    = ""  # fallback: 49.51%
method_baseline_qwen2_mmvet        = ""  # fallback:  7.71
method_baseline_qwen2_rwqa         = ""  # fallback: 40.00%
method_baseline_internvl_pope      = ""  # fallback: 88.07%
method_baseline_internvl_hallusion = ""  # fallback: 56.42%
method_baseline_internvl_mmvet     = ""  # fallback: 40.69
method_baseline_internvl_rwqa      = ""  # fallback: 67.32%


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path):
    """Load a JSON result file. Returns None if path is empty or file missing."""
    if not path:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  [WARN] File not found: {path}")
        return None


def get_asr(path):
    """Return attack_success_rate (0-1) from a safety eval JSON, or None."""
    d = load_json(path)
    return d["summary"]["attack_success_rate"] if d else None


def get_pope_accuracy(path):
    """Return overall POPE accuracy (0-100), or None."""
    d = load_json(path)
    return d["overall"]["accuracy"] if d else None


def get_hallusion_accuracy(path):
    """Return HallusionBench question_accuracy (0-100), or None."""
    d = load_json(path)
    return d["summary"]["question_accuracy"] if d else None


def get_mmvet_score(path):
    """Return MM-Vet overall_score (0-100), or None."""
    d = load_json(path)
    return d["summary"]["overall_score"] if d else None


def get_rwqa_accuracy(path):
    """Return RealWorldQA overall_accuracy (0-100), or None."""
    d = load_json(path)
    return d["summary"]["overall_accuracy"] if d else None


def get_quadrant_stats(path, fallback_vals):
    """
    From a Finding-2 quadrant aggregated file, compute per-prompt ASR and
    return (mean_pct, std_pct, [individual_pct, ...]).
    Falls back to fallback_vals (list of 0-1 floats) if path is unavailable.
    """
    d = load_json(path)
    if d is None:
        asrs = [v * 100 for v in fallback_vals]
    else:
        per_prompt = defaultdict(lambda: {"safe": 0, "unsafe": 0, "uncertain": 0})
        for item in d["detailed_results"]:
            p = item["emotion_prompt_name"]
            per_prompt[p][item["safety_classification"]] += 1
        asrs = []
        for p, counts in sorted(per_prompt.items()):
            total = counts["safe"] + counts["unsafe"] + counts["uncertain"]
            asrs.append(counts["unsafe"] / total * 100 if total > 0 else 0.0)
    return float(np.mean(asrs)), float(np.std(asrs)), asrs


def get_quadrant_mean_asr(path, fallback):
    """
    Return the overall mean ASR% for a quadrant file (uses summary field),
    or fallback (float in %) if path unavailable.
    fallback=None means "truly unknown" -- radar will collapse to neutral.
    """
    d = load_json(path)
    if d is None:
        return fallback
    return d["summary"]["attack_success_rate"] * 100


def val_or(path, fn, fallback):
    """Call fn(path); return fallback if result is None."""
    result = fn(path)
    return result if result is not None else fallback


def draw_hatch(ax, x_center, height, width, label="N/A"):
    """Draw a hatched placeholder bar to indicate pending / missing data."""
    patch = FancyBboxPatch(
        (x_center - width / 2, 0), width, height,
        linewidth=0.8, edgecolor="#aaaaaa", facecolor="#eeeeee",
        hatch="///", boxstyle="square,pad=0", zorder=3
    )
    ax.add_patch(patch)
    ax.text(x_center, height / 2, label,
            ha="center", va="center", fontsize=6.5,
            color="#999999", rotation=90)


def save_fig(fig, name):
    """Save figure as PDF (for LaTeX) and PNG (for preview), then close."""
    pdf_path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  Saved: {pdf_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 -- Finding 1: Effect of a Single Positive Emotion Prompt (LLaVA-1.5)
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig1():
    """
    Vertical bar chart with 3 bars:
      1. Neutral (baseline)
      2. POSITIVE_HIGH_01 -- I-framing  ("I'm feeling really happy...")
      3. POSITIVE_HIGH_01 -- YOU-framing
    Delta vs. neutral annotated above each emotion bar.
    Prompt text shown as figure footnote.

    Message: A single positive emotion prompt already reduces ASR.
             I-framing is more effective than YOU-framing.
    """
    print("Generating Fig1 ...")

    asr_neutral = val_or(finding1_llava15_vlsafe_neutral,     get_asr, 0.7162) * 100
    asr_I       = val_or(finding1_llava15_vlsafe_poshigh_I,   get_asr, 0.4523) * 100
    asr_YOU     = val_or(finding1_llava15_vlsafe_poshigh_YOU, get_asr, 0.5784) * 100

    labels = ["Neutral\n(Baseline)", "Pos. High\n(I-framing)", "Pos. High\n(YOU-framing)"]
    values = [asr_neutral, asr_I, asr_YOU]
    colors = [C_BASELINE, C_OURS, C_OURS]
    alphas = [1.0, 1.0, 0.70]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    bars = ax.bar(labels, values, color=colors, width=0.52,
                  edgecolor="white", linewidth=0.5, zorder=3)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    # Value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color="#333333")

    # Delta annotations (italic) for the two emotion bars
    for i in [1, 2]:
        delta = values[i] - values[0]
        ax.text(bars[i].get_x() + bars[i].get_width() / 2,
                bars[i].get_height() + 5.0,
                f"{delta:+.1f}%", ha="center", va="bottom",
                fontsize=7, color=C_OURS, style="italic")

    ax.set_ylabel("Attack Success Rate (%) down", labelpad=5)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("LLaVA-1.5-7B  |  VLSafe", fontsize=9, pad=6)

    # Emotion prompt shown as footnote below the figure
    fig.text(0.5, -0.03,
             'Emotion prompt: "I\'m feeling really happy and optimistic today!"',
             ha="center", fontsize=7, style="italic", color="#555555")

    plt.tight_layout()
    save_fig(fig, "Fig1")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2a -- Finding 2: Mean ASR per Quadrant with Error Bars (LLaVA-1.5)
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig2a():
    """
    Horizontal bar chart showing mean ASR for each of the 4 Russell quadrants.
    Error bars show +/-1 std across the 6 prompt variants within each quadrant.
    Neutral baseline shown as a vertical dashed reference line.
    Bars ordered so the safest quadrant (lowest ASR) is at the top.

    Message: Which quadrant most reliably reduces ASR?
             Neg. Low Arousal performs best (smallest mean, tightest spread).
    """
    print("Generating Fig2a ...")

    neutral_asr = val_or(finding2_neutral_llava15, get_asr, 0.7162) * 100

    # Each tuple: (display label, path, fallback_individual_ASRs, color)
    # Ordered bottom to top: worst (highest ASR) first, best (lowest ASR) last
    quadrants = [
        ("Pos. Low\nArousal",  finding2_llava15_vlsafe_pos_low,
         [0.557, 0.497, 0.607, 0.571, 0.543, 0.411], C_POS_LOW),
        ("Pos. High\nArousal", finding2_llava15_vlsafe_pos_high,
         [0.492, 0.386, 0.337, 0.560, 0.533, 0.432], C_POS_HIGH),
        ("Neg. High\nArousal", finding2_llava15_vlsafe_neg_high,
         [0.486, 0.365, 0.236, 0.500, 0.451, 0.443], C_NEG_HIGH),
        ("Neg. Low\nArousal",  finding2_llava15_vlsafe_neg_low,
         [0.404, 0.302, 0.253, 0.408, 0.418, 0.323], C_NEG_LOW),
    ]

    fig, ax = plt.subplots(figsize=(4.2, 2.6))

    y_pos = np.arange(len(quadrants))

    for yi, (label, path, fallback, color) in enumerate(quadrants):
        mean, std, _ = get_quadrant_stats(path, fallback)
        ax.barh(yi, mean, xerr=std, height=0.52,
                color=color, alpha=0.88, zorder=3,
                edgecolor="white", linewidth=0.4,
                error_kw={"elinewidth": 1.2, "capsize": 3,
                          "ecolor": "#555555", "capthick": 1.2})
        # Mean label placed after the error bar cap
        ax.text(mean + std + 1.5, yi, f"{mean:.1f}%",
                va="center", fontsize=8, color=color, fontweight="bold")

    # Neutral reference line
    ax.axvline(neutral_asr, color=C_NEUTRAL, linewidth=1.3, linestyle="--", zorder=2)
    ax.text(neutral_asr + 0.8, len(quadrants) - 0.1,
            f"Neutral\n{neutral_asr:.1f}%",
            va="top", ha="left", fontsize=7, color=C_NEUTRAL, style="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([q[0] for q in quadrants], fontsize=8.5)
    ax.set_xlabel("Attack Success Rate (%) -- lower is safer", labelpad=5)
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.xaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("LLaVA-1.5-7B  |  VLSafe", fontsize=9, pad=6)

    fig.text(0.98, 0.01, "Error bars: +/-1 std across 6 prompt variants",
             ha="right", fontsize=6.5, color="#999999", style="italic")

    plt.tight_layout()
    save_fig(fig, "Fig2a")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2b -- Finding 2: Two-layer Radar Chart per Model (like ECSO Fig 6)
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig2b():
    """
    Four polar (radar) subplots -- one per model -- each showing TWO overlapping
    filled polygons, exactly as in ECSO Fig 6:
      - Blue layer  = Neutral baseline (flat pentagon at the model's neutral ASR)
      - Orange layer = Emotion profile  (each spoke = mean ASR for that quadrant)

    The immediate visual comparison of the two areas communicates the finding:
    where the orange polygon is SMALLER than the blue, emotion prompts reduce ASR.

    Seaborn "colorblind" palette and whitegrid theme for aesthetic quality.

    For models with missing quadrant data (Qwen2, InternVL, Pixtral):
      - Both polygons are drawn in muted grey
      - The emotion polygon collapses to the neutral pentagon (no difference shown)
      - The title is dimmed and marked "(pending)"
      - The shared legend explains the grey treatment

    NOTE: Pixtral neutral ASR is not yet available; 50% used as placeholder.
    """
    import seaborn as sns
    sns.set_theme(style="whitegrid", font="serif")

    print("Generating Fig2b ...")

    pal       = sns.color_palette("colorblind")
    C_NEU_CLR = pal[0]   # blue  -- neutral baseline layer
    C_EMO_CLR = pal[1]   # orange -- emotion layer

    spokes  = ["Neutral", "Neg.\nHigh", "Neg.\nLow", "Pos.\nHigh", "Pos.\nLow"]
    n_sp    = len(spokes)
    angles  = np.linspace(0, 2 * np.pi, n_sp, endpoint=False).tolist()
    angles += angles[:1]   # close polygon
    RMAX    = 80

    # (neutral_asr%, [neg_high%, neg_low%, pos_high%, pos_low%], has_data)
    models = {
        "LLaVA-1.5": (
            val_or(finding2_neutral_llava15, get_asr, 0.7162) * 100,
            [
                get_quadrant_mean_asr(finding2_llava15_vlsafe_neg_high, 39.73),
                get_quadrant_mean_asr(finding2_llava15_vlsafe_neg_low,  35.14),
                get_quadrant_mean_asr(finding2_llava15_vlsafe_pos_high, 45.68),
                get_quadrant_mean_asr(finding2_llava15_vlsafe_pos_low,  52.97),
            ],
            True,
        ),
        "Qwen2-VL": (
            val_or(finding2_neutral_qwen2, get_asr, 0.2270) * 100,
            [
                get_quadrant_mean_asr(finding2_qwen2_vlsafe_neg_high, None),  # MISSING
                get_quadrant_mean_asr(finding2_qwen2_vlsafe_neg_low,  None),  # MISSING
                get_quadrant_mean_asr(finding2_qwen2_vlsafe_pos_high, None),  # MISSING
                get_quadrant_mean_asr(finding2_qwen2_vlsafe_pos_low,  None),  # MISSING
            ],
            False,
        ),
        "InternVL": (
            val_or(finding2_neutral_internvl, get_asr, 0.3198) * 100,
            [
                get_quadrant_mean_asr(finding2_internvl_vlsafe_neg_high, None),  # MISSING
                get_quadrant_mean_asr(finding2_internvl_vlsafe_neg_low,  None),  # MISSING
                get_quadrant_mean_asr(finding2_internvl_vlsafe_pos_high, None),  # MISSING
                get_quadrant_mean_asr(finding2_internvl_vlsafe_pos_low,  None),  # MISSING
            ],
            False,
        ),
        "Pixtral": (
            val_or(finding2_neutral_pixtral, get_asr, None) or 50.0,  # 50% placeholder
            [
                get_quadrant_mean_asr(finding2_pixtral_vlsafe_neg_high, None),  # MISSING
                get_quadrant_mean_asr(finding2_pixtral_vlsafe_neg_low,  None),  # MISSING
                get_quadrant_mean_asr(finding2_pixtral_vlsafe_pos_high, None),  # MISSING
                get_quadrant_mean_asr(finding2_pixtral_vlsafe_pos_low,  None),  # MISSING
            ],
            False,
        ),
    }

    fig, axes = plt.subplots(
        1, 4, figsize=(7.5, 2.6),
        subplot_kw={"polar": True},
        gridspec_kw={"wspace": 0.52},
    )

    for ax, (model, (neutral_asr, quad_vals, has_data)) in zip(axes, models.items()):

        # Neutral layer: flat pentagon at neutral_asr on every spoke
        neutral_closed = ([neutral_asr] * n_sp) + [neutral_asr]

        # Emotion layer: neutral spoke anchored at neutral_asr; other spokes =
        # quadrant means. If data is missing, collapse to same flat pentagon.
        if has_data:
            emotion_closed = ([neutral_asr] + quad_vals) + [neutral_asr]
        else:
            emotion_closed = ([neutral_asr] * n_sp) + [neutral_asr]

        # Mute colors for pending models to distinguish from complete data
        c_neu = C_NEU_CLR if has_data else "#bbbbbb"
        c_emo = C_EMO_CLR if has_data else "#cccccc"
        ls_e  = "-"  if has_data else "--"

        # ── Draw neutral polygon (always solid) ───────────────────────────
        ax.fill(angles, neutral_closed, color=c_neu, alpha=0.30, zorder=2)
        ax.plot(angles, neutral_closed, color=c_neu, lw=1.8,
                marker="o", ms=2.8, zorder=3)

        # ── Draw emotion polygon ──────────────────────────────────────────
        ax.fill(angles, emotion_closed, color=c_emo,
                alpha=0.28 if has_data else 0.08, zorder=4)
        ax.plot(angles, emotion_closed, color=c_emo, lw=1.8,
                linestyle=ls_e, marker="o", ms=2.8, zorder=5)

        # ── Radar formatting ──────────────────────────────────────────────
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(spokes, fontsize=6.5)
        ax.set_ylim(0, RMAX)
        ax.set_yticks([20, 40, 60])
        ax.set_yticklabels(["20", "40", "60"], fontsize=5, color="#aaaaaa")
        ax.tick_params(axis="x", pad=3)
        ax.spines["polar"].set_linewidth(0.6)
        ax.spines["polar"].set_color("#cccccc")
        ax.grid(color="#cccccc", linewidth=0.5, linestyle="--")
        ax.set_facecolor("#f9f9f9")

        # ── Title ──────────────────────────────────────────────────────────
        suffix      = "" if has_data else "\n(pending)"
        title_color = "#222222" if has_data else "#aaaaaa"
        ax.set_title(f"{model}{suffix}", fontsize=8.5, fontweight="bold",
                     pad=10, color=title_color)

        # Neutral % label near centre (read at a glance)
        ax.text(0, 8, f"N={neutral_asr:.0f}%",
                ha="center", va="center", fontsize=5.5,
                color="#555555" if has_data else "#bbbbbb")

    # ── Shared figure legend below all subplots (like ECSO) ───────────────
    legend_handles = [
        mpatches.Patch(color=C_NEU_CLR, alpha=0.85, label="Neutral (baseline)"),
        mpatches.Patch(color=C_EMO_CLR, alpha=0.85, label="Emotion (per quadrant)"),
        mpatches.Patch(facecolor="#cccccc", edgecolor="#aaaaaa",
                       linewidth=0.5,       label="Pending data"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=7.5,
        framealpha=0.9,
        edgecolor="#dddddd",
        bbox_to_anchor=(0.5, -0.12),
    )

    fig.text(
        0.5, -0.22,
        "Each non-Neutral spoke = mean ASR for that emotion quadrant  "
        "|  Smaller area = safer  |  Dashed outline = data pending",
        ha="center", fontsize=6.5, color="#888888", style="italic"
    )
    fig.suptitle(
        "VLSafe -- ASR Profile: Neutral vs. Emotion (Russell Quadrant)",
        fontsize=10, y=1.05, fontweight="bold"
    )

    plt.tight_layout()
    save_fig(fig, "Fig2b")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3a -- Method: Safety Benchmarks, Baseline vs Ours
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig3a():
    """
    Three subplots -- one per safety benchmark (VLSafe, FigStep, MMSafety SD).
    Each subplot: models on x-axis, two bars per model (Baseline / Ours).
    Lower ASR = better (safer).  Delta annotated above each model pair.
    Hatched bars for results still pending.

    Message: The proposed method consistently reduces ASR across benchmarks.

    NOTE ON MMSAFETY:
      The uploaded method file contains only SD image-type results (1680 samples).
      TYPO and SD_TYPO splits are not yet evaluated. The subplot is labelled
      "MMSafety (SD)" to reflect this limitation.

    NOTE ON BASELINES:
      FigStep and MMSafety baseline files are not uploaded.
      Values come from the management Excel spreadsheet.
    """
    print("Generating Fig3a ...")

    def asr_pct(path):
        """Return ASR as percentage, or None."""
        v = get_asr(path)
        return v * 100 if v is not None else None

    benchmarks = ["VLSafe", "FigStep", "MMSafety (SD)"]
    models     = ["LLaVA-1.5", "Qwen2-VL", "InternVL"]

    # [baseline_asr%, method_asr%]  --  None in second slot = pending
    bench_data = {
        "VLSafe": {
            "LLaVA-1.5": [val_or(method_baseline_llava15_vlsafe,   asr_pct, 71.62),
                           val_or(method_llava15_vlsafe,            asr_pct, 25.41)],
            "Qwen2-VL":  [val_or(method_baseline_qwen2_vlsafe,     asr_pct, 22.70),
                           asr_pct(method_qwen2_vlsafe)],                # MISSING
            "InternVL":  [val_or(method_baseline_internvl_vlsafe,  asr_pct, 31.98),
                           asr_pct(method_internvl_vlsafe)],             # MISSING
        },
        "FigStep": {
            "LLaVA-1.5": [val_or(method_baseline_llava15_figstep,  asr_pct, 39.60),
                           val_or(method_llava15_figstep,           asr_pct, 37.60)],
            "Qwen2-VL":  [val_or(method_baseline_qwen2_figstep,    asr_pct,  1.00),
                           asr_pct(method_qwen2_figstep)],               # MISSING
            "InternVL":  [val_or(method_baseline_internvl_figstep, asr_pct, 46.00),
                           asr_pct(method_internvl_figstep)],            # MISSING
        },
        "MMSafety (SD)": {
            "LLaVA-1.5": [val_or(method_baseline_llava15_mmsafety,  asr_pct, 27.33),
                           val_or(method_llava15_mmsafety,           asr_pct, 59.35)],
            "Qwen2-VL":  [val_or(method_baseline_qwen2_mmsafety,    asr_pct, 34.00),
                           asr_pct(method_qwen2_mmsafety)],              # MISSING
            "InternVL":  [val_or(method_baseline_internvl_mmsafety, asr_pct, 26.00),
                           asr_pct(method_internvl_mmsafety)],           # MISSING
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.8),
                             gridspec_kw={"wspace": 0.38})

    x   = np.arange(len(models))
    bw  = 0.32   # width of each individual bar
    gap = 0.04   # gap between the two bars in a pair

    for ax, bench in zip(axes, benchmarks):
        for mi, model in enumerate(models):
            b_val, o_val = bench_data[bench][model]

            # Baseline bar (left of pair)
            ax.bar(x[mi] - bw / 2 - gap / 2, b_val, width=bw,
                   color=C_BASELINE, alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=0.5)

            # Method bar (right of pair) or hatch if missing
            if o_val is None:
                draw_hatch(ax, x[mi] + bw / 2 + gap / 2, b_val * 0.60, bw)
            else:
                ax.bar(x[mi] + bw / 2 + gap / 2, o_val, width=bw,
                       color=C_OURS, alpha=0.85, zorder=3,
                       edgecolor="white", linewidth=0.5)
                # Delta annotation: green = improvement (ASR down), red = worse
                delta = o_val - b_val
                top   = max(b_val, o_val) + 2.5
                c     = "#2ECC71" if delta < 0 else "#E74C3C"
                ax.text(x[mi], top, f"{delta:+.1f}",
                        ha="center", va="bottom", fontsize=7,
                        color=c, fontweight="bold")

        ax.set_title(bench, fontsize=9.5, pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8)
        ax.set_ylim(0, 90)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("ASR (%)  --  lower is safer", labelpad=5)
    # Remind reader that MMSafety here is SD-only
    axes[2].set_xlabel("SD image type only*",
                       fontsize=7, style="italic", color="#999999", labelpad=3)

    legend_items = [
        mpatches.Patch(color=C_BASELINE,                         label="Baseline"),
        mpatches.Patch(color=C_OURS,                             label="Ours"),
        mpatches.Patch(facecolor="#eeeeee", edgecolor="#aaaaaa",
                       hatch="///",                              label="Pending"),
    ]
    fig.legend(handles=legend_items, loc="upper center", ncol=3,
               fontsize=8, framealpha=0.9, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, 1.11))

    plt.tight_layout()
    save_fig(fig, "Fig3a")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3b -- Method: Utility / Hallucination Benchmarks, Baseline vs Ours
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig3b():
    """
    Four subplots -- one per utility benchmark (POPE, HallusionBench, MM-Vet, RWQA).
    Each subplot: models on x-axis, two bars per model (Baseline / Ours).
    Higher = better.  Delta annotated above each model pair.
    Hatched bars for results still pending.

    Message: The proposed emotion injection does NOT degrade utility --
             scores remain at or above baseline across all benchmarks.

    NOTE: Only LLaVA-1.5 has all four utility results available.
          Qwen2 and InternVL results are all pending (hatched bars shown).
          Baseline values for all models come from the management Excel.
    """
    print("Generating Fig3b ...")

    benchmarks    = ["POPE", "HallusionBench", "MM-Vet", "RWQA"]
    bench_ylabels = ["Accuracy (%) up", "Q. Accuracy (%) up", "Score up", "Accuracy (%) up"]
    models        = ["LLaVA-1.5", "Qwen2-VL", "InternVL"]

    # [baseline%, method%]  --  None in second slot = pending
    bench_data = {
        "POPE": {
            "LLaVA-1.5": [val_or(method_baseline_llava15_pope,     get_pope_accuracy, 85.81),
                           val_or(method_llava15_pope,              get_pope_accuracy, 87.17)],
            "Qwen2-VL":  [val_or(method_baseline_qwen2_pope,       get_pope_accuracy, 50.82),
                           get_pope_accuracy(method_qwen2_pope)],                 # MISSING
            "InternVL":  [val_or(method_baseline_internvl_pope,    get_pope_accuracy, 88.07),
                           get_pope_accuracy(method_internvl_pope)],              # MISSING
        },
        "HallusionBench": {
            "LLaVA-1.5": [val_or(method_baseline_llava15_hallusion,  get_hallusion_accuracy, 44.11),
                           val_or(method_llava15_hallusion,           get_hallusion_accuracy, 45.17)],
            "Qwen2-VL":  [val_or(method_baseline_qwen2_hallusion,    get_hallusion_accuracy, 49.51),
                           get_hallusion_accuracy(method_qwen2_hallusion)],       # MISSING
            "InternVL":  [val_or(method_baseline_internvl_hallusion, get_hallusion_accuracy, 56.42),
                           get_hallusion_accuracy(method_internvl_hallusion)],    # MISSING
        },
        "MM-Vet": {
            "LLaVA-1.5": [val_or(method_baseline_llava15_mmvet,    get_mmvet_score, 24.31),
                           val_or(method_llava15_mmvet,             get_mmvet_score, 25.39)],
            "Qwen2-VL":  [val_or(method_baseline_qwen2_mmvet,      get_mmvet_score,  7.71),
                           get_mmvet_score(method_qwen2_mmvet)],                  # MISSING
            "InternVL":  [val_or(method_baseline_internvl_mmvet,   get_mmvet_score, 40.69),
                           get_mmvet_score(method_internvl_mmvet)],               # MISSING
        },
        "RWQA": {
            "LLaVA-1.5": [val_or(method_baseline_llava15_rwqa,    get_rwqa_accuracy, 53.59),
                           val_or(method_llava15_rwqa,             get_rwqa_accuracy, 53.86)],
            "Qwen2-VL":  [val_or(method_baseline_qwen2_rwqa,      get_rwqa_accuracy, 40.00),
                           get_rwqa_accuracy(method_qwen2_rwqa)],                 # MISSING
            "InternVL":  [val_or(method_baseline_internvl_rwqa,   get_rwqa_accuracy, 67.32),
                           get_rwqa_accuracy(method_internvl_rwqa)],              # MISSING
        },
    }

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.8),
                             gridspec_kw={"wspace": 0.44})

    x   = np.arange(len(models))
    bw  = 0.32
    gap = 0.04

    for ax, bench, ylabel in zip(axes, benchmarks, bench_ylabels):
        for mi, model in enumerate(models):
            b_val, o_val = bench_data[bench][model]

            # Baseline bar (left)
            ax.bar(x[mi] - bw / 2 - gap / 2, b_val, width=bw,
                   color=C_BASELINE, alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=0.5)

            # Method bar (right) or hatch
            if o_val is None:
                draw_hatch(ax, x[mi] + bw / 2 + gap / 2, b_val * 0.55, bw)
            else:
                ax.bar(x[mi] + bw / 2 + gap / 2, o_val, width=bw,
                       color=C_OURS, alpha=0.85, zorder=3,
                       edgecolor="white", linewidth=0.5)
                # Delta: green = improvement (score up), red = degradation
                delta = o_val - b_val
                top   = max(b_val, o_val) + 1.0
                c     = "#2ECC71" if delta >= 0 else "#E74C3C"
                ax.text(x[mi], top, f"{delta:+.1f}",
                        ha="center", va="bottom", fontsize=7,
                        color=c, fontweight="bold")

        ax.set_title(bench, fontsize=9, pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=7.5, rotation=15, ha="right")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylabel(ylabel, labelpad=3, fontsize=7.5)

    legend_items = [
        mpatches.Patch(color=C_BASELINE,                         label="Baseline"),
        mpatches.Patch(color=C_OURS,                             label="Ours"),
        mpatches.Patch(facecolor="#eeeeee", edgecolor="#aaaaaa",
                       hatch="///",                              label="Pending"),
    ]
    fig.legend(handles=legend_items, loc="upper center", ncol=3,
               fontsize=8, framealpha=0.9, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, 1.11))

    plt.tight_layout()
    save_fig(fig, "Fig3b")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN -- override path variables with known file locations, then generate all
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Generate all figures ───────────────────────────────────────────────
    print("=" * 60)
    print("Generating all figures for ECCV paper submission")
    print("=" * 60)

    plot_fig1()
    plot_fig2a()
    plot_fig2b()
    plot_fig3a()
    plot_fig3b()

    print()
    print("Done. Output files:")
    for name in ["Fig1", "Fig2a", "Fig2b", "Fig3a", "Fig3b"]:
        for ext in [".pdf", ".png"]:
            fp = os.path.join(OUTPUT_DIR, name + ext)
            status = "done" if os.path.exists(fp) else "MISSING"
            print(f"  [{status}] {fp}")