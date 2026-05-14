"""
Visualize iterative self-correction convergence — dual-axis line plots.
Two panels: (a) LLaVA-1.5-7B, (b) Qwen2-VL-7B.

Usage:
    python plot_loop_convergence.py \
        --input_llava  convergence_data.json \
        --input_qwen2  convergence_qwen2.json \
        --output loop_convergence.png
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    loops = data["loops"]
    asr = [l["asr"] * 100 for l in loops]
    safe = [l["safe_rate"] * 100 for l in loops]
    return asr, safe


def draw_dual_axis_panel(ax, x, asr_vals, safe_vals, title,
                          asr_ylim, safe_ylim):
    """Draw a single dual-axis line panel with non-overlapping labels."""

    color_asr = "#d63031"
    color_safe = "#00b894"
    n = len(x)

    # ── ASR line (left axis) ──
    ln1, = ax.plot(x, asr_vals, "o-", color=color_asr, linewidth=2.2,
                   markersize=7, markerfacecolor="white", markeredgewidth=2.0,
                   label="ASR ($\\downarrow$ better)", zorder=3)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=10, color=color_asr)
    ax.tick_params(axis="y", labelcolor=color_asr, labelsize=9)
    ax.set_ylim(asr_ylim)

    # ── Safe Rate line (right axis) ──
    ax2 = ax.twinx()
    ln2, = ax2.plot(x, safe_vals, "s-", color=color_safe, linewidth=2.2,
                    markersize=7, markerfacecolor="white", markeredgewidth=2.0,
                    label="Safe Rate ($\\uparrow$ better)", zorder=3)
    ax2.set_ylabel("Safe Rate (%)", fontsize=10, color=color_safe)
    ax2.tick_params(axis="y", labelcolor=color_safe, labelsize=9)
    ax2.set_ylim(safe_ylim)

    # ── X axis with padding so labels don't hit the y-axes ──
    ax.set_xlim(x[0] - 0.2, x[-1] + 0.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Loop {i}" for i in x], fontsize=9.5)
    ax.set_xlabel("Loop Iteration", fontsize=10)

    # ── Annotations with edge-aware alignment ──
    # First point: shift right; last point: shift left; middle: center
    for i, (xi, val) in enumerate(zip(x, asr_vals)):
        if i == 0:
            ha, x_off = "left", 1
        elif i == n - 1:
            ha, x_off = "right", -1
        else:
            ha, x_off = "center", 0
        ax.annotate(f"{val:.1f}%", (xi, val),
                    textcoords="offset points", xytext=(x_off, 10),
                    ha=ha, fontsize=8.5, color=color_asr, fontweight="bold")

    for i, (xi, val) in enumerate(zip(x, safe_vals)):
        if i == 0:
            ha, x_off = "left", 1
        elif i == n - 1:
            ha, x_off = "right", -1
        else:
            ha, x_off = "center", 0
        ax2.annotate(f"{val:.1f}%", (xi, val),
                     textcoords="offset points", xytext=(x_off, -14),
                     ha=ha, fontsize=8.5, color=color_safe, fontweight="bold")

    # ── Legend ──
    lines = [ln1, ln2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper center", fontsize=8.5, ncol=2,
              framealpha=0.9, edgecolor="#ddd",
              bbox_to_anchor=(0.5, 1.02))

    # ── Title ──
    ax.set_title(title, fontsize=11, fontweight="bold", pad=20)

    # ── Grid ──
    ax.grid(True, axis="y", alpha=0.2, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    return ax, ax2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_llava", type=str, required=True)
    parser.add_argument("--input_qwen2", type=str, required=True)
    parser.add_argument("--output", type=str, default="loop_convergence.png")
    args = parser.parse_args()

    asr_l, safe_l = load_data(args.input_llava)
    asr_q, safe_q = load_data(args.input_qwen2)

    n = len(asr_l)
    x = np.arange(1, n + 1)

    # ── Style ──
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 9.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })

    fig, (ax_l, ax_q) = plt.subplots(1, 2, figsize=(10.0, 3.8))

    # ── Panel (a): LLaVA-1.5-7B ──
    p = 3
    draw_dual_axis_panel(
        ax_l, x, asr_l, safe_l,
        title="(a) LLaVA-1.5-7B",
        asr_ylim=(min(asr_l) - p * 2.5, max(asr_l) + p),
        safe_ylim=(min(safe_l) - p, max(safe_l) + p * 2.5),
    )

    # ── Panel (b): Qwen2-VL-7B ──
    q = 1.5
    draw_dual_axis_panel(
        ax_q, x, asr_q, safe_q,
        title="(b) Qwen2-VL-7B",
        asr_ylim=(min(asr_q) - q * 2.5, max(asr_q) + q),
        safe_ylim=(min(safe_q) - q, max(safe_q) + q * 2.5),
    )

    plt.tight_layout(w_pad=3.0)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()