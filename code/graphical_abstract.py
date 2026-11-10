#!/usr/bin/env python
"""Graphical abstract, derived from Fig. 1 (same drawing helpers, colours and wording).

    python graphical_abstract.py                 # writes ../graphical_abstract/graphical_abstract.{pdf,png,jpg}
    python graphical_abstract.py --out /tmp/x

Numbers are the headline test-fold results of the manuscript (Abstract, Table II, Table III);
make_all.py asserts the same values from code/data before plotting.
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

import make_all as M

AMBER_D, BLUE_D = "#A15C07", "#0B5394"


def draw():
    fig = plt.figure(figsize=(8.0, 3.57))  # 2.24:1, the IEEE graphical-abstract aspect (660 x 295 px minimum)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(1.2, 44.6)
    ax.axis("off")
    ax.text(50, 43.3, "Task-aware 12-lead ECG transmission: which lead, and how to spend 16 channel uses",
            ha="center", va="top", fontsize=8.6, fontweight="bold", color=M.NAVY)

    # transmitter / receiver shading
    ax.add_patch(Rectangle((0.5, 12.2), 61.5, 28.4, fc="#F4F6F9", ec="none"))
    ax.text(1.4, 39.8, "Transmitter (all 12 leads acquired and encoded)", fontsize=6.8, fontweight="bold", color=M.NAVY, va="top")
    ax.add_patch(Rectangle((62.8, 12.2), 36.7, 28.4, fc="#FBF7F0", ec="none"))
    ax.text(63.7, 39.8, "Channel and receiver", fontsize=6.8, fontweight="bold", color=M.NAVY, va="top")

    # 12 leads
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    for i, l in enumerate(leads):
        yy = 36.4 - i * 1.95
        ax.text(2.6, yy + 0.25, l, fontsize=5.0, ha="right", va="center", color=M.INK2)
        M._ecg(ax, 3.0, yy, 6.0, 1.25, M.NAVY)
    M._arrow(ax, 9.6, 25.6, 11.4, 25.6)
    M._box(ax, 11.9, 21.4, 9.6, 8.4, "Shared\nper-lead\nencoder\n(frozen)", fc="#E3E9F2", fs=6.4)
    ax.text(16.7, 31.0, "12 × 32 values", fontsize=5.6, ha="center", va="bottom", color=M.INK)

    # split into the two questions
    ax.plot([22.1, 23.6], [25.6, 25.6], color=M.NAVY, lw=1.0)
    ax.plot([23.6, 23.6], [18.2, 33.0], color=M.NAVY, lw=1.0)
    M._arrow(ax, 23.6, 33.0, 25.2, 33.0)
    M._arrow(ax, 23.6, 18.2, 25.2, 18.2)

    # Q1: record-specific lead selection, one lead, clean link (Phase 2A; no channel, as in Fig. 1)
    M._box(ax, 25.7, 29.4, 14.4, 7.2, "Q1  Record-specific\nlead selection (one\nrepresentation sent)", fc="#DCEAF7",
           ec=BLUE_D, fs=6.3)
    M._blocks(ax, 44.0, 32.0, 1, 1.6, 2.2, ["#2C7BB6"])
    ax.text(44.8, 34.9, "1 × 32 values", fontsize=5.4, ha="center", va="bottom", color=M.INK2)
    M._arrow(ax, 40.6, 33.0, 43.5, 33.0, BLUE_D)
    M._arrow(ax, 46.2, 33.0, 78.0, 33.0, BLUE_D)
    ax.text(62.0, 33.6, "clean link", fontsize=5.6, ha="center", va="bottom", color=BLUE_D, fontstyle="italic")

    # Q2: how to spend a fixed channel budget
    M._box(ax, 25.7, 14.6, 14.4, 7.2, "Q2  Encoding method\nrepetition, digital, or\nlearned code (1 or 12 leads)",
           fc="#FDEBD3", ec=AMBER_D, fs=5.9)
    M._arrow(ax, 40.6, 18.2, 43.5, 18.2, AMBER_D)
    w = 1.05
    for i in range(16):
        x = 44.0 + (i % 8) * (w + 0.18)
        y = 18.6 if i < 8 else 17.0
        ax.add_patch(Rectangle((x, y), w, 1.25, fc="#E8A85C", ec="white", lw=0.4))
    ax.text(48.9, 21.0, "16 complex channel uses,\nenergy 16 per record", fontsize=5.6, ha="center", va="bottom",
            color=M.INK)
    ax.text(48.9, 16.3, "repetition: m × r = 16 (r = 1, 2 or 4)", fontsize=5.1, ha="center", va="top", color=M.INK2)
    M._arrow(ax, 54.6, 18.2, 64.6, 18.2, AMBER_D)

    # channel, receiver, five labels
    M._box(ax, 65.1, 14.8, 10.4, 6.8, "AWGN or\nblock Rayleigh\nfading", fc="white", ec=AMBER_D, fs=6.0)
    M._arrow(ax, 76.0, 18.2, 78.0, 18.2, AMBER_D)
    M._box(ax, 78.5, 14.0, 19.8, 22.6, "", fc="white", ec=M.NAVY, lw=1.2)
    ax.text(88.4, 35.0, "Receiver: five-label\nprediction (no waveform)", fontsize=6.0, fontweight="bold",
            ha="center", va="center")
    for i, (c, v) in enumerate(zip(M.CLASSES, (0.8, 0.45, 0.55, 0.35, 0.4))):
        yy = 30.4 - i * 3.3
        ax.text(80.3, yy, c, fontsize=5.8, va="center")
        ax.add_patch(Rectangle((85.0, yy - 0.75), 11.5 * v, 1.5, fc=M.NAVY, ec="none"))

    # findings (wording and numbers as in the Abstract)
    f1 = ("One lead at the receiver: record-specific selection vs matched fixed V5\n"
          "macro-AUROC +0.018 (95% CI 0.009 to 0.027); CD +0.067, MI +0.043, STTC −0.037 AUROC")
    f2 = ("All 12 leads at the transmitter, 16 channel uses, AWGN (−15 to 20 dB)\n"
          "joint learned code over 12 leads: +0.095 macro-AUROC vs single-lead repetition")
    for x, t, ec, fc in ((0.5, f1, BLUE_D, "#EEF4FB"), (50.3, f2, AMBER_D, "#FEF5EA")):
        ax.add_patch(FancyBboxPatch((x + 0.4, 3.0), 48.4, 7.4, boxstyle="round,pad=0.2,rounding_size=1.0",
                                    fc=fc, ec=ec, lw=0.9))
        ax.text(x + 24.6, 6.7, t, fontsize=6.4, ha="center", va="center", color=M.INK, linespacing=1.5)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(M.HERE, "..", "graphical_abstract"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig = draw()
    base = os.path.join(a.out, "graphical_abstract")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=600)
    fig.savefig(base + ".jpg", dpi=600, pil_kwargs={"quality": 95})
    plt.close(fig)
    print("wrote", base + ".{pdf,png,jpg}")


if __name__ == "__main__":
    main()
