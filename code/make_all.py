#!/usr/bin/env python
"""Generate every new figure and every data table of the manuscript and supplement.

Inputs: summary JSON files in ./data (copied unchanged from the locked evaluation outputs).
Outputs: ../source/figures/*.pdf and ../source/tables/*.tex (or the directory given by --out).
Nothing is trained, re-evaluated or resampled here; the script only reads and plots summaries.

    python make_all.py                 # writes into ../source
    python make_all.py --out /tmp/x    # writes into /tmp/x/figures and /tmp/x/tables
    python make_all.py --only-e4       # E4 extended-training outputs only (used for this revision round;
                                       # the other shipped tables/figures contain author edits and are not regenerated)

Fig. S1 is reused unchanged from the original submission (source/figures/figS1_repetition_vs_snr.pdf).
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

for f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf"):
    font_manager.fontManager.addfont(os.path.join(HERE, "fonts", f))
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "pdf.fonttype": 42, "axes.unicode_minus": True, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#3A3A3A", "xtick.color": "#3A3A3A", "ytick.color": "#3A3A3A",
    "text.color": "#1A1A1A", "axes.labelcolor": "#1A1A1A", "savefig.facecolor": "white", "lines.solid_capstyle": "round",
})
INK, INK2, MUTED, GRID = "#1A1A1A", "#555555", "#8A8A8A", "#E6E6E6"
PURPLE, PURPLE_L, AMBER, GRAY, REF, REF_L = "#5B4A8B", "#9C8FC4", "#C2851C", "#B4B4B4", "#2B2B2B", "#C9C9C9"
NAVY = "#1F3A5F"
# original-manuscript method colours (Okabe-Ito), kept for the lead-budget panels
OI = {"proposed": "#D55E00", "global_learned_k": "#0072B2", "fixed_clinical_k": "#E69F00", "random_k": "#999999"}
DOUBLE, SINGLE = 7.16, 3.5
MINUS = "−"
SNRS = [-15, -10, -5, 0, 5, 10, 20]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
OUT = {}


def load(name):
    with open(os.path.join(DATA, name)) as fh:
        return json.load(fh)


def fmt(v, d=3, sign=True):
    return (f"{v:+.{d}f}" if sign else f"{v:.{d}f}").replace("-", MINUS)


def ci_txt(ci, d=3):
    return f"[{fmt(ci[0], d, False)}, {fmt(ci[1], d, False)}]"


def tickfmt(v, _=None):
    t = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if t in ("0", "-0") else t.replace("-", MINUS)


def set_xticks(ax, step):
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(step))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(tickfmt))


def letter(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    fig.savefig(os.path.join(OUT["fig"], f"{name}.pdf"))
    plt.close(fig)
    print("figure", name)


# ====================================================================== Fig. 1: framework
def _box(ax, x, y, w, h, text, fc="white", ec=NAVY, fs=6.6, bold=False, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def _arrow(ax, x0, y0, x1, y1, color=NAVY):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                                                                     shrinkA=0, shrinkB=0, mutation_scale=7))


def _ecg(ax, x0, y0, w, amp, color):
    t = np.linspace(0, 1, 200)
    s = 0.15 * np.sin(2 * np.pi * t) * 0 + np.exp(-((t - 0.45) / 0.025) ** 2) - 0.35 * np.exp(-((t - 0.5) / 0.02) ** 2) \
        + 0.18 * np.exp(-((t - 0.72) / 0.06) ** 2)
    ax.plot(x0 + w * t, y0 + amp * s, color=color, lw=0.8)


def _blocks(ax, x, y, n, w, h, cols):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * (w + 0.25), y), w, h, fc=cols[i % len(cols)], ec="white", lw=0.4))


def fig1_framework():
    fig = plt.figure(figsize=(DOUBLE, 3.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 38.8)
    ax.axis("off")
    # transmitter side shading
    ax.add_patch(Rectangle((0.5, 0.6), 67.3, 37.6, fc="#F4F6F9", ec="none"))
    ax.text(1.5, 37.3, "Transmitter (all 12 leads acquired)", fontsize=6.8, fontweight="bold", color=NAVY, va="top")
    ax.add_patch(Rectangle((68.6, 0.6), 30.9, 37.6, fc="#FBF7F0", ec="none"))
    ax.text(69.6, 37.3, "Channel and receiver", fontsize=6.8, fontweight="bold", color=NAVY, va="top")
    # 12 leads
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    for i, l in enumerate(leads):
        yy = 32.6 - i * 2.45
        ax.text(2.2, yy + 0.3, l, fontsize=5.3, ha="right", va="center", color=INK2)
        _ecg(ax, 2.7, yy, 6.3, 1.5, "#1B9E8A" if l == "V5" else NAVY)
    _arrow(ax, 9.6, 19.2, 11.6, 19.2)
    _box(ax, 12.1, 14.6, 10.6, 9.2, "Shared per-lead\nencoder\n(1-D ResNet,\nd = 32)", fc="#E3E9F2", bold=False)
    ax.text(17.4, 13.0, "trained once on clean\n12-lead input, then frozen", fontsize=5.5, ha="center", va="top", color=INK2)
    ax.text(17.4, 25.2, "12 × 32 embeddings", fontsize=5.8, ha="center", va="bottom", color=INK)
    # split
    ax.plot([23.3, 25.0, 25.0], [19.2, 19.2, 19.2], color=NAVY, lw=1.0)
    ax.plot([25.0, 25.0], [9.2, 29.2], color=NAVY, lw=1.0)
    _arrow(ax, 25.0, 29.2, 27.0, 29.2)
    _arrow(ax, 25.0, 9.2, 27.0, 9.2)
    # ---- branch A: lead-selection model (lead budget, clean)
    ax.text(27.2, 35.4, "Q1  Lead budget (lead-selection model, clean link)", fontsize=6.6, fontweight="bold", color="#0B5394", va="center")
    _box(ax, 27.6, 25.6, 13.6, 7.2, "Lead selector\nrecord-specific top-K\n(K = 1, 3, 6)", fc="#DCEAF7", ec="#0B5394")
    _arrow(ax, 41.9, 29.2, 44.6, 29.2, "#0B5394")
    ax.text(48.2, 31.9, "K selected\nembeddings", fontsize=5.6, ha="center", va="bottom", color=INK2)
    _blocks(ax, 45.0, 28.0, 4, 1.45, 2.4, ["#2C7BB6", "#1A5A8E", "#3F8FCB", "#6AA9DA"])
    ax.text(48.2, 27.1, "K × 32 values, no noise", fontsize=5.5, ha="center", va="top", color=INK2)
    ax.text(48.2, 24.5, "Comparators: matched fixed V5,\nfixed subsets, learned ranking", fontsize=5.4, ha="center", va="top",
            color=INK2)
    _arrow(ax, 51.6, 29.2, 70.8, 29.2, "#0B5394")
    _box(ax, 71.3, 25.6, 11.0, 7.2, "Selector head\n(mean pool +\nlinear)", fc="#DCEAF7", ec="#0B5394")
    # ---- branch B: transmission at 16 complex uses
    ax.text(27.2, 19.3, "Q2  Transmission at 16 complex channel uses (K = 1)", fontsize=6.6, fontweight="bold",
            color="#A15C07", va="center")
    rows = [("Baseline: own selector + nested repetition, m·r = 16", "1 lead", True),
            ("SSCC: residual VQ → RM / uncoded → QPSK / 16-QAM", "1 lead", False),
            ("Learned JSCC (same selected lead)", "1 lead", False),
            ("Joint learned JSCC", "12 leads", False)]
    for i, (t, sees, orig) in enumerate(rows):
        yy = 12.0 - i * 3.15
        ax.add_patch(FancyBboxPatch((27.6, yy), 30.6, 2.5, boxstyle="round,pad=0.15,rounding_size=0.8",
                                    fc="#FDEBD3" if orig else "white", ec="#A15C07", lw=0.8))
        ax.text(28.3, yy + 1.25, t, fontsize=5.6, va="center", color=INK)
        ax.text(58.9, yy + 1.25, f"sees {sees}", fontsize=5.4, va="center", color="#A15C07", fontweight="bold")
    ax.text(42.9, 1.1, "each option: 16 complex symbols, energy 16 per record", fontsize=5.4, ha="center", va="center",
            color=INK2)
    _blocks(ax, 27.6, 15.6, 16, 0.75, 1.3, ["#F08A24", "#D86A20", "#E8A85C", "#F5C68F"])
    ax.text(44.2, 16.25, "Repetition baseline: selector, ranker and\nhead trained separately (AWGN training)",
            fontsize=5.3, va="center", color=INK2)
    for i in range(4):
        yy = 12.0 - i * 3.15 + 1.25
        ax.plot([65.6, 66.9], [yy, yy], color="#A15C07", lw=0.8)
    ax.plot([66.9, 66.9], [12.0 - 3 * 3.15 + 1.25, 12.0 + 1.25], color="#A15C07", lw=0.8)
    _arrow(ax, 66.9, 8.3, 70.8, 8.3, "#A15C07")
    _box(ax, 71.3, 5.0, 11.0, 6.6, "AWGN, block\nRayleigh or Rician\n(SNR −15…20 dB)", fc="white", ec="#A15C07")
    _arrow(ax, 82.8, 8.3, 85.0, 8.3, "#A15C07")
    _box(ax, 85.5, 5.0, 12.6, 6.6, "Receiver: equalize,\ncombine repeats or\ndecode; zero-fill; head", fc="#FDEBD3", ec="#A15C07")
    # diagnosis output
    _box(ax, 85.5, 19.0, 12.6, 14.0, "", fc="white", ec=NAVY, lw=1.2)
    ax.text(91.8, 31.6, "Multilabel diagnosis", fontsize=6.3, fontweight="bold", ha="center", va="center")
    for i, (c, v) in enumerate(zip(CLASSES, (0.8, 0.45, 0.55, 0.35, 0.4))):
        yy = 29.2 - i * 2.2
        ax.text(87.0, yy, c, fontsize=5.6, va="center")
        ax.add_patch(Rectangle((90.6, yy - 0.6), 6.4 * v, 1.2, fc=NAVY, ec="none"))
    _arrow(ax, 82.8, 29.2, 85.0, 29.2, "#0B5394")
    _arrow(ax, 91.8, 11.9, 91.8, 18.6, "#A15C07")
    save(fig, "fig1_framework")


# ====================================================================== Fig. 2: lead budget + per-class deltas
METHODS = [("proposed", "Record-specific selector", "o", "-"),
           ("global_learned_k", "Population-level learned ranking", "D", "-"),
           ("fixed_clinical_k", "Fixed clinical subset (K=1: II)", "s", "-"),
           ("random_k", "Random subset", "x", "-")]


def _budget_stats():
    cm = load("phase2a_clinical_metrics.json")["per_condition"]
    se = load("subset_enum_orig_test.json")["policies"]
    k1 = load("k1_eval_test.json")
    st = {}
    for m, *_ in METHODS:
        for k in (1, 3, 6):
            for met, key in (("auroc", "macro_auroc_consistency_check"), ("auprc", "macro_auprc")):
                v = [cm[m][str(s)][str(k)][key] for s in (0, 1, 2)]
                st[(m, k, met)] = (np.mean(v), np.std(v))
    for k in (1, 3, 6):
        for met in ("auroc", "auprc"):
            v = [se[str(k)][str(s)]["enumerated_best_fixed"][f"macro_{met}"] for s in (0, 1, 2)]
            st[("best_fixed", k, met)] = (np.mean(v), np.std(v))
    for met in ("auroc", "auprc"):
        v = [cm["ceiling_k12"][str(s)]["12"]["macro_auroc_consistency_check" if met == "auroc" else "macro_auprc"] for s in (0, 1, 2)]
        st[("k12", 12, met)] = (np.mean(v), np.std(v))
        v = [k1["absolute"]["orig_C"]["per_seed"][str(s)][f"macro_{met}"] for s in (0, 1, 2)]
        st[("v5", 1, met)] = (np.mean(v), np.std(v))
    return st


def fig2_budget_and_classes():
    st = _budget_stats()
    k1 = load("k1_eval_test.json")
    p = k1["paired"]["orig_A-orig_C"]
    hm = {r["row"]: r["n_records"] for r in load("selection_heatmap_test.json")["rows"]}
    assert abs(p["macro_auroc"]["mean"] - 0.0175) < 5e-5
    fig = plt.figure(figsize=(DOUBLE, 5.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.62, wspace=0.10, left=0.155, right=0.985, top=0.84,
                          bottom=0.085)
    ks = [1, 3, 6]
    for j, met in enumerate(("auroc", "auprc")):
        ax = fig.add_subplot(gs[0, j])
        for m, lab, mk, ls in METHODS:
            mu = [st[(m, k, met)][0] for k in ks]
            sd = [st[(m, k, met)][1] for k in ks]
            ax.errorbar(ks, mu, yerr=sd, color=OI[m], marker=mk, ls=ls, lw=1.3, ms=4.2, capsize=2, mec=OI[m] if mk == "x" else "white",
                        mew=0.9 if mk == "x" else 0.5)
        mu = [st[("best_fixed", k, met)][0] for k in ks]
        sd = [st[("best_fixed", k, met)][1] for k in ks]
        ax.errorbar(ks, mu, yerr=sd, color=PURPLE, marker="P", ls="--", lw=1.3, ms=5, capsize=2, mec="white", mew=0.5)
        v5 = st[("v5", 1, met)]
        ax.errorbar([1], [v5[0]], yerr=[v5[1]], color=INK, marker="*", ls="", ms=7, capsize=2, mec="white", mew=0.4)
        k12 = st[("k12", 12, met)][0]
        ax.axhline(k12, color=MUTED, ls=":", lw=1.1)
        ax.text(7.5, k12 + (0.002 if met == "auroc" else 0.004), "all 12 leads", fontsize=5.8, color=INK2, va="bottom", ha="right")
        ax.set_ylim(top=0.93 if met == "auroc" else 0.81)
        ax.set_xticks(ks)
        ax.set_xlim(0.6, 7.6)
        ax.set_xlabel("Number of transmitted leads K (16K complex channel uses)")
        ax.set_ylabel(f"Macro-{met.upper()}")
        ax.grid(color=GRID, lw=0.5)
        ax.set_title(f"Macro-{met.upper()} vs lead budget", loc="left", fontweight="bold")
        letter(ax, "ab"[j], x=-0.03 if j else -0.13)
        for m, k in (("proposed", 1), ("best_fixed", 6)):
            pass
    handles = [Line2D([], [], color=OI[m], marker=mk, ls=ls, lw=1.3, ms=4.2, mec=OI[m] if mk == "x" else "white", label=lab)
               for m, lab, mk, ls in METHODS]
    handles += [Line2D([], [], color=PURPLE, marker="P", ls="--", lw=1.3, ms=5, mec="white",
                       label="Best fixed subset (locked on validation)"),
                Line2D([], [], color=INK, marker="*", ls="", ms=7, mec="white", label="Matched fixed V5 (K = 1)"),
                Line2D([], [], color=MUTED, ls=":", lw=1.1, label="All 12 leads"),
                Line2D([], [], color=INK, marker="o", lw=1.4, ms=5, mec=INK, label="c, d: 95% CI excludes 0"),
                Line2D([], [], color=GRAY, marker="o", lw=1.4, ms=5, mfc="white", mec=GRAY, label="c, d: 95% CI includes 0")]
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.06, 0.998), columnspacing=1.1,
               handlelength=2.2)
    # ---- per-class deltas
    rows = [("Macro", f"n = {k1['n_records']:,}", "macro_auroc", "macro_auprc")] + \
           [(c, f"{hm[c]} pos. ({k1['prevalence'][c]:.0%})", f"auroc_{c}", f"auprc_{c}") for c in CLASSES]
    y = [0] + [i + 0.6 for i in range(1, len(rows))]
    for j, (met, lim, step) in enumerate((("auroc", (-0.075, 0.125), 0.04), ("auprc", (-0.13, 0.17), 0.05))):
        ax = fig.add_subplot(gs[1, j])
        ax.axvline(0, color=INK, lw=0.7)
        ax.grid(axis="x", color=GRID, lw=0.5)
        for yi, (lab, sub, ka, kp) in zip(y, rows):
            e = p[ka if met == "auroc" else kp]
            m, (lo, hi) = e["mean"], e["ci95"]
            sig = lo > 0 or hi < 0
            col = INK if sig else GRAY
            ax.plot([lo, hi], [yi, yi], color=col, lw=1.4)
            ax.plot(m, yi, "o", ms=5, color=col if sig else "white", mec=col, mew=1.1, zorder=3)
            txt = fmt(m, 4 if lab == "Macro" else 3) + (f" {ci_txt((lo, hi), 4)}" if lab == "Macro" else "")
            kw = dict(va="center", fontsize=6.3, color=INK, fontweight="bold" if lab in ("Macro", "CD", "MI", "STTC") else "normal")
            if m >= 0:
                ax.text(hi + 0.004, yi, txt, ha="left", **kw)
            else:
                ax.text(lo - 0.004, yi, txt, ha="right", **kw)
        ax.set_xlim(*lim)
        set_xticks(ax, step)
        ax.set_ylim(y[-1] + 0.6, -0.6)
        ax.set_yticks(y)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        if j == 0:
            ax.set_yticklabels([""] * len(rows))
            for yi, (lab, sub, *_) in zip(y, rows):
                ax.text(-0.02, yi - 0.15, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7,
                        fontweight="bold")
                ax.text(-0.02, yi + 0.27, sub, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=5.6,
                        color=INK2)
        else:
            ax.set_yticklabels([])
        ttl = "ΔAUROC" if met == "auroc" else "ΔAUPRC"
        ax.set_title(f"K = 1: selector − matched fixed V5, {ttl}", loc="left", fontweight="bold")
        ax.set_xlabel(f"{ttl} (positive favours the selector)")
        letter(ax, "cd"[j], x=-0.03 if j else -0.2)
    save(fig, "fig2_lead_budget_and_classes")
    return st


# ====================================================================== Fig. 3: transmission at 16 uses
COMM = [("rep_orig_policy", "Repetition baseline, encoder sees 1 lead", REF, "o", "-", 1.3),
        ("sscc_lookup", "SSCC (RVQ + RM/uncoded, QPSK/16-QAM), 1 lead", AMBER, "D", "--", 1.3),
        ("jscc_ctrl", "Learned JSCC, encoder sees 1 lead", PURPLE_L, "s", "-", 1.3),
        ("jscc_e2e", "Joint learned JSCC, encoder sees all 12 leads", PURPLE, "^", "-", 1.6)]
COND_T = {"awgn_perfect": "AWGN", "rayleigh_perfect": "Block Rayleigh, perfect CSI", "rayleigh_ls2": "Block Rayleigh, 2-pilot LS CSI"}
POLICY_TXT = {"awgn_perfect": "r = 4 at SNR ≤ −10 dB, else r = 2", "rayleigh_perfect": "r = 4 at SNR ≤ 0 dB, else r = 2"}


def _snr_axis(ax, xlabel=True):
    ax.set_xticks(SNRS)
    ax.set_xticklabels([fmt(s, 0, sign=False) for s in SNRS])
    ax.set_xlim(-17, 22)
    ax.grid(color=GRID, lw=0.5)
    if xlabel:
        ax.set_xlabel("SNR (dB)")


def fig3_transmission():
    rep = load("comm_report_test.json")["conditions"]
    der = load("comm_derived_contrasts_test.json")
    assert der["reproduction_check_max_abs_vs_locked_report"] == 0.0
    for cond, tau in (("awgn_perfect", -10), ("rayleigh_perfect", 0)):
        assert rep[cond]["policies"]["rep_orig_policy"] == {str(s): ("rep_orig_r4" if s <= tau else "rep_orig_r2") for s in SNRS}
    e2e = rep["awgn_perfect"]["contrasts"]["jscc_e2e-rep_orig_policy"]["policy"]["mean"]
    one = der["conditions"]["awgn_perfect"]["jscc_ctrl-rep_orig_policy"]["policy"]["mean"]
    ctl = rep["awgn_perfect"]["contrasts"]["jscc_ctrl-rep_ctrl_policy"]["policy"]["mean"]
    assert round(e2e, 3) == 0.095 and round(one, 3) == 0.037 and round(ctl, 3) == 0.013
    fig = plt.figure(figsize=(DOUBLE, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1], hspace=0.40, wspace=0.12, left=0.09, right=0.985, top=0.80, bottom=0.095)
    for j, cond in enumerate(("awgn_perfect", "rayleigh_perfect")):
        ax = fig.add_subplot(gs[0, j])
        for key, lab, col, mk, ls, lw in COMM:
            ax.plot(SNRS, [rep[cond]["absolute"][key]["per_snr_auroc"][str(s)] for s in SNRS], color=col, marker=mk, ls=ls, lw=lw,
                    ms=4, mec="white", mew=0.6, zorder=3)
        _snr_axis(ax, xlabel=False)
        ax.set_ylim(0.50, 0.95)
        if j == 0:
            ax.set_ylabel("Macro-AUROC")
        else:
            ax.set_yticklabels([])
        ax.set_title(COND_T[cond], loc="left", fontweight="bold")
        ax.text(0.99, 0.03, f"Repetition baseline policy: {POLICY_TXT[cond]}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=5.8, color=INK2)
        letter(ax, "ab"[j], x=-0.03 if j else -0.12)
        bx = fig.add_subplot(gs[1, j])
        bx.axhline(0, color=INK, lw=0.7)
        diffs = [("sscc_lookup-rep_orig_policy", der["conditions"][cond], AMBER, "D", "--", "SSCC, 1 lead"),
                 ("jscc_ctrl-rep_orig_policy", der["conditions"][cond], PURPLE_L, "s", "-", "JSCC, 1 lead"),
                 ("jscc_e2e-rep_orig_policy", rep[cond]["contrasts"], PURPLE, "^", "-", "Joint JSCC, 12 leads")]
        for key, src, col, mk, ls, _ in diffs:
            c = src[key]
            bx.fill_between(SNRS, [c[str(s)]["ci95"][0] for s in SNRS], [c[str(s)]["ci95"][1] for s in SNRS], color=col,
                            alpha=0.22, lw=0)
            bx.plot(SNRS, [c[str(s)]["mean"] for s in SNRS], color=col, marker=mk, ls=ls, lw=1.2, ms=3.5, mec="white", mew=0.5)
        _snr_axis(bx)
        bx.set_ylim(-0.27, 0.15)
        if j == 0:
            bx.set_ylabel("Δ macro-AUROC vs\nrepetition baseline")
        else:
            bx.set_yticklabels([])
        lines = ["Mean over the 7 SNRs (equal weight):"] + \
                [f"{short}  {fmt(src[key]['policy']['mean'])} {ci_txt(src[key]['policy']['ci95'])}" for key, src, *_, short in diffs[::-1]]
        bx.text(0.99, 0.05, "\n".join(lines), transform=bx.transAxes, ha="right", va="bottom", fontsize=5.8, linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.5))
        letter(bx, "cd"[j], x=-0.03 if j else -0.12)
    handles = [Line2D([], [], color=col, marker=mk, ls=ls, lw=lw, ms=4, mec="white", label=lab) for _, lab, col, mk, ls, lw in COMM]
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.08, 0.93), columnspacing=1.6,
               handlelength=2.6)
    fig.text(0.01, 0.99, "16 complex channel uses per record (energy 16), K = 1, PTB-XL test fold", ha="left", va="top",
             fontsize=7.5, fontweight="bold")
    save(fig, "fig3_transmission_16_uses")


# ====================================================================== Fig. 4: external
def fig4_external():
    k1 = load("k1_eval_test.json")
    ex = load("external_eval.json")["outcomes"]
    p = k1["paired"]["orig_A-orig_C"]
    cd = ex["CD"]["paired"]
    assert round(cd["A-C_auroc"]["mean"], 3) == 0.033 and round(cd["A-C_auprc"]["mean"], 3) == -0.091
    groups = [("CD", [("PTB-XL test fold", p["auroc_CD"], p["auprc_CD"], PURPLE, "o"),
                      ("External cohort", cd["A-C_auroc"], cd["A-C_auprc"], AMBER, "s")]),
              ("STTC", [("PTB-XL test fold: STTC", p["auroc_STTC"], p["auprc_STTC"], PURPLE, "o"),
                        ("External: partial STTC", ex["STTC_partial"]["paired"]["A-C_auroc"], ex["STTC_partial"]["paired"]["A-C_auprc"],
                         AMBER, "s")])]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), gridspec_kw={"wspace": 0.12})
    fig.subplots_adjust(left=0.20, right=0.985, top=0.82, bottom=0.19)
    ys = [0, 0.75, 2.05, 2.8]
    for k, (ax, met) in enumerate(zip(axes, ("auroc", "auprc"))):
        ax.axvline(0, color=INK, lw=0.7)
        ax.grid(axis="x", color=GRID, lw=0.5)
        i, labels = 0, []
        for g, items in groups:
            for lab, ea, ep, col, mk in items:
                e = ea if met == "auroc" else ep
                m, (lo, hi) = e["mean"], e["ci95"]
                ax.plot([lo, hi], [ys[i]] * 2, color=col, lw=1.5)
                ax.plot(m, ys[i], mk, ms=5, color=col, mec="white", mew=0.8, zorder=3)
                ext_cd = g == "CD" and col == AMBER
                txt = fmt(m) + (f" {ci_txt((lo, hi))}" if ext_cd else "")
                kw = dict(va="center", fontsize=6.4, color=INK, fontweight="bold" if ext_cd else "normal",
                          bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none") if ext_cd else None)
                if m >= 0 or (ext_cd and met == "auprc"):
                    ax.text(hi + 0.004, ys[i], txt, ha="left", **kw)
                else:
                    ax.text(lo - 0.004, ys[i], txt, ha="right", **kw)
                labels.append(lab)
                i += 1
        ax.set_xlim(*((-0.075, 0.125) if met == "auroc" else (-0.20, 0.17)))
        set_xticks(ax, 0.04 if met == "auroc" else 0.05)
        ax.set_ylim(3.35, -0.55)
        ax.set_yticks(ys)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.axhline(1.4, color=GRID, lw=0.6)
        if k == 0:
            ax.set_yticklabels(labels, fontsize=6.6)
            for yc, g in ((0.375, "CD"), (2.425, "STTC")):
                ax.text(-0.36, yc, g, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.5, fontweight="bold")
        else:
            ax.set_yticklabels([])
        ttl = "ΔAUROC" if met == "auroc" else "ΔAUPRC"
        ax.set_title(ttl, loc="left", fontweight="bold")
        ax.set_xlabel(f"{ttl}, selector − matched fixed V5 (within cohort)")
        letter(ax, "ab"[k], x=-0.03 if k else -0.45)
    handles = [Line2D([], [], color=PURPLE, marker="o", lw=1.5, mec="white", ms=5, label="PTB-XL test fold (patient-level bootstrap)"),
               Line2D([], [], color=AMBER, marker="s", lw=1.5, mec="white", ms=5, label="External cohort (record-level bootstrap)")]
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.19, 1.0))
    save(fig, "fig4_external")


# ====================================================================== supplementary figures
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def figS2_heatmap():
    hm = {r["row"]: r for r in load("selection_heatmap_test.json")["rows"]}
    rows = [("all records", "All records")] + [(c, c) for c in CLASSES]
    M = np.array([[hm[k]["freq"][l] for l in LEADS] for k, _ in rows]) * 100
    assert np.allclose(M.sum(1), 100)
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.45))
    fig.subplots_adjust(left=0.15, right=0.90, top=0.80, bottom=0.05)
    cmap = LinearSegmentedColormap.from_list("pur", ["#FFFFFF", "#E7E2F1", PURPLE_L, PURPLE, "#2E2350"])
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=50, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(12):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.4, color="white" if M[i, j] > 27 else INK)
    ax.set_xticks(range(12))
    ax.set_xticklabels(LEADS)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{n}  (n = {hm[k]['n_records']:,})" for k, n in rows])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.axvline(5.5, color="white", lw=2.5)
    ax.axhline(0.5, color="white", lw=2.5)
    for x0, x1, lab in ((-0.4, 5.4, "Limb leads"), (5.6, 11.4, "Chest leads")):
        ax.annotate("", xy=(x0, -0.98), xytext=(x1, -0.98), annotation_clip=False, arrowprops=dict(arrowstyle="-", color=INK2, lw=1.0))
        ax.text((x0 + x1) / 2, -1.12, lab, ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    cax = fig.add_axes([0.915, 0.05, 0.015, 0.75])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=6)
    cb.set_label("% of records in row", fontsize=6.5)
    save(fig, "figS2_selection_heatmap")


def figS3_comm_all():
    rep = load("comm_report_test.json")["conditions"]
    series = COMM[:1] + [("rep_ctrl_policy", "Repetition retrained on the selector's lead, 1 lead", REF_L, "v", ":", 1.4)] + COMM[1:]
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 4.4), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.80, bottom=0.10, hspace=0.30, wspace=0.10)
    for j, cond in enumerate(COND_T):
        for i, met in enumerate(("auroc", "auprc")):
            ax = axes[i, j]
            for key, lab, col, mk, ls, lw in series:
                ax.plot(SNRS, [rep[cond]["absolute"][key][f"per_snr_{met}"][str(s)] for s in SNRS], color=col, marker=mk, ls=ls, lw=lw,
                        ms=3.6, mec=INK2 if col == REF_L else "white", mew=0.5)
            _snr_axis(ax, xlabel=(i == 1))
            ax.set_ylim(*((0.50, 0.95) if met == "auroc" else (0.25, 0.85)))
            if j == 0:
                ax.set_ylabel(f"Macro-{met.upper()}")
            else:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(COND_T[cond] + (" (+2 uses)" if cond == "rayleigh_ls2" else ""), loc="left", fontweight="bold", fontsize=7)
            letter(ax, "abcdef"[i * 3 + j], x=-0.03 if j else -0.16)
    handles = [Line2D([], [], color=col, marker=mk, ls=ls, lw=lw, ms=4, mec=INK2 if col == REF_L else "white", label=lab)
               for _, lab, col, mk, ls, lw in series]
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.07, 0.995), columnspacing=1.4, handlelength=2.6)
    save(fig, "figS3_transmission_auprc_csi")


def figS4_controlled():
    rep = load("comm_report_test.json")["conditions"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.35), sharey=True)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.80, bottom=0.18, wspace=0.08)
    for j, cond in enumerate(COND_T):
        ax = axes[j]
        ax.axhline(0, color=INK, lw=0.7)
        lines = ["Mean over the 7 SNRs:"]
        for key, col, mk, ls, short in (("jscc_ctrl-rep_ctrl_policy", PURPLE_L, "s", "-", "JSCC"),
                                        ("sscc_lookup-rep_ctrl_policy", AMBER, "D", "--", "SSCC")):
            c = rep[cond]["contrasts"][key]
            ax.fill_between(SNRS, [c[str(s)]["ci95"][0] for s in SNRS], [c[str(s)]["ci95"][1] for s in SNRS], color=col, alpha=0.22, lw=0)
            ax.plot(SNRS, [c[str(s)]["mean"] for s in SNRS], color=col, marker=mk, ls=ls, lw=1.2, ms=3.5, mec="white", mew=0.5)
            lines.append(f"{short} {fmt(c['policy']['mean'])} {ci_txt(c['policy']['ci95'])}")
        ax.text(0.98, 0.05, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.5))
        _snr_axis(ax)
        ax.set_title(COND_T[cond], loc="left", fontweight="bold", fontsize=7)
        if j == 0:
            ax.set_ylabel("Δ macro-AUROC vs repetition\n(same lead and embedding)")
        letter(ax, "abc"[j], x=-0.03 if j else -0.2)
    handles = [Line2D([], [], color=PURPLE_L, marker="s", lw=1.2, mec="white", label="Learned JSCC − repetition (same lead)"),
               Line2D([], [], color=AMBER, marker="D", ls="--", lw=1.2, mec="white", label="SSCC − repetition (same lead)")]
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.095, 1.0))
    save(fig, "figS4_controlled_transmission")


def figS5_backbone():
    b = load("backbone_test.json")
    k1 = load("k1_eval_test.json")
    fig = plt.figure(figsize=(DOUBLE, 3.0))
    a1 = fig.add_axes([0.085, 0.17, 0.30, 0.71])
    a2 = fig.add_axes([0.62, 0.17, 0.365, 0.71])
    e4 = load("e4_eval_test.json")
    cols = [(0, [b["absolute"][f"orig_seed{s}"]["macro_auroc"] for s in (0, 1, 2)], GRAY, "o"),
            (1, [b["absolute"][f"new_seed{s}"]["macro_auroc"] for s in (0, 1, 2)], PURPLE_L, "^"),
            (2, e4["absolute"]["e4_12"]["macro_auroc"]["per_seed"], PURPLE, "^")]
    for x, v, col, mk in cols:
        a1.plot([x] * 3, v, mk, color=col, ms=5, mec="white", mew=0.6)
        a1.plot([x - 0.18, x + 0.18], [np.mean(v)] * 2, color=INK, lw=1.2)
    sub = b["absolute"]["submitted_seed42"]["macro_auroc"]
    a1.plot([-0.25], [sub], "*", color=AMBER, ms=7, mec="white", mew=0.4)
    a1.text(-0.33, sub, "submitted\nmodel", ha="right", va="center", fontsize=5.8, color=INK2)
    dd = b["new_minus_orig_retrained_12lead"]["macro_auroc"]
    de = e4["paired"]["e4_12-o40_12"]["macro_auroc"]
    a1.text(0.03, 0.97, f"Lead-aware 40 ep. − baseline: Δ {fmt(dd['mean'], 4)} {ci_txt(dd['ci95'], 4)}\n"
            f"≤120 ep. − 40 ep.: Δ {fmt(de['mean'], 4)} {ci_txt(de['ci95'], 4)}", transform=a1.transAxes, ha="left", va="top",
            fontsize=5.6)
    a1.set_xticks([0, 1, 2])
    a1.set_xticklabels(["Baseline arch.\n(12k, retrained)", "Lead-aware\n40 epochs", "Lead-aware\n≤120 epochs"], fontsize=6.0)
    a1.set_xlim(-1.05, 2.45)
    allv = [q for c in cols for q in c[1]] + [sub]
    a1.set_ylim(min(allv) - 0.002, max(allv) + 0.0075)
    a1.set_ylabel("12-lead macro-AUROC")
    a1.set_title("12 leads (dots: seeds 0–2)", loc="left", fontweight="bold", fontsize=7)
    a1.grid(axis="y", color=GRID, lw=0.5)
    letter(a1, "a", x=-0.22)
    comps = [("orig_A-orig_C", "Baseline encoder:\nselector − fixed V5", GRAY, k1),
             ("new_A-new_C", "Lead-aware, 40 ep.:\nselector − fixed V5", PURPLE_L, k1),
             ("new_A-new_BEST", f"Lead-aware, 40 ep.: selector\n− its best fixed lead ({k1['new_backbone_best_fixed_lead']})", PURPLE_L, k1),
             ("e4_A-e4_C", "Lead-aware, ≤120 ep.:\nselector − fixed V5", PURPLE, e4),
             ("e4_A-e4_BEST", f"Lead-aware, ≤120 ep.: selector\n− its best fixed lead ({e4['e4_best_fixed_lead']})", PURPLE, e4)]
    for yi, (k, lab, col, src) in enumerate(comps):
        for met, dy, mk in (("macro_auroc", -0.14, "o"), ("macro_auprc", 0.14, "s")):
            e = src["paired"][k][met]
            a2.plot(e["ci95"], [yi + dy] * 2, color=col, lw=1.4)
            a2.plot(e["mean"], yi + dy, mk, color=col, ms=4.5, mec="white", mew=0.6)
            a2.text(e["ci95"][1] + 0.003, yi + dy, fmt(e["mean"]), va="center", fontsize=6.1)
    a2.axvline(0, color=INK, lw=0.7)
    a2.set_yticks(range(len(comps)))
    a2.set_yticklabels([c[1] for c in comps], fontsize=6.0)
    a2.set_ylim(len(comps) - 0.4, -0.75)
    a2.tick_params(axis="y", length=0)
    a2.spines["left"].set_visible(False)
    a2.set_xlim(-0.01, 0.09)
    a2.grid(axis="x", color=GRID, lw=0.5)
    a2.set_xlabel("Paired difference, K = 1 (95% CI)")
    a2.set_title("K = 1 selection on each backbone", loc="left", fontweight="bold", fontsize=7)
    a2.legend(handles=[Line2D([], [], marker="o", color=INK2, ls="", ms=4.5, label="macro-AUROC"),
                       Line2D([], [], marker="s", color=INK2, ls="", ms=4.5, label="macro-AUPRC")],
              loc="upper right", frameon=False, fontsize=6, ncol=2, bbox_to_anchor=(1.0, 1.0))
    letter(a2, "b", x=-0.62)
    save(fig, "figS5_backbone")


def figS6_gate():
    f9 = load("gate_fold9_six_seeds.json")["summary"]["gate_eq_vs_margin@50"]
    l3 = load("gate_fold10_locked3.json")["summary"]["gate_eq_vs_margin@50"]
    a6 = load("gate_fold10_all6.json")["summary"]["gate_eq_vs_margin@50"]
    locked = load("gate_lock.json")["locked_seeds"]
    assert locked == [0, 3, 5]
    stages = [("Validation fold, six trained models\n(development; used to select three)", f9, list(range(6)), GRAY),
              ("Test fold, three models selected\non validation (seeds 0, 3, 5)", l3, locked, PURPLE),
              ("Test fold, all six models\n(pre-specified sensitivity analysis)", a6, list(range(6)), AMBER)]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.5), sharey=True, gridspec_kw={"wspace": 0.08})
    fig.subplots_adjust(left=0.235, right=0.985, top=0.78, bottom=0.18)
    for k, (ax, met, ttl) in enumerate(zip(axes, ("CD_auprc", "STTC_auprc", "macro_auroc"), ("CD AUPRC", "STTC AUPRC", "macro-AUROC"))):
        ax.axvline(0, color=INK, lw=0.7)
        ax.grid(axis="x", color=GRID, lw=0.5)
        for yi, (lab, S, seeds, col) in enumerate(stages):
            e = S[met]
            ax.plot(e["ci95"], [yi, yi], color=col, lw=1.6)
            ax.plot(e["mean"], yi, "o", color=col, ms=5.5, mec="white", mew=0.8, zorder=4)
            for sd, v in zip(seeds, e["per_seed"]):
                ax.plot(v, yi + 0.27, "o", ms=3.2, color=col if sd in locked else "white", mec=col, mew=0.7, zorder=3)
            ax.text(e["ci95"][1] + (0.003 if met != "macro_auroc" else 0.0008), yi, fmt(e["mean"]), va="center", fontsize=6.1,
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
        ax.set_yticks(range(3))
        ax.set_yticklabels([s[0] for s in stages], fontsize=6.2)
        ax.set_ylim(2.6, -0.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(*((-0.055, 0.085) if met != "macro_auroc" else (-0.025, 0.0125)))
        set_xticks(ax, 0.03 if met != "macro_auroc" else 0.01)
        ax.set_title(f"Δ {ttl}", loc="left", fontweight="bold", fontsize=7)
        letter(ax, "abc"[k], x=-0.03 if k else -0.72)
    axes[1].set_xlabel("Gate − selector-margin rule, 50% of switchable records switched")
    handles = [Line2D([], [], marker="o", color=INK2, ls="-", lw=1.6, ms=5.5, mec="white", label="Mean of models, 95% CI"),
               Line2D([], [], marker="o", color=INK2, ls="", ms=3.2, label="Single model, selected"),
               Line2D([], [], marker="o", color="white", mec=INK2, ls="", ms=3.2, label="Single model, not selected")]
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.23, 1.0))
    save(fig, "figS6_gate")


# ====================================================================== tables
def tex(v, d=3, sign=True):
    v = 0.0 if round(v, d) == 0 else v
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "$-$")


def tci(ci, d=3):
    return f"[{tex(ci[0], d, False)}, {tex(ci[1], d, False)}]"


def write_tab(name, body):
    with open(os.path.join(OUT["tab"], f"{name}.tex"), "w") as fh:
        fh.write(body)
    print("table", name)


def table_backbone():
    # ---- Table S-II: backbones (with the post hoc E4 extended-training rows)
    b = load("backbone_test.json")
    body = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule", "Model & Encoder params & Seed & Macro-AUROC / AUPRC \\\\", "\\midrule",
            f"Submitted model & {b['info']['orig_seed0']['params_encoder']:,} & 42 & "
            f"{b['absolute']['submitted_seed42']['macro_auroc']:.3f} / {b['absolute']['submitted_seed42']['macro_auprc']:.3f} \\\\"]
    for arch, lab in (("orig", "Baseline arch., retrained"), ("new", "Lead-aware ResNet")):
        for s in (0, 1, 2):
            a = b["absolute"][f"{arch}_seed{s}"]
            lab_s = lab + (", 40 epochs" if arch == "new" else "")
            body.append(f"{lab_s} & {b['info'][f'{arch}_seed{s}']['params_encoder']:,} & {s} & {a['macro_auroc']:.3f} / {a['macro_auprc']:.3f} \\\\")
    e4 = load("e4_eval_test.json")
    for s in (0, 1, 2):
        body.append(f"Lead-aware ResNet, $\\leq$120 epochs & {b['info'][f'new_seed{s}']['params_encoder']:,} & {s} & "
                    f"{e4['absolute']['e4_12']['macro_auroc']['per_seed'][s]:.3f} / {e4['absolute']['e4_12']['macro_auprc']['per_seed'][s]:.3f} \\\\")
    d = b["new_minus_orig_retrained_12lead"]
    de = e4["paired"]["e4_12-o40_12"]
    body += ["\\midrule", f"\\multicolumn{{4}}{{@{{}}l}}{{Paired difference (lead-aware, 40 epochs $-$ retrained baseline): AUROC {tex(d['macro_auroc']['mean'], 4)} "
             f"{tci(d['macro_auroc']['ci95'], 4)}, AUPRC {tex(d['macro_auprc']['mean'], 4)} {tci(d['macro_auprc']['ci95'], 4)}}} \\\\",
             f"\\multicolumn{{4}}{{@{{}}l}}{{Paired difference (lead-aware, $\\leq$120 $-$ 40 epochs): AUROC {tex(de['macro_auroc']['mean'], 4)} "
             f"{tci(de['macro_auroc']['ci95'], 4)}, AUPRC {tex(de['macro_auprc']['mean'], 4)} {tci(de['macro_auprc']['ci95'], 4)}}} \\\\",
             "\\bottomrule", "\\end{tabular}"]
    write_tab("tabS_backbone", "\n".join(body) + "\n")


def tables(st):
    k1 = load("k1_eval_test.json")
    P = k1["paired"]
    H = k1["key_secondary_holm"]
    # ---- Table II: K = 1 endpoints
    rows = [("orig_A-orig_C", "Selector $-$ matched fixed V5 (primary)"),
            ("orig_A-orig_B", "Same head: selector's lead $-$ V5"),
            ("orig_A-orig_GL", "Selector $-$ population-level ranking"),
            ("orig_A-orig_II", "Selector $-$ fixed Lead II"),
            ("new_A-new_C", "Lead-aware backbone: selector $-$ fixed V5"),
            ("new_A-new_BEST", f"Lead-aware backbone: selector $-$ fixed {k1['new_backbone_best_fixed_lead']}")]
    body = ["\\begin{tabular}{@{}lcc@{}}", "\\toprule", "Comparison & $\\Delta$AUROC [95\\% CI] & $\\Delta$AUPRC [95\\% CI] \\\\", "\\midrule"]
    for k, lab in rows:
        a, b = P[k]["macro_auroc"], P[k]["macro_auprc"]
        body.append(f"{lab} & {tex(a['mean'], 4)} {tci(a['ci95'], 4)} & {tex(b['mean'], 4)} {tci(b['ci95'], 4)} \\\\")
    body += ["\\midrule", "\\multicolumn{3}{@{}l}{Key secondary endpoints, selector $-$ matched fixed V5 (Holm-adjusted $p$)} \\\\"]
    for key, lab in (("auroc_CD", "CD AUROC"), ("auprc_CD", "CD AUPRC"), ("auroc_STTC", "STTC AUROC"), ("auprc_STTC", "STTC AUPRC"),
                     ("macro_auprc", "Macro-AUPRC")):
        e = H[key]
        p = e["p_holm"]
        ptxt = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        body.append(f"\\quad {lab} & {tex(e['mean'], 3)} {tci(e['ci95'], 3)} & $p$ = {ptxt} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tab("tab_k1_endpoints", "\n".join(body) + "\n")
    # ---- Table III: transmission summary
    rep = load("comm_report_test.json")["conditions"]
    der = load("comm_derived_contrasts_test.json")["conditions"]
    meth = [("rep_orig_policy", "Repetition baseline, 1 lead"), ("rep_ctrl_policy", "Repetition retrained on selected lead"),
            ("sscc_lookup", "SSCC, 1 lead"), ("jscc_ctrl", "Learned JSCC, 1 lead"), ("jscc_e2e", "Joint learned JSCC, 12 leads")]
    conds = ["awgn_perfect", "rayleigh_perfect", "rayleigh_ls2"]
    body = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule", "Method & AWGN & Rayleigh & Rayleigh, LS CSI \\\\", "\\midrule",
            "\\multicolumn{4}{@{}l}{Macro-AUROC, mean over the seven SNRs} \\\\"]
    for k, lab in meth:
        body.append(f"\\quad {lab} & " + " & ".join(f"{rep[c]['absolute'][k]['policy_auroc']:.3f}" for c in conds) + " \\\\")
    body += ["\\midrule", "\\multicolumn{4}{@{}l}{Paired differences [95\\% CI]} \\\\"]
    contr = [("jscc_e2e-rep_orig_policy", rep, "Joint JSCC $-$ repetition baseline"),
             ("jscc_ctrl-rep_orig_policy", der, "JSCC, 1 lead $-$ repetition baseline"),
             ("jscc_ctrl-rep_ctrl_policy", rep, "JSCC, 1 lead $-$ repetition, same lead"),
             ("sscc_lookup-rep_ctrl_policy", rep, "SSCC $-$ repetition, same lead"),
             ("rep_orig_policy-rep_orig_bestfixed", rep, "Adaptive $r$ $-$ best fixed $r$ (repetition baseline)")]
    for key, src, lab in contr:
        def get(c):
            return (src[c]["contrasts"][key] if src is rep else src[c][key])["policy"]
        d_ = 4 if key.startswith("rep_orig_policy-") else 3
        body.append(f"\\quad {lab} & " + " & ".join(f"{tex(get(c)['mean'], d_)} \\newline {{\\scriptsize {tci(get(c)['ci95'], d_)}}}" for c in conds) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tab("tab_transmission", "\n".join(body).replace(" \\newline ", "~") + "\n")
    # ---- Table IV: external
    ex = load("external_eval.json")["outcomes"]
    body = ["\\begin{tabular}{@{}lcc@{}}", "\\toprule", " & CD & Partial STTC \\\\", "\\midrule"]
    body.append("Records analysed & " + " & ".join(f"{ex[o]['n']:,}" for o in ("CD", "STTC_partial")) + " \\\\")
    body.append("Positives (prevalence) & " + " & ".join(f"{ex[o]['positives']:,} ({100 * ex[o]['prevalence']:.1f}\\%)" for o in ("CD", "STTC_partial")) + " \\\\")
    for m, lab in (("A", "Selector AUROC / AUPRC"), ("C", "Matched fixed V5 AUROC / AUPRC")):
        body.append(f"{lab} & " + " & ".join(f"{ex[o]['absolute'][m]['auroc']['mean']:.3f} / {ex[o]['absolute'][m]['auprc']['mean']:.3f}"
                                              for o in ("CD", "STTC_partial")) + " \\\\")
    for key, lab in (("A-C_auroc", "$\\Delta$AUROC [95\\% CI]"), ("A-C_auprc", "$\\Delta$AUPRC [95\\% CI]")):
        body.append(f"{lab} & " + " & ".join(f"{tex(ex[o]['paired'][key]['mean'])} {tci(ex[o]['paired'][key]['ci95'])}" for o in ("CD", "STTC_partial")) + " \\\\")
    body.append("Same-head $\\Delta$AUROC / $\\Delta$AUPRC & " + " & ".join(
        f"{tex(ex[o]['paired']['A-B_auroc']['mean'])} / {tex(ex[o]['paired']['A-B_auprc']['mean'])}" for o in ("CD", "STTC_partial")) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tab("tab_external", "\n".join(body) + "\n")
    table_backbone()
    # ---- Table S-III: subsets and oracle
    body = ["\\begin{tabular}{@{}llp{2.3cm}cccc@{}}", "\\toprule",
            "Backbone & $K$ & Locked best fixed subset & Best fixed & Clinical & Random & Selector \\\\", "\\midrule"]
    orc = []
    for bb, lab in (("orig", "Baseline"), ("new", "Lead-aware")):
        s = load(f"subset_enum_{bb}_test.json")
        lk = load(f"locked_subsets_{bb}.json")["locked"]
        for K in ("1", "3", "6"):
            def mean(key):
                return np.mean([s["policies"][K][str(sd)][key]["macro_auroc"] for sd in (0, 1, 2)])
            clin = "fixed_II" if K == "1" else "clinical_subset"
            body.append(f"{lab} & {K} & {', '.join(lk[K])} & {mean('enumerated_best_fixed'):.3f} & {mean(clin):.3f} & "
                        f"{mean('random_per_record_mean20'):.3f} & {mean('per_record_selector_choice'):.3f} \\\\")
        o = [s["oracle"][str(sd)]["mean_bce"] for sd in (0, 1, 2)]
        agree = [s["oracle"][str(sd)].get("selector_agrees_with_oracle", np.nan) for sd in (0, 1, 2)]
        orc.append((lab, np.mean([x["oracle"] for x in o]), np.mean([x["selector_choice"] for x in o]), np.mean([x["V5"] for x in o]),
                    np.nanmean(agree) if not np.all(np.isnan(agree)) else None))
    body += ["\\midrule", "\\multicolumn{7}{@{}l}{$K=1$ label-assisted oracle, mean BCE (oracle / selector's lead / V5)} \\\\"]
    for lab, a, b_, c, ag in orc:
        body.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\quad {lab}: {a:.3f} / {b_:.3f} / {c:.3f}}} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tab("tabS_subsets", "\n".join(body) + "\n")
    # ---- Table S-IV: operating points (CD, STTC + others) selector vs V5
    th = k1["thresholds"]
    body = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule", "Class & \\multicolumn{2}{c}{Selector} & \\multicolumn{2}{c}{Matched fixed V5} \\\\",
            " & Sensitivity & Specificity & Sensitivity & Specificity \\\\", "\\midrule"]
    for c in CLASSES:
        a, v = th["orig_A"][c], th["orig_C"][c]
        body.append(f"{c} & {a['sens_mean']:.3f} {tci(a['sens_ci95'])} & {a['spec_mean']:.3f} {tci(a['spec_ci95'])} & "
                    f"{v['sens_mean']:.3f} {tci(v['sens_ci95'])} & {v['spec_mean']:.3f} {tci(v['spec_ci95'])} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tab("tabS_operating_points", "\n".join(body) + "\n")
    # ---- Table S-V: calibration
    cal = k1["calibration"]
    body = ["\\begin{tabular}{@{}lccccc@{}}", "\\toprule", "Model & Seed & $T$ & ECE [95\\% CI] & ECE after $T$ [95\\% CI] & ECE, Platt \\\\", "\\midrule"]
    for m, lab in (("orig_A", "Selector"), ("orig_C", "Matched V5")):
        for r in cal[m]:
            body.append(f"{lab} & {r['seed']} & {r['T']:.2f} & {r['ece_before']:.3f} {tci(r['ece_before_ci95'])} & "
                        f"{r['ece_after']:.3f} {tci(r['ece_after_ci95'])} & {r['ece_platt_exploratory']:.3f} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tab("tabS_calibration", "\n".join(body) + "\n")
    # ---- Table S-VI: subgroups
    sg = k1["subgroups"]
    body = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule", "Group & $n$ (patients) & Selector [95\\% CI] & Matched V5 [95\\% CI] \\\\", "\\midrule"]
    order = [("sex", "0", "Male"), ("sex", "1", "Female"), ("age", "<40", "Age $<$40$^\\dagger$"), ("age", "40-59", "Age 40--59"),
             ("age", "60-74", "Age 60--74"), ("age", "75-89", "Age 75--89"), ("age", "90+", "Age $\\geq$90$^\\dagger$")]
    for g, k, lab in order:
        e = sg[g][k]
        assert e["descriptive_only"] == ("dagger" in lab)
        body.append(f"{lab} & {e['n']:,} ({e['n_patients']:,}) & {e['orig_A_macro_auroc_valid_classes']:.3f} {tci(e['orig_A_ci95'])} & "
                    f"{e['orig_C_macro_auroc_valid_classes']:.3f} {tci(e['orig_C_ci95'])} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tab("tabS_subgroups", "\n".join(body) + "\n")
    # ---- Table S-VII: external flow
    fl = load("label_flow.json")
    body = ["\\begin{tabular}{@{}lr@{}}", "\\toprule", "Step & Records \\\\", "\\midrule"]
    for lab, n in fl["flow"]:
        body.append(f"{lab[0].upper() + lab[1:]} & {n:,} \\\\")
    for o, lab in (("CD", "CD"), ("STTC_partial", "Partial STTC")):
        c = fl["outcome_counts"][o]
        body.append(f"{lab}: positive / negative / excluded (ambiguous code) & {c['positive']:,} / {c['negative']:,} / {c['excluded_ambiguous']:,} \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tab("tabS_external_flow", "\n".join(body) + "\n")


def table_stability():
    """Table S10: post hoc six-seed training stability of the primary comparison (orig_A - orig_C)."""
    d = load("stability6_test.json")
    locked = load("k1_eval_test.json")["primary"]["per_seed"]
    assert [d["per_seed"][str(s)]["diff"]["macro_auroc"] for s in (0, 1, 2)] == locked
    assert round(d["across_seeds"]["macro_auroc"]["mean"], 4) == 0.0161
    D = ["macro_auroc", "macro_auprc", "CD_auroc", "CD_auprc", "STTC_auroc", "STTC_auprc"]
    body = ["\\begin{tabular}{@{}c ccc ccc cc cc@{}}", "\\toprule",
            " & \\multicolumn{3}{c}{Macro-AUROC} & \\multicolumn{3}{c}{Macro-AUPRC} & \\multicolumn{2}{c}{CD $\\Delta$} "
            "& \\multicolumn{2}{c}{STTC $\\Delta$} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}",
            "Seed & Sel. & V5 & $\\Delta$ & Sel. & V5 & $\\Delta$ & AUROC & AUPRC & AUROC & AUPRC \\\\", "\\midrule"]
    for s in map(str, d["seeds"]):
        r = d["per_seed"][s]
        a, c, df = r["selector"], r["fixed_V5"], r["diff"]
        body.append(" & ".join([s, tex(a["macro_auroc"], 3, False), tex(c["macro_auroc"], 3, False), tex(df["macro_auroc"], 4),
                                tex(a["macro_auprc"], 3, False), tex(c["macro_auprc"], 3, False), tex(df["macro_auprc"], 4)]
                               + [tex(df[k], 3) for k in D[2:]]) + " \\\\")
    A = d["across_seeds"]
    dig = {k: (4 if k.startswith("macro") else 3) for k in D}

    def row(lab, f):
        return " & ".join([lab, "", "", f("macro_auroc"), "", "", f("macro_auprc")] + [f(k) for k in D[2:]]) + " \\\\"
    body += ["\\midrule",
             row("Mean", lambda k: tex(A[k]["mean"], dig[k])),
             row("SD", lambda k: tex(A[k]["sd_between_seeds"], dig[k], False)),
             row("Min", lambda k: tex(A[k]["min"], dig[k])),
             row("Max", lambda k: tex(A[k]["max"], dig[k])),
             "\\midrule",
             "\\multicolumn{11}{@{}l}{Patient-level paired bootstrap 95\\% CI of the six-seed mean difference: "
             + "; ".join(f"{lab} {tci(A[k]['patient_bootstrap_ci95'], dig[k])}" for k, lab in
                         (("macro_auroc", "macro-AUROC"), ("macro_auprc", "macro-AUPRC"))) + "} \\\\",
             "\\multicolumn{11}{@{}l}{\\quad " + "; ".join(f"{lab} {tci(A[k]['patient_bootstrap_ci95'], 3)}" for k, lab in
                         (("CD_auroc", "CD AUROC"), ("CD_auprc", "CD AUPRC"), ("STTC_auroc", "STTC AUROC"),
                          ("STTC_auprc", "STTC AUPRC"))) + "} \\\\",
             "\\bottomrule", "\\end{tabular}"]
    write_tab("tabS_stability", "\n".join(body) + "\n")


# ====================================================================== revision 2 (additional revision analyses)
R2 = {"split": "test"}
SERIES5 = [("rep_orig_policy", "Repetition baseline, 1 lead", REF, "o", "-", 1.3),
           ("rep_ctrl_policy", "Repetition retrained on the selector's lead, 1 lead", REF_L, "v", ":", 1.4),
           ("sscc_lookup", "SSCC, 1 lead", AMBER, "D", "--", 1.3),
           ("jscc_ctrl", "Learned JSCC, 1 lead", PURPLE_L, "s", "-", 1.3),
           ("jscc_e2e", "Joint learned JSCC, 12 leads", PURPLE, "^", "-", 1.6)]


def load_r2(name):
    return load(f"{name}_{R2['split']}.json")


def figS10_budget():
    """performance-budget curves, N = 8, 16, 32 complex channel uses (unit average power: energy N)"""
    b = load_r2("budget_report")
    rep16 = load("comm_report_test.json")["conditions"] if R2["split"] == "test" else None
    if rep16:  # N = 16 must equal the locked report
        for k in ("awgn_perfect", "rayleigh_perfect"):
            for m in ("rep_orig_policy", "jscc_e2e"):
                assert abs(b["absolute"][k][m]["policy_auroc"] - rep16[k]["absolute"][m]["policy_auroc"]) < 1e-9
    NS = [8, 16, 32]
    cn = lambda kind, n: f"{kind}_perfect" if n == 16 else f"{kind}_perfect_n{n}"
    fig = plt.figure(figsize=(DOUBLE, 3.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.42, wspace=0.12, left=0.09, right=0.985, top=0.79, bottom=0.165)
    xs = np.log2(NS)
    for j, kind in enumerate(("awgn", "rayleigh")):
        ax = fig.add_subplot(gs[0, j])
        for key, lab, col, mk, ls, lw in SERIES5:
            mu = [b["absolute"][cn(kind, n)][key]["policy_auroc"] for n in NS]
            sd = [b["absolute"][cn(kind, n)][key]["policy_auroc_sd_between_seeds"] for n in NS]
            ax.plot(xs, mu, color=col, ls=ls, lw=lw, zorder=2)
            ax.errorbar(xs, mu, yerr=sd, fmt="none", ecolor=col, elinewidth=0.8, capsize=1.6, zorder=2)
            for x, m_, n in zip(xs, mu, NS):
                frozen = key == "rep_orig_policy" and n != 16
                ax.plot(x, m_, mk, color=col, ms=4.2, mfc="white" if frozen else col, mec=col if frozen else ("white" if col != REF_L else INK2),
                        mew=0.9 if frozen else 0.6, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{n}\n(energy {n})" for n in NS])
        ax.set_xlim(2.75, 5.25)
        ax.set_ylim(0.55, 0.92)
        ax.grid(color=GRID, lw=0.5)
        if j == 0:
            ax.set_ylabel("Macro-AUROC, mean over 7 SNRs")
        else:
            ax.set_yticklabels([])
        ax.set_title("AWGN" if kind == "awgn" else "Block Rayleigh, perfect CSI", loc="left", fontweight="bold")
        letter(ax, "ab"[j], x=-0.03 if j else -0.12)
        bx = fig.add_subplot(gs[1, j])
        bx.axhline(0, color=INK, lw=0.7)
        for off, (key, col, mk, lab) in zip((-0.09, 0, 0.09), (("jscc_e2e-rep_orig_policy", PURPLE, "^", "Joint JSCC − repetition baseline"),
                                                               ("jscc_ctrl-rep_ctrl_policy", PURPLE_L, "s", "JSCC − repetition, same lead"),
                                                               ("sscc_lookup-rep_ctrl_policy", AMBER, "D", "SSCC − repetition, same lead"))):
            for x, n in zip(xs, NS):
                e = b["contrasts"][f"{cn(kind, n)}:{key}"]["policy"]["auroc"]
                bx.plot([x + off] * 2, e["ci95"], color=col, lw=1.4)
                bx.plot(x + off, e["mean"], mk, color=col, ms=4, mec="white", mew=0.6, zorder=3)
        bx.set_xticks(xs)
        bx.set_xticklabels([str(n) for n in NS])
        bx.set_xlim(2.75, 5.25)
        allc = [b["contrasts"][f"{cn(k2, n)}:{key}"]["policy"]["auroc"]["ci95"] for k2 in ("awgn", "rayleigh") for n in NS
                for key in ("jscc_e2e-rep_orig_policy", "jscc_ctrl-rep_ctrl_policy", "sscc_lookup-rep_ctrl_policy")]
        bx.set_ylim(min(-0.02, min(c[0] for c in allc) - 0.02), max(c[1] for c in allc) + 0.02)
        bx.grid(color=GRID, lw=0.5)
        bx.set_xlabel("Complex channel uses per record, N")
        if j == 0:
            bx.set_ylabel("Δ macro-AUROC")
        else:
            bx.set_yticklabels([])
        letter(bx, "cd"[j], x=-0.03 if j else -0.12)
    handles = [Line2D([], [], color=col, marker=mk, ls=ls, lw=lw, ms=4, mec="white" if col != REF_L else INK2, label=lab)
               for _, lab, col, mk, ls, lw in SERIES5]
    handles.append(Line2D([], [], color=REF, marker="o", ls="", ms=4, mfc="white", mec=REF, label="Open: frozen transfer (not retrained)"))
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.08, 0.95), columnspacing=1.2, handlelength=2.4)
    dh = [Line2D([], [], color=c, marker=m, lw=1.4, ms=4, mec="white", label=l) for c, m, l in
          ((PURPLE, "^", "Joint JSCC − repetition baseline"), (PURPLE_L, "s", "JSCC − repetition, same lead"),
           (AMBER, "D", "SSCC − repetition, same lead"))]
    fig.legend(handles=dh, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.53, -0.01))
    fig.text(0.01, 0.99, "Different channel-use budgets, K = 1, unit average symbol power (energy N per record), "
             + ("PTB-XL test fold" if R2["split"] == "test" else "validation fold"), ha="left", va="top", fontsize=7.5, fontweight="bold")
    save(fig, "figS10_budget")


RICIAN = [("rician3dB_perfect", "Rician 3 dB, perfect CSI"), ("rician3dB_ls2", "Rician 3 dB, 2-pilot LS CSI (+2 uses)"),
          ("rician10dB_perfect", "Rician 10 dB, perfect CSI"), ("rician10dB_ls2", "Rician 10 dB, 2-pilot LS CSI (+2 uses)")]


def figS7_rician():
    r = load_r2("rician_report")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.6), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.83, bottom=0.09, hspace=0.28, wspace=0.08)
    for i, (cond, title) in enumerate(RICIAN):
        ax = axes[i // 2, i % 2]
        for key, lab, col, mk, ls, lw in SERIES5:
            ax.plot(SNRS, [r["absolute"][cond][key]["per_snr_auroc"][str(s)] for s in SNRS], color=col, marker=mk, ls=ls, lw=lw,
                    ms=3.6, mec=INK2 if col == REF_L else "white", mew=0.5)
        _snr_axis(ax, xlabel=(i >= 2))
        ax.set_ylim(0.50, 0.95)
        if i % 2 == 0:
            ax.set_ylabel("Macro-AUROC")
        else:
            ax.set_yticklabels([])
        ax.set_title(title, loc="left", fontweight="bold", fontsize=7)
        letter(ax, "abcd"[i], x=-0.03 if i % 2 else -0.12)
    handles = [Line2D([], [], color=col, marker=mk, ls=ls, lw=lw, ms=4, mec=INK2 if col == REF_L else "white", label=lab)
               for _, lab, col, mk, ls, lw in SERIES5]
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.07, 0.995), columnspacing=1.2, handlelength=2.4)
    save(fig, "figS7_rician")


def figS8_timevar():
    t = load_r2("timevar_report")
    TV = [("rician3dB_ls2_rho1.0", "ρ = 1 (block)", "-"), ("rician3dB_ls2_rho0.95", "ρ = 0.95", "--"), ("rician3dB_ls2_rho0.0", "ρ = 0", ":")]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.55})
    fig.subplots_adjust(left=0.08, right=0.975, top=0.80, bottom=0.18)
    ax = axes[0]
    for key, col, mk in (("jscc_e2e", PURPLE, "^"), ("rep_ctrl_policy", REF, "v")):
        for cond, lab, ls in TV:
            ax.plot(SNRS, [t["absolute"][cond][key]["per_snr_auroc"][str(s)] for s in SNRS], color=col, marker=mk, ls=ls, lw=1.2, ms=3.3,
                    mec="white", mew=0.5)
    _snr_axis(ax)
    ax.set_ylim(0.50, 0.95)
    ax.set_ylabel("Macro-AUROC")
    ax.set_title("Rician 3 dB, 2-pilot LS CSI", loc="left", fontweight="bold", fontsize=7)
    letter(ax, "a", x=-0.12)
    bx = axes[1]
    rows = [("jscc_e2e:rician3dB_ls2_rho0.95-rho1.0", "Joint JSCC: ρ 0.95 − 1", PURPLE),
            ("jscc_e2e:rician3dB_ls2_rho0.0-rho1.0", "Joint JSCC: ρ 0 − 1", PURPLE),
            ("rep_ctrl_policy:rician3dB_ls2_rho0.95-rho1.0", "Repetition: ρ 0.95 − 1", REF),
            ("rep_ctrl_policy:rician3dB_ls2_rho0.0-rho1.0", "Repetition: ρ 0 − 1", REF)]
    for i, (k, lab, col) in enumerate(rows):
        e = t["contrasts"][k]["policy"]["auroc"]
        bx.plot(e["ci95"], [i, i], color=col, lw=1.5)
        bx.plot(e["mean"], i, "o", color=col, ms=4.5, mec="white", mew=0.6)
        bx.text(e["ci95"][1] + 0.0015, i, fmt(e["mean"]), ha="left", va="center", fontsize=6.2)
    lo = min(t["contrasts"][k]["policy"]["auroc"]["ci95"][0] for k, *_ in rows)
    hi = max(t["contrasts"][k]["policy"]["auroc"]["ci95"][1] for k, *_ in rows)
    bx.set_xlim(min(lo, 0) - 0.004, max(hi, 0) + 0.012)
    set_xticks(bx, 0.01)
    bx.axvline(0, color=INK, lw=0.7)
    bx.set_yticks(range(len(rows)))
    bx.set_yticklabels([lab for _, lab, _ in rows], fontsize=6.4)
    bx.set_ylim(len(rows) - 0.5, -0.5)
    bx.grid(axis="x", color=GRID, lw=0.5)
    bx.set_xlabel("Δ macro-AUROC, mean over 7 SNRs")
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    letter(bx, "b", x=-0.62)
    handles = [Line2D([], [], color=PURPLE, marker="^", lw=1.2, mec="white", label="Joint JSCC, 12 leads"),
               Line2D([], [], color=REF, marker="v", lw=1.2, mec="white", label="Repetition, same lead (policy)")] + \
              [Line2D([], [], color=INK2, ls=ls, lw=1.2, label=lab) for _, lab, ls in TV]
    fig.legend(handles=handles, loc="upper left", ncol=5, frameon=False, bbox_to_anchor=(0.06, 1.0), columnspacing=1.0, handlelength=2.2)
    save(fig, "figS8_timevar")


def figS9_quality():
    q = load_r2("quality")
    G = [("annotated", "With a quality annotation", AMBER, "s"), ("not_annotated", "Without a quality annotation", PURPLE, "o")]
    rows = ["macro"] + CLASSES
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), gridspec_kw={"wspace": 0.10})
    fig.subplots_adjust(left=0.10, right=0.985, top=0.80, bottom=0.17)
    for k, (ax, met) in enumerate(zip(axes, ("auroc", "auprc"))):
        ax.axvline(0, color=INK, lw=0.7)
        ax.grid(axis="x", color=GRID, lw=0.5)
        for gi, (g, lab, col, mk) in enumerate(G):
            for i, r in enumerate(rows):
                e = q["groups"][g]["paired"]["orig_A-orig_C"][f"macro_{met}" if r == "macro" else f"{met}_{r}"]
                y = i + (-0.17 if gi == 0 else 0.17)
                ax.plot(e["ci95"], [y, y], color=col, lw=1.4)
                ax.plot(e["mean"], y, mk, color=col, ms=4, mec="white", mew=0.6, zorder=3)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(["Macro"] + CLASSES if k == 0 else [])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ttl = "ΔAUROC" if met == "auroc" else "ΔAUPRC"
        ax.set_title(ttl, loc="left", fontweight="bold")
        ax.set_xlabel(f"{ttl}, selector − matched fixed V5")
        letter(ax, "ab"[k], x=-0.03 if k else -0.12)
    n = {g: (q["groups"][g]["records"], q["groups"][g]["patients"]) for g, *_ in G}
    handles = [Line2D([], [], color=col, marker=mk, lw=1.4, mec="white", ms=4, label=f"{lab} ({n[g][0]:,} records, {n[g][1]:,} patients)")
               for g, lab, col, mk in G]
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.09, 1.0))
    save(fig, "figS9_quality")


def tables_r2():
    # ---- Table S11: Rician
    r = load_r2("rician_report")
    meth = [("rep_orig_policy", "Repetition baseline"), ("rep_ctrl_policy", "Repetition, same lead"), ("sscc_lookup", "SSCC"),
            ("jscc_ctrl", "Learned JSCC, 1 lead"), ("jscc_e2e", "Joint learned JSCC, 12 leads")]
    conds = [c for c, _ in RICIAN]
    body = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule", " & \\multicolumn{2}{c}{$K_R=3$\\,dB} & \\multicolumn{2}{c}{$K_R=10$\\,dB} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Method & Perfect CSI & LS CSI & Perfect CSI & LS CSI \\\\", "\\midrule",
            "\\multicolumn{5}{@{}l}{Macro-AUROC / macro-AUPRC, mean over the seven SNRs} \\\\"]
    for k, lab in meth:
        body.append(f"\\quad {lab} & " + " & ".join(f"{r['absolute'][c][k]['policy_auroc']:.3f} / {r['absolute'][c][k]['policy_auprc']:.3f}" for c in conds) + " \\\\")
    body += ["\\midrule", "\\multicolumn{5}{@{}l}{Paired $\\Delta$AUROC [95\\% CI]; $\\Delta$AUPRC below} \\\\"]
    for key, lab in (("jscc_e2e-rep_orig_policy", "Joint JSCC $-$ repetition baseline"), ("jscc_ctrl-rep_orig_policy", "JSCC, 1 lead $-$ repetition baseline"),
                     ("jscc_ctrl-rep_ctrl_policy", "JSCC, 1 lead $-$ repetition, same lead"), ("sscc_lookup-rep_ctrl_policy", "SSCC $-$ repetition, same lead"),
                     ("rep_ctrl_policy-rep_ctrl_bestfixed", "Adaptive $r$ $-$ best fixed $r$ (same lead)")):
        d_ = 4 if key.startswith("rep_") else 3
        body.append(f"\\quad {lab} & " + " & ".join(f"{tex(r['contrasts'][c + ':' + key]['policy']['auroc']['mean'], d_)} {{\\scriptsize {tci(r['contrasts'][c + ':' + key]['policy']['auroc']['ci95'], d_)}}}" for c in conds) + " \\\\")
        body.append("\\quad & " + " & ".join(f"{{\\scriptsize {tex(r['contrasts'][c + ':' + key]['policy']['auprc']['mean'], d_)} {tci(r['contrasts'][c + ':' + key]['policy']['auprc']['ci95'], d_)}}}" for c in conds) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tab("tabS_rician", "\n".join(body) + "\n")
    # ---- Table S12: time variation
    t = load_r2("timevar_report")
    TV = [("rician3dB_ls2_rho1.0", "$\\rho=1$ (block)"), ("rician3dB_ls2_rho0.95", "$\\rho=0.95$"), ("rician3dB_ls2_rho0.0", "$\\rho=0$")]
    body = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule", " & " + " & ".join(l for _, l in TV) + " \\\\", "\\midrule",
            "\\multicolumn{4}{@{}l}{Macro-AUROC / macro-AUPRC, mean over the seven SNRs} \\\\"]
    for k, lab in (("rep_ctrl_policy", "Repetition, same lead (policy)"), ("rep_ctrl_r1", "\\quad fixed $r=1$"), ("rep_ctrl_r2", "\\quad fixed $r=2$"),
                   ("rep_ctrl_r4", "\\quad fixed $r=4$"), ("jscc_e2e", "Joint learned JSCC, 12 leads")):
        body.append(f"{lab} & " + " & ".join(f"{t['absolute'][c][k]['policy_auroc']:.3f} / {t['absolute'][c][k]['policy_auprc']:.3f}" for c, _ in TV) + " \\\\")
    body += ["\\midrule", "Joint JSCC $-$ repetition, $\\Delta$AUROC & " + " & ".join(
        f"{tex(t['contrasts'][c + ':jscc_e2e-rep_ctrl_policy']['policy']['auroc']['mean'])} {tci(t['contrasts'][c + ':jscc_e2e-rep_ctrl_policy']['policy']['auroc']['ci95'])}" for c, _ in TV) + " \\\\"]
    for k, lab in (("jscc_e2e", "Joint JSCC"), ("rep_ctrl_policy", "Repetition")):
        body.append(f"{lab}: change from $\\rho=1$, $\\Delta$AUROC & -- & " + " & ".join(
            f"{tex(t['contrasts'][f'{k}:{c}-rho1.0']['policy']['auroc']['mean'])} {tci(t['contrasts'][f'{k}:{c}-rho1.0']['policy']['auroc']['ci95'])}" for c, _ in TV[1:]) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tab("tabS_timevar", "\n".join(body) + "\n")
    # ---- Table S13: budget accounting and results
    b = load_r2("budget_report")
    acc = b["accounting"]
    cn = lambda kind, n: f"{kind}_perfect" if n == 16 else f"{kind}_perfect_n{n}"
    body = ["\\begin{tabular}{@{}rlcccll@{}}", "\\toprule",
            "$N$ & Method & Energy (mean) & Info.\\ bits & AWGN & Rayleigh & Model \\\\", "\\midrule"]
    for n in (8, 16, 32):
        for key, lab, opt in (("rep_orig_policy", "Repetition baseline", "rep_orig_r2"), ("rep_ctrl_policy", "Repetition, same lead", "rep_ctrl_r2"),
                              ("sscc_lookup", "SSCC (lookup)", None), ("jscc_ctrl", "Learned JSCC, 1 lead", "jscc_ctrl"),
                              ("jscc_e2e", "Joint learned JSCC", "jscc_e2e")):
            if key == "sscc_lookup":
                cfgs = sorted([k for k in acc if k.startswith(f"n{n}:sscc_")], key=lambda k: acc[k]["info_bits"])
                en = f"{min(acc[k]['energy_mean'] for k in cfgs):.2f}--{max(acc[k]['energy_mean'] for k in cfgs):.2f}"
                bits = ", ".join(str(acc[k]["info_bits"]) for k in cfgs)
                model = "refitted" if n != 16 else "existing"
            else:
                a = acc[f"n{n}:{opt}"]
                en, bits = f"{a['energy_mean']:.1f}", "--"
                model = {"frozen transfer of the submitted checkpoint": "frozen transfer", "existing model": "existing",
                         "retrained at this budget": "retrained"}[a["training"]]
            body.append(f"{n if key == 'rep_orig_policy' else ''} & {lab} & {en} & {bits} & "
                        f"{b['absolute'][cn('awgn', n)][key]['policy_auroc']:.3f} & {b['absolute'][cn('rayleigh', n)][key]['policy_auroc']:.3f} & {model} \\\\")
        if n != 32:
            body.append("\\midrule")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tab("tabS_budget", "\n".join(body) + "\n")
    # ---- Table S14: joint JSCC without noise (folds 9 and 10)
    body = ["\\begin{tabular}{@{}llcccc@{}}", "\\toprule", "Fold & Condition & Macro-AUROC & per seed & Macro-AUPRC & per seed \\\\", "\\midrule"]
    for sp, lab in (("val", "9"), ("test", "10")):
        try:
            nz = load(f"noiseless_check_{sp}.json")
        except FileNotFoundError:
            continue
        for k, l in (("A_bypass", "Joint JSCC, channel bypassed"), ("B_awgn_noise_off", "Joint JSCC, AWGN path, noise off"),
                     ("C_awgn_20dB", "Joint JSCC, AWGN 20\\,dB (30 draws)"), ("ref_clean_12lead", "12-lead mean pool + linear, clean")):
            ps = [nz["seeds"][str(s)]["metrics"][k] for s in (0, 1, 2)]
            pa = ", ".join("%.3f" % p["macro_auroc"] for p in ps)
            pp = ", ".join("%.3f" % p["macro_auprc"] for p in ps)
            body.append(f"{lab} & {l} & {nz['seed_mean'][k]['macro_auroc']:.4f} & {pa} & {nz['seed_mean'][k]['macro_auprc']:.4f} & {pp} \\\\")
        for k, l in (("C_minus_A", "20\\,dB $-$ noise-free, same checkpoint"), ("A_minus_R", "Joint JSCC (noise-free) $-$ 12-lead model$^\\ast$")):
            c = nz["contrasts"][k]
            body.append(f" & $\\Delta$ {l} & {tex(c['auroc']['mean'], 4)} & {{\\scriptsize {tci(c['auroc']['ci95'], 4)}}} & "
                        f"{tex(c['auprc']['mean'], 4)} & {{\\scriptsize {tci(c['auprc']['ci95'], 4)}}} \\\\")
        mx = max(max(v for kk, v in nz["seeds"][str(s)]["max_abs_dlogit"].items() if kk.startswith(("A_vs", "B_real"))) for s in (0, 1, 2))
        body.append(f" & \\multicolumn{{5}}{{l}}{{Max.\\ $|\\Delta\\text{{logit}}|$, bypassed vs noise off (all seeds): {mx:.1g}}} \\\\")
        if sp == "val":
            body.append("\\midrule")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tab("tabS_noiseless", "\n".join(body) + "\n")
    # ---- Table S15: signal-quality annotations
    q = load_r2("quality")
    c = q["counts"]
    body = ["\\begin{tabular}{@{}lcc@{}}", "\\toprule", " & With a quality annotation & Without \\\\", "\\midrule",
            f"Records (patients) & {q['groups']['annotated']['records']:,} ({q['groups']['annotated']['patients']:,}) & "
            f"{q['groups']['not_annotated']['records']:,} ({q['groups']['not_annotated']['patients']:,}) \\\\",
            "Positives NORM / MI / STTC / CD / HYP & " + " & ".join(" / ".join(str(q["groups"][g]["positives"][k]) for k in CLASSES)
                                                                    for g in ("annotated", "not_annotated")) + " \\\\"]
    for m, lab in (("orig_A", "Selector"), ("orig_C", "Matched fixed V5"), ("orig_II", "Fixed Lead II")):
        body.append(f"{lab}, macro-AUROC / AUPRC & " + " & ".join(
            f"{q['groups'][g]['absolute'][m]['mean']['macro_auroc']:.3f} / {q['groups'][g]['absolute'][m]['mean']['macro_auprc']:.3f}" for g in ("annotated", "not_annotated")) + " \\\\")
    for key, lab in (("orig_A-orig_C", "Selector $-$ V5"), ("orig_A-orig_II", "Selector $-$ Lead II")):
        for met in ("auroc", "auprc"):
            body.append(f"{lab}, $\\Delta$macro-{met.upper()} & " + " & ".join(
                f"{tex(q['groups'][g]['paired'][key][f'macro_{met}']['mean'], 4)} {tci(q['groups'][g]['paired'][key][f'macro_{met}']['ci95'], 4)}" for g in ("annotated", "not_annotated")) + " \\\\")
    it = q["interaction_annotated_minus_not_of_selector_minus_V5"]
    body += ["\\midrule", f"\\multicolumn{{3}}{{@{{}}l}}{{Difference between groups (selector $-$ V5): AUROC {tex(it['macro_auroc']['mean'], 4)} "
             f"{tci(it['macro_auroc']['ci95'], 4)}; AUPRC {tex(it['macro_auprc']['mean'], 4)} {tci(it['macro_auprc']['ci95'], 4)}}} \\\\"]
    ll = q["lead_level"]
    body += [f"\\multicolumn{{3}}{{@{{}}l}}{{Records with 1--11 flagged leads: {ll['records']:,} ({ll['patients']:,} patients). Selector sent a flagged lead in "
             f"{100 * np.mean([v['observed_flagged_pick_rate'] for v in ll['per_seed'].values()]):.1f}\\%,}} \\\\",
             f"\\multicolumn{{3}}{{@{{}}l}}{{\\quad expected from its lead frequencies {100 * np.mean([v['expected_from_unannotated_freq'] for v in ll['per_seed'].values()]):.1f}\\% "
             f"(difference {tex(100 * ll['observed_minus_expected_mean'], 1)} points {tci([100 * x for x in ll['observed_minus_expected_ci95']], 1)}); "
             f"V5 flagged {100 * ll['v5_flag_rate']:.1f}\\%; random lead {100 * ll['random_choice_rate']:.1f}\\%}} \\\\",
             "\\bottomrule", "\\end{tabular}"]
    write_tab("tabS_quality", "\n".join(body) + "\n")
    # counts for the caption/text
    print("quality counts", c["any_field"], c["not_annotated"])


# ====================================================================== E4: extended encoder training (post hoc)
E4_SEED_COLORS = ("#5B4A8B", "#C2851C", "#2B7A78")


def figS12_e4_training():
    tr = load("e4_training.json")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.25))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.8, wspace=0.38)
    panels = (("train_loss", "Training loss"), ("composite", "Validation composite AUROC"),
              ("val_auroc_K12", "Validation 12-lead macro-AUROC"))
    for ax, (key, title), lt in zip(axes, panels, "abc"):
        for s in (0, 1, 2):
            r = tr["e4"][str(s)]
            ep = [h["epoch"] + 1 for h in r["history"]]
            v = [h[key] for h in r["history"]]
            ax.plot(ep, v, color=E4_SEED_COLORS[s], lw=0.9, label=f"seed {s}")
            b = r["best_epoch"]
            ax.plot(b + 1, r["history"][b][key], "o", color=E4_SEED_COLORS[s], ms=4, mec="white", mew=0.5, zorder=4)
            ax.plot(ep[-1], v[-1], "x" if r["stopped_by_patience"] else ">", color=E4_SEED_COLORS[s], ms=4, zorder=4)
        ax.axvline(40, color=INK2, lw=0.7, ls="--")
        ax.text(38.5, 0.97, "first-run\n40-epoch limit", transform=ax.get_xaxis_transform(), fontsize=5.6, color=INK2, va="top", ha="right")
        ax.set_xlabel("Epoch")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=7)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_xlim(0, 121)
        letter(ax, lt, x=-0.12)
        if key != "train_loss":
            lo = min(h[key] for s in (0, 1, 2) for h in tr["e4"][str(s)]["history"][10:])
            hi = max(h[key] for s in (0, 1, 2) for h in tr["e4"][str(s)]["history"])
            ax.set_ylim(lo - 0.002, hi + 0.008)
    handles = [Line2D([], [], color=E4_SEED_COLORS[s], lw=1.2, label=f"seed {s}") for s in (0, 1, 2)] + [
        Line2D([], [], marker="o", color=INK2, ls="", ms=4, label="best epoch (checkpoint used)"),
        Line2D([], [], marker="x", color=INK2, ls="", ms=4, label="stopped by patience (8)")]
    if not all(tr["e4"][str(s)]["stopped_by_patience"] for s in (0, 1, 2)):
        handles.append(Line2D([], [], marker=">", color=INK2, ls="", ms=4, label="reached 120 epochs"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.53, 1.0), fontsize=6)
    save(fig, "figS12_e4_training")


def tables_e4():
    tr = load("e4_training.json")
    e4 = load("e4_eval_test.json")
    A, P = e4["absolute"], e4["paired"]
    body = ["\\begin{tabular}{@{}cccccccc@{}}", "\\toprule",
            "& \\multicolumn{3}{c}{Training (folds 1--8; early stopping on fold 9)} & \\multicolumn{2}{c}{Fold-9 composite} & \\multicolumn{2}{c}{Test 12-lead macro-AUROC} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
            "Seed & Epochs run & Best epoch & Stopped by patience & 40 epochs & $\\leq$120 epochs & 40 epochs & $\\leq$120 epochs \\\\", "\\midrule"]
    for s in (0, 1, 2):
        r, o = tr["e4"][str(s)], tr["o40"][str(s)]
        body.append(f"{s} & {r['epochs_run']} & {r['best_epoch'] + 1} & {'yes' if r['stopped_by_patience'] else 'no'} & "
                    f"{o['best_composite']:.4f} & {r['best_composite']:.4f} & "
                    f"{A['o40_12']['macro_auroc']['per_seed'][s]:.4f} & {A['e4_12']['macro_auroc']['per_seed'][s]:.4f} \\\\")
    body.append(f"Mean (SD) & & & & & & {A['o40_12']['macro_auroc']['mean']:.4f} ({A['o40_12']['macro_auroc']['sd_seeds']:.4f}) & "
                f"{A['e4_12']['macro_auroc']['mean']:.4f} ({A['e4_12']['macro_auroc']['sd_seeds']:.4f}) \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tab("tabS_e4_training", "\n".join(body) + "\n")
    rows = [("12-lead, $\\leq$120 $-$ 40 epochs", "e4_12-o40_12"),
            ("$K=1$, selector $-$ fixed V5 ($\\leq$120 epochs)", "e4_A-e4_C"),
            (f"$K=1$, selector $-$ fixed {e4['e4_best_fixed_lead']} ($\\leq$120 epochs)", "e4_A-e4_BEST"),
            ("$K=1$, selector $-$ fixed V5 (40 epochs)", "o40_A-o40_C")]
    mets = [("macro_auroc", "Macro-AUROC"), ("macro_auprc", "Macro-AUPRC"), ("auroc_CD", "CD AUROC"), ("auprc_CD", "CD AUPRC"),
            ("auroc_STTC", "STTC AUROC"), ("auprc_STTC", "STTC AUPRC")]
    body = ["\\begin{tabular}{@{}llccc@{}}", "\\toprule",
            "Comparison & Metric & Mean $\\Delta$ [95\\% CI] & Per seed (0, 1, 2) & Between-seed SD \\\\", "\\midrule"]
    for i, (lab, k) in enumerate(rows):
        for j, (m, mlab) in enumerate(mets):
            e = P[k][m]
            ps = ", ".join(tex(v, 4) for v in e["per_seed"])
            body.append(f"{lab if j == 0 else ''} & {mlab} & {tex(e['mean'], 4)} {tci(e['ci95'], 4)} & {ps} & {e['sd_seeds']:.4f} \\\\")
        if i < len(rows) - 1:
            body.append("\\midrule")
    body += ["\\bottomrule", "\\end{tabular}"]
    write_tab("tabS_e4_comparisons", "\n".join(body) + "\n")
    lab = e4["labelled_only_12lead"]
    print("E4 labelled-only", lab["n_records"], {m: round(lab[m]["macro_auroc"]["mean"], 4) for m in ("e4_12", "o40_12")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "outputs", "paper"))
    ap.add_argument("--only-e4", action="store_true",
                    help="write only the outputs changed by the E4 extended-training analysis (Fig. S5, Fig. S12, Table S2, Tables S17-S18)")
    ap.add_argument("--r2split", default="test", choices=("val", "test"), help="fold of the revision-2 summaries (val only for development)")
    a = ap.parse_args()
    R2["split"] = a.r2split
    OUT["fig"] = os.path.join(a.out, "figures")
    OUT["tab"] = os.path.join(a.out, "tables")
    os.makedirs(OUT["fig"], exist_ok=True)
    os.makedirs(OUT["tab"], exist_ok=True)
    if a.only_e4:
        figS5_backbone()
        figS12_e4_training()
        table_backbone()
        tables_e4()
        return
    fig1_framework()
    st = fig2_budget_and_classes()
    fig3_transmission()
    fig4_external()
    figS2_heatmap()
    figS3_comm_all()
    figS4_controlled()
    figS5_backbone()
    figS12_e4_training()
    figS6_gate()
    tables(st)
    table_stability()
    figS10_budget()
    figS7_rician()
    figS8_timevar()
    figS9_quality()
    tables_r2()
    tables_e4()
    for (m, k, met), (mu, sd) in sorted(st.items()):
        print(f"budget {m:18s} K={k:<2d} {met} {mu:.4f} +- {sd:.4f}")


if __name__ == "__main__":
    main()
