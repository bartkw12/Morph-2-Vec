"""
Past-tense step 4: slide figures for the midpoint meeting (16:9, large type). Reads only the small tables in
outputs/past_tense/tables/ and outputs/feasibility/tables/table9_regular_vs_irregular.csv; writes PNG + PDF to
outputs/past_tense/figures/:

    fig_A_reg_vs_irreg_top1        simplified feasibility Fig. 7: -s, regular past, irregular -> irregular, random pair
    fig_B_reg_to_irregular_top1    main result: regular -> each target class, top-1, with random-pair floors
    fig_C_reg_to_irregular_rank    the same, mean rank (log)
    fig_D_two_bin                  regular -> "adds a t/d" vs "no added stop" (verb-weighted)
    fig_E_allomorph_source         [t]- vs [d]-source regulars -> semi-regular / devoicing
    fig_F_similarity_scatter       per target verb: top-1 vs. distance from the regularized form

A model whose run is missing is left out of every figure.

    python scripts/past_tense/04_figures.py
"""

from datetime import datetime
from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt_common as P  # noqa: E402

C = P.C
SIZE = (12, 6.2)
INK, MUTED = "#222222", "#666666"
TICK_PT = 15.5  # x tick label size; offsets below the axis are computed from it
# Okabe-Ito, after the two model colours (common.MODEL_COLORS); one per irregular class in fig F
CLASS_COLORS = {"semi_regular": "#009E73", "devoicing": "#CC79A7", "vowel_change": "#56B4E9", "suppletive": "#D55E00"}
CLASS_MARKERS = {"semi_regular": "o", "devoicing": "s", "vowel_change": "^", "suppletive": "D"}


def slide_style():
    C.set_style()
    sns.set_theme(context="talk", style="whitegrid", font_scale=0.95)
    plt.rcParams.update({"figure.dpi": 100, "savefig.bbox": "tight", "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                         "grid.color": "#e3e3e3", "grid.linewidth": 0.8, "axes.edgecolor": "#bbbbbb",
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "xtick.labelsize": TICK_PT, "legend.frameon": False})


def title_and_legend(ax, title, handles, names):
    """Title top-left, legend in one row between the title and the plot (never over the data)."""
    ax.set_title(title, loc="left", pad=40)
    ax.legend(handles, names, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(names), fontsize=12,
              handlelength=1.5, columnspacing=1.4, borderaxespad=0.2)


def caption(ax, text, offset, width=150):
    """Grey caption under the axis, `offset` points below its bottom edge."""
    ax.annotate(textwrap.fill(text, width), (0, 0), xycoords="axes fraction", xytext=(0, offset),
                textcoords="offset points", ha="left", va="top", fontsize=10.5, color=MUTED)


def model_bars(ax, s: pd.DataFrame, cats: list, labels: list, est="top1", lo="top1_lo", hi="top1_hi",
               floor: pd.DataFrame = None, fmt="{:.2f}", log=False, notes: dict = None,
               zero: pd.Series = None) -> tuple[list, list, float]:
    """
    Grouped bars, one per model (common.MODEL_COLORS), CI whiskers, the value above each bar. `s` is indexed
    by (model, cat) with n_trials / n_target_verbs; `floor` (same index, column `est`) draws a grey tick
    across each bar, and `zero` (same index) a diamond for the zero-offset control. Under each group:
    "trials / verbs" (once if both models agree) and `notes[cat]`.
    Returns legend handles, names, and the lowest point (offset below the axis) used by the labels.
    """
    models = [m for m in C.MODELS if m in s.index.get_level_values(0)]
    width = 0.36
    x0 = np.arange(len(cats))
    for k, model in enumerate(models):
        x = x0 + (k - (len(models) - 1) / 2) * width
        d = s.loc[model].reindex(cats)
        err = np.vstack([d[est] - d[lo], d[hi] - d[est]]).clip(min=0)
        ax.bar(x, d[est], width * 0.92, color=C.MODEL_COLORS[model], label=model, zorder=2)
        ax.errorbar(x, d[est], yerr=err, fmt="none", ecolor=INK, elinewidth=1.4, capsize=4, zorder=3)
        for xi, v, h in zip(x, d[est], d[hi]):
            if np.isfinite(v):
                ax.text(xi, h * 1.15 if log else h + 0.015, fmt(v) if callable(fmt) else fmt.format(v),
                        ha="center", va="bottom", fontsize=12, color=INK, zorder=6,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))
        if floor is not None:
            f = floor.loc[model].reindex(cats)[est]
            ax.hlines(f, x - width * 0.46, x + width * 0.46, colors=C.CONTROL_COLOR, linewidth=4, zorder=4)
        if zero is not None and model in zero.index.get_level_values(0):
            ax.scatter(x, zero.loc[model].reindex(cats), marker="D", s=70, facecolor="white", edgecolor=INK,
                       linewidth=1.8, zorder=5)
    ax.set_xticks(x0, labels)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.grid(axis="x", visible=False)
    if log:
        ax.set_yscale("log")

    tick_lines = max(lab.count("\n") + 1 for lab in labels)
    n_top = -(8 + 1.25 * TICK_PT * tick_lines + 6)
    lowest = n_top
    for xi, cat in zip(x0, cats):
        ns = sorted({(int(s.loc[(m, cat), "n_trials"]), int(s.loc[(m, cat), "n_target_verbs"]))
                     for m in models if (m, cat) in s.index})
        lines = [f"{a:,} trials / {b} verbs" for a, b in ns] + ([notes[cat]] if notes and cat in notes else [])
        ax.annotate("\n".join(lines), (xi, 0), xycoords=("data", "axes fraction"), xytext=(0, n_top),
                    textcoords="offset points", ha="center", va="top", fontsize=10.5, color=MUTED)
        lowest = min(lowest, n_top - 13.5 * len(lines))

    handles, names = ax.get_legend_handles_labels()
    if floor is not None:
        handles.append(Line2D([], [], color=C.CONTROL_COLOR, linewidth=4))
        names.append("random-pair floor")
    if zero is not None:
        handles.append(Line2D([], [], marker="D", linestyle="none", markersize=8, markerfacecolor="white",
                              markeredgecolor=INK, markeredgewidth=1.8))
        names.append("base alone (no step)")
    return handles, names, lowest


def class_tick(c):
    return f"{P.CLASS_LABELS[c]}\n({', '.join(P.CLASS_EXAMPLES[c].split(', ')[:2])})"


SMALL_NOTE = {"suppletive": "descriptive only"}


# --------------------------------------------------------------------------- figures

def fig_a():
    t9 = pd.read_csv(C.TABLES / "table9_regular_vs_irregular.csv", encoding="utf-8")
    conds = {"plural NNS regular": "plural -s\n(cats)", "3sg VBZ regular": "3sg -s\n(runs)",
             "past regular (auto-class regular only)": "regular past\n(walked)",
             "past irregular (no modals, be, homographs)": "irregular →\nirregular",
             next(c for c in t9.condition if c.startswith("random pair")): "random pair\n(control)"}
    s = t9[t9.condition.isin(conds)].rename(columns={"n_items": "n_target_verbs"}).set_index(["model", "condition"])
    fig, ax = plt.subplots(figsize=SIZE)
    h, n, low = model_bars(ax, s, list(conds), list(conds.values()))
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("top-1 accuracy")
    title_and_legend(ax, "Analogies within one class: top-1 accuracy (layer 8)", h, n)
    caption(ax, "Source and target pairs from the same class (feasibility demo, table 9). Regular past: verbs the "
                "automatic check calls regular [t]/[d]/[ɪd]. Irregular → irregular excludes modals, be and homographs. "
                "Random pair: random word pairs applied to -s targets. 95% CIs resample target verbs.", low - 8)
    P.save_fig(fig, "fig_A_reg_vs_irreg_top1")
    plt.close(fig)


def by_target(t1: pd.DataFrame, prefix: str, classes: list) -> pd.DataFrame:
    s = t1[t1.experiment.isin([f"{prefix}{'reg' if c == 'regular' else c}" for c in classes])]
    return s.assign(cat=s.target).set_index(["model", "cat"])


def fig_b_c(t1: pd.DataFrame, t7: pd.DataFrame = None):
    cats = P.PAST_CLASSES
    s, floor = by_target(t1, "reg_to_", cats), by_target(t1, "random_to_", cats)
    zero = None if t7 is None else t7.set_index(["model", "target_class"]).zero_offset_top1
    labels = [class_tick(c) for c in cats]
    for name, est, lo, hi, log, ylabel, fmt in [
            ("fig_B_reg_to_irregular_top1", "top1", "top1_lo", "top1_hi", False, "top-1 accuracy", "{:.2f}"),
            ("fig_C_reg_to_irregular_rank", "mean_rank", "mean_lo", "mean_hi", True,
             "mean rank (log; lower is better)", C.fmt_rank)]:
        fig, ax = plt.subplots(figsize=SIZE)
        h, n, low = model_bars(ax, s, cats, labels, est, lo, hi, floor=floor, fmt=fmt, log=log, notes=SMALL_NOTE,
                               zero=None if log else zero)
        if log:
            ax.set_ylim(0.3, max(floor[est].max(), s[hi].max()) * 5)
        else:
            ax.set_ylim(0, 1.08)
        ax.set_ylabel(ylabel)
        title_and_legend(ax, "The regular step (walk → walked) applied to each kind of past tense", h, n)
        caption(ax, "Each trial: a regular source pair a : b and a target base c of the class shown; the guess b − a + c "
                    "is ranked against all 334,008 word tokens. Grey tick: the same target verbs with a random word pair "
                    "as the source." + ("" if log or zero is None else " Diamond: the guess is the base itself (no step), "
                    "base excluded; mean over target verbs.") + " Classes are automatic, not hand-checked. "
                    "95% CIs resample target verbs.", low - 8)
        P.save_fig(fig, name)
        plt.close(fig)


def fig_d(t2: pd.DataFrame):
    bins = list(dict.fromkeys(t2.bin))
    labels = ["regular →\nregular", "regular →\nadds a t/d\n(kept, sent)", "regular →\nno added stop\n(sang, went)"]
    tr = t2[t2.kind == "transfer"].assign(cat=lambda d: d.bin).set_index(["model", "cat"])
    fl = t2[t2.kind == "random floor"].assign(cat=lambda d: d.bin).set_index(["model", "cat"])
    fig, ax = plt.subplots(figsize=(10, 6.2))
    h, n, low = model_bars(ax, tr, bins, labels, "top1_verb_weighted", "lo", "hi", floor=fl)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("top-1 accuracy (mean over verbs)")
    title_and_legend(ax, "Two bins instead of four classes", h, n)
    caption(ax, "Irregular targets pooled by whether the past ends in an added t/d (semi-regular + devoicing) or not "
                "(vowel change + suppletive). Each target verb counts the same, so the 2 suppletive and 5 devoicing "
                "verbs do not dominate their bins. Grey tick: random-pair floor. 95% CIs resample verbs.", low - 8, 125)
    P.save_fig(fig, "fig_D_two_bin")
    plt.close(fig)


def fig_e(t1: pd.DataFrame):
    targets = ["semi_regular", "devoicing"]
    sources = [("reg", "all", 0.4, None), ("reg_T", "[t]", 1.0, None), ("reg_D", "[d]", 1.0, "//")]
    models = [m for m in C.MODELS if m in set(t1.model)]
    fig, axs = plt.subplots(1, len(models), figsize=SIZE, sharey=True, squeeze=False)
    for ax, model in zip(axs[0], models):
        d = t1[t1.model == model].set_index("experiment")
        xs, labels = [], []
        for g, tgt in enumerate(targets):
            for j, (src, lab, alpha, hatch) in enumerate(sources):
                r = d.loc[f"{src}_to_{tgt}"]
                x = g * (len(sources) + 1) + j
                ax.bar(x, r.top1, 0.82, color=C.MODEL_COLORS[model], alpha=alpha, hatch=hatch, edgecolor="white",
                       linewidth=0, zorder=2)
                ax.errorbar(x, r.top1, yerr=[[r.top1 - r.top1_lo], [r.top1_hi - r.top1]], fmt="none", ecolor=INK,
                            elinewidth=1.4, capsize=4, zorder=3)
                ax.text(x, r.top1_hi + 0.015, f"{r.top1:.2f}", ha="center", va="bottom", fontsize=12)
                xs.append(x)
                labels.append(lab)
            ax.annotate(f"→ {P.CLASS_LABELS[tgt]}\n{int(r.n_target_verbs)} verbs", (g * (len(sources) + 1) + 1, 0),
                        xycoords=("data", "axes fraction"), xytext=(0, -38), textcoords="offset points",
                        ha="center", va="top", fontsize=14, weight="bold")
        ax.set_xticks(xs, labels)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.grid(axis="x", visible=False)
        ax.set_title(model, loc="left", color=C.MODEL_COLORS[model], pad=12)
        ax.set_ylim(0, 1.05)
    axs[0, 0].set_ylabel("top-1 accuracy")
    handles = [Patch(color=C.CONTROL_COLOR, alpha=0.4), Patch(color=C.CONTROL_COLOR),
               Patch(facecolor=C.CONTROL_COLOR, hatch="//", edgecolor="white")]
    fig.legend(handles, ["source: all regular", "source ends in [t] (walked)", "source ends in [d] (played)"],
               loc="lower left", bbox_to_anchor=(0.06, 0.93), ncol=3, fontsize=12)
    fig.suptitle("Does a [t]-source step reach pasts ending in [t] better?", x=0.06, y=1.06, ha="left", weight="bold")
    caption(axs[0, 0], "Targets: semi-regular (kept, felt, told) and devoicing (sent, built) pasts. Up to 1,000 trials per bar; "
                       "95% CIs resample target verbs. Classes are automatic, not hand-checked.", -95)
    P.save_fig(fig, "fig_E_allomorph_source")
    plt.close(fig)


def fig_f(per: pd.DataFrame):
    d = per[per.target != "regular"]
    models = [m for m in C.MODELS if m in set(d.model)]
    fig, axs = plt.subplots(1, len(models), figsize=SIZE, sharey=True, sharex=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, model in zip(axs[0], models):
        m = d[d.model == model]
        for c in P.IRREGULAR_CLASSES:
            g = m[m.target == c].reset_index(drop=True)
            x = g.edit_dist + rng.uniform(-0.18, 0.18, len(g))
            ax.scatter(x, g.top1, s=75, color=CLASS_COLORS[c], marker=CLASS_MARKERS[c], edgecolor="white",
                       linewidth=1.2, alpha=0.9, label=P.CLASS_LABELS[c], zorder=3)
            # label sparingly: the suppletive verbs, and the best few of each class if they get any hits
            show = g.index[(c == "suppletive") | (g.top1.rank(ascending=False, method="first") <= 3) & (g.top1 >= 0.2)]
            for i in show:
                ax.annotate(g.gt_label[i], (x[i], g.top1[i]), xytext=(6, 4), textcoords="offset points", fontsize=10.5)
        means = m.groupby("edit_dist").top1.mean()
        ax.plot(means.index, means.to_numpy(), color=MUTED, linewidth=2, zorder=2, label="mean per distance")
        ax.set_title(model, loc="left", color=C.MODEL_COLORS[model], pad=12)
        ax.set_xticks(sorted(d.edit_dist.unique()))
        ax.grid(axis="x", visible=False)
    axs[0, 0].set_ylabel("top-1 accuracy per target verb")
    fig.subplots_adjust(top=0.80)
    h, n = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, n, loc="lower left", bbox_to_anchor=(0.06, 0.86), ncol=len(n), fontsize=12, handletextpad=0.3,
               columnspacing=1.2)
    fig.supxlabel("phones between the regularized form and the real past   (sing + [d] = [sɪŋd] → sang [sæŋ]: 2)",
                  fontsize=14, y=-0.02)
    fig.suptitle("Is transfer graded by similarity to a regular past? (exploratory)", x=0.06, y=0.97, ha="left",
                 weight="bold")
    caption(axs[0, 0], "Regular → irregular trials, one point per target verb (about 14–200 trials each; jittered "
                       "horizontally). Regularized form = base + the expected [t]/[d]/[ɪd]. Classes are automatic, "
                       "not hand-checked.", -75)
    P.save_fig(fig, "fig_F_similarity_scatter")
    plt.close(fig)


def main():
    start = datetime.now()
    slide_style()
    t1 = pd.read_csv(P.TABLES / "t1_transfer_summary.csv")
    t7 = P.TABLES / "t7_zero_offset.csv"
    fig_a()
    fig_b_c(t1, pd.read_csv(t7) if t7.exists() else None)
    fig_d(pd.read_csv(P.TABLES / "t2_two_bin.csv"))
    fig_e(t1)
    fig_f(pd.read_csv(P.TABLES / "t6_similarity.csv"))
    P.log_run("python scripts/past_tense/04_figures.py", start, datetime.now(), "figures A-F (png + pdf)")
    print("wrote", sorted(p.name for p in P.FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
