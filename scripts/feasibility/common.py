"""
Shared paths, constants and helpers for the feasibility scripts and notebook
(`scripts/feasibility/*.py`, `notebooks/feasibility/feasibility_demo.ipynb`).

Paths resolve against `MORPH2VEC_ROOT` if set (e.g. an unpacked share bundle that mirrors the
repo layout), otherwise against the repo root. Import from a script with

    sys.path.insert(0, str(Path(__file__).resolve().parent)); import common
"""

import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- paths

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("MORPH2VEC_ROOT", REPO_ROOT))

DATASET = "librispeech-train-clean-100"
BASE_MODEL = "w2v2_8"
MODEL = "ffff_32"
EQUIVALENCE = "word_broad_10frames_fixedlen25"

INPUTS = ROOT / "outputs/analogy/inputs" / DATASET / "w2v2"
RESULTS = {
    "Word": ROOT / "outputs/analogy/runs" / DATASET / BASE_MODEL / MODEL / EQUIVALENCE / DATASET / "experiment_results.csv",
    "Wav2Vec": ROOT / "outputs/analogy/runs_id" / DATASET / BASE_MODEL / "experiment_results.csv",
}
MODEL_DIR = ROOT / "outputs/models" / DATASET / BASE_MODEL / MODEL / EQUIVALENCE
BEST_CHECKPOINT = "checkpoint-3200"  # lowest eval loss; checked against trainer_state.json in step 1
EMBEDDINGS = ROOT / "outputs/model_embeddings" / DATASET / BASE_MODEL / MODEL / EQUIVALENCE / f"{DATASET}.npy"
HIDDEN_STATES = ROOT / "outputs/hidden_states" / BASE_MODEL / f"{DATASET}.h5"
HIDDEN_STATES_SSD = Path("F:/morph2vec_hot/outputs/hidden_states") / BASE_MODEL / f"{DATASET}.h5"
POS_COUNTS = ROOT / "outputs/pos_counts" / f"{DATASET}.pkl"
PAPER_PDF = ROOT / "Project-Research-Paper_Morpho.pdf"
FEASIBILITY_DOC = ROOT / "CAS784/Project_Feasibility/Project_Feasibility.md"

FEAS = ROOT / "outputs/feasibility"
CHECKS, DATA, FIGURES, TABLES, THEORY, BUNDLE = (FEAS / d for d in ("checks", "data", "figures", "tables", "theory", "bundle"))

MODELS = ["Word", "Wav2Vec"]


def checkpoints() -> list[Path]:
    """Probe checkpoints, sorted by step."""
    return sorted(MODEL_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))


# --------------------------------------------------------------------------- loading / parsing

def parse_group(x):
    """`group` holds tuple reprs with numpy scalars, e.g. "('VBD', np.True_)"; literal_eval can't read those."""
    if not isinstance(x, str):
        return None
    return ast.literal_eval(x.replace("np.True_", "True").replace("np.False_", "False"))


def load_results(model: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS[model], low_memory=False)
    df["model"] = model
    return df


def load_spec():
    from src.analysis.state_space import StateSpaceAnalysisSpec
    return StateSpaceAnalysisSpec.from_hdf5(str(INPUTS / "state_space_spec.h5"))


def cut_phonemic_forms(spec, labels=None) -> pd.Series:
    """Phone string of every token, indexed by (label, instance_idx), as in prepare_inputs.ipynb cells 5-6."""
    cuts = spec.cuts.xs("phoneme", level="level")
    if labels is not None:
        cuts = cuts[cuts.index.get_level_values("label").isin(set(labels))]
    return cuts.groupby(["label", "instance_idx"], sort=False).description.agg(" ".join)


def most_common_pron(forms: pd.Series) -> pd.Series:
    """Most frequent phone string per label."""
    return forms.groupby("label").agg(lambda xs: xs.value_counts().index[0])


# --------------------------------------------------------------------------- phonology

ALLOMORPH_IPA = {"T": "t", "D": "d", "IH D": "ɪd", "AH D": "ɪd", "S": "s", "Z": "z", "IH Z": "ɪz", "AH Z": "ɪz"}
# step 2: the alignments write the unstressed -ed / -es vowel as IH or AH; treat both as ɪd / ɪz
ALLOMORPH_CANON = {"AH D": "IH D", "AH Z": "IH Z"}
INFLECTION_NAMES = {"NNS": "Noun", "VBZ": "Verb", "VBD": "Past"}


def guess_nns_vbz_allomorph(base_phones):
    """Copied from prepare_inputs.ipynb cell 9."""
    last_phone = base_phones[-1]
    if last_phone in {"S", "Z", "SH", "CH", "JH", "ZH"}:
        return "IH Z"
    elif last_phone in {"P", "T", "K", "F", "TH"}:
        return "S"
    return "Z"


def guess_past_allomorph(base_phones):
    """Copied from prepare_inputs.ipynb cell 9 (returns "IH D"; the alignments also use "AH D")."""
    last_phone = base_phones[-1]
    if last_phone in {"T", "D"}:
        return "IH D"
    elif last_phone in {"P", "F", "K", "S", "SH", "CH", "TH"}:
        return "T"
    return "D"


# summary notebook cell 125, with one-letter phones mapped explicitly (the original lowercases them,
# which turns R into "r" and Y into "y"). Stress is stripped, so AH covers both ʌ and schwa.
CMUDICT_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɝ", "EY": "eɪ",
    "IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    "CH": "tʃ", "DH": "ð", "HH": "h", "JH": "dʒ", "NG": "ŋ", "R": "ɹ", "SH": "ʃ", "TH": "θ", "Y": "j", "ZH": "ʒ",
    **{c: c.lower() for c in "BDFGKLMNPSTVWZ"},
}


def to_ipa(phones: str) -> str:
    return "".join(CMUDICT_TO_IPA.get(p, f"<{p}>") for p in str(phones).split())


# --------------------------------------------------------------------------- authors' exclusion lists

# summary notebook cell 17: VBZ items homophonous with another category
VBZ_EXCLUSIONS = [("VBZ", b) for b in "know seem please write meet read".split()]
# summary notebook cell 70: irregulars that slipped into "regular", stress-shift verbs, multiple pasts
EXCLUDE_VBD_BASE = ["do", "hide", "make"] \
    + "record permit protest reject subject conduct contract conflict increase decrease contest insult impact address escort".split() \
    + "dream leap".split()
# summary notebook cell 120
FC_EXCLUDE = "ay ba c des eh p k pa ca na b co ben been shun own".split()
FC_EXCLUDE_INFLECTED = "look'd push'd los ince".split()
# summary notebook cell 93: VBD entries missing from summarize_analogy_results.STRONG_OVERRIDES
VBD_STRONG_OVERRIDES = {
    ("an", "aunt", "T"): False, ("cap", "kept", "T"): True, ("loss", "lost", "T"): True,
    ("men", "meant", "T"): False, ("plan", "plant", "T"): False, ("saw", "sought", "T"): False,
    ("stay", "state", "T"): False, ("the", "that", "T"): False, ("way", "wait", "T"): False,
    ("when", "wind", "D"): True, ("row", "wrote", "T"): False,
}

# our flags for the irregular VBD set (not the authors'); the manual annotation supersedes these
MODAL_PASTS = {"could", "would", "should", "might"}
SUPPLETIVE_PAIRS = {("be", "was"), ("be", "were"), ("go", "went")}
HOMOGRAPH_PASTS = {("wind", "wound"), ("lie", "lay"), ("rise", "rose")}


ARPABET_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
ENDING_CLASSES = {"T": "regular_t", "D": "regular_d", "IH D": "regular_id"}


def onset(pron: str) -> tuple[str, ...]:
    phones = pron.split()
    i = next((k for k, p in enumerate(phones) if p in ARPABET_VOWELS), len(phones))
    return tuple(phones[:i])


def auto_class(row) -> tuple[str, str]:
    """
    First guess at a VBD pair's class, with the reason (the manual annotation supersedes it). `row` needs
    base, inflected, base_pron, past_pron, is_regular, n_pasts_for_base, post_divergence_canon,
    expected_allomorph (the columns of pairs_vbd.parquet).
    """
    base, past, bp, pp = row.base, row.inflected, row.base_pron, row.past_pron
    if (base, past) in SUPPLETIVE_PAIRS:
        return "suppletive", "different root"
    if past in MODAL_PASTS:
        return "check", "modal; past use is marginal"
    if (base, past) in HOMOGRAPH_PASTS:
        return "check", f"'{past}' is also another word"
    if row.n_pasts_for_base > 1:
        return "check", "base has more than one past form"
    if row.is_regular:
        if past == base + "t":
            return "check", "pipeline calls this regular, but it is a base+t spelling"
        if base.endswith("ay") and past == base[:-1] + "id":
            return "check", "pipeline calls this regular, but it is ay -> aid"
        ending = row.post_divergence_canon
        if ending == row.expected_allomorph:
            return ENDING_CLASSES[ending], ""
        return "check", f"regular spelling, but the ending is '{ending}' where the rule predicts '{row.expected_allomorph}'"
    b, p = bp.split(), pp.split()
    if b[-1] == "D" and p == b[:-1] + ["T"]:
        return "devoicing", "final D becomes T"
    if p[-1] in ("T", "D") and p[-1] != b[-1] and onset(bp) == onset(pp):
        return "semi_regular_candidate", "past adds a final t/d, plus a stem change"
    return "irregular_unclassified", ""


def fmt_rank(v) -> str:
    """Compact rank label: 0.038, 5.4, 87, 1,830, 10,880."""
    if v is None or not np.isfinite(v):
        return "–"
    return f"{v:,.0f}" if v >= 100 else f"{v:.2g}"


# --------------------------------------------------------------------------- paper reference values
# Mean rank. Checked against the text layer of the paper's Figs 3-5 (PDF pages 5 and 7) on 2026-10-21.
PAPER_CONFIRMED = True

PAPER_FIG3 = {  # (model, source, target) -> mean rank
    ("Word", "NNS", "NNS"): 1, ("Word", "NNS", "VBZ"): 10, ("Word", "VBZ", "NNS"): 2.7, ("Word", "VBZ", "VBZ"): 7.9,
    ("Wav2Vec", "NNS", "NNS"): 6.7, ("Wav2Vec", "NNS", "VBZ"): 68, ("Wav2Vec", "VBZ", "NNS"): 29, ("Wav2Vec", "VBZ", "VBZ"): 37,
}
_FIG4_LABELS = ["Noun [s]", "Noun [z]", "Noun [ɪz]", "Verb [s]", "Verb [z]"]
_FIG4 = {
    "Word": [[0.24, 0.41, 1.0, 0.14, 7.3], [0.22, 0.30, 0.61, 0.087, 5.9], [4.1, 1.6, 0.52, 2.5, 12],
             [0.25, 0.52, 1.9, 0.076, 5.4], [0.71, 0.48, 1.7, 0.038, 11]],
    "Wav2Vec": [[41, 8.1, 4.2, 41, 40], [0.63, 1.2, 1.8, 37, 34], [1.9, 2.2, 1, 31, 35],
                [0.83, 19, 4.1, 0.24, 6], [37, 59, 57, 41, 47]],
}
PAPER_FIG4 = {model: pd.DataFrame(v, index=_FIG4_LABELS, columns=_FIG4_LABELS).rename_axis(index="src", columns="tgt")
              for model, v in _FIG4.items()}
PAPER_FIG5 = {  # (model, source, target) -> mean rank
    ("Word", "NNS", "NNS"): 1, ("Word", "NNS", "NNS-FF"): 5.3, ("Word", "NNS-FF", "NNS"): 1.4, ("Word", "NNS-FF", "NNS-FF"): 8.4,
    ("Wav2Vec", "NNS", "NNS"): 6.7, ("Wav2Vec", "NNS", "NNS-FF"): 47, ("Wav2Vec", "NNS-FF", "NNS"): 3, ("Wav2Vec", "NNS-FF", "NNS-FF"): 76,
    ("Word", "VBZ", "VBZ"): 7.9, ("Word", "VBZ", "VBZ-FF"): 4.3, ("Word", "VBZ-FF", "VBZ"): 18, ("Word", "VBZ-FF", "VBZ-FF"): 8.7,
    ("Wav2Vec", "VBZ", "VBZ"): 37, ("Wav2Vec", "VBZ", "VBZ-FF"): 61, ("Wav2Vec", "VBZ-FF", "VBZ"): 36, ("Wav2Vec", "VBZ-FF", "VBZ-FF"): 66,
}
PAPER_CHANCE_RANK = 94306
PAPER_N_TOKENS = 334008
PAPER_N_UNAMBIGUOUS = (753, 52)  # nouns, verbs (Sec. 3.5)

# What `scripts/summarize_analogy_results.py` prints for our run (no post-hoc exclusions), as recorded in
# Project_Feasibility.md §10.2-10.3; the notebook asserts that its unexcluded numbers reproduce these.
SCRIPT_FIG3 = {
    ("Word", "NNS", "NNS"): 0.9, ("Word", "NNS", "VBZ"): 5.4, ("Word", "VBZ", "NNS"): 3.2, ("Word", "VBZ", "VBZ"): 5.9,
    ("Wav2Vec", "NNS", "NNS"): 7.9, ("Wav2Vec", "NNS", "VBZ"): 51.2, ("Wav2Vec", "VBZ", "NNS"): 21.7, ("Wav2Vec", "VBZ", "VBZ"): 36.3,
}
_SCRIPT_FIG4 = {
    "Word": [[0.28, 0.26, 1.43, 0.30, 6.14], [0.16, 0.21, 0.47, 0.20, 6.27], [3.16, 1.69, 0.25, 5.28, 11.86],
             [0.33, 0.66, 1.62, 0.38, 6.31], [1.21, 2.74, 1.79, 0.14, 7.06]],
    "Wav2Vec": [[9.3, 11.7, 41.6, 11.5, 46.0], [0.99, 0.57, 2.9, 19.8, 49.1], [2.1, 0.78, 0.88, 30.7, 40.5],
                [1.6, 1.7, 4.5, 0.59, 4.0], [32.5, 38.4, 43.1, 49.5, 52.4]],
}
SCRIPT_FIG4 = {model: pd.DataFrame(v, index=_FIG4_LABELS, columns=_FIG4_LABELS) for model, v in _SCRIPT_FIG4.items()}

# Project facts not in the compact data (Project_Feasibility.md §9-10)
AUDIO_HOURS, N_UTTERANCES = 100.49, 28538

# --------------------------------------------------------------------------- statistics

def bootstrap_ci(df: pd.DataFrame, value: str, unit: str = "base_to", n: int = 2000, seed: int = 0,
                 ci: float = 0.95) -> dict:
    """
    Trial-weighted mean of `value` with a percentile CI from resampling `unit` (items), not trials.
    Works for ranks and for booleans (accuracy).
    """
    g = df.groupby(unit)[value].agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"mean": sums.sum() / counts.sum(), "lo": lo, "hi": hi, "n_trials": int(counts.sum()), "n_items": len(g)}


def rank_summary(df: pd.DataFrame, by: list[str], unit: str = "base_to", n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per cell: mean rank and top-1 accuracy with item-bootstrap CIs, median rank, n trials / items."""
    rows = []
    for key, rows_i in df.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        r = bootstrap_ci(rows_i, "gt_label_rank", unit, n, seed)
        a = bootstrap_ci(rows_i.assign(top1=rows_i.gt_label_rank == 0), "top1", unit, n, seed)
        rows.append({**dict(zip(by, key)),
                     "mean_rank": r["mean"], "mean_lo": r["lo"], "mean_hi": r["hi"],
                     "top1": a["mean"], "top1_lo": a["lo"], "top1_hi": a["hi"],
                     "median_rank": rows_i.gt_label_rank.median(),
                     "n_trials": r["n_trials"], "n_items": r["n_items"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plotting / output

# Okabe-Ito (colour-blind safe); one colour per model, used everywhere
MODEL_COLORS = {"Word": "#0072B2", "Wav2Vec": "#E69F00"}
CONTROL_COLOR = "#999999"


def set_style():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(context="notebook", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({"figure.dpi": 100, "savefig.bbox": "tight", "axes.titleweight": "bold"})


def heatmap(ax, values: pd.DataFrame, annot: pd.DataFrame, norm, cmap="magma_r", title=None,
            xlabel="Target", ylabel="Source", fontsize=9):
    """Annotated heatmap of `values` (NaN cells grey); `annot` holds the cell texts."""
    import matplotlib.pyplot as plt
    v = values.to_numpy(float)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#dddddd")
    im = ax.imshow(np.ma.masked_invalid(v), norm=norm, cmap=cm, aspect="auto")
    ax.set_xticks(range(v.shape[1]), [str(c) for c in values.columns])
    ax.set_yticks(range(v.shape[0]), [str(i) for i in values.index])
    ax.grid(False)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            dark = np.isfinite(v[i, j]) and norm(v[i, j]) > 0.55
            ax.text(j, i, annot.iat[i, j], ha="center", va="center", fontsize=fontsize,
                    color="white" if dark else "black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return im


def ci_bars(ax, summary: pd.DataFrame, cat: str, order: list, est="mean_rank", lo="mean_lo", hi="mean_hi",
            ylabel="", log=False, show_n=True, fmt=None):
    """Grouped bars (one colour per model) with CI error bars; n trials / items under each bar."""
    width = 0.8 / len(MODELS)
    x0 = np.arange(len(order))
    fmt = fmt or fmt_rank
    if isinstance(fmt, str):
        fmt = fmt.format
    for k, model in enumerate(MODELS):
        s = summary[summary.model == model].set_index(cat).reindex(order)
        x = x0 + (k - (len(MODELS) - 1) / 2) * width
        err = np.vstack([s[est] - s[lo], s[hi] - s[est]]).clip(min=0)
        ax.bar(x, s[est], width, color=MODEL_COLORS[model], label=model, yerr=err, capsize=3, error_kw={"lw": 1})
        for xi, (e, h) in zip(x, zip(s[est], s[hi])):
            if np.isfinite(e):
                ax.text(xi, h, " " + fmt(e), ha="center", va="bottom", fontsize=7, rotation=90)
        if show_n:
            for xi, (nt, ni) in zip(x, zip(s.n_trials, s.n_items)):
                if np.isfinite(nt):
                    ax.annotate(f"{int(nt)}\n{int(ni)}", (xi, 0), xycoords=("data", "axes fraction"),
                                xytext=(0, -4), textcoords="offset points", ha="center", va="top", fontsize=6, color="#555555")
    ax.set_xticks(x0, order)
    ax.tick_params(axis="x", pad=22 if show_n else 4)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    return ax


def save_fig(fig, name: str):
    """Save to outputs/feasibility/figures as png (dpi 200) and pdf."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200)
    fig.savefig(FIGURES / f"{name}.pdf")


def save_table(df: pd.DataFrame, name: str, index=False):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=index)


def md_table(df: pd.DataFrame, index=True, floatfmt="{:.3g}") -> str:
    """Markdown table without the `tabulate` dependency."""
    if index:
        df = df.reset_index()
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def human_size(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024 or unit == "TB":
            return f"{n_bytes:.1f} {unit}" if unit != "B" else f"{int(n_bytes)} B"
        n_bytes /= 1024
