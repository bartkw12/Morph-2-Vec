"""
Past-tense step 3: summarize the midpoint run. Reads the runs' experiment_results.csv and small files only, and
writes outputs/past_tense/data/ (parquet, gitignored) and outputs/past_tense/tables/:

    data/label_pron.parquet     most common pronunciation and token count of every word in the analogy set
    data/trials.parquet         all trials of both models, with source / target verb classes
    t1_transfer_summary.csv     per experiment x model: top-1 (strict and lenient), mean / median rank, CIs, n
    t2_two_bin.csv              regular -> "adds a t/d" vs "no added stop", verb-weighted
    t3_nearest_neighbours.csv   per target verb: top-1 rate and the most frequent wrong guesses
    t4_overreg_proxy.csv        wrong guesses ending in T/D, or that are base + t/d (exploratory)
    t5_vowel_split.csv          regular -> vowel change, split by whether the past ends in t/d
    t6_similarity.csv           per target verb: top-1 vs. edit distance from the regularized form
    t6_similarity_fit.csv       verb-level fit top-1 ~ edit distance + adds_stop (exploratory)
    sanity_checks.csv           the handoff's section 5 checklist with actual values

CIs are 95% percentile bootstraps over target base verbs (`base_to`, 2,000 resamples), as in the feasibility
notebook. "Lenient" top-1 also counts a guess that has the target's most common pronunciation (knew ~ new).

    python scripts/past_tense/03_summarize.py [--runs-root outputs/past_tense/runs]
"""

from argparse import ArgumentParser
import ast
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt_common as P  # noqa: E402

C = P.C
TD_ENDINGS = ("T", "D", "IH D", "AH D")
NN_EXPERIMENTS = ["reg_to_vowel_change", "reg_to_semi_regular", "reg_to_devoicing", "reg_to_suppletive"]


# --------------------------------------------------------------------------- data

def label_pron() -> pd.DataFrame:
    """Most common pronunciation of every label (cached: it needs the state-space spec)."""
    path = P.DATA / "label_pron.parquet"
    if path.exists():
        return pd.read_parquet(path)
    spec = C.load_spec()
    forms = C.cut_phonemic_forms(spec)
    vc = forms.groupby(level="label").value_counts()
    pron = vc.groupby(level=0).idxmax().map(lambda x: x[1])
    df = pd.DataFrame({"label": pron.index, "pron": pron.to_numpy()})
    df["n_tokens"] = df.label.map(spec.label_counts)
    P.DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_trials(runs_root: Path, classes: pd.DataFrame, prons: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    parts, order = [], None
    for model, d in P.RUN_DIRS.items():
        path = runs_root / d / "experiment_results.csv"
        if not path.exists():
            print(f"WARNING: no results for {model} ({path})")
            continue
        df = pd.read_csv(path, low_memory=False).rename(columns={"Unnamed: 1": "trial_idx"})
        df["model"] = model
        parts.append(df)
        order = order or list(json.loads((runs_root / d / "experiments.json").read_text())["experiments"])
    t = pd.concat(parts, ignore_index=True)
    t["inflected_from"] = t.from_equiv_label.map(lambda x: ast.literal_eval(x)[1])
    t["inflected_to"] = t.to_equiv_label.map(lambda x: ast.literal_eval(x)[1])
    assert (t.inflected_to == t.gt_label).all()

    meta = pd.DataFrame([{"experiment": e, **P.parse_experiment(e)} for e in order])
    t = t.merge(meta, on="experiment", how="left", validate="m:1")

    cl = classes.set_index(["base", "inflected"])
    src = cl[["past_class", "allomorph"]].rename(columns={"past_class": "source_class_obs", "allomorph": "source_ending"})
    t = t.merge(src, left_on=["base_from", "inflected_from"], right_index=True, how="left")
    t["source_class_obs"] = t.source_class_obs.fillna("random")
    t["source_ending"] = t.source_ending.fillna("none")
    tgt = cl[["past_class", "allomorph", "adds_stop", "past_ends_td", "edit_dist", "past_homophones", "base_pron"]] \
        .rename(columns={"past_class": "target_class_obs", "allomorph": "target_allomorph", "base_pron": "base_to_pron"})
    t = t.merge(tgt, left_on=["base_to", "gt_label"], right_index=True, how="left")

    # every trial must come from the classes (and source ending) its experiment asked for
    assert (t.source_class_obs == t.source).all(), t.loc[t.source_class_obs != t.source, ["experiment", "base_from"]].head()
    assert (t.target_class_obs == t.target).all(), t.loc[t.target_class_obs != t.target, ["experiment", "base_to"]].head()
    filtered = t.source_allomorph.notna()
    assert (t.source_ending[filtered] == t.source_allomorph[filtered]).all()
    t = t.drop(columns=["source_class_obs", "target_class_obs"])

    pron = prons.set_index("label").pron
    t["predicted_pron"] = t.predicted_label.map(pron)
    t["top1"] = t.gt_label_rank == 0
    hom = t.past_homophones.fillna("").str.split()
    t["top1_lenient"] = t.top1 | np.array([p in h for p, h in zip(t.predicted_label, hom)])
    t["experiment"] = pd.Categorical(t.experiment, categories=order, ordered=True)
    return t, order


# --------------------------------------------------------------------------- statistics

def verb_weighted(df: pd.DataFrame, value: str, unit: str = "base_to", n: int = 2000, seed: int = 0) -> dict:
    """Mean over items of the per-item mean of `value`, with a percentile CI from resampling items."""
    per = df.groupby(unit)[value].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = per[rng.integers(0, len(per), size=(n, len(per)))].mean(1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"mean": per.mean(), "lo": lo, "hi": hi, "n_trials": len(df), "n_items": len(per)}


def transfer_summary(t: pd.DataFrame) -> pd.DataFrame:
    s = C.rank_summary(t, ["model", "experiment"])
    extra = []
    for (model, exp), rows in t.groupby(["model", "experiment"], observed=True):
        le = C.bootstrap_ci(rows, "top1_lenient")
        vw = verb_weighted(rows, "top1")
        extra.append({"model": model, "experiment": exp, "top1_lenient": le["mean"], "top1_lenient_lo": le["lo"],
                      "top1_lenient_hi": le["hi"], "top1_verb_weighted": vw["mean"],
                      "n_target_pairs": rows[["base_to", "gt_label"]].drop_duplicates().shape[0],
                      "n_source_pairs": rows[["base_from", "inflected_from"]].drop_duplicates().shape[0]})
    s = s.merge(pd.DataFrame(extra), on=["model", "experiment"])
    s["lenient_minus_strict"] = s.top1_lenient - s.top1
    s = s.rename(columns={"n_items": "n_target_verbs"})
    meta = t.drop_duplicates("experiment")[["experiment", "source", "source_allomorph", "target"]].astype({"experiment": str})
    s = s.astype({"experiment": str}).merge(meta, on="experiment")
    s["small_target_class"] = s.n_target_verbs < 10
    s["experiment"] = pd.Categorical(s.experiment, categories=t.experiment.cat.categories, ordered=True)
    cols = ["model", "experiment", "source", "source_allomorph", "target", "top1", "top1_lo", "top1_hi",
            "top1_lenient", "top1_lenient_lo", "top1_lenient_hi", "lenient_minus_strict", "top1_verb_weighted",
            "mean_rank", "mean_lo", "mean_hi", "median_rank", "n_trials", "n_target_verbs", "n_target_pairs",
            "n_source_pairs", "small_target_class"]
    return s.sort_values(["experiment", "model"])[cols].reset_index(drop=True)


def two_bin(t: pd.DataFrame) -> pd.DataFrame:
    """Regular -> irregular pooled into 'adds a t/d' vs 'no added stop', each target verb weighted equally."""
    bins = {
        "regular -> regular": ("reg_to_reg", "random_to_reg", None),
        "regular -> adds a t/d (semi-regular + devoicing)": ("reg_to_", "random_to_", True),
        "regular -> no added stop (vowel change + suppletive)": ("reg_to_", "random_to_", False),
    }
    rows = []
    for model, tm in t.groupby("model"):
        for label, (exp, floor_exp, adds) in bins.items():
            for kind, prefix in [("transfer", exp), ("random floor", floor_exp)]:
                if adds is None:
                    d = tm[tm.experiment == prefix]
                else:
                    d = tm[tm.experiment.astype(str).isin([f"{prefix}{c}" for c in P.IRREGULAR_CLASSES])
                           & (tm.adds_stop == adds)]
                vw = verb_weighted(d, "top1")
                vl = verb_weighted(d, "top1_lenient")
                tw = C.bootstrap_ci(d, "top1")
                rows.append({"model": model, "bin": label, "kind": kind,
                             "top1_verb_weighted": vw["mean"], "lo": vw["lo"], "hi": vw["hi"],
                             "top1_lenient_verb_weighted": vl["mean"], "top1_trial_weighted": tw["mean"],
                             "n_trials": vw["n_trials"], "n_target_verbs": vw["n_items"],
                             "classes": ", ".join(sorted(d.target.unique()))})
    return pd.DataFrame(rows)


def nearest_neighbours(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = t[t.experiment.astype(str).isin(NN_EXPERIMENTS)]
    for (model, exp, base, past), g in d.groupby(["model", "experiment", "base_to", "gt_label"], observed=True):
        wrong = g[~g.top1]
        top = wrong.predicted_label.value_counts().head(3)
        prons = wrong.drop_duplicates("predicted_label").set_index("predicted_label").predicted_pron
        r = {"model": model, "experiment": exp, "target_class": g.target.iloc[0], "base": base, "past": past,
             "base_pron": g.base_to_pron.iloc[0], "n_trials": len(g), "top1": g.top1.mean(),
             "top1_lenient": g.top1_lenient.mean(), "mean_rank": g.gt_label_rank.mean()}
        for k, (lab, n) in enumerate(top.items(), 1):
            r[f"wrong{k}"], r[f"wrong{k}_n"], r[f"wrong{k}_pron"] = lab, int(n), prons.get(lab, "")
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["model", "experiment", "top1"]).reset_index(drop=True)


def overreg_proxy(t: pd.DataFrame, prons: pd.DataFrame) -> pd.DataFrame:
    """Among wrong guesses: share whose pronunciation ends in T/D, or is the target's base + t/d/ɪd."""
    w = t[~t.top1].copy()
    pp = w.predicted_pron.fillna("")
    w["ends_td"] = pp.str.split().str[-1].isin(["T", "D"])
    w["base_plus_td"] = [any(p == f"{b} {e}" for e in TD_ENDINGS) for p, b in zip(pp, w.base_to_pron)]
    w["starts_with_base"] = [p.startswith(b + " ") for p, b in zip(pp, w.base_to_pron)]
    lexicon_td = (prons.pron.str.split().str[-1].isin(["T", "D"]) * prons.n_tokens).sum() / prons.n_tokens.sum()
    rows = []
    for exp in ["reg_to_vowel_change", "random_to_vowel_change", "reg_to_semi_regular", "random_to_semi_regular",
                "reg_to_suppletive", "random_to_suppletive", "reg_to_reg", "random_to_reg"]:
        for model, g in w[w.experiment == exp].groupby("model"):
            r = {"model": model, "experiment": exp, "n_wrong": len(g), "n_target_verbs": g.base_to.nunique()}
            for col in ["ends_td", "base_plus_td", "starts_with_base"]:
                b = C.bootstrap_ci(g, col)
                r |= {f"share_{col}": b["mean"], f"share_{col}_lo": b["lo"], f"share_{col}_hi": b["hi"]}
            r["examples_base_plus_td"] = ", ".join(
                f"{bt}->{pl}" for bt, pl in g.loc[g.base_plus_td, ["base_to", "predicted_label"]].drop_duplicates().head(8).to_numpy())
            rows.append(r)
    df = pd.DataFrame(rows)
    df["lexicon_share_ends_td_tokens"] = lexicon_td
    df["note"] = "exploratory: wrong guesses only; CIs resample target verbs"
    return df


def vowel_split(t: pd.DataFrame) -> pd.DataFrame:
    d = t[t.experiment.astype(str).isin(["reg_to_vowel_change", "random_to_vowel_change", "vowel_change_to_vowel_change"])].copy()
    d["past_ends_td"] = d.past_ends_td.map({True: "past ends in t/d (met, wrote)", False: "past does not (sang, took)"})
    s = C.rank_summary(d, ["model", "experiment", "past_ends_td"])
    le = d.groupby(["model", "experiment", "past_ends_td"], observed=True).top1_lenient.mean().rename("top1_lenient").reset_index()
    return s.merge(le, on=["model", "experiment", "past_ends_td"]).rename(columns={"n_items": "n_target_verbs"})


def similarity(t: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = t[t.experiment.astype(str).isin(["reg_to_reg"] + [f"reg_to_{c}" for c in P.IRREGULAR_CLASSES])]
    per = d.groupby(["model", "target", "base_to", "gt_label"], observed=True).agg(
        adds_stop=("adds_stop", "first"), past_ends_td=("past_ends_td", "first"), edit_dist=("edit_dist", "first"),
        n_trials=("top1", "size"), top1=("top1", "mean"), top1_lenient=("top1_lenient", "mean"),
        mean_rank=("gt_label_rank", "mean")).reset_index()

    # verb level, irregular targets only: top1 = a + b * edit_dist + c * adds_stop, bootstrapped over verbs
    fits = []
    rng = np.random.default_rng(0)
    for model, g in per[per.target != "regular"].groupby("model"):
        X = np.column_stack([np.ones(len(g)), g.edit_dist.to_numpy(float), g.adds_stop.astype(float).to_numpy()])
        y = g.top1.to_numpy(float)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        boot = []
        for _ in range(2000):
            i = rng.integers(0, len(g), len(g))
            if np.linalg.matrix_rank(X[i]) == 3:
                boot.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
        lo, hi = np.quantile(np.array(boot), [0.025, 0.975], axis=0)
        from scipy.stats import spearmanr
        rho = spearmanr(g.edit_dist, g.top1).statistic
        for k, name in enumerate(["intercept", "edit_dist", "adds_stop"]):
            fits.append({"model": model, "term": name, "coef": coef[k], "lo": lo[k], "hi": hi[k], "n_verbs": len(g),
                         "spearman_edit_dist_top1": rho,
                         "note": "exploratory linear fit on per-verb top-1; percentile CI over verbs"})
    return per, pd.DataFrame(fits)


def sanity_checks(s: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    t9 = pd.read_csv(C.TABLES / "table9_regular_vs_irregular.csv", encoding="utf-8")
    ref = t9.set_index(["model", "condition"]).top1
    rows = []

    def get(model, exp):
        return s[(s.model == model) & (s.experiment == exp)].iloc[0]

    for model in s.model.unique():
        r = get(model, "reg_to_reg")
        v = ref[(model, "past regular (auto-class regular only)")]
        rows.append({"check": "reg_to_reg top-1 close to feasibility 'past regular (auto-class regular only)'",
                     "model": model, "value": r.top1, "lo": r.top1_lo, "hi": r.top1_hi, "reference": v,
                     "pass": r.top1_lo <= v <= r.top1_hi})
        r = get(model, "vowel_change_to_vowel_change")
        lo_ref = min(ref[(model, "past irregular")], ref[(model, "past irregular (no modals, be, homographs)")])
        hi_ref = max(ref[(model, "past irregular")], ref[(model, "past irregular (no modals, be, homographs)")])
        rows.append({"check": "vowel_change_to_vowel_change in the range of feasibility irregular -> irregular",
                     "model": model, "value": r.top1, "lo": r.top1_lo, "hi": r.top1_hi,
                     "reference": f"{lo_ref:.2f}-{hi_ref:.2f}", "pass": r.top1_lo <= hi_ref and r.top1_hi >= lo_ref})
        v = ref[(model, next(c for c in t9.condition if c.startswith("random pair")))]
        for exp in [e for e in s.experiment.cat.categories if e.startswith("random_to_")]:
            r = get(model, exp)
            rows.append({"check": f"{exp} near the feasibility random-pair control (-> NNS / VBZ)", "model": model,
                         "value": r.top1, "lo": r.top1_lo, "hi": r.top1_hi, "reference": v,
                         "pass": abs(r.top1 - v) <= 0.05 if model == "Word" else None})
    for _, r in s.drop_duplicates("experiment").iterrows():
        if r.small_target_class:
            rows.append({"check": f"{r.experiment}: fewer than 10 target verbs (flag)", "model": "both",
                         "value": r.n_target_verbs, "reference": 10, "pass": None})
    if t.model.nunique() == 2:
        k = ["experiment", "base_from", "inflected_from", "base_to", "gt_label"]
        a, b = (t[t.model == m].sort_values(["experiment", "trial_idx"])[k].reset_index(drop=True) for m in ["Word", "Wav2Vec"])
        rows.append({"check": "both models saw the same sampled trials (same seed)", "model": "both",
                     "value": bool(a.equals(b)), "reference": True, "pass": bool(a.equals(b))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main

def main(args):
    start = datetime.now()
    runs_root = Path(args.runs_root)
    classes = pd.read_csv(P.CLASSES_CSV, keep_default_na=False,
                          dtype={"adds_stop": "string"}).assign(
        adds_stop=lambda d: d.adds_stop.map({"True": True, "False": False}).astype("boolean"),
        past_ends_td=lambda d: d.past_ends_td.astype(str) == "True")
    prons = label_pron()
    t, order = load_trials(runs_root, classes, prons)
    P.DATA.mkdir(parents=True, exist_ok=True)
    keep = ["model", "experiment", "trial_idx", "source", "source_allomorph", "source_ending", "target", "base_from", "inflected_from",
            "base_to", "gt_label", "target_allomorph", "adds_stop", "past_ends_td", "edit_dist", "predicted_label",
            "predicted_pron", "gt_label_rank", "gt_distance", "top1", "top1_lenient"]
    t[keep].to_parquet(P.DATA / "trials.parquet", index=False)

    s = transfer_summary(t)
    P.save_table(s, "t1_transfer_summary")
    P.save_table(two_bin(t), "t2_two_bin")
    P.save_table(nearest_neighbours(t), "t3_nearest_neighbours")
    P.save_table(overreg_proxy(t, prons), "t4_overreg_proxy")
    P.save_table(vowel_split(t), "t5_vowel_split")
    per, fit = similarity(t)
    P.save_table(per, "t6_similarity")
    P.save_table(fit, "t6_similarity_fit")
    sc = sanity_checks(s, t)
    P.save_table(sc, "sanity_checks")

    show = s[["model", "experiment", "top1", "top1_lo", "top1_hi", "top1_lenient", "mean_rank", "median_rank",
              "n_trials", "n_target_verbs"]]
    print(C.md_table(show, index=False))
    print()
    print(C.md_table(sc, index=False))
    P.log_run("python scripts/past_tense/03_summarize.py " + " ".join(sys.argv[1:]), start, datetime.now(),
              f"{len(t):,} trials ({', '.join(sorted(t.model.unique()))}); tables t1-t6, sanity_checks")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", default=str(P.RUNS))
    main(p.parse_args())
