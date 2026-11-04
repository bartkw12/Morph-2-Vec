"""
Past-tense step 5: recompute every number on the obstacles slide from its source, instead of copying it from the
write-ups. Writes outputs/past_tense/tables/obstacles.csv (obstacle, item, value, n, source).

    python scripts/past_tense/05_obstacle_numbers.py
"""

from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt_common as P  # noqa: E402

C = P.C


def rel(path) -> str:
    return Path(path).relative_to(C.ROOT).as_posix()


def main():
    start = datetime.now()
    rows = []

    def add(obstacle, item, value, n="", source=""):
        rows.append({"obstacle": obstacle, "item": item, "value": value, "n": n, "source": source})

    # 1a. the [ɪd] ending is spelled IH D or AH D by the aligner; the authors' code only knew IH D
    O1 = "1. labels matter as much as the model"
    mca = pd.read_csv(C.INPUTS / "most_common_allomorphs.csv")
    v = mca[mca.inflection == "VBD"].most_common_allomorph.value_counts()
    n_id = int(v.get("IH D", 0) + v.get("AH D", 0))
    add(O1, "[ɪd] verbs whose most common ending is AH D (missed by the authors' IH D-only code)",
        f"{v.get('AH D', 0)} of {n_id} ({v.get('AH D', 0) / n_id:.0%})", f"{n_id} [ɪd] verbs", rel(C.INPUTS / "most_common_allomorphs.csv"))
    pairs = pd.read_parquet(C.DATA / "pairs_vbd.parquet")
    ii = pd.read_parquet(C.INPUTS / "inflection_instances.parquet", columns=["inflection", "base", "inflected", "post_divergence"])
    ii = ii[ii.inflection == "VBD"].merge(pairs[pairs.is_regular][["base", "inflected"]], on=["base", "inflected"])
    tok = ii.post_divergence.value_counts()
    n_tok = int(tok.get("IH D", 0) + tok.get("AH D", 0))
    add(O1, "[ɪd] tokens of regular past forms spelled AH D", f"{tok.get('AH D', 0):,} of {n_tok:,} ({tok.get('AH D', 0) / n_tok:.0%})",
        f"{len(ii):,} regular past tokens", rel(C.INPUTS / "inflection_instances.parquet"))

    # 1b. the pipeline's "regular" flag lets in stem-changing verbs, which dominate the mean rank
    vbd = pd.read_parquet(C.DATA / "vbd_trials.parquet")
    pairs[["auto_class", "auto_reason"]] = pairs.apply(C.auto_class, axis=1, result_type="expand")
    clean = set(map(tuple, pairs.loc[pairs.auto_class.str.startswith("regular_"), ["base", "inflected"]].to_numpy()))
    vr = vbd[(vbd.experiment == "regular") & (vbd.class_from == "regular") & ~vbd.authors_excluded]
    is_clean = np.array([(b, i) in clean and (b2, i2) in clean
                         for b, i, b2, i2 in zip(vr.base_from, vr.inflected_from, vr.base_to, vr.inflected_to)])
    for model in C.MODELS:
        m = vr.model == model
        r = vr.loc[m, "gt_label_rank"].sort_values(ascending=False).to_numpy(float)
        k = int(np.ceil(0.01 * len(r)))
        add(O1, f"{model}: regular-past mean rank, pipeline 'regular' flag -> auto-class regular only",
            f"{r.mean():.1f} -> {vr.loc[m & is_clean, 'gt_label_rank'].mean():.1f}",
            f"{m.sum()} -> {(m & is_clean).sum()} trials", rel(C.DATA / "vbd_trials.parquet"))
        add(O1, f"{model}: share of the total rank carried by the worst 1% of regular-past trials",
            f"{r[:k].sum() / r.sum():.0%}", f"worst {k} of {len(r)} trials", rel(C.DATA / "vbd_trials.parquet"))
    worst = vr[vr.model == "Word"].nlargest(8, "gt_label_rank")
    add(O1, "Word: the worst regular-past trials (target past, rank)",
        "; ".join(f"{t} {int(rk):,}" for t, rk in zip(worst.inflected_to, worst.gt_label_rank)), "8 trials",
        rel(C.DATA / "vbd_trials.parquet"))

    # 2. no hand classification
    O2 = "2. no hand classification of the verbs"
    x = pd.read_excel(C.THEORY / "vbd_pairs_for_annotation.xlsx")
    filled = int(x[["class", "sound_change_family", "exclude", "notes"]].notna().any(axis=1).sum())
    add(O2, "pairs with any manual annotation", f"{filled} of {len(x)}", "", rel(C.THEORY / "vbd_pairs_for_annotation.xlsx"))
    cc = pd.read_csv(P.TABLES / "class_counts.csv")
    for _, r in cc.iterrows():
        if not r["class"].startswith("  "):
            add(O2, f"class size: {r['class']}", f"{r.pairs_used} pairs used / {r.base_verbs_used} base verbs",
                f"{r.pairs_auto_class} before exclusions", rel(P.TABLES / "class_counts.csv"))
    add(O2, "pairs needing a human check (all non-regular plus 'check')",
        int((pairs.auto_class.isin(["semi_regular_candidate", "devoicing", "irregular_unclassified", "suppletive", "check"])).sum()),
        "", rel(C.DATA / "pairs_vbd.parquet"))

    # 3. compute on one machine
    O3 = "3. compute on one machine"
    h5 = C.ROOT / "outputs/hidden_states/w2v2_8/librispeech-train-clean-100.h5"
    size = h5.stat().st_size
    add(O3, "layer-8 hidden states, train-clean-100", f"{size / 2**30:.1f} GiB ({size / 1e9:.1f} GB)", "one layer", rel(h5))
    add(O3, "per extra layer (extract, train, embed, two analogy runs)", "about 3.5 h", "", "CAS784/Project_Feasibility/Project_Feasibility.md §9")
    add(O3, "out-of-memory failures in the original runs",
        "preprocessing (run 3), hidden-state extraction at 61% (run 3), raw-wav2vec analogy paged to disk at 17 GB "
        "(run 5: 49/120 experiments in 10.5 h); editor and NVIDIA service crashed when memory ran out",
        "", "CAS784/Project_Feasibility/Project_Feasibility.md §9, §14")
    add(O3, "run_pipeline traps", "analogy notebook resumes partial results by position; the runner skips a stage whose "
        "experiment_results.csv exists", "", "notebooks/analogy/run.ipynb cell 26; scripts/run_pipeline.py run_stage")
    for model, d in P.RUN_DIRS.items():
        t = P.RUNS / d / "experiment_timings.csv"
        if t.exists():
            tt = pd.read_csv(t)
            add(O3, f"this run, {model}: experiment loop", f"{tt.seconds.sum() / 60:.1f} min",
                f"{tt.n_trials.sum():,} trials, {len(tt)} experiments", rel(t))
    add(O3, "this run: wall-clock times per command", "see run_log.md", "", rel(P.RUN_LOG))

    # 4. statistics
    O4 = "4. statistics"
    add(O4, "probe seeds / layers analysed", "1 seed / 1 layer (8)", "", "outputs/models (one ffff_32 run)")
    t1 = P.TABLES / "t1_transfer_summary.csv"
    if t1.exists():
        s = pd.read_csv(t1)
        s = s[s.experiment.str.startswith("reg_to_")]
        for model, g in s.groupby("model"):
            top = g.loc[g.lenient_minus_strict.idxmax()]
            add(O4, f"{model}: lenient (homophone) minus strict top-1, regular -> each class",
                f"max +{top.lenient_minus_strict:.3f} ({top.experiment}); mean +{g.lenient_minus_strict.mean():.3f}",
                f"{len(g)} experiments", rel(t1))
            add(O4, f"{model}: median rank vs mean rank, regular -> vowel change",
                f"median {g.set_index('experiment').loc['reg_to_vowel_change', 'median_rank']:.0f}, "
                f"mean {g.set_index('experiment').loc['reg_to_vowel_change', 'mean_rank']:,.0f}", "", rel(t1))

    df = pd.DataFrame(rows)
    P.save_table(df, "obstacles")
    print(C.md_table(df, index=False))
    P.log_run("python scripts/past_tense/05_obstacle_numbers.py", start, datetime.now(), f"obstacles.csv ({len(df)} numbers)")


if __name__ == "__main__":
    main()
