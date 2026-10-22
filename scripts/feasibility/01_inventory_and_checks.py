"""
Feasibility step 1 (inventory and sanity checks) and step 2 (ɪd allomorph label check).
Reads existing pipeline outputs only; writes

    outputs/feasibility/checks/inventory.md
    outputs/feasibility/checks/id_label_check.md

    python scripts/feasibility/01_inventory_and_checks.py
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

sys.path.insert(0, str(C.REPO_ROOT))


def path_size(p: Path) -> int:
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return p.stat().st_size


# --------------------------------------------------------------------------- step 1

def check_paths() -> tuple[str, bool]:
    ckpts = C.checkpoints()
    paths = {
        "Probe analogy results": C.RESULTS["Word"],
        "Raw wav2vec analogy results": C.RESULTS["Wav2Vec"],
        "Probe run.ipynb (executed)": C.RESULTS["Word"].with_name("run.ipynb"),
        "Raw wav2vec run.ipynb (executed)": C.RESULTS["Wav2Vec"].with_name("run.ipynb"),
        **{f"INPUTS/{f}": C.INPUTS / f for f in ["inflection_results.parquet", "inflection_instances.parquet",
                                                  "all_cross_instances.parquet", "most_common_allomorphs.csv",
                                                  "false_friends.csv", "state_space_spec.h5"]},
        f"Best checkpoint ({C.BEST_CHECKPOINT})": C.MODEL_DIR / C.BEST_CHECKPOINT,
        f"Highest checkpoint ({ckpts[-1].name if ckpts else '?'})": ckpts[-1] if ckpts else C.MODEL_DIR / "checkpoint-?",
        "Probe embeddings (.npy)": C.EMBEDDINGS,
        "Raw layer-8 states (HDD)": C.HIDDEN_STATES,
        "Raw layer-8 states (SSD copy)": C.HIDDEN_STATES_SSD,
        "POS counts": C.POS_COUNTS,
        "Paper PDF": C.PAPER_PDF,
        "Project_Feasibility.md": C.FEASIBILITY_DOC,
    }
    rows = []
    for name, p in paths.items():
        exists = p.exists()
        rows.append({"what": name, "exists": "yes" if exists else "**MISSING**",
                     "size": C.human_size(path_size(p)) if exists else "", "path": f"`{p.relative_to(C.ROOT) if p.is_relative_to(C.ROOT) else p}`"})
    ok = all(p.exists() for p in C.RESULTS.values())

    state = json.loads((ckpts[-1] / "trainer_state.json").read_text())
    best = Path(state["best_model_checkpoint"]).name
    evals = [e for e in state["log_history"] if "eval_loss" in e]
    best_eval = next(e for e in evals if e["step"] == int(best.rsplit("-", 1)[1]))
    max_map = max(evals, key=lambda e: e["eval_mAP"])
    out = [C.md_table(pd.DataFrame(rows), index=False), "",
           f"- Checkpoints: {', '.join(p.name for p in ckpts)}.",
           f"- `{ckpts[-1].name}/trainer_state.json`: {len(state['log_history'])} log entries, {len(evals)} evaluations; "
           f"`best_model_checkpoint` = **{best}** (eval loss {best_eval['eval_loss']:.4f}, eval_mAP {best_eval['eval_mAP']:.3f}); "
           f"highest eval_mAP {max_map['eval_mAP']:.3f} at step {max_map['step']}.",
           f"- `common.BEST_CHECKPOINT` = {C.BEST_CHECKPOINT}: " + ("matches." if best == C.BEST_CHECKPOINT else "**differs, update common.py**.")]
    return "\n".join(out), ok


def summarize_results(results: dict[str, pd.DataFrame]) -> str:
    out = []
    for model, df in results.items():
        vbd_expts = sorted(e for e in df.experiment.unique() if "VBD" in e)
        sel = df[df.experiment.isin(["basic", "regular"]) & df.group.fillna("").str.contains("VBD")
                 | df.experiment.isin(vbd_expts)]
        counts = sel.groupby(["experiment", "group"], dropna=False).agg(
            trials=("gt_label_rank", "size"), source_items=("base_from", "nunique"), target_items=("base_to", "nunique"),
            mean_rank=("gt_label_rank", "mean"), median_rank=("gt_label_rank", "median"))
        out += [f"### {model}", "",
                f"- {len(df):,} trials, {df.experiment.nunique()} experiments.",
                f"- {len(vbd_expts)} experiments with `VBD` in the name: "
                + ", ".join(f"`{e}`" for e in vbd_expts) + ".",
                "- `basic` and `regular` also contain VBD groups (rows below).", "",
                C.md_table(counts, floatfmt="{:.1f}"), ""]
    return "\n".join(out)


def summarize_vbd_pairs(spec) -> tuple[str, pd.DataFrame]:
    ir = pd.read_parquet(C.INPUTS / "inflection_results.parquet")
    counts = spec.label_counts
    vbd = ir.loc["VBD"].reset_index(drop=True)
    vbd["base_tokens"] = vbd.base.map(counts)
    vbd["past_tokens"] = vbd.inflected.map(counts)
    vbd["min_tokens"] = vbd[["base_tokens", "past_tokens"]].min(axis=1)
    vbd["class"] = vbd.is_regular.map({True: "regular", False: "irregular"})
    vbd["n_pasts_for_base"] = vbd.groupby("base").inflected.transform("size")

    totals = vbd.groupby("class").agg(
        pairs=("base", "size"), distinct_bases=("base", "nunique"),
        base_tokens_min=("base_tokens", "min"), base_tokens_median=("base_tokens", "median"), base_tokens_max=("base_tokens", "max"),
        past_tokens_min=("past_tokens", "min"), past_tokens_median=("past_tokens", "median"), past_tokens_max=("past_tokens", "max"),
        pairs_ge20_both=("min_tokens", lambda x: int((x >= 20).sum())))

    irr = vbd[~vbd.is_regular]
    is_modal = irr.inflected.isin(C.MODAL_PASTS)
    is_supp = [(b, i) in C.SUPPLETIVE_PAIRS for b, i in zip(irr.base, irr.inflected)]
    is_homograph = [(b, i) in C.HOMOGRAPH_PASTS for b, i in zip(irr.base, irr.inflected)]
    multi = irr.n_pasts_for_base > 1
    clean = irr[~(is_modal | pd.Series(is_supp, index=irr.index) | pd.Series(is_homograph, index=irr.index) | multi)]
    excluded_reg = vbd[vbd.is_regular & vbd.base.isin(C.EXCLUDE_VBD_BASE)]

    out = ["All counts are over the analogy token universe (≤ 100 tokens per word type).",
           "`prepare_inputs.ipynb` (cell 11) only keeps labels with **more than 5 tokens**, so every pair has ≥ 6 tokens of both forms by construction; "
           "the informative numbers are the token distribution and the ≥ 20 count.", "",
           C.md_table(totals, floatfmt="{:.0f}"), "",
           f"- Irregular pairs: **{len(irr)}** over {irr.base.nunique()} bases. After removing modals ({int(is_modal.sum())}: "
           f"{', '.join(sorted(irr[is_modal].inflected))}), suppletives ({sum(is_supp)}), known homographs ({sum(is_homograph)}) "
           f"and bases with more than one past ({int(multi.sum())} pairs: "
           f"{', '.join(sorted(set(irr[multi].base)))}): **{len(clean)}** pairs, {int((clean.min_tokens >= 20).sum())} of them with ≥ 20 tokens of both forms.",
           f"- Regular pairs whose base is on the authors' `exclude_vbd_base` list: {len(excluded_reg)} "
           f"({', '.join(sorted(excluded_reg.base))}).", "",
           "**40 most frequent irregular pairs** (by the smaller of the two token counts, capped at 100):", "",
           C.md_table(irr.sort_values(["min_tokens", "past_tokens"], ascending=False).head(40)
                      [["base", "inflected", "base_tokens", "past_tokens", "n_pasts_for_base"]], index=False)]
    return "\n".join(out), vbd


# --------------------------------------------------------------------------- step 2

def id_label_check(spec, forms: pd.Series, results: dict[str, pd.DataFrame]) -> str:
    mca = pd.read_csv(C.INPUTS / "most_common_allomorphs.csv")
    vc = {infl: mca[mca.inflection == infl].most_common_allomorph.value_counts() for infl in ["VBD", "NNS", "VBZ"]}

    # token level: post_divergence vs. the final phone of the base's most common pronunciation
    ii = pd.read_parquet(C.INPUTS / "inflection_instances.parquet",
                         columns=["inflection", "base", "inflected", "post_divergence", "is_regular"])
    reg = ii[(ii.inflection == "VBD") & ii.is_regular].copy()
    base_pron = C.most_common_pron(forms[forms.index.get_level_values("label").isin(set(reg.base))])
    reg["base_final"] = reg.base.map(base_pron).str.split().str[-1]
    reg["expected"] = reg.base.map(base_pron).str.split().map(C.guess_past_allomorph)
    top_pd = ["T", "D", "IH D", "AH D"]
    reg["post_divergence"] = reg.post_divergence.where(reg.post_divergence.isin(top_pd), "other")
    xtab = pd.crosstab(reg.expected.rename("expected (from base)"), reg.post_divergence.rename("observed post_divergence"))
    xtab = xtab[[c for c in top_pd + ["other"] if c in xtab.columns]]
    final_td = reg[reg.base_final.isin(["T", "D"])]
    final_xtab = pd.crosstab(final_td.base_final.rename("base final phone"), final_td.post_divergence)
    per_pair = reg[reg.expected == "IH D"].groupby(["base", "inflected"]).post_divergence.agg(
        lambda x: "both" if {"IH D", "AH D"} <= set(x) else ("IH D only" if "IH D" in set(x) else ("AH D only" if "AH D" in set(x) else "neither")))

    # forced-choice stimuli as built, and how many an AH D spelling would add (approximate: stems
    # attested with both endings, no frequency filter; prepare_inputs also requires ≥ 2 tokens each)
    aci = pd.read_parquet(C.INPUTS / "all_cross_instances.parquet", columns=["inflection", "base", "inflected"])
    fc = aci[aci.inflection.str.startswith("FC-")].drop_duplicates()
    fc_counts = fc.groupby("inflection").agg(items=("inflected", "size"), bases=("base", "nunique"))
    attested = set(forms.unique())
    def n_triples(suffix1, suffix2):
        stems = {f[: -len(suffix1)].strip() for f in attested if f.endswith(" " + suffix1)}
        return sum(1 for s in stems if s in attested and f"{s} {suffix2}" in attested)
    approx = pd.DataFrame([{"pair": f"{a} / {b}", "stems with base, base+{a}, base+{b} attested": n_triples(a, b)}
                           for a, b in [("D", "T"), ("D", "IH D"), ("T", "IH D"), ("D", "AH D"), ("T", "AH D")]])

    ff = pd.read_csv(C.INPUTS / "false_friends.csv")
    ff_vc = ff[ff.inflection == "VBD"].post_divergence.value_counts()

    # Figure 6 cells with and without mapping AH D -> IH D (authors' filters otherwise)
    df = results["Word"]
    grid = df[(df.experiment == "regular") & (df.inflection_from == "VBD")]
    m = mca.rename(columns={"most_common_allomorph": "allomorph"})[["inflection", "base", "allomorph"]]
    for side in ("from", "to"):
        grid = grid.merge(m.rename(columns={"inflection": f"inflection_{side}", "base": f"base_{side}", "allomorph": f"allomorph_{side}"}),
                          on=[f"inflection_{side}", f"base_{side}"], how="left")
    grid = grid[~grid.base_from.isin(C.EXCLUDE_VBD_BASE) & ~grid.base_to.isin(C.EXCLUDE_VBD_BASE)
                & grid.group.str.contains("True")]
    grid_rows = []
    for mapped in (False, True):
        g = grid.copy()
        if mapped:
            g[["allomorph_from", "allomorph_to"]] = g[["allomorph_from", "allomorph_to"]].replace(C.ALLOMORPH_CANON)
        keep = g.allomorph_from.value_counts().head(3).index
        g = g[g.allomorph_from.isin(keep) & g.allomorph_to.isin(keep)]
        n = g.groupby(["allomorph_from", "allomorph_to"]).size()
        grid_rows.append(n.rename("AH D mapped" if mapped else "authors (raw labels)"))
    grid_n = pd.concat(grid_rows, axis=1).fillna(0).astype(int)

    n_ih, n_ah = vc["VBD"].get("IH D", 0), vc["VBD"].get("AH D", 0)
    tok_ih, tok_ah = int(reg.post_divergence.eq("IH D").sum()), int(reg.post_divergence.eq("AH D").sum())
    ah_common = n_ah / (n_ih + n_ah) > 0.1
    decision = (
        f"**Decision: `AH D` is common.** {n_ah} of {n_ih + n_ah} ɪd verbs ({n_ah / (n_ih + n_ah):.0%}) have `AH D` as their most "
        f"common ending, and {tok_ah:,} of {tok_ih + tok_ah:,} ɪd tokens of regular past forms ({tok_ah / (tok_ih + tok_ah):.0%}) are spelled `AH D`. "
        "Per the plan: the pipeline is **not** changed before Friday; all summaries map `IH D` and `AH D` to ɪd "
        "(and `IH Z` / `AH Z` to ɪz) via `common.ALLOMORPH_CANON`; this is a known issue, and fixing `guess_past_allomorph` "
        "plus regenerating the stimuli is a P2 item (needs an analogy rerun)."
        if ah_common else
        f"**Decision: `IH D` dominates** ({n_ih} vs {n_ah} verbs). The label is fine.")

    out = ["# ɪd allomorph label check (step 2)", "", decision, "",
           "## 1. Most common allomorph per base (`most_common_allomorphs.csv`)", "",
           C.md_table(pd.DataFrame({k: v.head(8) for k, v in vc.items()}).fillna(0).astype(int)
                      .pipe(lambda t: t.loc[t.sum(axis=1).sort_values(ascending=False).index]).rename_axis("allomorph"),
                      floatfmt="{:.0f}"), "",
           "## 2. Token level: regular VBD, observed vs. expected ending", "",
           f"{len(reg):,} past-form tokens of {reg.groupby(['base', 'inflected']).ngroups} regular pairs. "
           "Expected = `guess_past_allomorph()` of the base's most common pronunciation (from the state-space cuts).", "",
           C.md_table(xtab, floatfmt="{:.0f}"), "",
           "Bases ending in T or D only:", "", C.md_table(final_xtab, floatfmt="{:.0f}"), "",
           "Per ɪd pair (base ends in T/D), which spellings its tokens use:", "",
           C.md_table(per_pair.value_counts().rename("pairs").rename_axis("tokens spelled").to_frame()), "",
           "The aligner's dictionary lists both pronunciations for many -ed words, so the split is per token, not per verb.", "",
           "## 3. Forced-choice stimuli (`all_cross_instances.parquet`)", "",
           C.md_table(fc_counts.rename_axis("FC set")), "",
           "Approximate count of candidate stems (before the frequency filter), to show what the `IH D`-only search misses:", "",
           C.md_table(approx, index=False), "",
           "Including `AH D` would roughly double the ɪd forced-choice candidates, but they would stay tiny either way.", "",
           "## 4. VBD false friends (`false_friends.csv`)", "",
           C.md_table(ff_vc.rename("items").rename_axis("post_divergence").to_frame()), "",
           "False friends were generated only for the three most common VBD endings (D, IH D, T), so there are no `AH D` false friends.", "",
           "## Affected downstream sets", "",
           f"- **Allomorph-specific experiments** (`run.ipynb` cells 21–24) select tokens with `post_divergence == 'IH D'`. "
           f"Their ɪd sources therefore use only the `IH D` tokens: {(per_pair == 'AH D only').sum()} ɪd pairs whose tokens are all "
           "`AH D` are missing entirely, and pairs with both spellings contribute only part of their tokens.",
           f"- **Forced choice:** `FC-D_IH D` has {fc_counts.loc['FC-D_IH D', 'items'] if 'FC-D_IH D' in fc_counts.index else 0} items and "
           f"`FC-T_IH D` {fc_counts.loc['FC-T_IH D', 'items'] if 'FC-T_IH D' in fc_counts.index else 0}, vs "
           f"{fc_counts.loc['FC-D_T', 'items'] if 'FC-D_T' in fc_counts.index else 0} for `FC-D_T`. Section 7 shows D/T only.",
           f"- **False friends:** VBD-FF-IH D has {ff_vc.get('IH D', 0)} items; `false_friend_results()` skips experiments with < 10 bases, "
           "so VBD-FF means [t] and [d] only.",
           "- **`false_friends.csv` strong flags:** computed with `guess_past_allomorph` → `IH D`, consistent with the `IH D` items present; "
           "the problem is the missing `AH D` items, not wrong flags.",
           "- **`regular` experiment (Figure 6):** fixable post hoc, since the allomorph is merged in from `most_common_allomorphs.csv`. "
           "Probe trials per cell, `('VBD', True)` group, authors' base exclusions:", "",
           C.md_table(grid_n, floatfmt="{:.0f}"), "",
           "- **-s side:** NNS has `AH Z` for "
           f"{vc['NNS'].get('AH Z', 0)} bases and VBZ for {vc['VBZ'].get('AH Z', 0)}. The unambiguous Fig. 3/4 experiments use only Z, S and IH Z, "
           "so the reproduction figures are unaffected; `AH Z` bases are just absent from them."]
    return "\n".join(out)


def main(args):
    C.CHECKS.mkdir(parents=True, exist_ok=True)

    paths_md, ok = check_paths()
    if not ok:
        (C.CHECKS / "inventory.md").write_text("# Inventory\n\n" + paths_md, encoding="utf-8")
        sys.exit("experiment results missing; see outputs/feasibility/checks/inventory.md")

    results = {m: C.load_results(m) for m in C.MODELS}
    print("loading state-space spec ...")
    spec = C.load_spec()
    n_instances = int(spec.label_counts.sum())
    print("building phone strings ...")
    forms = C.cut_phonemic_forms(spec)
    pairs_md, vbd = summarize_vbd_pairs(spec)

    vbd_trials = {m: df[df.group.fillna("").str.contains("'VBD'") & df.experiment.isin(["basic", "regular"])]
                  .groupby(["experiment", "group"]).size().to_dict() for m, df in results.items()}
    group_names = {"('VBD',)": "all", "('VBD', np.True_)": "regular→regular", "('VBD', np.False_)": "irregular→irregular"}
    headline = "\n".join(
        [f"- **VBD pairs:** {int(vbd.is_regular.sum())} regular, {int((~vbd.is_regular).sum())} irregular "
         f"({vbd.base.nunique()} distinct bases)."]
        + [f"- **VBD trials, {m}:** " + "; ".join(f"`{e}` {group_names.get(g, g)}: {n}" for (e, g), n in t.items())
           for m, t in vbd_trials.items()])

    inventory = "\n".join([
        "# Inventory and sanity checks (step 1)", "",
        "Written by `scripts/feasibility/01_inventory_and_checks.py`.", "",
        "## Headline", "", headline, "",
        "## 1. Paths", "", paths_md, "",
        "## 2. Analogy results", "", summarize_results(results),
        "## 3. Analogy token universe", "",
        f"`INPUTS/state_space_spec.h5`: **{n_instances:,}** word tokens over **{len(spec.labels):,}** labels "
        f"(expected 334,008 / 32,046: {'OK' if (n_instances, len(spec.labels)) == (334008, 32046) else '**MISMATCH**'}); "
        f"`total_num_frames` = {spec.total_num_frames:,}.", "",
        "## 4. VBD pairs (`inflection_results.parquet`)", "", pairs_md, ""])
    (C.CHECKS / "inventory.md").write_text(inventory, encoding="utf-8")
    print("wrote", C.CHECKS / "inventory.md")

    (C.CHECKS / "id_label_check.md").write_text(id_label_check(spec, forms, results), encoding="utf-8")
    print("wrote", C.CHECKS / "id_label_check.md")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    main(p.parse_args())
