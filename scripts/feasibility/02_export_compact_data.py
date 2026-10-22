"""
Feasibility step 3: export the compact data files the feasibility notebook reads, so that it never
touches the raw pipeline outputs. Writes outputs/feasibility/data/:

    results_word.parquet, results_wav2vec.parquet   all analogy trials, parsed, with an `excluded` flag
    vbd_trials.parquet                              `basic` / `regular` VBD trials with allomorphs and classes
    fc_trials.parquet                               forced-choice trials, scored (summary notebook cells 118-133)
    pairs_vbd.parquet                               one row per VBD pair: token counts, pronunciations, homophones
    trainer_log.csv                                 probe training log_history
    manifest.json                                   sources, row counts, sizes, settings

    python scripts/feasibility/02_export_compact_data.py
"""

from argparse import ArgumentParser
import ast
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

sys.path.insert(0, str(C.REPO_ROOT))

FC_INFLECTIONS = ["FC-Z_S", "FC-D_T", "FC-D_IH D", "FC-T_IH D"]
# summary notebook cell 118
STRONG_GUESSERS = {"Z_S": C.guess_nns_vbz_allomorph, "D_T": C.guess_past_allomorph,
                   "T_IH D": C.guess_past_allomorph, "D_IH D": C.guess_past_allomorph}
WEAK_SPACES = {"Z_S": {"Z", "S", "IH Z"}, "D_T": {"D", "T", "IH D"},
               "T_IH D": {"D", "T", "IH D"}, "D_IH D": {"D", "T", "IH D"}}


def canon_pron(pron: str) -> str:
    """Spell a final unstressed AH D / AH Z as IH D / IH Z (step 2)."""
    for ah, ih in C.ALLOMORPH_CANON.items():
        if pron.endswith(" " + ah):
            return pron[: -len(ah)] + ih
    return pron


# --------------------------------------------------------------------------- 1. all trials

def export_results(model: str) -> pd.DataFrame:
    df = C.load_results(model).rename(columns={"Unnamed: 1": "trial_idx"})
    groups = df.group.map(C.parse_group)
    df["group0"] = groups.map(lambda g: g[0] if g else None)
    df["group1"] = groups.map(lambda g: g[1] if g and len(g) > 1 else None)
    df["inflected_from"] = df.from_equiv_label.map(lambda x: ast.literal_eval(x)[1])
    df["inflected_to"] = df.to_equiv_label.map(lambda x: ast.literal_eval(x)[1])
    excl = set(C.VBZ_EXCLUSIONS)  # summary notebook cells 17-18; flagged, not dropped
    df["excluded"] = [(f, bf) in excl or (t, bt) in excl
                      for f, bf, t, bt in zip(df.inflection_from, df.base_from, df.inflection_to, df.base_to)]
    return df


# --------------------------------------------------------------------------- 2. VBD trials

def export_vbd_trials(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """summary notebook cells 69-70, without the frequency merge; filters are flags, not drops."""
    mca = pd.read_csv(C.INPUTS / "most_common_allomorphs.csv")[["inflection", "base", "most_common_allomorph"]]
    parts = []
    for model, df in results.items():
        v = df[df.experiment.isin(["basic", "regular"]) & (df.group0 == "VBD")].copy()
        for side in ("from", "to"):
            v = v.merge(mca.rename(columns={"inflection": f"inflection_{side}", "base": f"base_{side}",
                                            "most_common_allomorph": f"allomorph_{side}"}),
                        on=[f"inflection_{side}", f"base_{side}"], how="left")
            v[f"allomorph_{side}_canon"] = v[f"allomorph_{side}"].replace(C.ALLOMORPH_CANON)
            v[f"class_{side}"] = v.group1.map({True: "regular", False: "irregular"})
        v["authors_excluded"] = v.base_from.isin(C.EXCLUDE_VBD_BASE) | v.base_to.isin(C.EXCLUDE_VBD_BASE)
        parts.append(v)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- 3. forced choice

def export_fc_trials(results: dict[str, pd.DataFrame], forms: pd.Series, canonical: bool) -> pd.DataFrame:
    # pronunciation -> labels with a token of that pronunciation (cut_phonemic_forms lookups in cell 131)
    pron2labels = defaultdict(set)
    for (label, _), pron in forms.items():
        pron2labels[pron].add(label)
        if canonical and canon_pron(pron) != pron:
            pron2labels[canon_pron(pron)].add(label)

    # cell 122: one metadata row per FC (inflection, base, inflected)
    aci = pd.read_parquet(C.INPUTS / "all_cross_instances.parquet",
                          columns=["inflection", "base", "base_phones", "inflected", "inflected_phones",
                                   "inflected2", "inflected2_phones", "post_divergence"],
                          filters=[("inflection", "in", FC_INFLECTIONS)])
    meta = aci.groupby(["inflection", "base", "inflected"]).head(1).rename(columns={
        "base": "base_to", "base_phones": "base_to_phones", "inflected": "inflected_to",
        "inflected_phones": "inflected_to_phones", "inflected2": "inflected2_to",
        "inflected2_phones": "inflected2_to_phones", "post_divergence": "post_divergence_to"})
    meta["fc_pair"] = meta.inflection.str.extract(r"FC-([\w\s_]+)")[0]
    meta = meta.drop(columns=["inflection"])

    # cell 128, from the token-level allomorphs: most common post_divergence of each source pair
    # (all_cross_instances repeats each inflected token once per base token, which doesn't change the mode)
    ii = pd.read_parquet(C.INPUTS / "inflection_instances.parquet", columns=["inflection", "base", "inflected", "post_divergence"])
    from_meta = ii.groupby(["inflection", "base", "inflected"]).post_divergence \
        .agg(lambda xs: xs.value_counts().index[0]).rename("post_divergence_from").reset_index() \
        .rename(columns={"inflection": "inflection_from", "base": "base_from", "inflected": "inflected_from"})

    parts = []
    for model, df in results.items():
        fc = df[df.inflection_to.str.startswith("FC-")].copy()  # cell 121
        n0 = len(fc)
        fc = fc[~fc.base_to.isin(C.FC_EXCLUDE)]
        fc["fc_pair"] = fc.inflection_to.str.extract(r"FC-([\w\s_]+)")[0]
        fc = fc[~fc.inflected_to.isin(C.FC_EXCLUDE_INFLECTED)]
        fc["source_allomorph"] = fc.experiment.str.extract(r"-from_\w+-(.+)$")[0]
        fc = fc.merge(meta, on=["fc_pair", "base_to", "inflected_to"])  # cell 123

        # cell 124
        fc["strong_expected"] = [STRONG_GUESSERS[p](b.split(" ")) for p, b in zip(fc.fc_pair, fc.base_to_phones)]
        fc["inflected_to_strong"] = [i[len(b) + 1:] == s for i, b, s in zip(fc.inflected_to_phones, fc.base_to_phones, fc.strong_expected)]
        fc["inflected2_to_strong"] = [i[len(b) + 1:] == s for i, b, s in zip(fc.inflected2_to_phones, fc.base_to_phones, fc.strong_expected)]

        fc = fc.merge(from_meta, on=["inflection_from", "base_from", "inflected_from"])  # cell 128
        # cell 129: keep the 4 most frequent source suffixes per FC pair
        keep = fc.groupby("fc_pair").post_divergence_from.apply(lambda xs: set(xs.value_counts().head(4).index))
        fc = fc[[pd_ in keep[p] for p, pd_ in zip(fc.fc_pair, fc.post_divergence_from)]]

        # cell 131: strong = labels pronounced base + expected suffix; weak = base + any other suffix in the space
        def strong_items(pair, base_phones):
            return " ".join(sorted(pron2labels.get(f"{base_phones} {STRONG_GUESSERS[pair](base_phones.split())}", ())))

        def weak_items(pair, base_phones):
            expected = STRONG_GUESSERS[pair](base_phones.split())
            labels = set().union(*(pron2labels.get(f"{base_phones} {s}", set()) for s in WEAK_SPACES[pair] - {expected}))
            return " ".join(sorted(labels))

        key = fc[["fc_pair", "base_to_phones"]].drop_duplicates()
        key["strong_item_to"] = [strong_items(p, b) for p, b in zip(key.fc_pair, key.base_to_phones)]
        key["weak_item_to"] = [weak_items(p, b) for p, b in zip(key.fc_pair, key.base_to_phones)]
        fc = fc.merge(key, on=["fc_pair", "base_to_phones"])

        # cell 132: drop items whose strong and weak orthographic forms overlap
        fc = fc[[set(s.split()).isdisjoint(w.split()) for s, w in zip(fc.strong_item_to, fc.weak_item_to)]]

        # cell 133 (word-boundary match, as the authors do, with the label regex-escaped)
        def chose(pred, items):
            return re.search(rf"\b{re.escape(pred)}\b", items) is not None
        fc["chose_strong"] = [chose(p, s) for p, s in zip(fc.predicted_label, fc.strong_item_to)]
        fc["chose_weak"] = [chose(p, w) for p, w in zip(fc.predicted_label, fc.weak_item_to)]
        fc["chose_strong_or_weak"] = fc.chose_strong | fc.chose_weak
        print(f"  {model}: {n0} FC trials -> {len(fc)} after the authors' filters")
        parts.append(fc)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- 4. VBD pairs

def export_pairs_vbd(spec, forms: pd.Series) -> pd.DataFrame:
    ir = pd.read_parquet(C.INPUTS / "inflection_results.parquet")
    vbd = ir.loc["VBD"].reset_index(drop=True)[["base", "inflected", "is_regular", "base_idx", "inflected_idx"]]
    counts = spec.label_counts
    vbd["base_tokens"] = vbd.base.map(counts)
    vbd["past_tokens"] = vbd.inflected.map(counts)

    # most common pronunciation of every label, and the labels sharing it
    vc = forms.groupby(level="label").value_counts()
    pron = vc.groupby(level=0).idxmax().map(lambda x: x[1])
    pron_labels = pron.groupby(pron).apply(lambda xs: set(xs.index))
    vbd["base_pron"] = vbd.base.map(pron)
    vbd["past_pron"] = vbd.inflected.map(pron)
    vbd["past_homophones"] = [" ".join(sorted(pron_labels.get(p, set()) - {w})) for w, p in zip(vbd.inflected, vbd.past_pron)]
    vbd["base_homophones"] = [" ".join(sorted(pron_labels.get(p, set()) - {w})) for w, p in zip(vbd.base, vbd.base_pron)]

    ii = pd.read_parquet(C.INPUTS / "inflection_instances.parquet", columns=["inflection", "base", "inflected", "post_divergence"])
    ii = ii[ii.inflection == "VBD"]
    pdv = ii.groupby(["base", "inflected"]).post_divergence.agg(lambda xs: xs.value_counts().index[0]).rename("post_divergence")
    share_ah = ii.assign(ah=ii.post_divergence == "AH D").groupby(["base", "inflected"]).ah.mean().rename("share_AH_D")
    vbd = vbd.merge(pdv, left_on=["base", "inflected"], right_index=True, how="left") \
             .merge(share_ah, left_on=["base", "inflected"], right_index=True, how="left")
    vbd["post_divergence_canon"] = vbd.post_divergence.replace(C.ALLOMORPH_CANON)
    vbd["expected_allomorph"] = vbd.base_pron.str.split().map(C.guess_past_allomorph)
    vbd["n_pasts_for_base"] = vbd.groupby("base").inflected.transform("size")
    vbd["authors_excluded"] = vbd.base.isin(C.EXCLUDE_VBD_BASE)
    return vbd.sort_values(["is_regular", "past_tokens"], ascending=[True, False]).reset_index(drop=True)


# --------------------------------------------------------------------------- 5. training log

def export_trainer_log() -> tuple[pd.DataFrame, Path]:
    path = C.checkpoints()[-1] / "trainer_state.json"
    log = pd.DataFrame(json.loads(path.read_text())["log_history"])
    cols = [c for c in ["step", "epoch", "loss", "eval_loss", "eval_mAP"] if c in log.columns]
    return log[cols], path


def main(args):
    C.DATA.mkdir(parents=True, exist_ok=True)
    sources, outputs = {}, {}

    def write(name, df, src):
        path = C.DATA / name
        df.to_csv(path, index=False) if name.endswith(".csv") else df.to_parquet(path, index=False)
        outputs[name] = {"rows": len(df), "columns": len(df.columns), "bytes": path.stat().st_size}
        sources[name] = [str(Path(s).relative_to(C.ROOT)) if Path(s).is_relative_to(C.ROOT) else str(s) for s in src]
        print(f"wrote {name}: {len(df):,} rows, {C.human_size(path.stat().st_size)}")

    print("1. results ...")
    results = {m: export_results(m) for m in C.MODELS}
    for m, df in results.items():
        write(f"results_{m.lower()}.parquet", df, [C.RESULTS[m]])

    print("2. VBD trials ...")
    write("vbd_trials.parquet", export_vbd_trials(results),
          [*C.RESULTS.values(), C.INPUTS / "most_common_allomorphs.csv"])

    print("loading state-space spec ...")
    spec = C.load_spec()
    forms = C.cut_phonemic_forms(spec)

    ir = pd.read_parquet(C.INPUTS / "inflection_results.parquet").reset_index()
    unamb = ir[ir.is_regular & (ir.base_ambig_NN_VB == False)]  # noqa: E712 (object column)
    summary = {"n_tokens": int(spec.label_counts.sum()), "n_labels": len(spec.labels),
               "total_num_frames": int(spec.total_num_frames),
               "unambiguous_pairs": unamb.groupby("inflection").size().to_dict(),
               "unambiguous_VBZ_after_exclusions": int(((unamb.inflection == "VBZ")
                                                        & ~unamb.base.isin([b for _, b in C.VBZ_EXCLUSIONS])).sum())}
    (C.DATA / "spec_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote spec_summary.json:", summary)

    print("3. forced choice ...")
    write("fc_trials.parquet", export_fc_trials(results, forms, args.canonical_id),
          [*C.RESULTS.values(), C.INPUTS / "all_cross_instances.parquet", C.INPUTS / "inflection_instances.parquet",
           C.INPUTS / "state_space_spec.h5"])

    print("4. VBD pairs ...")
    write("pairs_vbd.parquet", export_pairs_vbd(spec, forms),
          [C.INPUTS / f for f in ["inflection_results.parquet", "inflection_instances.parquet", "state_space_spec.h5"]])

    print("5. trainer log ...")
    log, log_src = export_trainer_log()
    write("trainer_log.csv", log, [log_src])

    total = sum(o["bytes"] for o in outputs.values())
    manifest = {
        "written_by": "scripts/feasibility/02_export_compact_data.py",
        "root": str(C.ROOT),
        "settings": {"fc_canonical_id": args.canonical_id,
                     "notes": ["`excluded` flags the authors' post-hoc VBZ exclusions (summary cells 17-18); nothing is dropped",
                               "vbd_trials: allomorph_* are the raw labels, allomorph_*_canon map AH D -> IH D",
                               "fc_trials: authors' filters of cells 120-132 applied; fc_canonical_id also counts AH D / AH Z "
                               "spellings as the IH D / IH Z strong/weak homophones"]},
        "files": {k: {**v, "sources": sources[k]} for k, v in outputs.items()},
        "total_bytes": total,
    }
    (C.DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"total {C.human_size(total)}; wrote manifest.json")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--no-canonical-id", dest="canonical_id", action="store_false",
                   help="score forced choice with the authors' exact IH D / IH Z spellings only")
    main(p.parse_args())
