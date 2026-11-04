"""
Past-tense step 1: the automatic verb classes for the midpoint run. Writes outputs/past_tense/tables/:

    vbd_classes.csv   one row per VBD pair: past_class (regular / semi_regular / devoicing / vowel_change /
                      suppletive / exclude), allomorph, adds_stop, past_ends_td, phone edit distance from
                      the regularized form (base + expected [t]/[d]/[ɪd]) to the real past, token counts
    class_counts.csv  pairs, base verbs and tokens per class, before and after exclusions ("Table 1")

The classes come from common.auto_class: automatic, not hand-checked. `exclude` = auto_class "check"
plus the authors' VBD exclusions (common.EXCLUDE_VBD_BASE).

    python scripts/past_tense/01_make_classes.py
"""

from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt_common as P  # noqa: E402

C = P.C

# handoff §1.4, before any exclusions
EXPECTED_AUTO_COUNTS = {"regular_t": 160, "regular_d": 415, "regular_id": 223, "semi_regular_candidate": 21,
                        "devoicing": 5, "irregular_unclassified": 72, "suppletive": 3, "check": 50}


def make_classes() -> pd.DataFrame:
    p = pd.read_parquet(C.DATA / "pairs_vbd.parquet")
    p[["auto_class", "auto_reason"]] = p.apply(C.auto_class, axis=1, result_type="expand")

    # must reproduce the auto_class column of the annotation spreadsheet exactly
    x = pd.read_excel(C.THEORY / "vbd_pairs_for_annotation.xlsx").rename(columns={"past": "inflected"})
    m = p.merge(x[["base", "inflected", "auto_class", "base_corpus_count", "past_corpus_count"]],
                on=["base", "inflected"], how="outer", suffixes=("", "_xlsx"), validate="1:1", indicator=True)
    assert (m._merge == "both").all(), m._merge.value_counts()
    assert (m.auto_class == m.auto_class_xlsx).all(), m.loc[m.auto_class != m.auto_class_xlsx]
    counts = p.auto_class.value_counts().to_dict()
    assert counts == EXPECTED_AUTO_COUNTS, counts
    p = m.drop(columns=["auto_class_xlsx", "_merge"])

    p["past_class"] = p.auto_class.map(P.AUTO_TO_PAST)
    assert p.past_class.notna().all()
    p["exclude_reason"] = ""
    p.loc[p.auto_class == "check", "exclude_reason"] = "check: " + p.auto_reason
    p.loc[p.authors_excluded & (p.past_class != "exclude"), "exclude_reason"] = "authors' exclusion list"
    p.loc[p.authors_excluded, "past_class"] = "exclude"

    p["allomorph"] = p.auto_class.map(P.AUTO_ALLOMORPH).fillna("none")
    p.loc[p.past_class != "regular", "allomorph"] = "none"
    p["adds_stop"] = p.past_class.map(P.ADDS_STOP).astype("boolean")
    p["past_ends_td"] = p.past_pron.str.split().str[-1].isin(["T", "D"])
    p["regularized_pron"] = p.base_pron + " " + p.expected_allomorph
    p["edit_dist"] = [P.edit_distance(r.split(), q.split()) for r, q in zip(p.regularized_pron, p.past_pron)]

    cols = ["base", "inflected", "past_class", "allomorph", "adds_stop", "past_ends_td", "base_tokens", "past_tokens",
            "base_corpus_count", "past_corpus_count", "auto_class", "auto_reason", "authors_excluded", "exclude_reason",
            "base_pron", "past_pron", "regularized_pron", "edit_dist", "past_homophones"]
    order = {c: i for i, c in enumerate(P.PAST_CLASSES + ["exclude"])}
    return p[cols].sort_values(["past_class", "past_tokens"], key=lambda s: s.map(order) if s.name == "past_class" else -s) \
        .reset_index(drop=True)


def class_counts(v: pd.DataFrame) -> pd.DataFrame:
    """'Table 1': per class, pairs before the authors' exclusions, and pairs / base verbs / tokens used."""
    before = v.auto_class.map(P.AUTO_TO_PAST)
    used = v[v.past_class != "exclude"]
    rows = []

    def row(label, before_mask, used_rows, examples=""):
        rows.append({"class": label, "examples": examples, "pairs_auto_class": int(before_mask.sum()),
                     "pairs_used": len(used_rows), "base_verbs_used": used_rows.base.nunique(),
                     "base_tokens_used": int(used_rows.base_tokens.sum()), "past_tokens_used": int(used_rows.past_tokens.sum()),
                     "past_corpus_count_used": int(used_rows.past_corpus_count.sum())})

    for c in P.PAST_CLASSES:
        row(P.CLASS_LABELS[c], before == c, used[used.past_class == c], P.CLASS_EXAMPLES[c])
        if c == "regular":
            for a in ["T", "D", "IH D"]:
                auto = {v_: k for k, v_ in P.AUTO_ALLOMORPH.items()}[a]
                row(f"  regular {P.ALLOMORPH_LABELS[a]}", v.auto_class == auto, used[used.allomorph == a])
    excl = v[v.past_class == "exclude"]
    rows.append({"class": "excluded: auto_class 'check'", "examples": "could, lay, rose, said, meant, dealt, bathed",
                 "pairs_auto_class": int((v.auto_class == "check").sum()), "pairs_used": 0,
                 "base_verbs_used": 0, "base_tokens_used": 0, "past_tokens_used": 0, "past_corpus_count_used": 0})
    rows.append({"class": "excluded: authors' list (other classes)", "examples": ", ".join(excl.loc[excl.auto_class != "check", "base"]),
                 "pairs_auto_class": int((excl.auto_class != "check").sum()), "pairs_used": 0,
                 "base_verbs_used": 0, "base_tokens_used": 0, "past_tokens_used": 0, "past_corpus_count_used": 0})
    row("total", pd.Series(True, index=v.index), used)
    t = pd.DataFrame(rows)
    t["note"] = ""
    t.loc[0, "note"] = P.CAVEAT + " Tokens are those in the analogy set (at most 100 per word); corpus counts are all of train-clean-100."
    return t


def main():
    start = datetime.now()
    v = make_classes()
    P.save_table(v, "vbd_classes")
    t = class_counts(v)
    P.save_table(t, "class_counts")
    print(C.md_table(t.drop(columns=["note"]), index=False))
    print("\npairs used per class:", v[v.past_class != "exclude"].past_class.value_counts().to_dict())
    print("past ends in T/D, vowel_change:", v[v.past_class == "vowel_change"].past_ends_td.value_counts().to_dict())
    P.log_run("python scripts/past_tense/01_make_classes.py", start, datetime.now(),
              f"vbd_classes.csv ({len(v)} pairs) and class_counts.csv; xlsx auto_class reproduced 949/949")


if __name__ == "__main__":
    main()
