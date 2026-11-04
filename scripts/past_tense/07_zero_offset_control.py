"""
Past-tense step 7 (added after the first results): zero-offset control. Is the target past simply the nearest
word to its base, so that any small step would land on it?

For every target pair of the midpoint run, the "guess" is the base itself (b - a = 0): each base token is compared
with all 334,008 word tokens exactly as in analogy.run_experiment_equiv_level (cosine distance, mean over the
base's tokens, the base's own tokens excluded), and the rank of the nearest token of the past form is recorded.
The regular step only adds something where it beats this control. Writes outputs/past_tense/tables/:

    t7_zero_offset_per_verb_<model>.csv   per target pair: zero-offset rank / top-1 / nearest word, regular-step top-1
    t7_zero_offset.csv                    per model and class: zero-offset vs. regular-step top-1 (verb-weighted,
                                          CIs over target pairs); rebuilt from the per-model files present

    python scripts/past_tense/07_zero_offset_control.py --model word
    python scripts/past_tense/07_zero_offset_control.py --model wav2vec   # reads the 52 GB hidden states (SSD copy)
"""

from argparse import ArgumentParser
from datetime import datetime
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt_common as P  # noqa: E402

C = P.C
MODEL_NAMES = {"word": "Word", "wav2vec": "Wav2Vec"}


def token_vectors(spec, model: str):
    """One vector per word token, as in run.ipynb cells 7-9 (mean over the token's frames)."""
    from src.analysis.state_space import prepare_state_trajectory, flatten_trajectory
    if model == "word":
        reps = np.load(C.EMBEDDINGS)
    else:
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        from src.datasets.speech_equivalence import SpeechHiddenStateDataset
        path = C.HIDDEN_STATES_SSD if C.HIDDEN_STATES_SSD.exists() else C.HIDDEN_STATES
        reps = SpeechHiddenStateDataset.from_hdf5(str(path)).states
    assert spec.is_compatible_with(reps)
    traj = prepare_state_trajectory(reps, spec, agg_fn_spec="mean", agg_fn_dimension=1)
    agg, src = flatten_trajectory(traj)
    del traj, reps
    return agg.astype(np.float32), src


def zero_offset(agg, src, spec, pairs: pd.DataFrame, device: str) -> pd.DataFrame:
    labels = spec.labels
    label_idx = {l: i for i, l in enumerate(labels)}
    A = torch.tensor(agg, device=device)
    A_norm = A / A.norm(dim=1, keepdim=True).clamp_min(1e-8)
    S = torch.tensor(src[:, 0], device=device)
    rows = []
    for base, past in zip(pairs.base, pairs.inflected):
        b, p = label_idx[base], label_idx[past]
        base_flat = torch.where(S == b)[0]
        pred = A[base_flat] / A[base_flat].norm(dim=1, keepdim=True)
        dists = (1 - pred @ A_norm.T).mean(0)
        dists[base_flat] = torch.inf
        order = dists.argsort()
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(len(order), device=device)
        gt_rank = ranks[S == p].min().item()
        rows.append({"base": base, "inflected": past, "zero_rank": gt_rank, "zero_top1": gt_rank == 0,
                     "zero_nearest": labels[S[order[0]].item()]})
    return pd.DataFrame(rows)


def class_summary(z: pd.DataFrame, model: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    out = []
    for c in P.PAST_CLASSES:
        d = z[(z.past_class == c) & z.step_top1.notna()]
        zt, st = d.zero_top1.to_numpy(float), d.step_top1.to_numpy(float)
        idx = rng.integers(0, len(d), size=(2000, len(d)))
        ci = lambda x: np.quantile(x[idx].mean(1), [0.025, 0.975])
        out.append({"model": model, "target_class": c, "n_target_pairs": len(d),
                    "zero_offset_top1": zt.mean(), "zero_lo": ci(zt)[0], "zero_hi": ci(zt)[1],
                    "regular_step_top1": st.mean(), "step_lo": ci(st)[0], "step_hi": ci(st)[1],
                    "step_minus_zero": (st - zt).mean(), "diff_lo": ci(st - zt)[0], "diff_hi": ci(st - zt)[1],
                    "zero_median_rank": d.zero_rank.median(),
                    "note": "verb-weighted means; 95% CIs resample target pairs"})
    return pd.DataFrame(out)


def main(args):
    start = datetime.now()
    model = MODEL_NAMES[args.model]
    v = pd.read_csv(P.CLASSES_CSV, keep_default_na=False)
    v = v[v.past_class != "exclude"]
    spec = C.load_spec()
    agg, src = token_vectors(spec, args.model)
    z = zero_offset(agg, src, spec, v, args.device)
    z = z.merge(v[["base", "inflected", "past_class"]], on=["base", "inflected"])

    # regular-step top-1 per target pair, from the midpoint run
    t = pd.read_parquet(P.DATA / "trials.parquet", columns=["model", "experiment", "base_to", "gt_label", "top1"])
    t = t[(t.model == model) & t.experiment.astype(str).isin(
        ["reg_to_reg"] + [f"reg_to_{c}" for c in P.IRREGULAR_CLASSES])]
    step = t.groupby(["base_to", "gt_label"]).top1.agg(step_top1="mean", step_trials="size").reset_index()
    z = z.merge(step, left_on=["base", "inflected"], right_on=["base_to", "gt_label"], how="left") \
        .drop(columns=["base_to", "gt_label"])
    z.insert(0, "model", model)
    P.save_table(z, f"t7_zero_offset_per_verb_{args.model}")

    per_model = [pd.read_csv(p) for p in sorted(P.TABLES.glob("t7_zero_offset_per_verb_*.csv"))]
    s = pd.concat([class_summary(d, d.model.iloc[0]) for d in per_model], ignore_index=True)
    P.save_table(s, "t7_zero_offset")
    print(C.md_table(s.drop(columns=["note"]), index=False))
    print(z[z.past_class != "regular"].sort_values("zero_top1", ascending=False).head(25).to_string(index=False))
    P.log_run("python scripts/past_tense/07_zero_offset_control.py " + " ".join(sys.argv[1:]), start, datetime.now(),
              f"t7_zero_offset_per_verb_{args.model}.csv ({len(z)} target pairs) and t7_zero_offset.csv")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(MODEL_NAMES), default="word")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
