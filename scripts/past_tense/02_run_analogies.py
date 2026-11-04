"""
Past-tense step 2: run the midpoint analogy experiments (`notebooks/past_tense/run_past_tense.ipynb`) through
papermill, for the word probe ("Word", --model word) or raw wav2vec layer 8 ("Wav2Vec", --model wav2vec). Paths
and parameters come from `scripts/run_pipeline.py` (imported, not copied). Writes <out-root>/<model>/:

    experiment_results.csv   all trials (columns of analogy.run_experiment_equiv_level, plus `correct`)
    experiment_timings.csv   seconds and trials per experiment
    experiments.json         the experiment set, num_samples and seed (the notebook refuses to mix runs)
    partial_results/         one CSV per finished experiment, named by experiment (resume after a failure)
    run_past_tense.ipynb     the executed notebook

    # dry run: 20 trials per experiment, probe only
    python scripts/past_tense/02_run_analogies.py --model word --num-samples 20 --out-root outputs/past_tense/runs_dryrun
    # real run: probe first; close browsers before the wav2vec run, which reads the 52 GB hidden states
    python scripts/past_tense/02_run_analogies.py --model word
    python scripts/past_tense/02_run_analogies.py --model wav2vec --hot-dir F:/morph2vec_hot
"""

from argparse import ArgumentParser
from datetime import datetime
import os
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt_common as P  # noqa: E402

sys.path.insert(0, str(P.REPO_ROOT / "scripts"))
import run_pipeline as RP  # noqa: E402

SPLIT = "train-clean-100"
LAYER = 8
PROBE = "ffff_32"


def main(args):
    # the environment run_pipeline.main sets up: notebook kernels import `src`, temp files go to tmp_dir
    os.chdir(RP.REPO_ROOT)
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(RP.REPO_ROOT), os.environ.get("PYTHONPATH")]))
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    RP.set_tmp_dir(Path(args.tmp_dir).resolve())

    paths = RP.Paths("w2v2", Path(args.hot_dir) if args.hot_dir else None)
    outdir = Path(args.out_root) / args.model
    if outdir.exists() and any(outdir.iterdir()) and not args.resume:
        raise SystemExit(f"{outdir} is not empty; pass --resume to continue that run, or use another --out-root")

    if args.model == "word":
        hidden_states = paths.hidden_states(SPLIT, LAYER)  # not read: the notebook uses the probe embeddings
        embeddings = str(paths.embeddings(SPLIT, SPLIT, LAYER, PROBE))
    else:
        hidden_states = paths.hot(paths.hidden_states(SPLIT, LAYER))
        embeddings = "ID"
    params = RP.analogy_parameters(paths, SPLIT, outdir, hidden_states, embeddings)
    params |= {"classes_path": P.CLASSES_CSV.relative_to(P.REPO_ROOT).as_posix(), "num_samples": args.num_samples}

    cmd = "python scripts/past_tense/02_run_analogies.py " + " ".join(sys.argv[1:])
    start = datetime.now()
    try:
        RP.run_notebook(P.NOTEBOOK, outdir / "run_past_tense.ipynb", params)
    except BaseException as e:
        first_line = str(e).strip().splitlines()[0][:200] if str(e).strip() else ""
        P.log_run(cmd, start, datetime.now(), f"FAILED: {type(e).__name__} {first_line}")
        raise

    res = pd.read_csv(outdir / "experiment_results.csv", usecols=["experiment"])
    t = pd.read_csv(outdir / "experiment_timings.csv")
    P.log_run(cmd, start, datetime.now(),
              f"done: {res.experiment.nunique()} experiments, {len(res):,} trials; experiment loop "
              f"{t.seconds.sum() / 60:.1f} min of {(datetime.now() - start).total_seconds() / 60:.1f} min total")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__, formatter_class=lambda prog: __import__("argparse").RawDescriptionHelpFormatter(prog))
    p.add_argument("--model", choices=["word", "wav2vec"], required=True)
    p.add_argument("--num-samples", type=int, default=1000, help="trials per experiment (paper: 1000)")
    p.add_argument("--out-root", default=str(P.RUNS.relative_to(P.REPO_ROOT)))
    p.add_argument("--hot-dir", default=None, help="SSD directory holding a copy of the hidden states (wav2vec run)")
    p.add_argument("--tmp-dir", default=str(RP.REPO_ROOT / "outputs" / "tmp"))
    p.add_argument("--resume", action="store_true", help="continue a run in a non-empty output folder")
    main(p.parse_args())
