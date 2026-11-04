"""
Shared paths, constants and helpers for the midpoint past-tense scripts (`scripts/past_tense/*.py`).

Everything else comes from `scripts/feasibility/common.py` (imported here as `C`). Its `save_fig` /
`save_table` are NOT used: they write into outputs/feasibility/, which this work must not change.
Import from a script with

    sys.path.insert(0, str(Path(__file__).resolve().parent)); import pt_common as P; C = P.C
"""

from datetime import datetime
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "feasibility"))
import common as C  # noqa: E402

REPO_ROOT = C.REPO_ROOT
sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- paths

OUT = C.ROOT / "outputs/past_tense"
DATA, FIGURES, TABLES = (OUT / d for d in ("data", "figures", "tables"))
RUNS = OUT / "runs"
RUN_LOG = OUT / "run_log.md"
CLASSES_CSV = TABLES / "vbd_classes.csv"
NOTEBOOK = "notebooks/past_tense/run_past_tense.ipynb"
RUN_DIRS = {"Word": "word", "Wav2Vec": "wav2vec"}

# --------------------------------------------------------------------------- verb classes

CAVEAT = "Verb classes are automatic (common.auto_class), not hand-checked."

# common.auto_class -> the class used in the midpoint run ("check" pairs are excluded)
AUTO_TO_PAST = {
    "regular_t": "regular", "regular_d": "regular", "regular_id": "regular",
    "semi_regular_candidate": "semi_regular", "devoicing": "devoicing",
    "irregular_unclassified": "vowel_change", "suppletive": "suppletive", "check": "exclude",
}
AUTO_ALLOMORPH = {"regular_t": "T", "regular_d": "D", "regular_id": "IH D"}
PAST_CLASSES = ["regular", "semi_regular", "devoicing", "vowel_change", "suppletive"]
IRREGULAR_CLASSES = PAST_CLASSES[1:]
# does the past end in an added t/d? (regulars do; the two-bin summary uses the irregular classes only)
ADDS_STOP = {"regular": True, "semi_regular": True, "devoicing": True, "vowel_change": False, "suppletive": False}
CLASS_LABELS = {
    "regular": "regular", "semi_regular": "semi-regular", "devoicing": "devoicing",
    "vowel_change": "vowel change", "suppletive": "suppletive",
}
CLASS_EXAMPLES = {
    "regular": "walked, played", "semi_regular": "kept, felt, told", "devoicing": "sent, built",
    "vowel_change": "sang, took, came", "suppletive": "went, was",
}
ALLOMORPH_LABELS = {"T": "[t]", "D": "[d]", "IH D": "[ɪd]"}


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two phone sequences."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def parse_experiment(name: str) -> dict:
    """`reg_T_to_semi_regular` -> source/target class (and source allomorph) of a midpoint experiment."""
    src, tgt = name.split("_to_")
    allomorph = None
    if m := re.fullmatch(r"reg_(T|D)", src):
        src, allomorph = "reg", m.group(1)
    fix = lambda c: "regular" if c == "reg" else c
    return {"source": fix(src), "source_allomorph": allomorph, "target": fix(tgt)}


# --------------------------------------------------------------------------- output

def save_fig(fig, name: str):
    """Save to outputs/past_tense/figures as png (dpi 200) and pdf."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200)
    fig.savefig(FIGURES / f"{name}.pdf")


def save_table(df: pd.DataFrame, name: str, index=False):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=index)


def log_run(command: str, start: datetime, end: datetime, outcome: str):
    """Append one row to the command table in run_log.md."""
    text = RUN_LOG.read_text(encoding="utf-8")
    n = len(re.findall(r"^\| \d+ \|", text, flags=re.M)) + 1
    row = f"| {n} | {start:%m-%d %H:%M:%S} | {end:%m-%d %H:%M:%S} | `{command}` | {outcome} |\n"
    RUN_LOG.write_text(text.rstrip("\n") + "\n" + row, encoding="utf-8")
