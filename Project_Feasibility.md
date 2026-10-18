# Project Feasibility: Reproducing *Emergent morpho-phonological representations in self-supervised speech models*

CAS 784 (Speech Recognition in Humans and Machines), course project, feasibility step.

> Feasibility step brief: *"Load the data, run the model, reproduce a figure, compute a baseline, or pilot the experiment. This is where most projects change shape; what you learn here feeds the proposal."*

This is a working document, not the polished submission. It records everything done between 2026-10-15 and 2026-10-18 to get the paper's code running and reproduce its main results:
- how the repo works,
- what was broken and how it was fixed,
- what was downloaded and where it lives,
- every run and its results,
- what is still left to do.

- **Paper:** Gauthier, Breiss, Leonard & Chang (2025), *Emergent morpho-phonological representations in self-supervised speech models*, EMNLP 2025, pp. 28067–28086. PDF: `I:\Python\CAS 784\Project-Research-Paper_Morpho.pdf`
- **Repo:** `I:\Python\CAS 784\Morph-2-Vec` (fork: https://github.com/bartkw12/Morph-2-Vec)

---

## Table of contents

1. [TL;DR](#1-tldr)
2. [The paper in one page](#2-the-paper-in-one-page)
3. [Repository architecture](#3-repository-architecture)
4. [How the pipeline works (stage by stage)](#4-how-the-pipeline-works-stage-by-stage)
5. [Setup: environment, VS Code, machine](#5-setup-environment-vs-code-machine)
6. [How to run it (commands)](#6-how-to-run-it-commands)
7. [What was downloaded and where everything lives](#7-what-was-downloaded-and-where-everything-lives)
8. [What was broken and what we changed](#8-what-was-broken-and-what-we-changed)
9. [Run log (everything we ran)](#9-run-log-everything-we-ran)
10. [Results and comparison with the paper](#10-results-and-comparison-with-the-paper)
11. [Deviations from the paper and open questions](#11-deviations-from-the-paper-and-open-questions)
12. [What still needs to be done](#12-what-still-needs-to-be-done)
13. [Lessons for the proposal](#13-lessons-for-the-proposal)
14. [Troubleshooting notes](#14-troubleshooting-notes)

---

## 1. TL;DR

- **The pipeline runs end to end on our machine** (Windows 10, RTX 3080 Ti, 32 GB RAM):
  - LibriSpeech audio → wav2vec 2.0 hidden states → word-probe training → probe embeddings → analogy experiments.
  - It took about 20 fixes (§8): Python package incompatibilities, Windows incompatibilities, memory problems, and two real bugs that surfaced at full scale.
- **No pretrained probe weights are released.** We trained the paper's word probe ourselves on `train-clean-100`, using the paper's hyperparameters (Table 9). That is 24,576 parameters and about 25 minutes on our GPU.
- **We reproduced the paper's main analysis for layer 8 of wav2vec2-base on `train-clean-100`:**
  - Our analogy set has **334,008 word tokens**, exactly the number the paper reports.
  - **Fig. 3** (noun/verb analogies): word probe 0.9 / 5.4 / 3.2 / 5.9 vs. the paper's 1 / 10 / 2.7 / 7.9. Raw wav2vec 7.9 / 51.2 / 21.7 / 36.3 vs. the paper's 6.7 / 68 / 29 / 37.
  - **Fig. 4** (allomorphs): the same pattern as the paper, cell by cell.
  - **Fig. 5** (false friends): the same direction as the paper, with larger absolute ranks. This is probably a missing frequency filter; see §11.
- **The paper's main claim holds.** A linear probe trained only to tell spoken words apart gives a representation where noun plurals and 3rd-person-singular verbs share one linear direction, largely regardless of morphology or allomorph. Raw wav2vec states are much more sensitive to both.
- **Not done yet:**
  - All 12 layers (Fig. 1): about 620 GB of disk and roughly 2 days of compute.
  - Forced choice (Fig. 6) and the regression (Fig. 7 / Table 7), which need the authors' summary notebook adapted plus the WorldLex frequency file.
  - The held-out and appendix analyses.
  - Swapping in a newer speech model.

---

## 2. The paper in one page

**Question.** What linguistic structure does a self-supervised speech model (S3M) *need* in order to recognize spoken words? In particular, how does it represent word-final [z], [s] and [ɪz]? These sounds:
- mark English noun plurals (NNS: *dogs, cats, dishes*) and 3rd-person-singular verbs (VBZ: *runs, jumps, finishes*);
- follow a phonological rule, given in the paper's Box 1:
  - /z/ is pronounced [z] after a voiced non-sibilant (*dogs*),
  - [s] after a voiceless non-sibilant (*cats*),
  - [ɪz] after a sibilant (*dishes*);
- also appear in words that are not inflections at all (*haze, fleece, besides*).

**Models compared.**

1. **"Wav2Vec":** the raw hidden states of `facebook/wav2vec2-base` (12 transformer layers, 768-dim, one frame per 20 ms, pretrained on 960 h of LibriSpeech with no labels).
2. **"Word" probe:** a *linear* map z = W·x from 768 to 32 dimensions, applied to one wav2vec layer. It is trained with a contrastive hinge loss: frames of the same word should be close in cosine distance, frames of different words far apart (paper Eq. 1–2). Training data is the 100 h `train-clean-100` split with forced word/phone alignments.

**Method.**
- **Word embeddings.** Each spoken word token gets an embedding: the mean of its frames (Eq. 3).
- **Analogies.** Word2vec-style analogies over these embeddings: *shirt : shirts :: cheese : ?*, i.e. predict d̂ = b − a + c (Eq. 4).
- **Metric.** The **rank** of the correct target word among all ~334k word tokens, by cosine distance. 0 means the correct word is the nearest neighbour. Lower is better.

**Experiments.**

| § | Test | Example | Finding (word probe) |
|---|---|---|---|
| 4.1 | across morphology (NNS↔VBZ) | *daughter:daughters :: own:owns* | works almost as well across categories as within; wav2vec does not |
| 4.2 | across allomorphs ([z]/[s]/[ɪz]) | *daughter:daughters :: lip:lips* | largely invariant; wav2vec is not |
| 4.3 | to/from "false friends" | *daughter:daughters :: beside:besides* | largely invariant; wav2vec is not |
| 4.4 | forced choice | *daughter:daughters :: bay : {bays, base}?* | prefers the form consistent with the Box 1 rule |
| 4.5 | regression on trial ranks | — | interaction (sensitivity) terms shrink for the probe |

**Conclusion.** Optimizing for word recognition produces a representation that tracks an abstract *distributional/phonological regularity*: "add a sibilant that agrees in voicing". It does not cleanly encode separate morphological or phonological levels.

---

## 3. Repository architecture

The repo had no README. This is the structure, annotated. Files marked 🆕 are ones we added; ✏️ are ones we modified.

```
Morph-2-Vec/
├── Project_Feasibility.md          🆕 this document
├── environment.yml                 🆕 conda env (python 3.11 + pip)
├── requirements.txt                🆕 pip dependencies (install torch separately, CUDA wheel)
├── config.yaml                     list of models for the (Linux-only) Snakemake workflow; mostly
│                                      pseudo-causal "-pc-" variants from a related project, not this paper
├── Snakefile                       ORIGINAL pipeline (Snakemake + bash). Does not run on Windows.
├── workflows/librispeech/Snakefile ORIGINAL download/preprocess rules (wget/tar/ffmpeg/GNU parallel)
├── train_decoder.py                Hydra entry point: trains the word probe  -> src/train.py
├── word_recognition.py             Hydra entry point for a word-recognition classifier (not used by this paper)
│
├── conf/                           Hydra configs (composed by every script)
│   ├── config.yaml              ✏️ defaults + Trainer args (batch 32, eval every 200 steps, early stopping)
│   ├── base_model/w2v2_{0..11}.yaml   which wav2vec2-base layer to extract (model_ref, layer)
│   ├── model/ffff_32.yaml          THE PAPER'S PROBE: linear 768->32, hinge loss, Table 9 hyperparameters
│   ├── model/ffff_32-pc-*.yaml     pseudo-causal variants (other project)
│   ├── model/randomff_32.yaml      untrained random-projection baseline
│   ├── equivalence/word_broad_10frames_fixedlen25.yaml   classes = words (used by the paper)
│   ├── equivalence/phoneme_10frames_fixedlen25.yaml      classes = phonemes (not used here)
│   ├── dataset/timit_syllables.yaml   placeholder; scripts override dataset.processed_data_dir
│   └── recognition_model/linear.yaml  for word_recognition.py (unused)
│
├── scripts/
│   ├── run_pipeline.py          🆕 Windows-friendly replacement for the Snakefiles (see §6)
│   ├── make_pos_counts.py       🆕 builds the missing POS-count file (spaCy en_core_web_trf)
│   ├── summarize_analogy_results.py 🆕 prints Fig. 3/4/5-style tables from experiment results
│   ├── extract_hidden_states.py ✏️ run wav2vec over all audio -> HDF5 of frame states (streaming)
│   ├── generate_state_space_specs.py ✏️ word/phoneme/syllable frame spans ("state space specs")
│   ├── make_equivalence_dataset.py   frame -> word-class labels for contrastive training
│   └── extract_model_embeddings.py ✏️ run the trained probe over all frames -> .npy
│
├── src/
│   ├── train.py                 ✏️ probe training (HuggingFace Trainer); hyper-param search via Ray (optional)
│   ├── models/integrator.py     ✏️ ContrastiveEmbeddingModel (the probe), hinge/ratio losses, metrics (mAP)
│   ├── models/transformer.py    ✏️ feature extractor helper (padding)
│   ├── models/__init__.py          get_best_checkpoint()
│   ├── datasets/huggingface_librispeech.py ✏️ HF dataset loading script: LibriSpeech + TextGrid alignments
│   ├── datasets/speech_equivalence.py ✏️ SpeechHiddenStateDataset (HDF5), SpeechEquivalenceDataset (Q/S labels)
│   ├── analysis/state_space.py     StateSpaceAnalysisSpec (word spans), trajectory pooling
│   ├── analysis/analogy.py      ✏️ stimuli construction (lemminflect, false friends) + analogy experiment loop
│   ├── analysis/analogy_pseudocausal.py  variant for pseudo-causal models (unused)
│   ├── utils/syllabifier.py        CMU-phone syllabifier (used in preprocessing)
│   └── utils/__init__.py           helpers for globbing result CSVs (used by summary notebooks)
│
├── notebooks/
│   ├── preprocessing/librispeech.ipynb ✏️ builds the HF dataset (alignments grouped by word, syllables, audio)
│   ├── analogy/prepare_inputs.ipynb    builds analogy stimuli (pairs, allomorphs, false friends, forced choice)
│   ├── analogy/run.ipynb            ✏️ runs all analogy experiments for one model/layer -> experiment_results.csv
│   ├── analogy/*_v3.ipynb              "-er" suffix extension (needs a hand-annotated er_forms.csv that is NOT in the repo)
│   ├── geometry/*.ipynb                exploratory PCA/geometry notebooks (Fig. 2-style), not pipeline steps
│   └── word_discrimination.ipynb       exploratory
│
├── summary_notebooks/
│   ├── analogy v4.ipynb            the authors' figure/statistics notebook for the main results (Figs 1, 3-7, Table 7)
│   ├── analogy v4 heldout.ipynb    same for the held-out test-clean evaluation (App. B.2)
│   ├── analogy v2.ipynb, word_discrimination.ipynb   older versions
│
├── data/        (gitignored) raw downloads: LibriSpeech, alignments   -> see §7
└── outputs/     (gitignored) everything the pipeline produces          -> see §7
```

### Key classes and data structures

| Object | Where | What it is |
|---|---|---|
| Preprocessed dataset | `outputs/preprocessed_data/librispeech-<split>` | HF `datasets` arrow dataset, one row per utterance. Columns: `audio`, `text`, `input_values` (normalized waveform), `word_detail` (word start/stop/label), `word_phonemic_detail` (per-word list of phones with sample offsets), `word_syllable_detail`, `idx`. |
| `SpeechHiddenStateDataset` | `src/datasets/speech_equivalence.py` | HDF5 file: `states [num_frames, 1, 768]` float32, `flat_idxs [num_frames, 2]` (utterance idx, frame idx), `compression_ratios` (frames per audio sample, per utterance). All utterances are concatenated into one long frame axis. |
| `StateSpaceAnalysisSpec` | `src/analysis/state_space.py` | For every label (e.g. word type), the list of `(start_frame, end_frame)` spans of its tokens, plus `cuts` (phoneme/syllable spans within each token, with phone labels). |
| `SpeechEquivalenceDataset` | `src/datasets/speech_equivalence.py` | `Q[frame]` = class id (word) or −1; `S[frame]` = start frame of the input window (fixed 25 frames back); `class_to_frames`. Defines positives and negatives for contrastive training. |
| `ContrastiveEmbeddingModel` | `src/models/integrator.py` | The probe. With `num_layers: 0` it is `Identity` followed by `Linear(768 → 32)` applied to the last frame of the window, so it is a purely linear per-frame map. |
| `all_cross_instances.parquet` | `outputs/analogy/inputs/...` | One row per (base token, inflected token) instance pair for every stimulus pair, with labels such as `inflection` (NNS/VBZ/VBD/VBG/random/…-FF-…/FC-…), `post_divergence` (allomorph), `is_regular`, `base_ambig_NN_VB`. 14.5M rows on train-clean-100. |
| `experiment_results.csv` | `outputs/analogy/runs*/...` | One row per analogy trial: source pair, target pair, predicted nearest word, `gt_label_rank`, `gt_distance`. |

---

## 4. How the pipeline works (stage by stage)

```
LibriSpeech FLAC + MFA TextGrid alignments
   │  [download]       tar/zip extract, FLAC -> 16 kHz WAV
   ▼
data/librispeech/<split>, data/librispeech_alignments
   │  [preprocess]     notebooks/preprocessing/librispeech.ipynb
   ▼
outputs/preprocessed_data/librispeech-<split>       (HF dataset: audio + word/phone/syllable alignments)
   │  [extract]        scripts/extract_hidden_states.py   (wav2vec2-base forward pass, keep layer L)
   ▼
outputs/hidden_states/w2v2_L/librispeech-<split>.h5  (one 768-d vector per 20 ms frame)
   ├──[specs]   scripts/generate_state_space_specs.py  -> outputs/state_space_specs/... (word token frame spans)
   ├──[equiv]   scripts/make_equivalence_dataset.py    -> outputs/equivalence_datasets/... (frame -> word class)
   │
   │  [train]   train_decoder.py  (word probe, contrastive hinge loss, early stopping)
   ▼
outputs/models/<train split>/w2v2_L/ffff_32/word_broad_10frames_fixedlen25/checkpoint-*
   │  [embed]   scripts/extract_model_embeddings.py   (probe applied to every frame)
   ▼
outputs/model_embeddings/.../<split>.npy             (one 32-d vector per frame)

[pos]    scripts/make_pos_counts.py                -> outputs/pos_counts/<split>.pkl   (spaCy POS per word type)
[inputs] notebooks/analogy/prepare_inputs.ipynb    -> outputs/analogy/inputs/<split>/w2v2/  (stimuli)
[analogy] notebooks/analogy/run.ipynb  x2:
           embeddings = probe  -> outputs/analogy/runs/<train split>/w2v2_L/ffff_32/.../<split>/experiment_results.csv
           embeddings = "ID"   -> outputs/analogy/runs_id/<split>/w2v2_L/experiment_results.csv   (raw wav2vec)
[summary] scripts/summarize_analogy_results.py  (our quick tables)  /  summary_notebooks/analogy v4.ipynb (authors' figures)
```

### 4.1 Data (download + preprocess)

- **Audio:** LibriSpeech `test-clean` (5.4 h, 2,620 utterances) for smoke tests, and `train-clean-100` (100.5 h, 28,538 utterances) for the main run. FLAC is converted to 16-bit WAV, which is already 16 kHz mono.
- **Alignments:** Lugosch (2019) Montreal-Forced-Aligner TextGrids, with word and phone tiers in CMU ARPAbet with stress digits.
- **Preprocessing notebook** (`notebooks/preprocessing/librispeech.ipynb`):
  1. Loads the audio and TextGrids through the HF loading script.
  2. Maps British spellings to American ones.
  3. Strips stress digits.
  4. Groups phones into the word that contains their onset.
  5. Syllabifies each word with the P2TK syllabifier.
  6. Normalizes the waveform (zero mean, unit variance) into `input_values`.
  7. Saves the result as an HF dataset.
  - It also runs sanity checks against CMUdict.

### 4.2 Hidden-state extraction

`scripts/extract_hidden_states.py`:
- Runs `facebook/wav2vec2-base` on batches of 12 zero-padded utterances, with an attention mask, and keeps `output.hidden_states[L]`.
- `hidden_states[0]` is the input to the first transformer layer and `[L]` is the output of layer L. The paper's "layer 8" is config `w2v2_8`; the summary notebook labels it layer = index + 1.
- Frames are un-padded per utterance and concatenated, about 50 frames per second of audio.
- Our version streams to HDF5 and can save several layers from one forward pass (see §8).

### 4.3 State-space specs and equivalence classes

- **`generate_state_space_specs.py`** records, for every word type, the frame span of each of its tokens, plus the phoneme and syllable cuts inside each token. The analogy step uses this to mean-pool word tokens.
- **`make_equivalence_dataset.py`** (config `word_broad_10frames_fixedlen25`):
  - Labels up to 10 uniformly spaced frames per phoneme with their word's orthographic label.
  - Drops rare words: below the 10th percentile, or fewer than 5 tokens.
  - Sets each frame's input window to start 25 frames earlier.
  - On train-clean-100 this gives **11,512 word classes over 13.6M labeled frames**.

### 4.4 Word-probe training (the "Word" model)

- **Model** (`conf/model/ffff_32.yaml`): `num_layers: 0`, `hidden_dim: 768`, `output_dim: 32`, so z = W·x_t + b. That is 24,576 + 32 learnable parameters.
  - The 25-frame input window is passed in, but only the last frame is used, because there is no recurrent layer.
- **Batches.** Each training example is a random labelled frame plus a random *positive*: another frame of the same word class. *Negatives* are the other examples in the batch of 32 with a different class (`in_batch_soft_negatives`).
- **Loss** (`ContrastiveEmbeddingHingeObjective`): mean over valid pairs of max(0, m − [cos(z, z⁺) − cos(z, z⁻)]), with margin m = 0.3759.
- **Optimization.** HF `Trainer` (AdamW), lr 0.00108, weight decay 0.00607 (paper Table 9), batch 32, gradient accumulation 2.
  - Evaluates every 200 steps on ≤10k held-out frames.
  - **Early stopping:** stops after 3 evaluations without eval-loss improvement, then reloads the best checkpoint.
  - The eval metrics include **mAP**: the mean average precision of a logistic-regression word classifier on the probe outputs. This is the metric the authors used for hyperparameter selection.
- The paper's ~12 h/layer figure is for a **hyperparameter search**, which we did not need because the tuned values are in the config. Plain training takes about 25 min on our GPU.

### 4.5 Probe embeddings

`extract_model_embeddings.py` loads the best checkpoint and applies the probe to every frame of a split. The result is an `[num_frames, 32]` float32 `.npy`: 18.1M × 32 for train-clean-100.

### 4.6 Stimuli (analogy inputs)

`notebooks/analogy/prepare_inputs.ipynb`, using `src/analysis/analogy.py`:
1. **Subsample** to at most 100 tokens per word type, giving the word-token universe used for ranking. On train-clean-100: 32,046 types and **334,008 tokens**.
2. **Candidate pairs.** For words with more than 5 tokens, `lemminflect` proposes inflected forms (NNS, VBZ, VBD, VBG, plus a negative-prefix "NOT-latin" class). `is_regular()` keeps regular orthographic inflections.
3. **Noun/verb ambiguity.** The spaCy POS counts mark whether a base or inflected form was ever tagged as the other category (`base_ambig_NN_VB`). The paper's "unambiguous" nouns and verbs are the pairs with `base_ambig_NN_VB == False`.
4. **Allomorph** (`post_divergence`): the phones after the point where the inflected form's pronunciation diverges from the base (S, Z, IH Z, …).
5. **Random word pairs** as a control.
6. **False friends:** pairs whose phonological relation mimics an inflection but which are not inflections (*beside/besides*, *though/those*), with manual exclusion lists.
7. **Forced-choice triples** (*bay → bays vs. base*).

### 4.7 Analogy experiments

`notebooks/analogy/run.ipynb` + `analogy.run_experiment_equiv_level`:

- **Token representations.** Each word token is the mean of its frames, from either the probe embeddings or the raw wav2vec states (`embeddings_path = "ID"`).
- **Experiments.** About 120 are generated: basic, morphology transfer, unambiguous NNS/VBZ × allomorph combinations, false-friend transfers, forced choice, and random controls.
- **Each experiment:**
  1. Samples up to 1,000 (source pair, target pair) combinations.
  2. For each combination, draws up to 250 token-level instances of b_i − a_i + c_i and normalizes them.
  3. Computes the cosine distance from each prediction to all 334k tokens and averages over instances.
  4. Removes tokens of the base word c.
  5. Records the **rank of the best-ranked token of the target word d** (`gt_label_rank`).
- **Reporting.** The figures report the **mean rank**; 0 means the target is the nearest neighbour. The paper gives random chance as a rank of 94,306.

---

## 5. Setup: environment, VS Code, machine

### Machine (constraints that shaped everything)

| Resource | Value | Consequence |
|---|---|---|
| OS | Windows 10 Pro | Snakemake/bash pipeline unusable; several Windows-only bugs |
| GPU | RTX 3080 Ti, 12 GB, driver 595.97 | fast float32, **very slow float64** (1/64 rate) |
| RAM | 32 GB (+ pagefile on C:, commit limit ~47–59 GB) | memory pressure; GPU memory counts toward commit on Windows (WDDM) |
| C: | NVMe, **only ~10 GB free** | keep envs, caches, temp **off C:** |
| F: | SATA SSD, ~230 GB free ("Backup") | conda env, HF cache, fast copy of the HDF5 being trained on |
| I: | 10 TB HDD, ~3.6 TB free ("Big10") | repo, raw data, all outputs |

### Conda environment (`morph2vec`, on F:)

Installed versions: Python 3.11.17, **torch 2.11.0+cu128** (CUDA 12.8), transformers 4.49.0, datasets 3.6.0, accelerate 1.15.0, pandas 2.3.3 (pinned < 3), numpy 2.4.6, spaCy 3.8.16 + `en_core_web_trf` 3.8.0, h5py 3.16.0, tables 3.11.1, hydra-core 1.3.7, papermill 2.7.0, lemminflect 0.2.3, librosa 0.11.0, dask 2026.8.0, scikit-learn 1.9.1, scipy 1.17.1.

Recreating it from scratch, in PowerShell:

```powershell
# one-time: keep conda envs/packages off C:
conda config --add envs_dirs F:\conda\envs
conda config --add pkgs_dirs F:\conda\pkgs

cd "I:\Python\CAS 784\Morph-2-Vec"
conda env create -f environment.yml
conda activate morph2vec
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128   # CUDA build first
pip install -r requirements.txt
python -m spacy download en_core_web_trf

# environment variables stored in the env (active after re-activating)
conda env config vars set -n morph2vec HF_HOME=F:\hf_cache TMPDIR=I:\tmp TMP=I:\tmp TEMP=I:\tmp `
    PIP_CACHE_DIR=F:\pip_cache HDF5_USE_FILE_LOCKING=FALSE "PYTHONPATH=I:\Python\CAS 784\Morph-2-Vec"
conda deactivate; conda activate morph2vec

# check
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

Why those environment variables:
- `HF_HOME` puts model and dataset caches on F:.
- `TMPDIR`/`TEMP`/`TMP` keep `datasets` temp files off C:. Python checks `TMPDIR` first.
- `HDF5_USE_FILE_LOCKING` avoids HDF5 lock errors, as in the original Snakefile.
- `PYTHONPATH` makes `import src...` work from scripts and notebooks.

### VS Code

- Extensions: Python, Pylance, Jupyter, YAML.
- **Open the `Morph-2-Vec` folder as the workspace root**, not `CAS 784`. The settings live in `Morph-2-Vec/.vscode/`.
- `.vscode/settings.json` (gitignored) sets:
  - the interpreter to `F:\conda\envs\morph2vec\python.exe`;
  - `jupyter.notebookFileRoot = ${workspaceFolder}`, so notebooks resolve `src.*` and `outputs/...`;
  - terminal environment variables;
  - excludes `data/` and `outputs/` from the file watcher and search, since they hold hundreds of GB.
- `.vscode/.env` holds the same environment variables for the Python extension.
- If imports show as unresolved, run *Python: Select Interpreter* and pick `morph2vec`.

---

## 6. How to run it (commands)

All commands run from the repo root in an activated `morph2vec` terminal:

```powershell
conda activate morph2vec
cd "I:\Python\CAS 784\Morph-2-Vec"
```

### 6.1 The runner: `scripts/run_pipeline.py`

The runner replaces the Snakemake workflow:
- It calls the same scripts and notebooks with the same Hydra overrides, and writes to the same output paths.
- Every finished stage writes a `<output>.done` marker. Re-running skips finished stages, and a half-written output is deleted and redone, so you can always just re-run after a failure.

```powershell
# Smoke test (everything on test-clean, layer 8, probe trained on 1000 frames only)
python scripts/run_pipeline.py --split test-clean --layers 8 --smoke

# Full pipeline on test-clean (real probe training; ~1 h total)
python scripts/run_pipeline.py --split test-clean --layers 8 --hot-dir F:/morph2vec_hot

# MAIN REPRODUCTION (what we ran): layer 8 on train-clean-100
python scripts/run_pipeline.py --split train-clean-100 --layers 8 --hot-dir F:/morph2vec_hot

# Fig. 1: all 12 layers (NOT run yet; ~620 GB, ~2 days)
python scripts/run_pipeline.py --split train-clean-100 --layers 0-11 --hot-dir F:/morph2vec_hot --hot-cleanup

# Held-out evaluation (App. B.2): probe trained on train-clean-100, analyzed on test-clean (NOT run yet)
python scripts/run_pipeline.py --split test-clean --train-split train-clean-100 --layers 8

# Run only some stages (comma-separated)
python scripts/run_pipeline.py --split train-clean-100 --layers 8 --stage train,embed,analogy --hot-dir F:/morph2vec_hot
```

| Option | Meaning |
|---|---|
| `--split` | LibriSpeech split to analyze: `test-clean`, `train-clean-100`, … |
| `--train-split` | split the probe is trained on (default: same as `--split`) |
| `--layers` | `8`, `0-11`, `6,8,10` |
| `--stage` | `all` or a subset of `download,preprocess,extract,specs,equiv,pos,inputs,train,embed,analogy` |
| `--model` | probe config in `conf/model` (paper: `ffff_32`) |
| `--base-model-class` | prefix of `conf/base_model/{class}_{layer}.yaml` (default `w2v2`) |
| `--smoke` | train the probe on the first 1000 frames only (pipeline check) |
| `--hot-dir` | SSD folder; the layer's HDF5 is copied there before training and the raw-model analogy, because random reads on the I: HDD are slow |
| `--hot-cleanup` | delete each layer's SSD copy after that layer finishes |
| `--parallel-specs` | compute state-space specs with a multi-process dask cluster (default: sequential, which is safer on memory) |
| `--tmp-dir` | temp files (default `outputs/tmp`, can reach tens of GB during preprocessing) |

Logs: we redirected output to `outputs/logs/*.log`, e.g. `... *> outputs\logs\my_run.log`. Each notebook stage also saves an executed copy of the notebook next to its outputs, e.g. `outputs/analogy/runs_id/.../run.ipynb`, so you can open it to see every cell's output.

### 6.2 Running individual steps by hand (Hydra)

Each script is a Hydra app configured by `conf/config.yaml`. Override config groups with `key=value`, and add new keys with `+key=value`. Quote path values containing `{`, `:` or `\` with single quotes inside double quotes. For example, training the probe manually:

```powershell
python train_decoder.py `
  "hydra.run.dir='outputs/models/librispeech-train-clean-100/w2v2_8/ffff_32/word_broad_10frames_fixedlen25'" `
  "dataset.processed_data_dir='outputs/preprocessed_data/librispeech-train-clean-100'" `
  base_model=w2v2_8 "+base_model.hidden_state_path='F:/morph2vec_hot/outputs/hidden_states/w2v2_8/librispeech-train-clean-100.h5'" `
  model=ffff_32 equivalence=word_broad_10frames_fixedlen25 `
  "+equivalence.path='outputs/equivalence_datasets/librispeech-train-clean-100/w2v2/word_broad_10frames_fixedlen25/equivalence.pkl'" `
  device=cuda
```

Extract several layers in one forward pass:

```powershell
python scripts/extract_hidden_states.py "hydra.run.dir='outputs/hidden_states/_logs'" base_model=w2v2_0 `
  "+base_model.extract_layers=[0,1,2,3,4,5,6,7,8,9,10,11]" `
  "+base_model.hidden_state_path='outputs/hidden_states/w2v2_{layer}/librispeech-train-clean-100.h5'" `
  "dataset.processed_data_dir='outputs/preprocessed_data/librispeech-train-clean-100'"
```

The exact command for every stage is printed in the runner logs (lines starting with `[run_pipeline]`).

### 6.3 Looking at results

```powershell
# Fig. 3/4/5-style tables for the probe and raw wav2vec
python scripts/summarize_analogy_results.py --dataset librispeech-train-clean-100 --layer 8
python scripts/summarize_analogy_results.py --dataset librispeech-test-clean --layer 8
```

- **Raw results:** `experiment_results.csv`, one row per trial, under `outputs/analogy/runs*/...`. Load them in a notebook with pandas.
- **Training curves:** `outputs/models/.../checkpoint-*/trainer_state.json` (`log_history`, with eval loss and eval mAP every 200 steps).
- **Exploring the embedding space:** the `notebooks/geometry/*.ipynb` notebooks do PCA of the difference vectors (paper Fig. 2). They predate the runner and will need their path parameters updated.
- **Authors' figures:** `summary_notebooks/analogy v4.ipynb` produces all main figures. It is **not yet adapted to our setup**: it expects all 12 layers and `data/WorldLex_Eng_US.Freq.2.txt` (see §12).

---

## 7. What was downloaded and where everything lives

### 7.1 Downloads

| What | Source | Saved as | Size |
|---|---|---|---|
| LibriSpeech `test-clean` | https://www.openslr.org/resources/12/test-clean.tar.gz | `data\downloads\test-clean.tar.gz` | 0.32 GB |
| LibriSpeech `train-clean-100` | https://www.openslr.org/resources/12/train-clean-100.tar.gz | `data\downloads\train-clean-100.tar.gz` | 5.95 GB |
| LibriSpeech MFA alignments (Lugosch 2019) | https://zenodo.org/records/2619474/files/librispeech_alignments.zip | `data\downloads\librispeech_alignments.zip` | 0.58 GB |
| wav2vec2-base weights | HuggingFace `facebook/wav2vec2-base` (auto-download on first use) | `F:\hf_cache\hub\models--facebook--wav2vec2-base` | 0.71 GB |
| spaCy `en_core_web_trf` 3.8.0 | `python -m spacy download en_core_web_trf` | inside the conda env (`site-packages\en_core_web_trf`) | 0.47 GB |
| PyTorch 2.11 (CUDA 12.8) + all pip packages | download.pytorch.org, PyPI, conda-forge | conda env `F:\conda\envs\morph2vec` | 6.07 GB |
| British→American spelling map | GitHub `hyperreality/American-British-English-Translator` (fetched by the preprocessing notebook) | not saved | — |
| CMUdict | GitHub `cmusphinx/cmudict` (fetched by the preprocessing notebook, sanity checks only) | not saved | — |
| *Not downloaded yet:* Worldlex English frequencies (Gimenes & New 2016) | Worldlex/Lexique website | should go to `data\WorldLex_Eng_US.Freq.2.txt` | — |

### 7.2 Where everything is on disk (as of 2026-10-18)

Repo-relative paths are under `I:\Python\CAS 784\Morph-2-Vec\`.

| Location | Contents | Size |
|---|---|---|
| `data\downloads\` | the three archives above | 6.9 GB |
| `data\librispeech\test-clean\LibriSpeech\test-clean\<speaker>\<chapter>\*.wav` | extracted test-clean audio (WAV) + transcripts | 0.58 GB |
| `data\librispeech\train-clean-100\LibriSpeech\train-clean-100\...` | extracted train-clean-100 audio (WAV) | 10.8 GB |
| `data\librispeech_alignments\<split>\<speaker>\<chapter>\*.TextGrid` | alignments for **all 7** LibriSpeech splits | 3.2 GB |
| `outputs\preprocessed_data\librispeech-test-clean` | preprocessed HF dataset | 1.8 GB |
| `outputs\preprocessed_data\librispeech-train-clean-100` | preprocessed HF dataset | 33.0 GB |
| `outputs\preprocessing\librispeech-<split>\notebook.ipynb` | executed preprocessing notebooks | small |
| `outputs\hidden_states\w2v2_8\librispeech-test-clean.h5` | wav2vec layer-8 states, 970,953 frames | 2.8 GB |
| `outputs\hidden_states\w2v2_8\librispeech-train-clean-100.h5` | wav2vec layer-8 states, 18,087,600 frames | 51.9 GB |
| `outputs\state_space_specs\librispeech-<split>\w2v2_8\state_space_specs.h5` | word/phoneme/syllable spans | 0.96 GB total |
| `outputs\equivalence_datasets\librispeech-<split>\w2v2\word_broad_10frames_fixedlen25\equivalence.pkl` | frame → word-class labels | 0.02 / 0.33 GB |
| `outputs\pos_counts\librispeech-<split>.pkl` | spaCy POS counts per word type | < 1 MB |
| `outputs\analogy\inputs\librispeech-<split>\w2v2\` | stimuli: `all_cross_instances.parquet`, `inflection_results.parquet`, `false_friends.csv`, `most_common_allomorphs.csv`, `state_space_spec.h5`, executed `prepare_inputs.ipynb` | 0.01 / 0.15 GB |
| `outputs\models\librispeech-<split>\w2v2_8\ffff_32\word_broad_10frames_fixedlen25\checkpoint-*` | **trained probes** (best: test-clean `checkpoint-5200`, train-clean-100 `checkpoint-3200`) | < 1 MB each |
| `outputs\model_embeddings\librispeech-<split>\w2v2_8\ffff_32\...\librispeech-<split>.npy` | probe embeddings for every frame | 0.12 / 2.16 GB |
| `outputs\analogy\runs\librispeech-<split>\w2v2_8\ffff_32\...\librispeech-<split>\experiment_results.csv` | **probe analogy results** | ~10 MB |
| `outputs\analogy\runs_id\librispeech-<split>\w2v2_8\experiment_results.csv` (+ `partial_results\`) | **raw-wav2vec analogy results** | ~20 MB |
| `outputs\logs\*.log` | logs of every runner invocation (see §9) | 10 MB |
| `outputs\tmp\` | temp files (cleaned automatically) | ~0 |
| **Repo total** | `data\` 21.4 GB + `outputs\` 93.2 GB | **~115 GB** |
| `F:\morph2vec_hot\outputs\hidden_states\w2v2_8\` | SSD copies of the layer-8 states (train-clean-100 51.9 GB + test-clean 2.8 GB) for fast random access. **Safe to delete.** | 54.7 GB |
| `F:\hf_cache\hub\` | wav2vec2-base weights | 0.71 GB |
| `F:\hf_cache\datasets\generator\` | cached eval sets built during probe training. **Safe to delete.** | 3.8 GB |
| `F:\hf_cache\datasets\huggingface_librispeech\` | HF loading-script cache | 0.12 GB |
| `F:\conda\envs\morph2vec\` | the conda environment | 6.1 GB |
| `F:\conda\pkgs\`, `F:\pip_cache\` | package download caches. **Safe to delete.** | 0.6 + 3.3 GB |
| `C:\Users\BBM\.condarc` | conda config (we added the F: `envs_dirs` and `pkgs_dirs`) | — |
| `C:\Users\BBM\Documents\WindowsPowerShell\profile.ps1` | `conda init` hook (it was already there) | — |

Everything under `data/` and `outputs/` is gitignored. Raw audio and alignments can be re-downloaded; everything in `outputs/` can be regenerated by the runner.

---

## 8. What was broken and what we changed

The original code was written for a Linux cluster with Snakemake, older library versions, and a TITAN V GPU. Below is everything we had to change, why, and where.

### 8.1 Problems in the original repo (any platform, with current libraries)

| # | Problem | Symptom / cause | Fix | File(s) |
|---|---|---|---|---|
| 1 | Hydra defaults point to configs that don't exist (`model: rnn_8`, `equivalence: phoneme`) | every script that doesn't override both fails with "Could not find 'model/rnn_8'" | defaults → `model: ffff_32`, `equivalence: word_broad_10frames_fixedlen25`, `base_model: w2v2_8`. Also removed a `compute_objective` target pointing at a module that doesn't exist (`src.train_decoder`) | `conf/config.yaml` |
| 2 | `torch.load` defaults to `weights_only=True` since PyTorch 2.6 | loading the pickled `SpeechEquivalenceDataset` fails | `torch.load(..., weights_only=False)` at all 3 call sites | `src/train.py`, `src/models/integrator.py`, `scripts/extract_model_embeddings.py` |
| 3 | `datasets.tasks` was removed in `datasets` 3.0 | the LibriSpeech loading script fails on import | removed the import and `task_templates`. Pinned `datasets<4` (4.x drops loading scripts and changes audio decoding). Added `trust_remote_code=True` | `src/datasets/huggingface_librispeech.py`, preprocessing notebook |
| 4 | `ray` imported at the top of `src/train.py` | training needs Ray installed even though Ray is only used for hyperparameter search | moved the import into the search-space functions | `src/train.py` |
| 5 | **Hidden-state extraction keeps every frame in RAM** and concatenates at the end | train-clean-100 at one layer is ~56 GB of float32 against 32 GB RAM, so it is guaranteed to run out of memory. It also redoes the full forward pass for each layer | new `HiddenStateWriter`: streams batches into a chunked, resizable HDF5 with **the same file schema**. `extract_layers` saves several layers from one forward pass. `AutoModel` instead of `Wav2Vec2Model` (ready for other models). Byte-for-byte identical output verified on test-clean | `scripts/extract_hidden_states.py`, `src/models/transformer.py` |
| 6 | Extraction needed the `charsiu/tokenizer_en_cmu` tokenizer | an unnecessary download; only the feature extractor's padding is used | `prepare_feature_extractor()` replaces `prepare_processor()` | `src/models/transformer.py` |
| 7 | **Hinge loss is NaN for a batch with no negative pairs** (e.g. a final eval batch of size 1) | our train-clean-100 eval set had 7,753 examples = 969 × 8 + **1**, so `eval_loss` was NaN from the very first evaluation. Early stopping never sees an improvement and stopped training after 4 evaluations (7 min, mAP ~0.76 instead of 0.83). It only bites when eval size ≡ 1 (mod 8), so test-clean (9,844) was unaffected | return 0 loss when there are no pairs, as the ratio loss in the same file already does | `src/models/integrator.py` |
| 8 | Analogy math runs in float64 | consumer GPUs run float64 at 1/64 of float32 speed. The raw-wav2vec (768-dim) analogy run was heading for ~30 h | cast to float32 before the GPU cosine/ranking. This could only change near-tie orderings | `src/analysis/analogy.py` |
| 9 | Analogy notebook memory | loads all 18 columns of a 14.5M-row table and keeps two float64 copies of all 334k × 768 token vectors. On our machine the raw-wav2vec run grew to 17 GB, Windows paged it to disk, and 49/120 experiments took 10.5 h. Results were only written at the very end | loads only the 11 columns used; frees the float64 trajectories; **saves each experiment to `partial_results/NNN.csv` and resumes from them** (each experiment reseeds, so resuming gives identical results). The rerun took 1 h 41 m | `notebooks/analogy/run.ipynb` |
| 10 | `data/pos_counts.pkl` is not in the repo | `prepare_inputs.ipynb` needs it for the noun/verb ambiguity filter (paper App. A) | new `scripts/make_pos_counts.py`: tags each utterance's alignment words with spaCy `en_core_web_trf` (pre-tokenized, so there is exactly one tag per alignment word) and pickles word → Counter(POS) | 🆕 `scripts/make_pos_counts.py` |
| 11 | `data/WorldLex_Eng_US.Freq.2.txt` is not in the repo | `run.ipynb` loads it but never uses it; the summary notebook needs it for the regression and the false-friend analysis | `run.ipynb` loads it only if present. **Still needs downloading for Fig. 7** | `notebooks/analogy/run.ipynb` |
| 12 | Forced-choice pairs missing on small datasets | `run.ipynb` raised `ValueError` when a forced-choice pair had no stimuli | warn and skip | `notebooks/analogy/run.ipynb` |
| 13 | Preprocessing runs out of memory on train-clean-100 | `ArrowMemoryError: realloc of size 1895301120 failed` in the `prepare_audio` map, which buffers 1,000 utterances of float audio per write | `writer_batch_size=100` for that map (no change to the output) | preprocessing notebook |
| 14 | Missing dependencies | not declared anywhere: `librosa` (`datasets` audio decoding), `tables` (pandas `to_hdf` for specs), `accelerate` (HF Trainer), spaCy model. `pandas` 3.0 changes copy-on-write and `groupby.apply` semantics the code relies on | `environment.yml` + `requirements.txt`, `pandas<3`, `transformers<5`, `datasets<4` | 🆕 `environment.yml`, `requirements.txt` |

### 8.2 Windows-specific problems

| # | Problem | Fix | File(s) |
|---|---|---|---|
| 15 | The Snakemake rules are bash (`export`, backticks, `find \| parallel`, `ffmpeg`, `tar`, `unzip`, `mkdir`) | new runner `scripts/run_pipeline.py`. It downloads with `requests` (with resume), extracts with `tarfile`/`zipfile`, converts FLAC→WAV with `soundfile` in a process pool, runs scripts via `subprocess` and notebooks via the papermill API, uses the same output paths, and adds `.done` markers | 🆕 `scripts/run_pipeline.py` |
| 16 | Zenodo download fails with Python's `urllib` (`SSL: UNEXPECTED_EOF_WHILE_READING`) | switched to `requests`, with HTTP-range resume and retries | `scripts/run_pipeline.py` |
| 17 | `datasets.disable_caching()` writes every `.map()` result to `%TEMP%` on C:, and on Windows can't delete those folders (memory-mapped files) | C: filled up during preprocessing. The runner forces `TMPDIR`/`TEMP`/`TMP` to `outputs/tmp` on I: and deletes leftover `hf_datasets-*` folders after each notebook. `TMPDIR` is also set in the conda env | `scripts/run_pipeline.py`, conda env vars, `.vscode` |
| 18 | `NamedTemporaryFile()` can't be reopened by name on Windows (preprocessing notebook's CMUdict check) | read the download with `requests` instead | preprocessing notebook |
| 19 | dask `LocalCluster` (multi-process) in `generate_state_space_specs.py` is fragile on Windows and memory-hungry | `+analysis.scheduler=synchronous` option (the runner's default; `--parallel-specs` restores the cluster) | `scripts/generate_state_space_specs.py` |
| 20 | **On Windows (WDDM), GPU memory held by PyTorch counts against system RAM+pagefile ("commit")** | the train-clean-100 extraction died at 61% with "not enough memory"; peak commit was 20 GB on test-clean. Fixes: `torch.cuda.empty_cache()` after each batch, and `flat_idxs` stored as compact numpy arrays instead of ~18M Python tuples (~2 GB). Peak dropped to 10.8 GB, output unchanged | `scripts/extract_hidden_states.py` |
| 21 | Random HDF5 reads from the I: HDD are slow (probe training reads random 25-frame windows) | `--hot-dir`: copy the layer's `.h5` to the F: SSD for training and the raw-model analogy | `scripts/run_pipeline.py` |

### 8.3 Known issues we did *not* fix (not needed for our runs)

- The `Snakefile` uses `re` without importing it, and references files that don't exist:
  - `conf/base_model/w2v2_pc_8.yaml`
  - `notebooks/analogy/run_pseudocausal.ipynb`
  - most of the `ALL_MODEL_NOTEBOOKS`

  These belong to related projects. We don't use Snakemake.
- `summary_notebooks/*` hard-code paths and model names (`w2v2_8`, `ffff_32`, layers 0–11) and need WorldLex.
- `conf/config.yaml` still lists the unused `tokenizer` entry, which is harmless.
- `run.ipynb`'s default parameter cell still names an old model (`discrim-rnn_32-mAP1`). The runner always overrides it.

### 8.4 Where these changes are in git

All the code changes above have been committed to the fork (the assistant was deliberately not allowed to run `git add`/`commit`):

`d08ca1d` conda env · `7939ccd` requirements · `8c9ccff` POS counts · `c442f6f` runner · `f080cd1` hidden-state writer · `bb2117b` compact `flat_idxs` · `e9864e1` float32 analogy · `bab1d12` NaN hinge-loss fix · `4e765e4` run.ipynb memory/resume, plus the notebook and config commits in between (`f247599`, `8b7f706`, `1faac30`, `09dea6d`, `6d93d63`, `a5b5114`).

Not committed yet: `scripts/summarize_analogy_results.py` and this document.

---

## 9. Run log (everything we ran)

All times are local (Eastern), 2026-10-15 to 10-18. Logs are in `outputs/logs/`.

| # | When | What | Outcome |
|---|---|---|---|
| 0 | 10-15 | env + VS Code setup, code fixes 1–6, 10, 14 | env OK; `torch.cuda.is_available() == True` |
| 1 | 10-15 → 10-16 | **Smoke test, test-clean** (`--smoke`), `smoke_test-clean.log` | Failed five times, each failure exposing a fix: Zenodo SSL (#16), missing `librosa`, C: disk full from temp files (#17), missing `tables`, missing `accelerate`. Then **succeeded end to end.** The probe was trained for ~33 s on 1,000 frames only, as a pipeline check |
| 2 | 10-16 15:00–15:36 | **Full probe on test-clean**, `full_test-clean.log` | trained 17.5 min (5,800 steps, 29 evals, best mAP 0.684 at step 5,200); embeddings 2 min; analogy runs done |
| 3 | 10-16 → 10-17 | **train-clean-100 data prep**, `data_train-clean-100.log` | 1st attempt: preprocessing ran out of memory (#13). 2nd: extraction ran out of memory at 61% (#20). 3rd: preprocessing done 11:34, extraction 13:57 (~2 h), specs 14:29, equivalence 14:39, POS 14:52, stimuli 15:00 |
| — | 10-17 13:05, 14:34 | *(side effects)* | system memory ran out during runs 3–4. **The Claude Code process crashed** (`0xC0000409`) and **NVIDIA's background service `nvcontainer.exe` crashed** (the pop-ups). This was not a driver crash; the GPU stayed healthy |
| 4 | 10-17 15:04–19:40 | **Main run, attempt 1**, `main_train-clean-100_L8_nan-run.log` | probe "trained" for only 7 min because of the NaN eval loss (#7). The raw-wav2vec analogy ran in float64 (#8) at 7/120 experiments after 2 h. **Discarded** |
| 5 | 10-17 19:50 → 10-18 08:45 | **Main run, attempt 2**, `main_train-clean-100_L8_part1.log` | probe trained properly: 25 min, 3,800 steps, 19 evals, best eval loss 0.0564 / **mAP 0.832** at step 3,200 (first eval mAP 0.737). Embeddings 31 min. **Probe analogy done at 22:03 (73 min).** The raw-wav2vec analogy paged to disk at 17 GB (#9): 49/120 after 10.5 h, stopped |
| 6 | 10-18 08:47–10:28 | **Main run, attempt 3** (raw-wav2vec analogy only), `main_train-clean-100_L8.log` | with fix #9, **done in 1 h 41 m** |

**Total compute time for one layer of train-clean-100**, excluding failed attempts:
- download/convert ~0.5 h
- preprocess ~1.5 h
- extraction ~2 h
- specs + equivalence + POS + stimuli ~0.5 h
- training ~0.5 h
- embedding ~0.5 h
- probe analogy ~1.2 h
- raw-wav2vec analogy ~1.7 h

That is **about 8–9 h**. Of this, about 3.5 h is per-layer cost (extract, train, embed, two analogy runs) and about 5 h is one-time.

---

## 10. Results and comparison with the paper

All numbers are **mean rank of the target word**: lower is better, 0 means the target is the nearest neighbour. "Word" is the trained probe; "Wav2Vec" is raw layer-8 hidden states. Reproduce the tables with `python scripts/summarize_analogy_results.py --dataset librispeech-train-clean-100 --layer 8`.

### 10.1 Sanity checks on data and training

| Quantity | Ours | Paper |
|---|---|---|
| train-clean-100 audio | 100.49 h, 28,538 utterances, 18,087,600 frames | 100 h |
| word tokens in the analogy universe (≤100 per type) | **334,008** (32,046 types) | **334,008** |
| unambiguous nouns / verbs with regular inflection | 776 / 59 | 753 / 52 (after manual exclusion) |
| verb list | contains all 52 of the paper's verbs, plus *indicate, know, meet, please, read, seem, write* | 52 (Table 6) |
| probe dims / params | 768 → 32 / 24,576 (+32 bias) | same |
| probe hyperparameters | m = 0.3759, lr = 1.08e-3, wd = 6.07e-3 | Table 9: same |
| probe eval mAP (train-clean-100, layer 8) | 0.737 → **0.832** | (not reported) |
| word classes / labelled frames | 11,512 / 13.6M | — |

The exact token count is strong evidence that our preprocessing and stimulus construction match the authors' own. Several of our 7 extra verbs are the kind they removed by hand, e.g. *knows* ~ *nose* (paper footnote 4).

### 10.2 Fig. 3: morphology (NNS ↔ VBZ), train-clean-100, layer 8

Rows are the source pair (difference vector); columns are the target pair.

| | Word: NNS | Word: VBZ | Wav2Vec: NNS | Wav2Vec: VBZ |
|---|---|---|---|---|
| **NNS →** | **0.9** (paper 1) | **5.4** (paper 10) | **7.9** (paper 6.7) | **51.2** (paper 68) |
| **VBZ →** | **3.2** (paper 2.7) | **5.9** (paper 7.9) | **21.7** (paper 29) | **36.3** (paper 37) |

- **Reproduced.** The word probe is near-perfect for nouns and almost as good *across* morphological categories (NNS→VBZ, VBZ→NNS) as within.
- Raw wav2vec is 5–10× worse overall and much more sensitive to a category mismatch (NNS→VBZ = 51).
- The paper's main effect also holds: nouns are predicted better than verbs. The paper attributes this to verbs' larger inflectional paradigm (*pleasing* vs. *pleases* errors).
- Our probe numbers are slightly *better* than the paper's in 3 of 4 cells. That is within what you'd expect from retraining (different random sampling and checkpoint).

### 10.3 Fig. 4: allomorphs, train-clean-100, layer 8

Word probe, ours vs. paper (the paper's 5×5 grid; our run also has a "Verb [ɪz]" row and column, not shown):

| src \ tgt | Noun [s] | Noun [z] | Noun [ɪz] | Verb [s] | Verb [z] |
|---|---|---|---|---|---|
| Noun [s] | 0.28 (0.24) | 0.26 (0.41) | 1.43 (1.0) | 0.30 (0.14) | 6.14 (7.3) |
| Noun [z] | 0.16 (0.22) | 0.21 (0.30) | 0.47 (0.61) | 0.20 (0.087) | 6.27 (5.9) |
| Noun [ɪz] | 3.16 (4.1) | 1.69 (1.6) | 0.25 (0.52) | 5.28 (2.5) | 11.86 (12) |
| Verb [s] | 0.33 (0.25) | 0.66 (0.52) | 1.62 (1.9) | 0.38 (0.076) | 6.31 (5.4) |
| Verb [z] | 1.21 (0.71) | 2.74 (0.48) | 1.79 (1.7) | 0.14 (0.038) | 7.06 (11) |

Raw wav2vec, ours vs. paper:

| src \ tgt | Noun [s] | Noun [z] | Noun [ɪz] | Verb [s] | Verb [z] |
|---|---|---|---|---|---|
| Noun [s] | 9.3 (41) | 11.7 (8.1) | 41.6 (4.2) | 11.5 (41) | 46.0 (40) |
| Noun [z] | 0.99 (0.63) | 0.57 (1.2) | 2.9 (1.8) | 19.8 (37) | 49.1 (34) |
| Noun [ɪz] | 2.1 (1.9) | 0.78 (2.2) | 0.88 (1) | 30.7 (31) | 40.5 (35) |
| Verb [s] | 1.6 (0.83) | 1.7 (19) | 4.5 (4.1) | 0.59 (0.24) | 4.0 (6) |
| Verb [z] | 32.5 (37) | 38.4 (59) | 43.1 (57) | 49.5 (41) | 52.4 (47) |

- **Reproduced, almost cell by cell, for the word probe.** It is largely invariant to which allomorph the source and target use: most cells are below 2. The two weak spots the paper calls out reappear: the **"Verb [z]" target column** (~6–12) and the **"Noun [ɪz]" source row**.
- **Wav2vec reproduces the paper's qualitative pattern:**
  - Noun→Verb transfers are bad (20–50).
  - Verb [z] as a source is bad everywhere (32–52).
  - Verb [s] as a source is good.
  - The cell-level differences, e.g. Noun [s]→Noun [s] 9.3 vs. 41, are larger than for the probe. Raw wav2vec ranks are noisier, with means dominated by a few failures; the paper's median-rank figure, Fig. 13, is all 0–3.

### 10.4 Fig. 5: false friends (*beside:besides*, *though:those*), train-clean-100, layer 8

Using the authors' filtering (excluded items plus "strong" false friends only; see §11):

| | Word NNS | Word NNS-FF | Wav2Vec NNS | Wav2Vec NNS-FF |
|---|---|---|---|---|
| **NNS →** | 0.9 (paper 1) | 45.5 (paper 5.3) | 7.9 (6.7) | 267.5 (47) |
| **NNS-FF →** | 10.6 (1.4) | 61.6 (8.4) | 3.5 (3) | 211.5 (76) |

| | Word VBZ | Word VBZ-FF | Wav2Vec VBZ | Wav2Vec VBZ-FF |
|---|---|---|---|---|
| **VBZ →** | 5.9 (7.9) | 24.2 (4.3) | 36.3 (37) | 182.4 (61) |
| **VBZ-FF →** | 26.1 (18) | 59.2 (8.7) | 39.2 (36) | 212.0 (66) |

- **Partially reproduced.** The direction matches: false friends cost wav2vec far more (rank ~180–270) than the word probe (~24–62). This is consistent with the paper's claim that the probe is relatively insensitive to whether a final [s]/[z]/[ɪz] is a true inflection.
- **The absolute numbers are 3–8× higher than the paper's.** The most likely reason is a filter we couldn't apply: the authors' notebook inner-joins the false-friend trials with the Worldlex frequency table, which silently drops rare words, names and odd tokens that produce huge ranks. Means are very sensitive to such outliers. Other possible reasons: the false-friend lists depend on our POS counts, and the manual exclusion lists were tuned to the authors' items.
- Before applying the authors' filters, the naive averages were 691/1407. Filtering matters a lot for this figure.

### 10.5 Random-pair control

This is the rank of the target when the difference vector comes from a *random* word pair:
- word probe: 10,665 (NNS) / 11,097 (VBZ)
- wav2vec: 784 / 926

All real analogies are orders of magnitude better than this control.

The paper reports random chance as 94,306. Our control numbers are much lower, particularly for wav2vec. A raw wav2vec prediction b − a + c stays close to c, and inflected forms sound like their base, so even a random difference vector lands near the right word. We have not yet worked out exactly how the paper computed 94,306; see §11.

### 10.6 test-clean runs (pipeline validation)

Probe **trained and evaluated on test-clean** (5.4 h). This is *not* the paper's held-out setup (App. B.2), which trains on train-clean-100.

| Fig. 3 cell | Word (test-clean probe) | Wav2Vec | Paper Fig. 17 Word / Wav2Vec (held-out) |
|---|---|---|---|
| NNS→NNS | 0.2 | 31.1 | 0.12 / 4.5 |
| NNS→VBZ | 11.6 | 536.8 | 6.6 / 92 |
| VBZ→NNS | 2.4 | 29.6 | 2 / 8.5 |
| VBZ→VBZ | 15.6 | 706.6 | 4.7 / 35 |

- Smoke-test probe (1,000 frames, 33 s of training): 149.8 / 387.6 / 262.0 / 839.7. This is worse than raw wav2vec, as expected for an almost untrained linear projection.
- On test-clean the word-probe pattern matches the paper's held-out figure. Our raw-wav2vec ranks are much higher than the paper's, most likely because our test-clean stimuli were built from test-clean's own POS counts and word lists (only 44 NNS / 33 VBZ pairs). The paper's held-out analysis presumably used the train-clean-100 stimulus definitions.

### 10.7 What the results mean

1. **The core claim replicates.** A *linear* 32-dim projection of wav2vec layer 8, trained only to separate word identities, yields a space where "add the final sibilant" is one shared direction across noun plurals and verb inflections, and across [s]/[z]/[ɪz]. Raw wav2vec encodes the same words with much more allomorph- and category-specific structure.
2. **Morphology is not what the probe tracks.** Cross-category transfer (NNS↔VBZ) is nearly as good as within-category transfer, even though the noun and verb sets share no morphological relation.
3. **The "phonological rule" interpretation** (§4.4 of the paper) is the part we have not checked yet: we haven't summarized the forced-choice experiment (Fig. 6). The raw forced-choice trials are already in our results files.
4. **Feasibility is established.** The full chain can be re-run on new layers or new models with one command. The numbers are stable enough to compare models: our probe tracks the paper within about 2× in every Fig. 3/4 cell.

---

## 11. Deviations from the paper and open questions

| Deviation | Why | Expected impact |
|---|---|---|
| Probe retrained by us (no released weights) | not available | small; Fig. 3/4 match closely |
| Analogy math in float32 (paper: float64) | 64× faster on our GPU | negligible; could only reorder near-ties |
| POS ambiguity filter from our own spaCy run; no manual exclusion pass | `pos_counts.pkl` and the manual lists were not released | 776/59 vs 753/52 items; a few ambiguous or homophonous items remain (e.g. *knows* ~ *nose*) |
| False-friend analysis without the Worldlex join | data file missing | probably the main reason Fig. 5 absolute values are higher |
| Fig. 5 averaging is our port of the authors' logic, not their notebook | the notebook needs all 12 layers plus Worldlex | cell definitions match theirs, but we haven't verified the numbers against their notebook |
| Random-chance baseline (paper 94,306) vs. our random-pair control (~10.7k probe, ~0.8k wav2vec) | — | **open question**: check how the summary notebook computes chance |
| Eval-batch NaN fix | bug | none on results; it makes training work |
| Batch size 12 for extraction kept on purpose | wav2vec2-base's first conv layer normalizes over the padded batch, so changing the batching would slightly change features | identical extraction to the authors' setup |

---

## 12. What still needs to be done

Roughly in order of usefulness for the proposal:

1. **Fig. 1: all 12 layers.**
   - Command: `python scripts/run_pipeline.py --split train-clean-100 --layers 0-11 --hot-dir F:/morph2vec_hot --hot-cleanup`
   - **Disk:** 12 × 52 GB ≈ **620 GB** on I: (3.6 TB free).
   - **Time:** one extraction pass for all layers (~3–4 h, mostly HDD writes), then ~3.5 h per layer for train, embed and the two analogy runs, so **~2 days**.
   - Close Firefox and extra VS Code windows first (see §14). The runner resumes after a crash.
   - To save time, run a subset first, e.g. `--layers 1,4,8,11`.
2. **Get WorldLex** (`data/WorldLex_Eng_US.Freq.2.txt`, Gimenes & New 2016). It's needed for the regression (Fig. 7 / Table 7) and the authors' exact false-friend filtering.
3. **Adapt `summary_notebooks/analogy v4.ipynb`** to our paths, available layers and data, for the official figures:
   - Fig. 2: PCA of difference vectors
   - Fig. 6: forced choice, scored as chose-consistent ÷ (consistent + inconsistent) per word
   - Fig. 7 / Table 7: regression
   - Appendix Figs 9–13 and 16
4. **Held-out evaluation (App. B.2).** Analyze test-clean with the train-clean-100 probe: `--split test-clean --train-split train-clean-100`. The test-clean stimuli should probably use train-clean-100 POS counts; check how the authors did it.
5. **Optional appendix analyses:**
   - phoneme-level pooling (B.3)
   - same-word evaluation (B.4)
   - median ranks (Figs 13, 16)

   The code for B.3 and B.4 is not obviously present in the repo and may need writing.
6. **Exact replication of the item lists.** Filter stimuli to the paper's printed noun and verb lists (Tables 4–6) to remove the residual item differences.
7. **Model upgrade** (our planned extension):
   - Extraction already uses `AutoModel`. A new model needs `conf/base_model/<family>_<L>.yaml` files, and the runner takes `--base-model-class <family>`.
   - **Watch out:** `input_values` are normalized at preprocessing (`do_normalize=True`, which wav2vec2-base expects). Check each new model's expected input normalization.
   - Drop-in candidates (768-d, 12 layers): `microsoft/wavlm-base-plus`, `facebook/hubert-base-ls960`.
   - Large models (1024-d, 24 layers): about 2.7× the disk per layer.
   - `facebook/w2v-bert-2.0` uses filterbank input and needs more work.
8. **Housekeeping.** Delete `F:\morph2vec_hot` (55 GB), `F:\hf_cache\datasets\generator` (3.8 GB) and `F:\pip_cache` (3.3 GB) when you need space.

---

## 13. Lessons for the proposal

This is where the project "changes shape":

- **The code is reproducible, but not turnkey.** About 20 fixes were needed (§8). The biggest risks weren't the science; they were scale and memory: 100 h of audio means 52 GB per layer, and a 14.5M-row stimulus table.
- **Compute budget.** About 3.5 h per (model, layer) once the data is prepared, plus ~620 GB per 12-layer model for float32 hidden states. Ideas to reduce this:
  - store hidden states in float16 (halves the disk);
  - since the probe only uses the last frame of each window, read single frames instead of 25-frame windows (much faster training I/O);
  - analyze a subset of layers.
- **The headline results are robust; the false-friend numbers are sensitive to item filtering.** Any extension that relies on false friends needs careful, documented stimulus curation.
- **Extensions that are cheap given what's built:**
  1. **Newer S3Ms** (WavLM, HuBERT; multilingual XLS-R or w2v-BERT 2.0). Does the shared "inflection direction" appear in other models, and at which layers?
  2. **Past tense (VBD: [t]/[d]/[ɪd]).** The stimuli and false friends are **already computed** and in our results files (VBD experiments and the D/T forced choice). It's an untested second morpho-phonological alternation in the same data.
  3. **The "-er" suffix** (comparative *bigger* vs. agentive *buyer*): the paper's own suggested future work. The `*_v3.ipynb` notebooks implement it, but they need a hand-annotated `er_forms.csv` (morphologically related or not).
  4. **A non-English language,** to test whether the geometry reflects English phonology specifically. This needs aligned non-English speech; Multilingual LibriSpeech plus MFA would be one option.

---

## 14. Troubleshooting notes

- **Memory ("commit") is the binding constraint on this PC.**
  - Windows has ~47–59 GB of commit (32 GB RAM + pagefile on C:), and C: is too full for the pagefile to grow.
  - Firefox (~9 GB) and VS Code (~8 GB, several windows) take a big share.
  - On Windows, **GPU memory used by PyTorch also counts as commit**.
  - When commit runs out, *other* programs crash first. We saw Claude Code (`0xC0000409`) and NVIDIA's `nvcontainer.exe` popups.
  - Before long runs, close browsers and extra editors. Optionally add a pagefile on F: or I: (System → Advanced → Performance → Virtual memory).
- **Check progress** without opening the notebooks:
  - `Get-Content outputs\logs\<log>.log -Tail 5`
  - for analogy runs, count the files in `...\partial_results\`
- **Resume after any failure** by re-running the same `run_pipeline.py` command. Finished stages are skipped, and the analogy notebook resumes from its partial results.
- **`eval_loss = nan` during probe training** should no longer happen. If it does, early stopping will cut training short; check `trainer_state.json`.
- **"Out of memory" from WMI/PowerShell** (`Get-CimInstance`) is a sign the system is out of commit, not just the Python process.
- **`TMPDIR`** may be set by other tools. The runner overrides it. In interactive notebooks, make sure the `morph2vec` env vars are active, or temp files go to C:.
- **Paths with spaces** (`CAS 784`): always quote them in PowerShell. In Hydra overrides, wrap path values as `"key='value'"`.
