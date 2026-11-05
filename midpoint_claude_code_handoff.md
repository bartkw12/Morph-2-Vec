# Handoff for Claude Code: past-tense results for the midpoint lab meeting

**Project:** CAS 784 (Speech Recognition in Humans and Machines), Group 3: Bart, Tanya, Nigel.
**Repo:** `Morph-2-Vec` (local clone on Bart's PC, e.g. `I:\Python\CAS 784\Morph-2-Vec`).
**Written:** Wednesday 2026-11-04. **Results needed by:** Thursday 2026-11-05 evening. The lab meeting is Friday 2026-11-06.
**For:** Claude Code running on Bart's PC, which holds all the data. Bart reviews everything you produce.

Read this whole file before you change anything. Then read the files listed in Section 2.

---

## 0. What you are producing, in one paragraph

The project is now at **Milestone 5, the midpoint lab meeting**. Bart presents the "current evidence" part (about 5 minutes), plus the obstacles from his part of the work and one decision the group wants advice on. He needs **figures, tables and exact numbers** for four slides. Two slides reuse past-tense results that already exist. One slide needs **one new, short GPU analogy run**: does the regular past-tense step (*walk → walked*) transfer to irregular verbs, split by verb type? The fourth slide is the discussion. Your job is to run that experiment, summarize it, regenerate one existing figure in a simpler form, and collect the numbers for the obstacles and decision slides. Write everything into `outputs/past_tense/` and finish with one results file, `outputs/past_tense/MIDPOINT_RESULTS.md` (Section 7).

You are **not** making slides. Do not edit the feasibility notebook or any existing results.

---

## 1. Where the project stands

### 1.1 The research question

Gauthier et al. (2025), "Emergent morpho-phonological representations in self-supervised speech models" (`Project-Research-Paper_Morpho.pdf`), took **wav2vec 2.0 base**, read **layer 8**, and trained a small **linear word probe** (768 → 32, trained only to tell spoken words apart). They used word analogies (*a : b :: c : ?*, guess = *b − a + c*, ranked by cosine among all 334,008 spoken word tokens) to show that English **-s** is mostly one shared step: across plural nouns and verbs, and across [s]/[z]/[ɪz].

**Our extension is the English past tense.** Regular verbs add *-ed* ([t] *walked*, [d] *played*, [ɪd] *wanted*). Irregular verbs change the stem (*sang*, *took*, *went*), and some do both (*kept*, *sent*). The question is whether a model that learned only from audio represents the past tense as one shared step, as several sound-based patterns, or verb by verb, and **where irregular verbs fall relative to the regular step**. Three accounts make different predictions:

| If the regular step, applied to irregular bases… | It suggests |
|---|---|
| fails on all irregular types about equally | a dual-route-like split between regulars and irregulars (Pinker) |
| works better the more an irregular resembles a regular | one graded representation (single-route, Rumelhart & McClelland) |
| works on pasts that end in an added [t]/[d] (*kept, sent*) but not on vowel-change pasts (*sang*) or suppletive ones (*went*) | the model tracks a surface sound pattern, as Gauthier et al. found for -s |

### 1.2 Milestones so far

1. Research question: done.
2. **Feasibility demo (Milestone 3), done 22 Oct.** We reproduced the paper on LibriSpeech train-clean-100, layer 8. The token set matches exactly (334,008). Paper Fig. 3: 8 of 8 cells within 2×. Paper Fig. 4: 41 of 50 cells within 2×. Probe best checkpoint: `checkpoint-3200`, mAP 0.832. We also summarized the past-tense trials already contained in the authors' runs:
   - **Fig. 6** (`fig06_vbd_allomorph_grid`): regular past transfers across [t]/[d]/[ɪd]. Probe top-1 accuracy is 0.58–0.96 per cell.
   - **Fig. 7** (`fig07_regular_vs_irregular`, table 9): top-1 accuracy for the probe (Word) and raw wav2vec. Clean regular past 0.77 / 0.71. Irregular → irregular 0.22 / 0.16. Plural -s 0.90 / 0.80. 3sg -s 0.87 / 0.78. Random-pair control 0.02 / 0.12.
3. Proposal: done. It plans regular ↔ irregular transfer as the main experiment.
4. **Midpoint lab meeting (Milestone 5): now.** The course asks for "current evidence (figures of preliminary results, even if incomplete), obstacles, and at least one decision on which the group needs advice." The group split it three ways. Tanya is working on alternative probe loss functions; Nigel's part is to be confirmed. Bart covers current evidence, plus the obstacles and the advice decision for his own area.
5. After this: paper draft, peer review, final talk, final paper.

**Nothing has been run since 22 Oct.** The repo's last commit is `751e238`.

### 1.3 Why the new run matters

On its own, irregular → irregular failing (0.22) is not evidence for any theory. *sing → sang* and *take → took* are different changes, so every account predicts that analogy will fail. What separates the accounts is **regular → irregular transfer, split by irregular type**. The authors' code already defines `regular_to_irregular` and `irregular_to_regular`, commented out in `notebooks/analogy/run.ipynb` (cell 3). We extend that with our own verb classes.

### 1.4 The verb-class situation (important)

Nobody has filled in the hand annotation (`outputs/feasibility/theory/vbd_pairs_for_annotation.xlsx`: the `class`, `sound_change_family`, `exclude` and `notes` columns are empty for all 949 rows). Nobody has time to do it before Friday. **So use the automatic classes from `scripts/feasibility/common.py::auto_class`**, which already populate the `auto_class` column in that spreadsheet:

| auto_class | Verbs | Examples | Use as |
|---|---|---|---|
| regular_t / regular_d / regular_id | 160 / 415 / 223 | asked, called, wanted | `regular` (also keep the allomorph) |
| semi_regular_candidate | 21 | kept, felt, told, thought, made, had | `semi_regular` (the past ends in an added t/d, plus a stem change) |
| devoicing | 5 | sent, spent, built, bent, lent | `devoicing` |
| irregular_unclassified | 72 | came, found, took, sang, knew, rode | `vowel_change` (mostly vowel change; label it "other irregular, mostly vowel change") |
| suppletive | 3 | was, were, went | `suppletive` (2 base verbs only: report it descriptively) |
| check | 50 | modals (could), lay/rise homographs, multi-past verbs, dealt, meant, bathed | **exclude** from all new experiments |

Also drop the authors' exclusions (`common.EXCLUDE_VBD_BASE`, the `authors_excluded` column). Say in every output that the classes are **automatic, not hand-checked**.

---

## 2. Read these first

| File | Why |
|---|---|
| `README.md`, `CAS784/Project_Feasibility/Project_Feasibility.md` (§8–10, §14) | Pipeline, every fix made so far, run history, known pitfalls |
| `CAS784/Project_Feasibility/Feasibility_Results_Simplified.md` §6 (Figs 6–7) and §8 | What the past-tense pilot showed, and what is open |
| `scripts/run_pipeline.py` (`analogy_parameters`, `stage_analogy`, `Paths`) | Exactly how `run.ipynb` is launched through papermill, with which paths |
| `notebooks/analogy/run.ipynb` | The analogy runner. Experiments are a dict of pandas queries over `all_cross_instances` (cell 3). The main loop is in cell 26 |
| `src/analysis/analogy.py::run_experiment_equiv_level` | Trial logic and output columns (`base_from`, `base_to`, `gt_label`, `predicted_label`, `gt_label_rank`) |
| `scripts/feasibility/common.py` | Paths, `auto_class`, `bootstrap_ci`, `rank_summary`, `MODEL_COLORS`, `set_style`, `ci_bars`, `save_fig`. **Reuse these.** |
| `scripts/feasibility/02_export_compact_data.py` | The pattern for exporting compact data; `export_pairs_vbd` builds the columns `auto_class` needs |
| `scripts/feasibility/build_notebook.py` (around line 374) | How fig07 was drawn |
| `outputs/feasibility/tables/table9_regular_vs_irregular.csv` | The numbers behind fig07 |
| `outputs/feasibility/data/*.parquet` | Compact data on Bart's PC (gitignored): `results_word.parquet`, `results_wav2vec.parquet`, `vbd_trials.parquet`, `pairs_vbd.parquet`, `fc_trials.parquet` |

---

## 3. Ground rules

- **Work on a new git branch** (e.g. `midpoint-past-tense`). Commit in small steps with clear messages. Do not push or merge unless Bart asks.
- **Do not modify or overwrite** anything in `outputs/feasibility/`, `outputs/analogy/runs*/`, `notebooks/feasibility/`, or the default `experiments` dict in `notebooks/analogy/run.ipynb`. New code goes in `scripts/past_tense/` and `notebooks/past_tense/`, and new outputs in `outputs/past_tense/`.
- **Use a fresh output directory for the new run.** Two traps in the existing setup:
  1. `run.ipynb` resumes from `partial_results/NNN.csv` **by position**. Running a different experiment list in an old folder silently loads old results under new names.
  2. `run_pipeline.py`'s `run_stage` skips a stage whose `experiment_results.csv` already exists, so re-running `--stage analogy` does nothing.
- Keep the paper's settings: `seed = 42`, `num_samples = 1000` trials per experiment, `metric = "cosine"`, mean pooling, layer 8, probe `ffff_32`. Do not retrain anything.
- Machine: Windows 10, RTX 3080 Ti (12 GB), 32 GB RAM. The raw-wav2vec run loads the 52 GB layer-8 hidden states and has crashed from memory before (`Project_Feasibility.md` §14). Use the SSD hot copy that `run_pipeline.py` uses (`--hot-dir`, e.g. `F:/morph2vec_hot`), run the probe model first, and ask Bart to close browsers before the wav2vec run.
- **Never delete files.** If something must be redone, write to a new folder.
- Add `!outputs/past_tense` and `!outputs/past_tense/**` to `.gitignore`, right after the `outputs/feasibility` lines, so the CSV and PNG outputs can be committed. Parquet stays ignored, which is fine.
- Log every command you run, with start and end times, in `outputs/past_tense/run_log.md`. The timings are evidence for the obstacles slide.
- If something does not match this file (a column name, a path, a count), stop and tell Bart rather than guessing.

---

## 4. Tasks, in order

### Task 1. Build the verb-class table (CPU, about 15 min)

Write `scripts/past_tense/01_make_classes.py`:

1. Load `outputs/feasibility/data/pairs_vbd.parquet` and apply `common.auto_class` row by row. It should reproduce the `auto_class` column of the xlsx exactly; assert that, and that the counts match Section 1.4.
2. Map to `past_class`: `regular` (any `regular_*`), `semi_regular`, `devoicing`, `vowel_change`, `suppletive`, `exclude` (`check` + `authors_excluded`). Keep `allomorph` (`T` / `D` / `IH D`, using the canonical ending, where `AH D` counts as `IH D`) for regulars.
3. Also add a two-bin column, `adds_stop`: `True` for semi_regular and devoicing (the past ends in an added t/d), `False` for vowel_change and suppletive. The decision slide needs it (Section 6).
4. Write `outputs/past_tense/tables/vbd_classes.csv` (base, inflected, past_class, allomorph, adds_stop, base_tokens, past_tokens) and a count table, `class_counts.csv` (verbs and tokens per class). This count table is a Methods-ready "Table 1".

### Task 2. Set up the new analogy run (CPU, about 30 min)

1. Copy `notebooks/analogy/run.ipynb` to `notebooks/past_tense/run_past_tense.ipynb`. In the copy:
   - Add a parameter `classes_path`. After `all_cross_instances` is loaded (cell 14), merge `vbd_classes.csv` on (`base`, `inflected`) **for `inflection == 'VBD'` rows only**, adding `past_class` and `allomorph`. Leave other rows with `past_class = "none"`, so string queries don't fail on NaN. Print the class counts after the merge, and assert that each class has at least one row with `exclude_main == False`.
   - Replace the `experiments` dict with **only** the experiments below. Remove or skip the generated-experiment cells (22–24: unambiguous transfer, false friends, forced choice). Keep everything else, including the cell-26 loop with its per-experiment `partial_results` and the final `experiment_results.csv`.
2. Experiments. Every one has `all_query = "not exclude_main"`, and no `group_by`. `R` = `inflection == 'VBD' and past_class == 'regular'`. `X(c)` = `inflection == 'VBD' and past_class == 'c'`. Random = `inflection == 'random'`.

| Name | base_query (source pairs) | inflected_query (target pairs) | Purpose |
|---|---|---|---|
| `reg_to_reg` | R | R | Reference. Should land near the existing 0.77 / 0.71 top-1 (sanity check) |
| `reg_to_semi_regular` | R | X(semi_regular) | **Main**: does the regular step reach *kept / felt / told*? |
| `reg_to_devoicing` | R | X(devoicing) | **Main**: *sent / built* |
| `reg_to_vowel_change` | R | X(vowel_change) | **Main**: *sang / took / came* |
| `reg_to_suppletive` | R | X(suppletive) | **Main**: *went / was* (descriptive only: 2 bases) |
| `semi_regular_to_reg`, `devoicing_to_reg`, `vowel_change_to_reg` | X(c) | R | Reverse direction (secondary) |
| `semi_regular_to_semi_regular`, `vowel_change_to_vowel_change` | X(c) | X(c) | Within-class reference (vowel→vowel should be near the existing irregular→irregular 0.22) |
| `random_to_reg`, `random_to_semi_regular`, `random_to_devoicing`, `random_to_vowel_change`, `random_to_suppletive` | Random | R or X(c) | **Floor for each target class.** It differs by target set, so each class needs its own |
| `reg_T_to_semi_regular`, `reg_D_to_semi_regular`, `reg_T_to_devoicing`, `reg_D_to_devoicing` | R plus `and allomorph == 'T'` (or `'D'`) | X(c) | Does a [t]-source step transfer better to pasts ending in [t]? |

   That is 21 experiments. Add `devoicing_to_devoicing` only if it runs. With 5 verbs, `sample_non_identical_pairs` may have very few distinct pairs; report whatever it returns.

3. Write `scripts/past_tense/02_run_analogies.py`. It should reuse `run_pipeline.Paths` and `analogy_parameters` (import them; don't copy paths by hand) and call `papermill` on the new notebook twice, with `classes_path` added:
   - **Probe ("Word")**: `embeddings_path` = the probe embeddings (`paths.embeddings(...)`, which is `outputs/model_embeddings/librispeech-train-clean-100/w2v2_8/ffff_32/word_broad_10frames_fixedlen25/librispeech-train-clean-100.npy`). `output_dir` = `outputs/past_tense/runs/word/`.
   - **Raw wav2vec ("Wav2Vec")**: `embeddings_path = "ID"`, hidden states from the hot copy. `output_dir` = `outputs/past_tense/runs/wav2vec/`.
   - The executed notebooks are saved into those folders.
4. Before the full run, do a **dry run** with `num_samples = 20` into `outputs/past_tense/runs_dryrun/word/` (probe only). Check that every experiment returns trials and that the class merge worked. Then do the real run.

### Task 3. Run it (GPU)

Run the probe model first, then wav2vec. My rough estimate, based on the 22 Oct runs (120 experiments: 73 min for the probe, 1 h 41 min for wav2vec, most of it in the experiment loop), is **under 2 hours total**. That is a guess, not a measurement: log the real times. If the wav2vec run fails from memory, the probe results alone are still worth having; report that and keep going.

### Task 4. Summarize the new run (CPU, about 1–2 h)

Write `scripts/past_tense/03_summarize.py` (or a notebook, `notebooks/past_tense/past_tense_midpoint.ipynb`, that reads only small exported files, like the feasibility notebook does):

1. Export compact trials: concatenate both models' `experiment_results.csv` with `model` (`Word`/`Wav2Vec`) and `experiment` columns. Join `past_class`, `allomorph` and `adds_stop` for source (`base_from`) and target (`base_to` + `gt_label`). Save as `outputs/past_tense/data/trials.parquet`.
2. **Scores per experiment × model**, using `common.rank_summary` (item bootstrap over target base verbs `base_to`, 2,000 resamples): top-1, mean rank, median rank, n trials, n target verbs. Save `tables/t1_transfer_summary.csv`.
3. **Lenient (homophone) top-1** as a robustness column. Count a trial as correct if `predicted_label` is the target or a word with the **same most-common pronunciation**. Several irregular pasts are homophones: *knew/new, threw/through, blew/blue, won/one, ate/eight, rode/road*. Use the `past_homophones` column in `pairs_vbd.parquet`, or rebuild it the way `export_pairs_vbd` does. Note: the authors build a homophone map in `run.ipynb` cells 18–19 but never pass it to the loop, so all existing scores are strict.
4. **Two-bin summary** (`adds_stop` True vs False) for `reg_to_*`, so the decision slide can show whether the conclusion survives coarser classes. Save `tables/t2_two_bin.csv`.
5. **"Where did the guess land?"** For `reg_to_vowel_change`, `reg_to_semi_regular` and `reg_to_suppletive` (Word model), list per target verb: the correct past, its top-1 rate, and the three most frequent wrong `predicted_label`s with counts. Save `tables/t3_nearest_neighbours.csv`. Then pick 4–6 telling examples for the slide (for instance, does *sing* + the regular step land on a word that sounds like *sing* plus [d], the model's version of a child's *goed*?). Also compute one number: among wrong guesses, the share whose most-common pronunciation ends in `T` or `D`, for `reg_to_vowel_change` vs. `random_to_vowel_change`. Call it the "over-regularization proxy" and say plainly that it is exploratory.

### Task 5. Figures (CPU)

Use `common.set_style()`, `MODEL_COLORS` (Word `#0072B2`, Wav2Vec `#E69F00`), `CONTROL_COLOR` for floors, and `save_fig` (PNG + PDF) into `outputs/past_tense/figures/`. Keep figures readable on a slide at 16:9: large fonts, few words, n under each bar as "trials / verbs".

| File | Content | For |
|---|---|---|
| `fig_A_reg_vs_irreg_top1.png` | **Simplified fig07**: top-1 accuracy only, five conditions (plural -s, 3sg -s, regular past (auto-clean), irregular → irregular (no modals, *be*, homographs), random pair), Word vs Wav2Vec bars with 95% CIs. Numbers from `outputs/feasibility/tables/table9_regular_vs_irregular.csv`; no new run needed | Slide 2 |
| `fig_B_reg_to_irregular_top1.png` | **The main new figure.** x = target class (regular, semi-regular, devoicing, vowel change, suppletive). Bars = top-1 of `reg_to_<class>` for Word and Wav2Vec with CIs. A grey marker per class = the `random_to_<class>` floor. Mark suppletive as "n = 2 verbs" | Slide 3 |
| `fig_C_reg_to_irregular_rank.png` | Same layout, mean rank on a log axis | Backup |
| `fig_D_two_bin.png` | Regular → "adds a t/d" vs "no added stop" vs regular → regular, top-1 | Decision slide |
| `fig_E_allomorph_source.png` | [t]-source vs [d]-source → semi-regular / devoicing | Backup |

Show the figures to Bart (or at least list them with one-line captions) before you call the work finished.

### Task 6. Obstacles and decision material (CPU, about 30 min)

Collect exact numbers, with their source file, for Bart's obstacles and advice slide (Section 6). Most are already in `Project_Feasibility.md`, `outputs/feasibility/checks/id_label_check.md` and table 9. Verify each one rather than copying it.

### Task 7 (optional, only if time remains). Pre-fill the annotation for a human check

In a **new file** (`outputs/past_tense/annotation_prefill.csv`; do not edit the xlsx), write a suggested `class` and `sound_change_family` for the ~150 non-regular and `check` rows, with a one-line reason each. Suggested families: *sing/sang/sung*, *ring*, *drink*; *take/took*, *shake*; *know/knew*, *grow*, *throw*, *blow*; *keep/kept*, *sleep*, *weep*, *creep*, *sweep*; *teach/taught*, *catch*, *bring*, *think*, *buy*, *seek*; *send/sent*, *spend*, *bend*, *lend*, *build*. This turns the hand annotation into roughly a 30–45 minute review for one person, which makes option B of the decision (Section 6) realistic.

### Task 8. Contribution log

Add rows to `CONTRIBUTIONS.md`: a Meetings row for Monday 2026-11-02 (leave "Present" and "Decisions" for Bart to fill), and a "Work between meetings" row for this run (Bart; AI coding assistant used, reviewed by Bart).

---

## 5. Checks before you report

- [ ] `reg_to_reg` top-1 is close to the existing "past regular (auto-class regular only)" (Word 0.77, Wav2Vec 0.71). A big gap means the class merge or the queries are wrong.
- [ ] `vowel_change_to_vowel_change` is in the range of the existing irregular → irregular (0.22 / 0.16).
- [ ] `random_to_*` top-1 for Word is near the existing random-pair control (about 0.02).
- [ ] Every experiment has its n trials and n target verbs reported. Flag any class with fewer than 10 target verbs.
- [ ] The figures open, the labels are readable, and the CIs are item-bootstrap.
- [ ] Nothing outside `outputs/past_tense/`, `scripts/past_tense/`, `notebooks/past_tense/`, `.gitignore` and `CONTRIBUTIONS.md` changed (`git status`).

---

## 6. What Bart presents beyond the evidence

### 6.1 Obstacles (Bart's part)

Collect a verified number for each.

1. **Labels matter as much as the model.**
   - The aligner spells the [ɪd] ending two ways (`IH D` / `AH D`), and the authors' code only knew one. That dropped 28% of [ɪd] verbs (64 of 230); 35% of [ɪd] tokens are `AH D`.
   - The pipeline's automatic "regular" flag lets in stem-changing verbs (*dealt, meant, bathed*). That moves the regular mean rank from 5.1 to 87; the worst 1% of trials carry about 95% of the total rank.
2. **No hand classification of the verbs.** 0 of 949 pairs are annotated, so all classes are rule-based. Small classes: devoicing 5 verbs, suppletive 3 pairs (2 bases), 50 `check` verbs excluded.
3. **Compute on one machine.** 52 GB of hidden states per layer, about 3.5 h per extra layer, out-of-memory crashes during the original runs, and the resume and skip traps above. Add the actual times of this run.
4. **Statistics.** One probe seed, one layer. Mean ranks are dominated by a few failures, which is why top-1 is the primary measure. Homophones are scored as wrong (give the strict vs. lenient difference from Task 4).

### 6.2 The decision we need advice on (Bart's part)

**Recommended question:** *"Nobody in the group has time to hand-classify 949 verbs. How should we classify irregular verbs for the paper?"*

| Option | What it means | Supporting number to provide |
|---|---|---|
| A. Keep the automatic classes | Rule-based, reproducible, stated as a limitation | Class counts; how many `check` verbs are excluded |
| B. Auto-suggest, then one person checks (recommended) | Task 7 pre-fill, about 30–45 min of review for ~150 rows | How many rows need review |
| C. Collapse to two bins: "past adds a t/d" vs "no added stop" | Coarser, but the rules get it right, and it is the contrast the surface-sound account turns on | Whether `fig_D` tells the same story as `fig_B` |

**Backup question, if the group prefers:** we average all 20 ms frames of a word into one vector. A final [t]/[d] is only a few frames, so averaging may dilute the ending. Should we test pooling only the final frames? (Not run for Friday; it needs re-embedding from the 52 GB hidden states.)

---

## 7. Final report: `outputs/past_tense/MIDPOINT_RESULTS.md`

Bart will share this file back so the slides can be built from it. Keep it factual and short. Every number states its n and its source table.

1. **Run summary:** what ran, the exact commands, start and end times, the branch and commit hash, and any failures.
2. **Sanity checks:** the Section 5 checklist with actual values.
3. **Main result table:** for each `reg_to_<class>`: top-1 (95% CI), lenient top-1, mean rank, median rank, n trials / n target verbs, for Word and Wav2Vec, plus the matching `random_to_<class>` floor.
4. **Secondary:** reverse direction, within-class, [t]-source vs [d]-source, and the two-bin table.
5. **Nearest-neighbour examples:** 4–6 rows, plus the over-regularization proxy with its caveat.
6. **Which prediction pattern from Section 1.1 the results look closest to,** in two or three sentences, hedged appropriately (automatic classes, one seed, one layer, small classes).
7. **Obstacles:** Section 6.1 with verified numbers.
8. **Decision material:** Section 6.2 with the supporting numbers.
9. **Figure list:** file name and a one-line caption for each.
10. **Open issues:** anything that looked wrong or that you could not do.
