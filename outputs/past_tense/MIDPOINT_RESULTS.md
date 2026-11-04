# Midpoint results: does the regular past-tense step reach irregular verbs?

CAS 784, Group 3, Milestone 5 (lab meeting Fri 2026-11-06). Produced 2026-11-04 on branch `midpoint-past-tense`. Every number below gives its n and its source table in `outputs/past_tense/tables/`.

**Read this first.**
- The verb classes are **automatic** (`common.auto_class`), not hand-checked.
- **Model setup:** one probe seed (`ffff_32`, `checkpoint-3200`), one layer (8), LibriSpeech train-clean-100.
- **Small classes:** devoicing has 5 verbs; suppletive has 3 pairs from 2 base verbs, so it is descriptive only.
- **Main measure:** top-1 accuracy, with 95% CIs that resample target verbs. Mean ranks are dominated by a few failures.

---

## 1. Run summary

**What ran.** One analogy run per model, with the paper's settings: seed 42, up to 1,000 trials per experiment, cosine distance, mean pooling, layer 8, probe `ffff_32`.
- **Experiments:** 20, each pairing source pairs from one class with target pairs from another, *a : b :: c : ?*, guess = *b − a + c*, ranked against all 334,008 word tokens.
  - The handoff's table lists 19 experiments (it says 21, but its rows add up to 19); `devoicing_to_devoicing` brings it to 20.
  - The *n* is below 1,000 where fewer distinct pairs exist: `semi_regular_to_semi_regular` has 342 trials, `devoicing_to_devoicing` 20, `reg_T_to_devoicing` 790.
- **Same trials in both models.** Both models saw exactly the same 18,152 sampled trials (same seed; `sanity_checks.csv`), so the Word vs Wav2Vec comparisons are paired.
- **Code:** `notebooks/past_tense/run_past_tense.ipynb` (a copy of `notebooks/analogy/run.ipynb`), launched by `scripts/past_tense/02_run_analogies.py`. Both are committed in `921d2e4` and unchanged since the runs.

| Step | Command | Start → end (2026-11-04) | Outcome |
|---|---|---|---|
| classes | `python scripts/past_tense/01_make_classes.py` | 17:36:45 → 17:36:46 | 949 pairs; the xlsx `auto_class` column reproduced 949/949 |
| dry run 1 | `... 02_run_analogies.py --model word --num-samples 20 --out-root outputs/past_tense/runs_dryrun` | 17:38:43 → 17:39:42 | **failed**: my own count assertion compared an empty "none" class (0) with NaN. The class merge itself was correct; fixed and rerun in a new folder |
| dry run 2 | same, `--out-root outputs/past_tense/runs_dryrun2` | 17:40:50 → 17:42:18 | all 20 experiments returned trials; class counts after the merge equal `vbd_classes.csv` |
| **probe** | `python scripts/past_tense/02_run_analogies.py --model word` | 17:42:54 → 17:55:52 (**13.0 min**; loop 12.1 min) | 20 experiments, 18,152 trials |
| **raw wav2vec** | `python scripts/past_tense/02_run_analogies.py --model wav2vec --hot-dir F:/morph2vec_hot` | 18:13:34 → 18:36:51 (**23.3 min**; loop 15.3 min, about 8 min building word vectors from the 52 GB hidden states) | 20 experiments, 18,152 trials; no memory problems with browsers closed (40 GB commit free at the start) |
| summaries | `python scripts/past_tense/03_summarize.py` | seconds | tables t1–t6, `sanity_checks.csv` |
| zero-offset control | `python scripts/past_tense/07_zero_offset_control.py --model word` / `--model wav2vec` | see `run_log.md` | `t7_zero_offset.csv` (added after the first results; see §3) |

GPU time was **36 min for both models**, against the handoff's estimate of under 2 h. The full command log with times is in `run_log.md`. Both GPU runs were launched as detached processes with output to `outputs/past_tense/logs/` (gitignored), so they would survive an editor crash.

## 2. Sanity checks (`sanity_checks.csv`)

| Check | Word | Wav2Vec | Verdict |
|---|---|---|---|
| `reg_to_reg` top-1 vs. feasibility "past regular (auto-class regular only)" (0.77 / 0.71) | 0.78 [0.75, 0.81] | 0.73 [0.69, 0.76] | pass: reference inside the CI for both |
| `vowel_change_to_vowel_change` vs. feasibility irregular → irregular (0.22–0.25 / 0.16–0.19) | 0.26 [0.22, 0.31] | 0.19 [0.14, 0.24] | pass: CIs overlap the reference range |
| `random_to_*` (Word) near the random-pair control (0.02) | 0.000–0.022 for all five | (0.000–0.127) | pass for Word. Wav2Vec floors depend on the target class: 0.13 for regular targets, 0.10 for devoicing, 0.02 for semi-regular and vowel change, against 0.12 for the feasibility -s control |
| n trials and n target verbs reported | yes (`t1_transfer_summary.csv`) | yes | — |
| classes with fewer than 10 target verbs | devoicing (5), suppletive (2) | same | flagged throughout |
| CIs are item bootstrap (target verbs, 2,000 resamples) | yes | yes | — |
| only `outputs/past_tense/`, `scripts/past_tense/`, `notebooks/past_tense/`, `.gitignore`, `CONTRIBUTIONS.md` changed | yes (compared with the baseline in `run_log.md`) | | pass |

## 3. Main result: the regular step applied to each kind of past (`t1_transfer_summary.csv`, `t7_zero_offset.csv`)

Source pairs: regular verbs (*walk : walked*). Target: a base verb of the class shown. Each row has 1,000 trials.

| Target class | target verbs | Word top-1 [95% CI] | Word lenient | Word mean / median rank | Word random floor | Word base alone (no step) | Wav2Vec top-1 [95% CI] | Wav2Vec lenient | Wav2Vec mean / median rank | Wav2Vec random floor | Wav2Vec base alone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| regular | 560 | **0.78** [0.75, 0.81] | 0.78 | 45 / 0 | 0.014 | 0.50 | **0.73** [0.69, 0.76] | 0.73 | 8.4 / 0 | 0.127 | 0.60 |
| semi-regular (*kept, felt, told*) | 19 | **0.04** [0.01, 0.09] | 0.04 | 2,039 / 134 | 0.012 | 0.00 | **0.06** [0.00, 0.16] | 0.06 | 3,942 / 137 | 0.018 | 0.11 |
| devoicing (*sent, built*) | 5 | **0.43** [0.23, 0.62] | 0.43 | 9.6 / 1 | 0.022 | 0.40 | **0.47** [0.22, 0.71] | 0.47 | 9.3 / 1 | 0.095 | 0.60 |
| vowel change (*sang, took*) | 71 | **0.14** [0.09, 0.20] | 0.14 | 375 / 36 | 0.021 | 0.13 | **0.07** [0.03, 0.11] | 0.07 | 857 / 37 | 0.018 | 0.07 |
| suppletive (*went, was, were*) | 2 (3 pairs) | 0.00 | 0.00 | 17,322 / 14,800 | 0.000 | 0.00 | 0.00 | 0.00 | 10,753 / 7,822 | 0.000 | 0.00 |

- **Random floor:** the same target verbs, with a random word pair as the source.
- **Base alone (no step):** a control added after the first results. The guess is the target base itself (*b − a* = 0), with the base's own tokens excluded; the value is the mean over target verbs (`t7_zero_offset.csv`). It asks whether the past is simply the nearest word to its base. Note that the random floor cannot answer this, because random steps are large.

**What the numbers say.**
- **Regular targets.** The regular step clearly works: Word 0.78, against 0.50 with no step.
  - The step's own contribution is **+0.28 [0.24, 0.32]** (Word) and **+0.13 [0.09, 0.16]** (Wav2Vec, from 0.60 with no step to 0.73).
  - Half of the regular pasts are already the nearest word to their base in the probe; 60% are in raw wav2vec.
- **Irregular targets.** Against the no-step control, the regular step adds nothing in either model:

  | Class | Word: step minus no step | Wav2Vec: step minus no step |
  |---|---|---|
  | semi-regular | +0.04 [+0.01, +0.09] | −0.05 [−0.15, +0.01] |
  | devoicing | +0.03 [−0.32, +0.38] (5 verbs) | −0.14 [−0.49, +0.24] (5 verbs) |
  | vowel change | +0.02 [−0.04, +0.08] | −0.00 [−0.04, +0.03] |
  | suppletive | 0 | 0 |

  The raw "successes" are verbs whose past is already the base's nearest word:
  - vowel change: *threw, shrank, struck, ate, ran, bade, chose, drank, began*
  - devoicing: *lent, built*
- **Semi-regulars are the lowest non-suppletive class in both models** (Word 0.04, Wav2Vec 0.06), although their past ends in an added t/d.
- **Lenient scoring (homophones count) changes nothing:** at most +0.004 in any `reg_to_*` cell, and +0.008 in any experiment (`devoicing_to_reg`; `t1`, `lenient_minus_strict`).
- **Mean ranks** are dominated by a few far misses (e.g. Word semi-regular: mean 2,039, median 134); use top-1.

## 4. Secondary results

**Reverse direction:** an irregular step applied to regular bases (`t1`). For comparison, the no-step control for regular targets is 0.50 (Word; measured on the 560 regular targets of `reg_to_reg`, a similar but not identical set).

| Experiment | Word top-1 | Wav2Vec top-1 | n trials / target verbs |
|---|---|---|---|
| `semi_regular_to_reg` | 0.24 [0.21, 0.27] | 0.44 [0.40, 0.48] | 1,000 / 580 |
| `devoicing_to_reg` | 0.36 [0.32, 0.40] | 0.57 [0.54, 0.61] | 1,000 / 611 |
| `vowel_change_to_reg` | 0.29 [0.26, 0.32] | 0.51 [0.47, 0.55] | 1,000 / 579 |

The probe's irregular steps land on the regular past *less* often than no step at all (0.24–0.36 against 0.50). So an irregular step does not contain the regular step. Raw wav2vec scores higher here because *b − a + c* stays close to *c*, and a regular past sounds like its base.

**Within class** (`t1`):

| Experiment | Word | Wav2Vec | n trials / verbs |
|---|---|---|---|
| `semi_regular_to_semi_regular` | 0.27 [0.19, 0.34] | 0.30 [0.20, 0.40] | 342 / 19 |
| `vowel_change_to_vowel_change` | 0.26 [0.22, 0.31] | 0.19 [0.14, 0.24] | 1,000 / 71 |
| `devoicing_to_devoicing` | 1.00 | 0.90 [0.80, 1.00] | 20 / 5 (too few to read) |

Semi-regular → semi-regular (0.27) beats regular → semi-regular (0.04). In the probe, *kept*-type verbs share a step with each other that they do not share with regulars.

**[t]-source vs. [d]-source regulars** (`t1`, fig E): no difference.

| Target | Word: all / [t] / [d] sources | Wav2Vec: all / [t] / [d] |
|---|---|---|
| semi-regular (19 verbs) | 0.04 / 0.04 / 0.06 | 0.06 / 0.07 / 0.06 |
| devoicing (5 verbs; the past ends in [t]) | 0.43 / 0.52 / 0.58 | 0.47 / 0.57 / 0.53 |

A [t]-source step does not reach pasts ending in [t] better, and every devoicing CI spans about ±0.25.

**Two bins instead of four classes** (`t2_two_bin.csv`, fig D). Each target verb weighs the same; the trial-weighted values are in brackets.

| | Word | Wav2Vec | target verbs |
|---|---|---|---|
| regular → regular | 0.79 [0.76, 0.82] | 0.73 [0.70, 0.76] | 560 |
| regular → "adds a t/d" (semi-regular + devoicing) | 0.13 [0.05, 0.21] (0.24) | 0.14 [0.05, 0.25] (0.26) | 24 |
| regular → "no added stop" (vowel change + suppletive) | 0.14 [0.09, 0.21] (0.07) | 0.07 [0.03, 0.10] (0.03) | 73 |

The two-bin view tells the same story as the four classes. In the probe, pasts that add a stop are not reached more often (0.13 vs 0.14). Weighting by trials instead would have made "adds a t/d" look 3× better (0.24 vs 0.07), only because the 5 devoicing verbs got half that bin's trials.

**Vowel-change pasts that end in [t]/[d]** (*met, wrote, found*; `t5_vowel_split.csv`):

| Model | ends in [t]/[d] (23 verbs) | doesn't (48 verbs) |
|---|---|---|
| Word | 0.13 [0.05, 0.24] | 0.15 [0.08, 0.22] |
| Wav2Vec | 0.04 [0.00, 0.11] | 0.08 [0.04, 0.14] |

A final [t]/[d] kept from the base does not help.

**Similarity to the regularized form** (`t6_similarity*.csv`, fig F; exploratory). Per target verb, top-1 is unrelated to the phone distance between the regularized form (base + expected [t]/[d]/[ɪd]) and the real past:
- Word: Spearman ρ = 0.05; fit coefficient per phone +0.05 [−0.02, +0.13]; "adds a stop" +0.01 [−0.10, +0.11].
- Wav2Vec: ρ = −0.05.
- 98 irregular target pairs per model.

## 5. Where the guesses land (`t3_nearest_neighbours.csv`, `t4_overreg_proxy.csv`)

Examples from the probe, regular step applied to an irregular base. The counts are wrong guesses out of that verb's trials.

| Target | Correct | Top-1 | Most frequent wrong guesses |
|---|---|---|---|
| catch | caught | 0.00 (42 trials) | **catch'd** 22, ketchup 8, catching 3 |
| come | came | 0.00 (11) | **comed** [kʌmd] 6, become 3 |
| build | built | 0.53 (201) | **builded** 66, mingled 8 |
| feel | felt | 0.00 (51) | **failed** 19, **filled** 15, fell 13 |
| think | thought | 0.00 (58) | **thanked** 27, thank 24 |
| be | was / were | 0.00 (335 / 324) | **bade** [beɪd] 206 / 201, maybe 65 / 54 |
| go | went | 0.00 (341) | grew 110, ago 99, got 30 |

The probe's regular step behaves like **"add a final [t]/[d] to whatever the base sounds like"**. It lands on over-regularized forms (*catch'd, comed, builded, throwed, telled*) and on base-like words with -ed (*failed, thanked*), not on the irregular past. This is the model's version of a child's *goed*.

**Over-regularization proxy (exploratory).** Among wrong guesses on vowel-change targets (`reg_to_vowel_change` vs `random_to_vowel_change`; CIs resample target verbs):

| | Word: regular step | Word: random floor | Wav2Vec: regular step | Wav2Vec: random floor |
|---|---|---|---|---|
| wrong guess ends in T/D | **0.66** [0.58, 0.74] | 0.29 [0.25, 0.32] | 0.24 [0.17, 0.32] | 0.18 [0.16, 0.21] |
| wrong guess is a real word = base + [t]/[d]/[ɪd] (*see → seed, throw → throwed*) | **0.06** [0.03, 0.10] | 0.00 | 0.02 [0.00, 0.04] | 0.00 |
| wrong guess begins with the base's sounds | 0.19 | 0.04 | **0.57** [0.49, 0.65] | 0.18 |
| n wrong guesses | 859 | 979 | 933 | 982 |

For reference, 26% of all word tokens end in T/D. This is a crude, post-hoc measure built on the most common pronunciation of each word.

The probe's step pushes toward words ending in a stop. The raw wav2vec step mostly stays near the base's sound. This matches the paper's -s finding: the probe abstracts the inflection's sound pattern, and raw wav2vec does not.

## 6. Which prediction pattern this is closest to

Section 1.1 of the handoff listed three predictions. The results are closest to the first, **"fails on all irregular types about equally"**: once the no-step control is subtracted, the regular step adds about +0.28 for regular verbs (raw wav2vec +0.13) and between −0.14 and +0.04 for every irregular class, in both models. Semi-regulars, the most regular-like irregulars, are not reached better than vowel-change verbs, and there is no gradient with phonological distance. That argues against the graded single-route prediction.

At the same time, what the step *does* is surface-phonological, as Gauthier et al. found for -s: in the probe it adds a final stop and produces over-regularized look-alikes. It does not reach *kept* or *felt*, so the surface account's prediction for pasts that add a t/d fails too.

A cautious one-line summary for the slide: **in the probe, the regular past is a sound-based "add [t]/[d]" step that does not extend to any kind of irregular past.** This is consistent with a regular/irregular split, but it is not evidence for a separate lexical route as such. All of this rests on:
- automatic classes;
- one probe seed and one layer;
- 5 devoicing verbs, where the result is inconclusive;
- a no-step control that was added after seeing the first results.

## 7. Obstacles (verified numbers: `obstacles.csv`)

1. **Labels matter as much as the model.**
   - The aligner spells the [ɪd] ending two ways (`IH D` / `AH D`), and the authors' code only knows `IH D`. **64 of 230 [ɪd] verbs (28%)** have `AH D` as their most common ending (`most_common_allomorphs.csv`). **2,102 of 6,066 [ɪd] tokens (35%)** are spelled `AH D` (`inflection_instances.parquet`, regular pasts).
   - The pipeline's "regular" flag lets in stem-changing verbs:
     - **Mean rank:** the regular-past mean rank goes from **87.3 to 5.1** (Word; Wav2Vec 52.3 → 7.1) when only automatic-class regulars are kept (960 → 886 trials, `vbd_trials.parquet`).
     - **Worst 1%:** the worst 10 of 960 trials (1%) carry **96%** (Word) / **95%** (Wav2Vec) of the total rank.
     - **Culprits:** *bathed* (rank 38,857 and 30,422), *dealt* (6,509), *meant*.
   - **New this run:** without a no-step control, *sent* or *threw* look like transfer, when the past is simply the nearest word to its base. For regular verbs that applies to half the targets: 0.50 of regular pasts are the base's nearest word in the probe.
2. **No hand classification.**
   - **0 of 949** pairs have any annotation.
   - All classes are rule-based. Small classes: devoicing 5 verbs, suppletive 3 pairs / 2 bases. The `check` class (50 pairs) and the authors' list (14 more) are excluded.
   - The automatic classes contain mistakes a person would catch in seconds:
     - *bit* is mostly "a bit";
     - *crew* is almost always the noun;
     - the aligner's *dove* [dʌv] is the bird;
     - *laid* and *paid* are phonologically regular.
3. **Compute on one machine.**
   - **Data size:** the layer-8 hidden states are **51.9 GiB (55.7 GB)** per layer, and each extra layer costs about **3.5 h** (`Project_Feasibility.md` §9).
   - **Past memory failures:** preprocessing, extraction at 61%, and the raw-wav2vec analogy, which paged to disk at 17 GB (49 of 120 experiments in 10.5 h). When memory ran out, the editor and NVIDIA's background service crashed.
   - **Pipeline traps:** the analogy notebook resumes partial results by position, and the runner skips any stage whose output exists. Both are worked around in the new notebook and launcher.
   - **This run:** 13.0 min (probe) + 23.3 min (wav2vec), with the browser closed and the runs detached.
4. **Statistics.**
   - **Data:** one probe seed, one layer.
   - **Means vs. medians:** mean ranks are dominated by a few failures (Word regular → vowel change: median 36, mean 375), so top-1 is the primary measure.
   - **Homophones:** they are scored as wrong. Counting them as correct changes `reg_to_*` top-1 by at most +0.004 (+0.008 in any experiment), so they are not what drives the results.
   - **Small classes:** with 5 devoicing verbs, any devoicing CI spans about ±0.2–0.3.

## 8. Decision material: how should we classify irregular verbs for the paper?

| Option | What it means | Supporting numbers |
|---|---|---|
| A. Keep the automatic classes | Rule-based and reproducible; stated as a limitation | 885 usable pairs: 787 regular, 19 semi-regular, 5 devoicing, 71 vowel change, 3 suppletive; 64 excluded (50 `check` + 14 on the authors' list) (`class_counts.csv`). Known errors stay in (*bit, crew, dove*) |
| **B. Auto-suggest, then one person checks (recommended)** | `annotation_prefill.csv` has a suggested class, family and reason for all **151** non-regular and `check` rows; checking it is a 30–45 min review | Of the 50 `check` rows, **26 come back as regulars** (18 [d], 6 [ɪd], 2 [t]), 6 as semi-regular, 3 as vowel change (ablaut) and 3 as "other"; 12 stay excluded. In the vowel-change class, 3 rows (*bit, crew, dove*) should be excluded. 41 sound-change families are suggested (largest: *keep-kept* 10, *sing-sang* 9, *bring-brought* 6, *cling-clung* 6) |
| C. Two bins: "adds a t/d" vs. "no added stop" | Coarser, but the rules get it right, and it is the contrast the surface-sound account turns on | Fig D tells the same story as fig B (Word: 0.13 vs 0.14 verb-weighted). This only holds with verb weighting; trial weighting would invert the picture |
| D. *(new)* A continuous measure instead of classes | Phone distance between the regularized form and the real past, per verb; no annotation needed | It shows **no** relation to transfer here (ρ = 0.05 Word, −0.05 Wav2Vec; fig F). Usable as a robustness check, but on its own it says nothing about families |

Today's conclusion does not depend on the classification scheme: four classes, two bins and the continuous measure all show no transfer to irregulars beyond the no-step control. The hand check therefore matters most for the exact paper numbers, for the exclusions (homographs like *bit, crew, dove*), and for the sound-change families. The families are needed for the next test, transfer *within* a family (*sing : sang :: ring : ?*).

**Backup question (not run):** we average all 20 ms frames of a word into one vector, and a final [t]/[d] is only a few frames, so averaging may dilute the ending. Should we test pooling only the final frames? That needs re-embedding from the 52 GB hidden states (about 3.5 h).

## 9. Figures (`outputs/past_tense/figures/`, PNG + PDF, 16:9)

| File | Caption | For |
|---|---|---|
| `fig_A_reg_vs_irreg_top1` | Within-class analogies (feasibility table 9). Top-1: plural -s 0.90 / 0.80, 3sg -s 0.87 / 0.78, regular past 0.77 / 0.71, irregular → irregular (no modals, *be*, homographs) 0.25 / 0.19, random pair 0.02 / 0.12 (Word / Wav2Vec) | Slide 2 |
| `fig_B_reg_to_irregular_top1` | **Main figure.** The regular step applied to each class, top-1 for both models, with the random-pair floor (grey tick) and the base-alone control (diamond). Suppletive is marked "descriptive only" | Slide 3 |
| `fig_C_reg_to_irregular_rank` | The same, as mean rank on a log axis | Backup |
| `fig_D_two_bin` | Regular → "adds a t/d" vs "no added stop" vs regular → regular, verb-weighted top-1 | Decision slide |
| `fig_E_allomorph_source` | All / [t]-source / [d]-source regulars → semi-regular and devoicing; no source effect | Backup |
| `fig_F_similarity_scatter` | Per target verb: top-1 vs. phone distance from the regularized form; no gradient (exploratory) | Backup / option D |

**Note on fig A:** the handoff quotes irregular → irregular as 0.22 / 0.16, which is the row with *all* irregulars. Fig A uses the "no modals, *be*, homographs" row (0.25 / 0.19), as the handoff's figure spec asks. Quote 0.25 / 0.19 alongside this figure.

## 10. Open issues

- **The no-step control was added after the first results.** It is a new analysis, not part of the handoff, and it changes the reading of the devoicing and vowel-change bars. It was run for both models.
- **Devoicing (5 verbs) is inconclusive.** Some pasts are already their base's nearest word: *lent, built* in the probe; *sent, spent, lent* in raw wav2vec.
- **The class counts in the handoff are before the authors' exclusions.** After them: regular 787 (not 798), semi-regular 19 (not 21; *make, do* excluded), vowel change 71 (not 72; *hide* excluded).
- **Wav2Vec random floors differ from the -s control by target class** (0.02–0.13), so the "near 0.02" check applies to the probe only.
- **Pronunciation errors** remain in the automatic data, e.g. *practiced* aligned as [præktaɪzd]; it is excluded as `check`.
- **The 52 GB figure is 51.9 GiB.**
- **Leftovers:**
  - `outputs/past_tense/runs_dryrun/` (the failed dry run) and `tables/t7_zero_offset_per_verb.csv` (an unsuffixed first version, superseded by `t7_zero_offset_per_verb_word.csv`) were left in place, per the "never delete" rule. The first is gitignored; the second is not committed.
  - The 6 feasibility PDFs and the root `Project_Feasibility.md` deletion were already uncommitted on `main` before this work. They are untouched and unstaged.
