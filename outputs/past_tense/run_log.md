# Run log: midpoint past-tense run

Every command run for `outputs/past_tense/`, with local start and end times (Eastern). Python is the `morph2vec` conda env, `F:\conda\envs\morph2vec\python.exe`, called without activating the env (the scripts set `PYTHONPATH`, `TMPDIR` and `HDF5_USE_FILE_LOCKING` themselves).

## Baseline

- 2026-11-04 17:35. Branch `midpoint-past-tense` created from `main` at `751e238`.
- Uncommitted changes already on `main` before this work, left untouched and never staged:
  - `D Project_Feasibility.md` (the root copy; the document now lives in `CAS784/Project_Feasibility/`)
  - `M outputs/feasibility/figures/fig01–fig06 *.pdf` (6 regenerated PDFs)
  - `?? midpoint_claude_code_handoff.md` (the handoff itself)

## Commands

| # | Start | End | Command | Outcome |
|---|---|---|---|---|
| 1 | 11-04 17:36:45 | 11-04 17:36:46 | `python scripts/past_tense/01_make_classes.py` | vbd_classes.csv (949 pairs) and class_counts.csv; xlsx auto_class reproduced 949/949 |
| 2 | 11-04 17:38:43 | 11-04 17:39:42 | `python scripts/past_tense/02_run_analogies.py --model word --num-samples 20 --out-root outputs/past_tense/runs_dryrun` | FAILED: PapermillExecutionError --------------------------------------------------------------------------- |
| 3 | 11-04 17:40:50 | 11-04 17:42:18 | `python scripts/past_tense/02_run_analogies.py --model word --num-samples 20 --out-root outputs/past_tense/runs_dryrun2` | done: 20 experiments, 400 trials; experiment loop 0.6 min of 1.5 min total |
| 4 | 11-04 17:43:00 | 11-04 17:43:15 | `python scripts/past_tense/03_summarize.py --runs-root outputs/past_tense/runs_dryrun2` | 400 trials (Word); tables t1-t6, sanity_checks |
| 5 | 11-04 17:43:47 | 11-04 17:43:53 | `python scripts/past_tense/04_figures.py` | figures A-F (png + pdf) |
| 6 | 11-04 17:45:32 | 11-04 17:45:36 | `python scripts/past_tense/04_figures.py` | figures A-F (png + pdf) |
| 7 | 11-04 17:46:35 | 11-04 17:46:36 | `python scripts/past_tense/05_obstacle_numbers.py` | obstacles.csv (25 numbers) |
| 8 | 11-04 17:48:56 | 11-04 17:48:56 | `python scripts/past_tense/06_annotation_prefill.py` | annotation_prefill.csv (151 rows; xlsx not edited) |
| 9 | 11-04 17:42:54 | 11-04 17:55:52 | `python scripts/past_tense/02_run_analogies.py --model word` | done: 20 experiments, 18,152 trials; experiment loop 12.1 min of 13.0 min total |
| 10 | 11-04 18:13:40 | 11-04 18:13:42 | `python scripts/past_tense/03_summarize.py ` | 18,152 trials (Word); tables t1-t6, sanity_checks |
| 11 | 11-04 18:13:34 | 11-04 18:36:51 | `python scripts/past_tense/02_run_analogies.py --model wav2vec --hot-dir F:/morph2vec_hot` | done: 20 experiments, 18,152 trials; experiment loop 15.3 min of 23.3 min total |
| 12 | 11-04 18:36:57 | 11-04 18:37:01 | `python scripts/past_tense/03_summarize.py ` | 36,304 trials (Wav2Vec, Word); tables t1-t6, sanity_checks |
| 13 | 11-04 18:37:04 | 11-04 18:37:35 | `python scripts/past_tense/07_zero_offset_control.py ` | t7_zero_offset*.csv (885 target pairs, Word) |
| 14 | 11-04 18:38:53 | 11-04 18:39:07 | `python scripts/past_tense/07_zero_offset_control.py --model word` | t7_zero_offset_per_verb_word.csv (885 target pairs) and t7_zero_offset.csv |
| 15 | 11-04 18:39:53 | 11-04 18:39:57 | `python scripts/past_tense/04_figures.py` | figures A-F (png + pdf) |
| 16 | 11-04 18:43:09 | 11-04 18:43:13 | `python scripts/past_tense/04_figures.py` | figures A-F (png + pdf) |
| 17 | 11-04 18:43:38 | 11-04 18:43:39 | `python scripts/past_tense/05_obstacle_numbers.py` | obstacles.csv (29 numbers) |
| 18 | 11-04 18:39:15 | 11-04 18:46:24 | `python scripts/past_tense/07_zero_offset_control.py --model wav2vec` | t7_zero_offset_per_verb_wav2vec.csv (885 target pairs) and t7_zero_offset.csv |
| 19 | 11-04 18:47:09 | 11-04 18:47:13 | `python scripts/past_tense/04_figures.py` | figures A-F (png + pdf) |
