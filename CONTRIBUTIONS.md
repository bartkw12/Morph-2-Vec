# Contribution log: CAS 784, Group 3

**Members:** Bart (pipeline), Tanya (AI implementation), Nigel (linguistics theory).

As the course requires, this log records our meetings, who was present, and what each person did between meetings. Add a row as you go, with the newest entry at the bottom. Link to commits, files or documents where possible.

## Meetings

| Date | Present | Agenda / decisions | Action items (who) |
|---|---|---|---|
| YYYY-MM-DD | | | |
| | | | |

## Work between meetings

| Date(s) | Who | What | Output / evidence |
|---|---|---|---|
| 2026-10-15 | Bart | Environment setup (conda env on F:, VS Code). Code fixes for current libraries: Hydra defaults, `torch.load`, `datasets` 3.x, lazy Ray import, missing dependencies, POS-count script | Project_Feasibility.md §8.1 (#1–6, 10, 14); commits `d08ca1d`, `7939ccd`, `8c9ccff` |
| 2026-10-15 → 16 | Bart | Windows pipeline runner replacing the Snakefiles; smoke test on `test-clean` (5 failures fixed, then end to end) | `scripts/run_pipeline.py` (`c442f6f`); §9 run 1 |
| 2026-10-16 | Bart | Full probe trained and analyzed on `test-clean` (pipeline validation) | §9 run 2; §10.6 |
| 2026-10-16 → 17 | Bart | `train-clean-100` data preparation. Fixed out-of-memory errors in preprocessing and hidden-state extraction (streaming HDF5 writer, compact indices, GPU-memory release) | §9 run 3; §8 #5, #13, #20 (`f080cd1`, `bb2117b`) |
| 2026-10-17 → 18 | Bart | Main reproduction run, layer 8. Found and fixed the NaN eval-loss bug (#7), float64 analogy slowness (#8) and analogy-notebook memory/resume (#9) | §9 runs 4–6; commits `bab1d12`, `e9864e1`, `4e765e4` |
| 2026-10-18 | Bart | Results summary script; reproduction of Figs. 3–5; Project_Feasibility.md write-up | `scripts/summarize_analogy_results.py`; Project_Feasibility.md §10–13 |
| 2026-10-21 → 22 | Bart | Feasibility demo (Milestone 3):<br>• checks and inventory, including the ɪd label check;<br>• compact data export;<br>• annotation materials for the theory work (`outputs/feasibility/theory/`);<br>• feasibility notebook: reproduction figures, baselines and the past-tense pilot;<br>• README.<br>AI coding assistant (Claude Code) used for the code and drafts, reviewed by Bart. | `scripts/feasibility/`, `notebooks/feasibility/feasibility_demo.ipynb`, `README.md` |
| | Tanya | | |
| | Nigel | | |
