# Inventory and sanity checks (step 1)

Written by `scripts/feasibility/01_inventory_and_checks.py`.

## Headline

- **VBD pairs:** 837 regular, 112 irregular (941 distinct bases).
- **VBD trials, Word:** `basic` all: 1000; `regular` irregular→irregular: 1000; `regular` regular→regular: 1000
- **VBD trials, Wav2Vec:** `basic` all: 1000; `regular` irregular→irregular: 1000; `regular` regular→regular: 1000

## 1. Paths

| what | exists | size | path |
|---|---|---|---|
| Probe analogy results | yes | 15.9 MB | `outputs\analogy\runs\librispeech-train-clean-100\w2v2_8\ffff_32\word_broad_10frames_fixedlen25\librispeech-train-clean-100\experiment_results.csv` |
| Raw wav2vec analogy results | yes | 15.9 MB | `outputs\analogy\runs_id\librispeech-train-clean-100\w2v2_8\experiment_results.csv` |
| Probe run.ipynb (executed) | yes | 3.1 MB | `outputs\analogy\runs\librispeech-train-clean-100\w2v2_8\ffff_32\word_broad_10frames_fixedlen25\librispeech-train-clean-100\run.ipynb` |
| Raw wav2vec run.ipynb (executed) | yes | 3.1 MB | `outputs\analogy\runs_id\librispeech-train-clean-100\w2v2_8\run.ipynb` |
| INPUTS/inflection_results.parquet | yes | 102.6 KB | `outputs\analogy\inputs\librispeech-train-clean-100\w2v2\inflection_results.parquet` |
| INPUTS/inflection_instances.parquet | yes | 301.8 KB | `outputs\analogy\inputs\librispeech-train-clean-100\w2v2\inflection_instances.parquet` |
| INPUTS/all_cross_instances.parquet | yes | 61.8 MB | `outputs\analogy\inputs\librispeech-train-clean-100\w2v2\all_cross_instances.parquet` |
| INPUTS/most_common_allomorphs.csv | yes | 106.0 KB | `outputs\analogy\inputs\librispeech-train-clean-100\w2v2\most_common_allomorphs.csv` |
| INPUTS/false_friends.csv | yes | 12.0 KB | `outputs\analogy\inputs\librispeech-train-clean-100\w2v2\false_friends.csv` |
| INPUTS/state_space_spec.h5 | yes | 91.5 MB | `outputs\analogy\inputs\librispeech-train-clean-100\w2v2\state_space_spec.h5` |
| Best checkpoint (checkpoint-3200) | yes | 383.5 KB | `outputs\models\librispeech-train-clean-100\w2v2_8\ffff_32\word_broad_10frames_fixedlen25\checkpoint-3200` |
| Highest checkpoint (checkpoint-3800) | yes | 396.0 KB | `outputs\models\librispeech-train-clean-100\w2v2_8\ffff_32\word_broad_10frames_fixedlen25\checkpoint-3800` |
| Probe embeddings (.npy) | yes | 2.2 GB | `outputs\model_embeddings\librispeech-train-clean-100\w2v2_8\ffff_32\word_broad_10frames_fixedlen25\librispeech-train-clean-100.npy` |
| Raw layer-8 states (HDD) | yes | 51.9 GB | `outputs\hidden_states\w2v2_8\librispeech-train-clean-100.h5` |
| Raw layer-8 states (SSD copy) | yes | 51.9 GB | `F:\morph2vec_hot\outputs\hidden_states\w2v2_8\librispeech-train-clean-100.h5` |
| POS counts | yes | 771.5 KB | `outputs\pos_counts\librispeech-train-clean-100.pkl` |
| Paper PDF | yes | 792.0 KB | `Project-Research-Paper_Morpho.pdf` |
| Project_Feasibility.md | yes | 60.0 KB | `CAS784\Project_Feasibility\Project_Feasibility.md` |

- Checkpoints: checkpoint-3000, checkpoint-3200, checkpoint-3400, checkpoint-3600, checkpoint-3800.
- `checkpoint-3800/trainer_state.json`: 400 log entries, 19 evaluations; `best_model_checkpoint` = **checkpoint-3200** (eval loss 0.0564, eval_mAP 0.832); highest eval_mAP 0.836 at step 3800.
- `common.BEST_CHECKPOINT` = checkpoint-3200: matches.

## 2. Analogy results

### Word

- 108,060 trials, 120 experiments.
- 30 experiments with `VBD` in the name: `FC-D_IH D-from_VBD-D`, `FC-D_IH D-from_VBD-IH D`, `FC-D_IH D-from_VBD-T`, `FC-D_T-from_VBD-D`, `FC-D_T-from_VBD-IH D`, `FC-D_T-from_VBD-T`, `FC-T_IH D-from_VBD-D`, `FC-T_IH D-from_VBD-IH D`, `FC-T_IH D-from_VBD-T`, `VBD-FF-D-to-VBD-FF-IH D`, `VBD-FF-D-to-VBD-FF-T`, `VBD-FF-D-to-VBD_D`, `VBD-FF-D-to-VBD_IH D`, `VBD-FF-D-to-VBD_T`, `VBD-FF-IH D-to-VBD-FF-T`, `VBD-FF-IH D-to-VBD_D`, `VBD-FF-IH D-to-VBD_IH D`, `VBD-FF-IH D-to-VBD_T`, `VBD-FF-T-to-VBD_D`, `VBD-FF-T-to-VBD_IH D`, `VBD-FF-T-to-VBD_T`, `VBD_D-to-VBD-FF-D`, `VBD_D-to-VBD-FF-IH D`, `VBD_D-to-VBD-FF-T`, `VBD_IH D-to-VBD-FF-D`, `VBD_IH D-to-VBD-FF-IH D`, `VBD_IH D-to-VBD-FF-T`, `VBD_T-to-VBD-FF-D`, `VBD_T-to-VBD-FF-IH D`, `VBD_T-to-VBD-FF-T`.
- `basic` and `regular` also contain VBD groups (rows below).

| experiment | group | trials | source_items | target_items | mean_rank | median_rank |
|---|---|---|---|---|---|---|
| FC-D_IH D-from_VBD-D | nan | 1000 | 413 | 4 | 5.8 | 0.0 |
| FC-D_IH D-from_VBD-IH D | nan | 780 | 195 | 4 | 51.7 | 1.0 |
| FC-D_IH D-from_VBD-T | nan | 684 | 171 | 4 | 5.4 | 0.0 |
| FC-D_T-from_VBD-D | nan | 1000 | 392 | 75 | 439.8 | 1.0 |
| FC-D_T-from_VBD-IH D | nan | 1000 | 195 | 75 | 973.0 | 10.0 |
| FC-D_T-from_VBD-T | nan | 1000 | 170 | 75 | 519.3 | 3.0 |
| FC-T_IH D-from_VBD-D | nan | 1000 | 401 | 7 | 4514.6 | 17.0 |
| FC-T_IH D-from_VBD-IH D | nan | 1000 | 195 | 7 | 7227.7 | 56.0 |
| FC-T_IH D-from_VBD-T | nan | 1000 | 171 | 7 | 5651.8 | 18.0 |
| VBD-FF-D-to-VBD-FF-IH D | nan | 296 | 36 | 8 | 3035.6 | 405.5 |
| VBD-FF-D-to-VBD-FF-T | nan | 1000 | 36 | 76 | 1506.6 | 128.5 |
| VBD-FF-D-to-VBD_D | nan | 1000 | 36 | 383 | 173.4 | 1.0 |
| VBD-FF-D-to-VBD_IH D | nan | 1000 | 36 | 191 | 108.9 | 1.0 |
| VBD-FF-D-to-VBD_T | nan | 1000 | 36 | 165 | 412.3 | 3.0 |
| VBD-FF-IH D-to-VBD-FF-T | nan | 608 | 8 | 76 | 3063.3 | 310.5 |
| VBD-FF-IH D-to-VBD_D | nan | 1000 | 8 | 399 | 2149.7 | 56.0 |
| VBD-FF-IH D-to-VBD_IH D | nan | 1000 | 8 | 192 | 625.6 | 3.0 |
| VBD-FF-IH D-to-VBD_T | nan | 1000 | 8 | 165 | 2381.9 | 186.5 |
| VBD-FF-T-to-VBD_D | nan | 1000 | 76 | 388 | 1312.3 | 29.0 |
| VBD-FF-T-to-VBD_IH D | nan | 1000 | 76 | 191 | 1176.4 | 35.5 |
| VBD-FF-T-to-VBD_T | nan | 1000 | 76 | 164 | 830.0 | 23.0 |
| VBD_D-to-VBD-FF-D | nan | 1000 | 391 | 36 | 642.1 | 2.0 |
| VBD_D-to-VBD-FF-IH D | nan | 1000 | 402 | 8 | 4532.4 | 367.0 |
| VBD_D-to-VBD-FF-T | nan | 1000 | 389 | 76 | 1983.6 | 109.5 |
| VBD_IH D-to-VBD-FF-D | nan | 1000 | 191 | 36 | 668.6 | 14.0 |
| VBD_IH D-to-VBD-FF-IH D | nan | 1000 | 192 | 8 | 3399.6 | 141.0 |
| VBD_IH D-to-VBD-FF-T | nan | 1000 | 191 | 76 | 2674.8 | 236.5 |
| VBD_T-to-VBD-FF-D | nan | 1000 | 164 | 36 | 702.9 | 4.5 |
| VBD_T-to-VBD-FF-IH D | nan | 1000 | 165 | 8 | 5484.0 | 610.5 |
| VBD_T-to-VBD-FF-T | nan | 1000 | 164 | 76 | 1980.9 | 127.5 |
| basic | ('VBD',) | 1000 | 627 | 623 | 254.6 | 0.0 |
| regular | ('VBD', np.False_) | 1000 | 110 | 110 | 2082.2 | 48.5 |
| regular | ('VBD', np.True_) | 1000 | 558 | 579 | 87.4 | 0.0 |

### Wav2Vec

- 108,060 trials, 120 experiments.
- 30 experiments with `VBD` in the name: `FC-D_IH D-from_VBD-D`, `FC-D_IH D-from_VBD-IH D`, `FC-D_IH D-from_VBD-T`, `FC-D_T-from_VBD-D`, `FC-D_T-from_VBD-IH D`, `FC-D_T-from_VBD-T`, `FC-T_IH D-from_VBD-D`, `FC-T_IH D-from_VBD-IH D`, `FC-T_IH D-from_VBD-T`, `VBD-FF-D-to-VBD-FF-IH D`, `VBD-FF-D-to-VBD-FF-T`, `VBD-FF-D-to-VBD_D`, `VBD-FF-D-to-VBD_IH D`, `VBD-FF-D-to-VBD_T`, `VBD-FF-IH D-to-VBD-FF-T`, `VBD-FF-IH D-to-VBD_D`, `VBD-FF-IH D-to-VBD_IH D`, `VBD-FF-IH D-to-VBD_T`, `VBD-FF-T-to-VBD_D`, `VBD-FF-T-to-VBD_IH D`, `VBD-FF-T-to-VBD_T`, `VBD_D-to-VBD-FF-D`, `VBD_D-to-VBD-FF-IH D`, `VBD_D-to-VBD-FF-T`, `VBD_IH D-to-VBD-FF-D`, `VBD_IH D-to-VBD-FF-IH D`, `VBD_IH D-to-VBD-FF-T`, `VBD_T-to-VBD-FF-D`, `VBD_T-to-VBD-FF-IH D`, `VBD_T-to-VBD-FF-T`.
- `basic` and `regular` also contain VBD groups (rows below).

| experiment | group | trials | source_items | target_items | mean_rank | median_rank |
|---|---|---|---|---|---|---|
| FC-D_IH D-from_VBD-D | nan | 1000 | 413 | 4 | 3.7 | 1.0 |
| FC-D_IH D-from_VBD-IH D | nan | 780 | 195 | 4 | 13.5 | 1.0 |
| FC-D_IH D-from_VBD-T | nan | 684 | 171 | 4 | 11.6 | 0.0 |
| FC-D_T-from_VBD-D | nan | 1000 | 392 | 75 | 1043.2 | 6.0 |
| FC-D_T-from_VBD-IH D | nan | 1000 | 195 | 75 | 1441.8 | 8.0 |
| FC-D_T-from_VBD-T | nan | 1000 | 170 | 75 | 1288.6 | 6.0 |
| FC-T_IH D-from_VBD-D | nan | 1000 | 401 | 7 | 170.3 | 15.0 |
| FC-T_IH D-from_VBD-IH D | nan | 1000 | 195 | 7 | 599.6 | 20.0 |
| FC-T_IH D-from_VBD-T | nan | 1000 | 171 | 7 | 260.2 | 15.0 |
| VBD-FF-D-to-VBD-FF-IH D | nan | 296 | 36 | 8 | 2110.2 | 25.0 |
| VBD-FF-D-to-VBD-FF-T | nan | 1000 | 36 | 76 | 2365.5 | 49.0 |
| VBD-FF-D-to-VBD_D | nan | 1000 | 36 | 383 | 19.9 | 0.0 |
| VBD-FF-D-to-VBD_IH D | nan | 1000 | 36 | 191 | 30.2 | 0.0 |
| VBD-FF-D-to-VBD_T | nan | 1000 | 36 | 165 | 55.0 | 0.0 |
| VBD-FF-IH D-to-VBD-FF-T | nan | 608 | 8 | 76 | 3574.9 | 101.5 |
| VBD-FF-IH D-to-VBD_D | nan | 1000 | 8 | 399 | 98.7 | 0.0 |
| VBD-FF-IH D-to-VBD_IH D | nan | 1000 | 8 | 192 | 142.6 | 0.0 |
| VBD-FF-IH D-to-VBD_T | nan | 1000 | 8 | 165 | 201.3 | 1.0 |
| VBD-FF-T-to-VBD_D | nan | 1000 | 76 | 388 | 58.6 | 0.0 |
| VBD-FF-T-to-VBD_IH D | nan | 1000 | 76 | 191 | 141.6 | 2.0 |
| VBD-FF-T-to-VBD_T | nan | 1000 | 76 | 164 | 59.5 | 0.0 |
| VBD_D-to-VBD-FF-D | nan | 1000 | 391 | 36 | 1804.9 | 4.0 |
| VBD_D-to-VBD-FF-IH D | nan | 1000 | 402 | 8 | 4014.0 | 45.5 |
| VBD_D-to-VBD-FF-T | nan | 1000 | 389 | 76 | 3213.9 | 42.5 |
| VBD_IH D-to-VBD-FF-D | nan | 1000 | 191 | 36 | 808.2 | 7.0 |
| VBD_IH D-to-VBD-FF-IH D | nan | 1000 | 192 | 8 | 1646.8 | 24.5 |
| VBD_IH D-to-VBD-FF-T | nan | 1000 | 191 | 76 | 3327.0 | 67.5 |
| VBD_T-to-VBD-FF-D | nan | 1000 | 164 | 36 | 2024.4 | 8.0 |
| VBD_T-to-VBD-FF-IH D | nan | 1000 | 165 | 8 | 4865.5 | 53.5 |
| VBD_T-to-VBD-FF-T | nan | 1000 | 164 | 76 | 3079.2 | 36.0 |
| basic | ('VBD',) | 1000 | 627 | 623 | 306.9 | 0.0 |
| regular | ('VBD', np.False_) | 1000 | 110 | 110 | 2100.2 | 31.5 |
| regular | ('VBD', np.True_) | 1000 | 558 | 579 | 50.4 | 0.0 |

## 3. Analogy token universe

`INPUTS/state_space_spec.h5`: **334,008** word tokens over **32,046** labels (expected 334,008 / 32,046: OK); `total_num_frames` = 18,087,600.

## 4. VBD pairs (`inflection_results.parquet`)

All counts are over the analogy token universe (≤ 100 tokens per word type).
`prepare_inputs.ipynb` (cell 11) only keeps labels with **more than 5 tokens**, so every pair has ≥ 6 tokens of both forms by construction; the informative numbers are the token distribution and the ≥ 20 count.

| class | pairs | distinct_bases | base_tokens_min | base_tokens_median | base_tokens_max | past_tokens_min | past_tokens_median | past_tokens_max | pairs_ge20_both |
|---|---|---|---|---|---|---|---|---|---|
| irregular | 112 | 110 | 7 | 68 | 100 | 6 | 71 | 100 | 84 |
| regular | 837 | 833 | 6 | 24 | 100 | 6 | 19 | 100 | 299 |

- Irregular pairs: **112** over 110 bases. After removing modals (4: could, might, should, would), suppletives (3), known homographs (3) and bases with more than one past (6 pairs: be, hang, light, spring): **98** pairs, 71 of them with ≥ 20 tokens of both forms.
- Regular pairs whose base is on the authors' `exclude_vbd_base` list: 14 (address, conduct, contract, dream, dream, escort, increase, insult, leap, permit, protest, record, reject, subject).

**40 most frequent irregular pairs** (by the smaller of the two token counts, capped at 100):

| base | inflected | base_tokens | past_tokens | n_pasts_for_base |
|---|---|---|---|---|
| speak | spoke | 100 | 100 | 1 |
| meet | met | 100 | 100 | 1 |
| run | ran | 100 | 100 | 1 |
| lie | lay | 100 | 100 | 1 |
| shall | should | 100 | 100 | 1 |
| get | got | 100 | 100 | 1 |
| will | would | 100 | 100 | 1 |
| keep | kept | 100 | 100 | 1 |
| can | could | 100 | 100 | 1 |
| be | were | 100 | 100 | 2 |
| be | was | 100 | 100 | 2 |
| fall | fell | 100 | 100 | 1 |
| take | took | 100 | 100 | 1 |
| understand | understood | 100 | 100 | 1 |
| send | sent | 100 | 100 | 1 |
| sit | sat | 100 | 100 | 1 |
| make | made | 100 | 100 | 1 |
| stand | stood | 100 | 100 | 1 |
| tell | told | 100 | 100 | 1 |
| find | found | 100 | 100 | 1 |
| know | knew | 100 | 100 | 1 |
| have | had | 100 | 100 | 1 |
| bring | brought | 100 | 100 | 1 |
| leave | left | 100 | 100 | 1 |
| think | thought | 100 | 100 | 1 |
| give | gave | 100 | 100 | 1 |
| see | saw | 100 | 100 | 1 |
| hold | held | 100 | 100 | 1 |
| come | came | 100 | 100 | 1 |
| become | became | 100 | 100 | 1 |
| may | might | 100 | 100 | 1 |
| go | went | 100 | 100 | 1 |
| feel | felt | 100 | 100 | 1 |
| do | did | 100 | 100 | 1 |
| begin | began | 93 | 100 | 1 |
| lose | lost | 91 | 100 | 1 |
| catch | caught | 82 | 100 | 1 |
| write | wrote | 93 | 82 | 1 |
| break | broke | 78 | 100 | 1 |
| rise | rose | 76 | 100 | 1 |
