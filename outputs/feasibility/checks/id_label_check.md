# ɪd allomorph label check (step 2)

**Decision: `AH D` is common.** 64 of 230 ɪd verbs (28%) have `AH D` as their most common ending, and 2,102 of 6,066 ɪd tokens of regular past forms (35%) are spelled `AH D`. Per the plan: the pipeline is **not** changed before Friday; all summaries map `IH D` and `AH D` to ɪd (and `IH Z` / `AH Z` to ɪz) via `common.ALLOMORPH_CANON`; this is a known issue, and fixing `guess_past_allomorph` plus regenerating the stimuli is a P2 item (needs an analogy rerun).

## 1. Most common allomorph per base (`most_common_allomorphs.csv`)

| allomorph | VBD | NNS | VBZ |
|---|---|---|---|
| Z | 0 | 728 | 407 |
| S | 0 | 290 | 211 |
| D | 427 | 0 | 0 |
| IH Z | 0 | 101 | 82 |
| T | 166 | 0 | 0 |
| IH D | 166 | 0 | 0 |
| AH D | 64 | 0 | 0 |
| AH Z | 0 | 23 | 13 |
| UW | 9 | 0 | 0 |
| V Z | 0 | 7 | 0 |
| EH P T | 6 | 0 | 0 |
| UH D | 6 | 0 | 0 |
| IH N | 0 | 5 | 0 |
| EH D | 5 | 0 | 0 |
| AH T S | 0 | 3 | 0 |
| IH T S | 0 | 2 | 1 |
| AH N T S | 0 | 0 | 1 |
| IH K EY T S | 0 | 0 | 1 |
| AO R Z | 0 | 0 | 1 |

## 2. Token level: regular VBD, observed vs. expected ending

26,420 past-form tokens of 837 regular pairs. Expected = `guess_past_allomorph()` of the base's most common pronunciation (from the state-space cuts).

| expected (from base) | T | D | IH D | AH D | other |
|---|---|---|---|---|---|
| D | 36 | 14038 | 20 | 14 | 581 |
| IH D | 0 | 0 | 3944 | 2027 | 78 |
| T | 5340 | 228 | 0 | 61 | 53 |

Bases ending in T or D only:

| base final phone | AH D | IH D | other |
|---|---|---|---|
| D | 520 | 1296 | 7 |
| T | 1507 | 2648 | 71 |

Per ɪd pair (base ends in T/D), which spellings its tokens use:

| tokens spelled | pairs |
|---|---|
| both | 138 |
| IH D only | 52 |
| AH D only | 35 |
| neither | 4 |

The aligner's dictionary lists both pronunciations for many -ed words, so the split is per token, not per verb.

## 3. Forced-choice stimuli (`all_cross_instances.parquet`)

| FC set | items | bases |
|---|---|---|
| FC-D_IH D | 4 | 4 |
| FC-D_T | 88 | 75 |
| FC-T_IH D | 9 | 7 |
| FC-Z_S | 54 | 43 |

Approximate count of candidate stems (before the frequency filter), to show what the `IH D`-only search misses:

| pair | stems with base, base+{a}, base+{b} attested |
|---|---|
| D / T | 144 |
| D / IH D | 8 |
| T / IH D | 9 |
| D / AH D | 8 |
| T / AH D | 10 |

Including `AH D` would roughly double the ɪd forced-choice candidates, but they would stay tiny either way.

## 4. VBD false friends (`false_friends.csv`)

| post_divergence | items |
|---|---|
| T | 77 |
| D | 38 |
| IH D | 8 |

False friends were generated only for the three most common VBD endings (D, IH D, T), so there are no `AH D` false friends.

## Affected downstream sets

- **Allomorph-specific experiments** (`run.ipynb` cells 21–24) select tokens with `post_divergence == 'IH D'`. Their ɪd sources therefore use only the `IH D` tokens: 35 ɪd pairs whose tokens are all `AH D` are missing entirely, and pairs with both spellings contribute only part of their tokens.
- **Forced choice:** `FC-D_IH D` has 4 items and `FC-T_IH D` 9, vs 88 for `FC-D_T`. Section 7 shows D/T only.
- **False friends:** VBD-FF-IH D has 8 items; `false_friend_results()` skips experiments with < 10 bases, so VBD-FF means [t] and [d] only.
- **`false_friends.csv` strong flags:** computed with `guess_past_allomorph` → `IH D`, consistent with the `IH D` items present; the problem is the missing `AH D` items, not wrong flags.
- **`regular` experiment (Figure 6):** fixable post hoc, since the allomorph is merged in from `most_common_allomorphs.csv`. Probe trials per cell, `('VBD', True)` group, authors' base exclusions:

| allomorph_from | allomorph_to | authors (raw labels) | AH D mapped |
|---|---|---|---|
| D | D | 235 | 235 |
| D | IH D | 94 | 136 |
| D | T | 96 | 96 |
| IH D | D | 93 | 133 |
| IH D | IH D | 32 | 67 |
| IH D | T | 37 | 61 |
| T | D | 101 | 101 |
| T | IH D | 32 | 53 |
| T | T | 26 | 26 |

- **-s side:** NNS has `AH Z` for 23 bases and VBZ for 13. The unambiguous Fig. 3/4 experiments use only Z, S and IH Z, so the reproduction figures are unaffected; `AH Z` bases are just absent from them.