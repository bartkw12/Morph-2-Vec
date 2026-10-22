# Annotation guide: past-tense items

These files are for the linguistic annotation of the past-tense verbs in our data. No code is needed: open them in Excel.
**The yellow columns are the ones to fill in**; everything else was filled in automatically.
A first pass of **2–3 hours on the top ~150 rows of the first file is enough for now.**

## Background in two sentences

The model turns each spoken word into a point in a 32-dimensional space. We test whether "make it past tense"
is one consistent direction in that space (*walk → walked* ≈ *play → played* ≈ *want → wanted*), and how
semi-regular and irregular verbs relate to it. The annotated classes decide which verbs are grouped together in that test.

## The files

### 1. `vbd_pairs_for_annotation.xlsx` (main task)

There is one row per verb pair found in the audio: 949 pairs, irregulars first, then by how common the past form is.

| Column | Meaning |
|---|---|
| base, past | the two word forms |
| base_tokens, past_tokens | how many spoken examples the model sees (capped at 100) |
| base_corpus_count, past_corpus_count | how often the word occurs in the whole 100-hour corpus |
| base_pron / past_pron | the pronunciation the speech aligner used most often, in ARPAbet (the CMU dictionary alphabet) |
| base_ipa / past_ipa | the same, converted to IPA. Stress marks were removed, so **ʌ also stands for schwa** |
| ending | what the past form adds after the point where it stops sounding like the base |
| share_ending_AH_D | for -ed syllables: the share of examples the aligner wrote as [ʌd] rather than [ɪd] (both mean ɪd) |
| auto_regular | the program's spelling-based guess: does the past end in -ed/-d/-t added to the base? |
| auto_class, auto_reason | an automatic first guess at the class (below), and why. `check` means "please look at this one" |
| authors_excluded | the original paper's authors left this verb out (stress-shift verbs like *record*, or odd cases like *do/did*) |
| past_sounds_like | other words in the data with the same pronunciation as the past form (*made* ~ *maid*). These make the test harder for that verb |
| n_pasts_for_base | more than 1 means the program found two past forms (*spring → sprang, sprung*) |
| **class** | the class, from the dropdown |
| **sound_change_family** | free text, e.g. `sing-sang`, `keep-kept`, `bring-brought`; use the same label for verbs that change the same way |
| **exclude** | `yes` if the pair should not be used at all (not really a past tense, a different word, an alignment error) |
| **notes** | anything else |

**Classes**, with two examples each:

| class | meaning | examples |
|---|---|---|
| regular_t | regular, ending pronounced [t] | walked, missed |
| regular_d | regular, ending pronounced [d] | played, called |
| regular_id | regular, ending pronounced [ɪd] | wanted, needed |
| semi_regular | adds [t]/[d] *and* changes the stem vowel | kept, slept |
| devoicing | final [d] of the base becomes [t] | sent, built |
| ablaut | vowel change only, no added ending | sang, took |
| suppletive | a different root | went, was |
| other | anything that fits none of these (*brought*? *made*?), with a note | — |
| exclude | not a usable past-tense pair | could (from *can*), wound (from *wind*) |

### 2. `vbd_false_friends_for_review.xlsx`

These are word pairs that *sound* like a verb and its past but aren't one, e.g. *bold* ~ *bowl* + [d], *wild* ~ *wile* + [d], *plant* ~ *plan* + [t].
The model's -s results in the original paper depend on such "false friends", so we need a clean list for -ed.
- `pipeline_strong` / `authors_strong_override`: whether the added sound is the one the regular rule would predict (*cold* = *coal* + [d] after a voiced sound: yes).
- `in_analysis`: whether the pair is currently used. The [ɪd] set is too small to use (8 pairs).
- **keep**: `yes`/`no`. Is this a genuine look-alike, i.e. not secretly a real inflection and not a name or foreign word? **notes**: why.

### 3. `forced_choice_items_vbd.xlsx`

These are triples where both a [d] and a [t] (or [ɪd]) version of a base exist as words, e.g. *coal → cold / colt*.
The test asks which one the model prefers when we apply a past-tense change to the base.
The `consistent_form` is the one the regular voicing rule predicts.
- Please check the pronunciations and say whether the triple makes sense (**ok**: yes/no/unsure).
- Rows with `form_1` / `form_2` filled are cases where neither form matches the rule's prediction.

## Hypotheses

`hypotheses_draft.md` has the table of predictions per theory. Please refine the wording and the predictions,
especially for semi-regular and devoicing verbs.

Please don't rename the columns: the scripts read the completed files back by column name.
