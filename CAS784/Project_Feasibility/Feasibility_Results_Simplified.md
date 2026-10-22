# Feasibility Results, Explained Simply

**CAS 784, Group 3.** A plain-language guide to the feasibility demo: what we built, what we ran, what every figure and table shows, and what it means.

This guide is written for someone who is not a machine-learning specialist. It explains the same material as `Project_Feasibility.md`, the feasibility notebook (`notebooks/feasibility/feasibility_demo.ipynb` in the repo) and the workshop slide brief, but more slowly and with pictures. Every number here comes from those sources. Nothing was re-run to write this guide.

**How to use it.**
- If you only have five minutes, read [Section 1](#1-the-short-version) and [Section 7](#7-what-it-all-means).
- When a word is unfamiliar, look it up in the [word bank](#2-word-bank).
- [Section 4](#4-the-training-pipeline-step-by-step) answers "what is layer 8, and why are there 12 layers?"
- [Section 6](#6-the-results-figure-by-figure) goes through every figure and table.

---

## Contents

1. [The short version](#1-the-short-version)
2. [Word bank](#2-word-bank)
3. [What question are we asking?](#3-what-question-are-we-asking)
4. [The training pipeline, step by step](#4-the-training-pipeline-step-by-step)
5. [Every run we did, and why](#5-every-run-we-did-and-why)
6. [The results, figure by figure](#6-the-results-figure-by-figure)
7. [What it all means](#7-what-it-all-means)
8. [What is not done yet, and open questions](#8-what-is-not-done-yet-and-open-questions)
9. [Where everything lives](#9-where-everything-lives)

---

## 1. The short version

- **What the paper did.** Gauthier et al. (2025) took a speech model called **wav2vec 2.0**, which learned from raw audio with no labels at all. They looked inside it at **layer 8** (one of its 12 internal processing stages). They trained a tiny add-on, the **word probe**, whose only job is to tell spoken words apart. Then they tested whether "adding -s" (*cat → cats*, *run → runs*) is the **same step** for every word inside the model, using word analogies like *shirt : shirts :: cheese : ?*.
- **What they found.** In the probe's space, "add -s" is largely one shared step. It works almost the same for plural nouns and for verbs, and for all three ways of pronouncing -s: [s] in *cats*, [z] in *dogs*, [ɪz] in *dishes*. The raw wav2vec numbers are much more sensitive to those differences.
- **What we did.** We got the authors' code running on one Windows PC. That took about 20 fixes. We trained our own word probe, because the authors did not release theirs, and re-ran the paper's main experiments.
- **Did it reproduce?** Yes, for the main results. Our analogy word set has **exactly the same number of words as the paper (334,008)**. All 8 cells of the paper's Fig. 3 and 41 of the 50 cells of its Fig. 4 land within a factor of 2 of the paper's numbers. The "false friends" result (paper Fig. 5) reproduces only in direction, not in size.
- **Our new part: the past tense.** The paper only studied -s. Our runs already contained past-tense analogies (*walk : walked :: play : ?*) that nobody had summarized. The first look says that the **regular** past tense behaves a lot like -s: one mostly shared step across [t] *walked*, [d] *played* and [ɪd] *wanted*. **Irregular** verbs (*sang*, *took*) mostly do not share a step with each other. That is expected under every theory, so the more informative tests are still to come.
- **The big lesson.** The pipeline works and the measuring tool is trustworthy. The biggest risks were computer memory and disk space, not the science. For the past tense, how the verbs are **labelled** (regular, irregular, pronunciation) matters as much as the model, so the verbs need a careful classification by hand.

---

## 2. Word bank

The terms are grouped by topic. Terms in **bold** in the explanations are usually defined here.

### Linguistics

| Term | Plain meaning |
|---|---|
| **Morphology** | How words are built from meaningful pieces. *walk + -ed = walked*. |
| **Morpheme** | The smallest piece of a word that carries meaning. *-s* (plural) and *-ed* (past) are morphemes. |
| **Inflection** | Changing a word's form for grammar without making a new word: *dog → dogs*, *walk → walked*. |
| **Base form** | The plain word before the inflection is added: *dog*, *walk*. |
| **Phonology** | The sound system of a language: which sounds exist and the rules for how they combine. |
| **Morpho-phonology** | Where word-building and sound rules meet. For example, the sound of *-s* depends on the sound before it. |
| **Allomorph** | One of the different pronunciations of the same morpheme. Plural *-s* has three: [s] *cats*, [z] *dogs*, [ɪz] *dishes*. Past *-ed* also has three: [t] *walked*, [d] *played*, [ɪd] *wanted*. |
| **Voiced / voiceless** | Whether your vocal cords buzz when you make the sound. [z], [d], [g] are voiced; [s], [t], [k] are voiceless. The ending agrees with the sound before it: voiceless + voiceless (*cats* [s]), voiced + voiced (*dogs* [z]). |
| **Sibilant** | A hissing sound: [s], [z], [ʃ] (*sh*), [ʒ], [tʃ] (*ch*), [dʒ] (*j*). After a sibilant, -s becomes [ɪz] (*dishes*), because two hisses in a row are hard to say. |
| **IPA / ARPAbet** | Two ways of writing sounds. IPA uses symbols in square brackets like [ɪz]. ARPAbet is the computer-friendly version the aligner uses: `IH Z`, `IH D`, `AH D`. |
| **NNS** | Tag for a **plural noun** (*dogs*). |
| **VBZ** | Tag for a **3rd-person-singular present verb** (*he runs*). |
| **VBD** | Tag for a **past-tense verb** (*walked*, *sang*). |
| **Regular verb** | A verb that forms its past by adding *-ed*: *walk → walked*. |
| **Irregular verb** | A verb that changes its stem instead: *sing → sang*, *take → took*, *go → went*. |
| **Semi-regular verb** | A mix: a stem change plus a *-t/-d* ending: *keep → kept*, *sleep → slept*. |
| **False friend** | A pair of words that *looks* like an inflection but is not one: *beside / besides*, *though / those*, *bold / bowl*. The final sound matches, but there is no grammar relationship. |
| **Homophone** | Two words that sound the same: *rowed / road*, *missed / mist*, *knows / nose*. They confuse a model that only hears sound. |
| **Dual-route theory** | The idea (Pinker, 1991) that the brain uses a **rule** for regular verbs and **memory** for irregular ones. |
| **Single-route theory** | The idea (Rumelhart & McClelland, 1986) that one learned network handles all verbs, with no explicit rules. |

### Speech data

| Term | Plain meaning |
|---|---|
| **LibriSpeech** | A free collection of about 1,000 hours of people reading audiobooks aloud, with transcripts. |
| **train-clean-100** | A 100-hour slice of LibriSpeech (28,538 recordings). This is the main data for the paper and for us. |
| **test-clean** | A separate 5.4-hour slice used to check things. wav2vec never trained on it. |
| **Forced alignment** | Automatic timestamps that say exactly when each word and each speech sound starts and stops in a recording. We use public ones (Lugosch, 2019). |
| **Frame** | A 20-millisecond slice of audio. wav2vec produces one output per frame, so 1 second of speech = about 50 frames. |
| **Token vs. type** | A **type** is a word in the dictionary ("cat"). A **token** is one time someone says it. "cat" said 40 times = 1 type, 40 tokens. |

### Machine learning

| Term | Plain meaning |
|---|---|
| **Model** | A big mathematical function with adjustable numbers (weights) that turns an input into an output. |
| **Weights / parameters** | The adjustable numbers inside a model. "Training" means adjusting them. |
| **Self-supervised learning** | Learning from raw data with no human labels, by inventing a puzzle from the data itself. wav2vec hides bits of audio and learns to guess them. |
| **S3M** | "Self-supervised speech model". wav2vec 2.0, HuBERT and WavLM are examples. |
| **wav2vec 2.0 (base)** | The speech model we study, made by Facebook/Meta. It was trained on 960 hours of LibriSpeech audio without transcripts. "Base" is the smaller version, with 12 layers. |
| **Pretrained / frozen** | wav2vec was already trained by Meta. We never change its weights; we only read its outputs. |
| **Layer** | One processing stage inside the model. The output of one layer is the input to the next. wav2vec-base has 12 stacked layers (see [Section 4.2](#42-step-2-run-wav2vec-and-keep-layer-8)). |
| **Transformer** | The kind of layer wav2vec uses. Each layer looks at the whole recording at once and mixes information across time. |
| **Hidden state** | The list of numbers a layer outputs for one frame. For wav2vec-base, it is 768 numbers. |
| **Vector** | A list of numbers, which can be thought of as a point or an arrow in space. A 768-number vector is a point in 768-dimensional space. |
| **Dimension (768-d, 32-d)** | How many numbers are in the vector. |
| **Embedding** | A vector that stands for something, here one spoken word. Similar things should have nearby embeddings. |
| **Representation / representation space** | The "map" where a model places its embeddings. "Wav2Vec space" = raw 768-number vectors; "Word space" = the probe's 32-number vectors. |
| **Probe** | A small, simple model trained on top of a big model's hidden states, to find out what information is in them. |
| **Linear** | Only multiplying and adding: no curves, no hidden layers. A linear probe cannot invent new information; it can only re-weight what is already there. |
| **Contrastive learning** | Training by comparison: pull examples of the same thing together, push examples of different things apart. |
| **Positive / negative example** | In contrastive learning, a positive is another frame of the *same* word; a negative is a frame of a *different* word. |
| **Hinge loss / margin** | The training penalty. It is zero once the positive is closer than the negative by at least a safety gap, the "margin" (0.3759 here). |
| **Loss** | A number that says how badly the model is doing. Training tries to make it go down. |
| **Training step** | One small update of the weights, after looking at one batch. |
| **Batch** | The handful of examples looked at in one step (32 here). |
| **Learning rate / weight decay** | Settings that control how big each update is, and how strongly weights are kept small. We used the paper's values (0.00108 and 0.00607). |
| **Hyperparameters** | Settings chosen before training (learning rate, margin, size). Not learned. |
| **Evaluation (eval) set** | Examples held back from training, used to check progress honestly. |
| **Early stopping** | Stop training when the eval score has not improved for a while (3 checks here), then keep the best version. |
| **Checkpoint** | A saved copy of the model at some training step. Our best one is `checkpoint-3200`. |
| **mAP (mean average precision)** | A 0-to-1 score of how well you can tell which word a frame belongs to from the probe's output. 1.0 = perfect. Ours went from 0.74 to 0.83. |
| **NaN** | "Not a number": what a computer gives you for an impossible calculation such as 0 ÷ 0. One of these silently broke our first training run. |
| **Cosine similarity / distance** | A measure of how much two vectors point the same way. Cosine similarity 1 = same direction. The distance is 1 minus that. |
| **Mean-pooling** | Averaging many vectors into one. We average all the frames of a spoken word into one word embedding. |
| **GPU** | The graphics card, which does the heavy maths fast. Ours is an RTX 3080 Ti with 12 GB of memory. |
| **float32 / float64** | How precisely numbers are stored. float32 is much faster on gaming GPUs and was precise enough. |
| **HDF5 (.h5), .npy, parquet** | File formats for storing big tables of numbers. |

### Terms specific to this project

| Term | Plain meaning |
|---|---|
| **"Word" (the probe model)** | Word embeddings built from the **trained probe's** 32-number output. Shown in **blue** in all figures. |
| **"Wav2Vec" (the raw model)** | Word embeddings built from **raw layer-8** 768-number states, with no probe. Shown in **orange** in all figures. |
| **Analogy** | A puzzle *a : b :: c : ?*, such as *shirt : shirts :: cheese : ?*. The model's guess is the vector *b − a + c*. |
| **Source pair / target pair** | The source pair (*shirt → shirts*) supplies the step; the target pair (*cheese → cheeses*) is where we apply it. |
| **Offset / difference vector** | The arrow from base to inflected form, *b − a*. If "add -s" is one shared step, all these arrows point the same way. |
| **Rank** | Our score. We sort all 334,008 spoken words by closeness to the guess, and record where the correct word lands. **0 = perfect** (the correct word was the closest). Lower is better. |
| **Mean rank** | The average rank over many trials. The paper's main number. |
| **Median rank** | The middle rank. Usually 0 here, because most trials are perfect. |
| **Top-1 accuracy** | The share of trials with rank 0, i.e. how often the guess is exactly right. Easier to read than the mean. |
| **Trial** | One analogy test (one source pair applied to one target pair). |
| **Experiment** | A set of up to 1,000 trials of one kind, such as "noun -s applied to verbs". There are about 120. |
| **Random-pair control** | A baseline: use the step from a *random* pair of words (*table → river*) instead of a real inflection. It tells us how good "doing nothing meaningful" looks. |
| **Chance level (94,306)** | The paper's stated random baseline. We do not yet know how it was computed (see [Section 8](#8-what-is-not-done-yet-and-open-questions)). |
| **95% confidence interval (CI)** | The error bars. A range where the true value probably lies. Ours come from "bootstrapping": re-sampling the target words many times and seeing how much the result moves. |
| **n = trials / items** | How many trials and how many different target words a number is based on. Small n means less certainty. |
| **Forced choice** | A test where the model must pick between two candidate words, such as *bays* vs. *base* for *bay*. |
| **Unambiguous noun / verb** | A word whose -s form can only be a plural noun or only a verb. *stacks* is ambiguous (a noun or a verb); *daughters* is not. The paper uses only unambiguous ones. |

---

## 3. What question are we asking?

### The paper's question

When a speech model learns to recognize words, does it also learn how English words are built? In particular, does it learn the **-s** ending?

English **-s** is a good test because:
- the same sound marks two different grammar jobs: plural nouns (*dogs*) and verbs (*runs*);
- it has three pronunciations, chosen by a sound rule ([s] after voiceless sounds, [z] after voiced ones, [ɪz] after hissing sounds);
- the same sounds also end many words that are not inflections at all (*haze*, *besides*).

So you can ask whether the model's "add -s" depends on grammar (noun vs. verb), on pronunciation ([s]/[z]/[ɪz]), or on neither.

### Our question (the extension)

We ask the same thing for the **past tense**, the classic test case in the long debate between **dual-route** theory ("a rule for regulars, memory for irregulars") and **single-route** theory ("one learned network, no rules").

1. Is the past tense encoded the same way across **[t] / [d] / [ɪd]** (*walked / played / wanted*)?
2. Is there **one general pattern**, separate patterns by sound or verb type, or is it **word by word**?
3. Does the pattern generalize to **held-out verbs** the analysis has not seen?

wav2vec is an interesting test subject because it never saw rules, letters, suffixes or even phonemes. It only heard audio. If a rule-like past tense shows up inside it, that pattern was learned from sound alone.

### Why analogies answer the question

If the past tense is a rule, then in the model's space *walk → walked* and *play → played* should be **the same step** (the same arrow). An analogy *walk : walked :: play : ?* checks exactly that. Take the *walk → walked* arrow, attach it to *play*, and see whether you land on *played*. If you land right on it (rank 0), the step is shared.

---

## 4. The training pipeline, step by step

Here is the whole chain from audio to results. Each numbered box is explained below, with what it does and why.

![The pipeline, step by step](explainer_figures/d1_pipeline.png)

There are two parallel paths. The **"Word" path** (blue) goes through the trained probe. The **"Wav2Vec" path** (orange, dashed) skips the probe and uses wav2vec's raw numbers. Comparing the two tells us what the probe, which only learned to tell words apart, keeps and throws away.

### 4.1 Step 1: get the audio and the word timestamps

- **What.** We downloaded LibriSpeech `train-clean-100` (100.5 hours, 28,538 recordings) and `test-clean` (5.4 hours), plus the forced alignments that mark when each word and sound starts and ends. A preprocessing notebook cleans this up: it converts British spellings to American, groups speech sounds into words, splits words into syllables, and normalizes the audio volume.
- **Why.** The paper uses exactly this data. Word timestamps are essential: without them we would not know which frames belong to which word.

### 4.2 Step 2: run wav2vec and keep layer 8

This is the part that answers "what does layer 8 mean?".

![Inside wav2vec 2.0: 12 layers](explainer_figures/d2_wav2vec_layers.png)

**What wav2vec does.** You feed it a raw audio waveform (16,000 numbers per second). It works in two stages:

1. A **feature encoder** chops the audio into 20 ms frames and turns each frame into a list of numbers.
2. Those lists then pass up through a **stack of 12 transformer layers**. Each layer takes the previous layer's output, looks across the whole recording, and produces a new, more processed list of **768 numbers per frame**.

**Why are there 12 layers?** That is simply how the model was designed. The "base" version of wav2vec 2.0 has 12 layers and the "large" version has 24. Stacking layers lets a model build understanding gradually, much like the visual system goes from edges to shapes to objects. Every layer outputs the same shape (768 numbers per frame), so you can read the output of **any** layer and ask what information it holds.

**What changes from layer to layer?** Roughly:
- **Early layers (1–4)** are close to the raw sound: loudness, pitch, acoustic texture.
- **Middle layers (about 5–9)** hold the most readable information about speech sounds and words.
- **Late layers (10–12)** become specialized for wav2vec's own training puzzle (guessing hidden bits of audio). General linguistic information becomes a bit harder to read out there.

**So what is "layer 8"?** It means: we take the 768 numbers that come **out of the 8th transformer layer** for every 20 ms frame, and ignore the other layers. The paper's own Fig. 1 tested all 12 layers and found that the analogies work best in the middle, peaking at **layers 8 and 9**. This matches earlier studies that found speech-sound information peaks there. So the paper ran all its detailed experiments on layer 8, and so did we.

**Why only one layer for us?** Cost. Saving one layer for all 100 hours of audio produces 18,087,600 frames × 768 numbers = **52 GB**, and each extra layer costs about 3.5 hours of computing. All 12 layers would need about **620 GB and about 2 days**. For a feasibility check, the paper's best layer is the right choice.

> **Fine print on the numbering.** In the code, "layer 8" is the setting `w2v2_8`, which reads the model's `hidden_states[8]`, the output after 8 transformer layers (`hidden_states[0]` is the feature encoder's output, before any transformer layer). The authors' plotting code for their Fig. 1 adds 1 to this index, so on that one plot the same setting would sit at "9". The paper's text and Table 9 call it the 8th layer, and we used the authors' exact configuration, so our results are directly comparable to theirs.

**Output of this step:** a 52 GB file of layer-8 numbers, one 768-number list per 20 ms frame (18.1 million frames). This took about 2 hours on our GPU.

### 4.3 Steps 3 and 5: from frames to one vector per spoken word

![From frames to one word vector](explainer_figures/d3_frames_to_word.png)

- **What.** A spoken word lasts many frames: *cats* might be 15 frames. To get **one vector per spoken word**, we average all of its frames. This is called mean-pooling, and the result is a **word embedding**.
- We do this on both paths: averaging the probe's 32-number outputs gives a "Word" embedding, and averaging the raw 768-number states gives a "Wav2Vec" embedding.
- **The word list (334,008 tokens).** To keep things manageable, each word type is capped at 100 spoken examples. This leaves **334,008 spoken word tokens of 32,046 different words**. That is our "universe" for ranking. It matches the paper's number exactly, which strongly suggests our data preparation matches theirs.
- **Why.** Analogies need one vector per word. Averaging is the paper's method (its Eq. 3), and it is simple and robust.

### 4.4 Step 4: train the word probe

![What the probe is and how it is trained](explainer_figures/d4_probe.png)

**What the probe is.** A single table of 768 × 32 = **24,576 numbers** (plus 32 more). It turns each 768-number wav2vec frame into a shorter 32-number list by multiplying and adding. It is **linear**: no hidden layers and no curves. That matters, because it means the probe cannot invent new knowledge. It can only choose which of wav2vec's existing information to keep and how to weight it.

**How it is trained (contrastive learning).**
1. Pick a random frame from some word, say *dog*.
2. Pick a **positive**: another frame from another spoken *dog*.
3. The other examples in the batch, which come from different words, are **negatives**.
4. The loss penalizes the probe unless the positive is closer (more similar) than each negative by at least the margin (0.3759).
5. Adjust the 24,576 numbers a little to reduce the penalty. That is one **training step**. Each step looks at 64 examples (two batches of 32).

The probe is told only **"same word or different word?"**. It never sees grammar labels such as "plural", "past tense" or "noun". So if a shared "add -s" step shows up in its space, the probe discovered it as a side effect of telling words apart.

**Training data.** A helper table (the "equivalence dataset") says which word every frame belongs to. Very rare words (fewer than 5 examples, or in the rarest 10%) are dropped. That leaves **11,512 different words over 13.6 million labelled frames**.

**Settings.** We used the paper's tuned settings (its Table 9): margin 0.3759, learning rate 0.00108, weight decay 0.00607, and 32 output numbers.

**Checking progress.** Every 200 steps, the probe is scored on frames it never trained on (the eval set): the eval loss (should go down) and the mAP (should go up). If the eval loss does not improve for 3 checks in a row, training stops and the best checkpoint is kept.

**Why we had to train it ourselves.** The authors did not release their trained probe. Retraining with their exact settings took about **25 minutes** on our GPU. (The paper's "12 hours per layer" was for *searching* for the best settings, which we did not need to repeat.)

**Why use a probe at all?** Raw wav2vec states hold a lot of detail that word recognition does not need: the speaker's voice, recording noise, exact pronunciation. The probe finds the slice of wav2vec that is **useful for telling words apart**. Comparing that slice with raw wav2vec shows which information word recognition keeps (for example, "a sibilant was added") and which it ignores (for example, exactly which allomorph).

### 4.5 Step 6: build the analogy tests

![Analogy as vector arithmetic, scored with a rank](explainer_figures/d5_analogy_and_rank.png)

- **Word pairs.** A notebook finds base and inflected pairs in the data (*dog / dogs*, *run / runs*, *walk / walked*, *sing / sang*), and labels each pair with:
  - its grammar type (NNS, VBZ, VBD);
  - its allomorph ([s]/[z]/[ɪz], [t]/[d]/[ɪd]);
  - whether it is regular;
  - whether the noun or verb reading is ambiguous. This uses an automatic part-of-speech tagger (spaCy), because the authors' file for it was not released.

  It also builds false-friend pairs (*beside / besides*), random pairs, and forced-choice triples (*bay → bays* or *base*?).
- **Running the analogies.** There are about 120 experiments, for example "noun [z] pairs applied to verb [s] pairs". Each runs up to 1,000 trials. In each trial:
  1. Compute the guess *b − a + c*, averaged over up to 250 random spoken examples of each word.
  2. Measure how close each of the 334,008 word tokens is to the guess (cosine distance).
  3. Remove the base word *c* itself from the list, since it is trivially close.
  4. Record the **rank** of the correct word *d*.
- This is done twice: once with **Word** (probe) embeddings and once with **Wav2Vec** (raw) embeddings.
- **Why ranks?** A rank is easy to understand: 0 means the correct word was the single closest of 334,008, and 5 means four other words were closer. It is also the paper's metric, so we can compare directly.

### 4.6 Step 7: summarize

The feasibility notebook reads the analogy results and makes Figures 1–7: heatmaps and bar charts of mean rank, top-1 accuracy and error bars. It runs in under 5 minutes on any laptop, because it reads only small exported files and never the 52 GB data.

### 4.7 Why each design choice was made

| Choice | Why |
|---|---|
| wav2vec 2.0 base | It is the paper's model, and it learned from audio only, with no rules or labels. |
| Layer 8 only | It is the paper's best layer, and each layer costs 52 GB and about 3.5 h. |
| A **linear** probe | It cannot create new knowledge, so anything it shows was already in wav2vec. |
| Trained only to tell words apart | No grammar labels, so any grammar-like structure is emergent. |
| Averaging frames per word | The paper's method; it gives one vector per spoken word. |
| Analogies + rank | They directly test "is it the same step?", and match the paper's metric. |
| Comparing Word vs. Wav2Vec | Shows what word recognition keeps and what it throws away. |
| Paper's settings, not re-tuned | Saves about 12 h per layer and keeps us comparable to the paper. |

---

## 5. Every run we did, and why

All heavy computing happened between 15 and 18 October 2026 on Bart's PC (Windows 10, RTX 3080 Ti with 12 GB, 32 GB RAM). The feasibility demo itself (the notebook and figures) used **no new GPU runs**. It summarizes results that were already on disk.

| # | Run | Purpose | What happened |
|---|---|---|---|
| 0 | Environment setup | Install Python, PyTorch with GPU support, and the other libraries; fix outdated code | It worked after several code fixes (old library versions, missing files). |
| 1 | **Smoke test** on test-clean, with the probe trained on only 1,000 frames for about 33 s | Check that every stage connects end to end, without caring about quality | Failed five times, and each failure revealed a fix (a download error, missing libraries, the C: drive filling up with temporary files). Then it succeeded. As expected, this barely-trained probe was worse than raw wav2vec (ranks 150–840). |
| 2 | **Full run on test-clean** (5.4 h of audio) | First real probe training and analogy run on a small dataset | Trained in 17.5 min (5,800 steps, best mAP 0.684). Analogies ran. The pattern matched the paper's held-out results for the probe. |
| 3 | **train-clean-100 data preparation** | Preprocess 100 h of audio, run wav2vec and save layer 8 (52 GB), build the word spans, training labels, part-of-speech counts and analogy word lists | It ran out of memory twice: first in preprocessing, then at 61% of extraction. Both were fixed. The third attempt finished, with extraction taking about 2 h. |
| 4 | Main run, attempt 1 | Train the probe on train-clean-100 and run both analogy sets | **Discarded.** The probe "trained" for only 7 minutes because of the NaN bug (see Fig. 1 below), and the raw-wav2vec analogies were heading for about 30 h because of slow float64 maths. |
| 5 | **Main run, attempt 2** | Same, with the NaN and float32 fixes | The probe trained properly: **25 min, 3,800 steps, best mAP 0.832 at step 3,200**. Probe outputs for all 18.1M frames took 31 min. **Probe analogies done in 73 min.** The raw-wav2vec analogies used too much memory and slowed to a crawl (49 of 120 experiments in 10.5 h), so we stopped them. |
| 6 | **Main run, attempt 3** | Raw-wav2vec analogies only, with a memory fix and the ability to resume | **Done in 1 h 41 min.** |
| 7 | **Feasibility demo** (no GPU) | Four scripts plus the notebook | (a) Inventory and sanity checks: every file exists, 334,008 tokens confirmed, past-tense pairs counted. (b) The [ɪd] spelling check. (c) Export of small data files for the notebook. (d) Spreadsheets for Nigel to hand-label the verbs. (e) The notebook with Figures 1–7. |

**Where the past-tense results came from.** The analogy runs in #5 and #6 already included past-tense experiments, because the authors' code generates them. The paper just never reported them. So the past-tense pilot needed no new computing; we only had to summarize existing results using the authors' own unpublished analysis code.

**Time and size for one layer**, end to end and not counting failed attempts: about **8–9 hours**. About 5 h of that is one-time setup (download, preprocessing, word lists). About 3.5 h repeats for each extra layer (extraction, probe training, probe outputs, both analogy runs). Everything takes about 115 GB of disk.

**The fixes, in brief.** The code was written for a Linux computer cluster. About 20 changes were needed. The important ones:
- A Windows-friendly runner script that replaces the Linux-only workflow and can resume after a crash.
- Saving wav2vec's output to disk as it goes, instead of holding all 52 GB in 32 GB of memory.
- The **NaN bug** (explained under Fig. 1).
- Faster float32 maths and resumable analogy runs: about 30 h became 1 h 41 min.
- Recreating a missing part-of-speech file with spaCy.

---

## 6. The results, figure by figure

### First: how to read the figures

Most figures are **heatmaps**: grids of coloured squares.

![How to read a heatmap cell](explainer_figures/d6_read_heatmap.png)

- **Rows** are where the step comes from (the source pair). **Columns** are where it is applied (the target pair).
- **Each cell** is the mean rank over up to 1,000 analogies.
- **Colour:** light/yellow = low rank = good; dark/purple/black = high rank = bad. The colour scale is **logarithmic**: each step on the scale is 10× bigger (0.1, 1, 10, 100), because the numbers range from almost 0 to over 100.
- **"ours (paper)"** means our number, with the paper's number in brackets for comparison. "–" means the paper did not report that cell.
- **Panels:** "Word" on the left is the trained probe (blue in bar charts). "Wav2Vec" on the right is raw layer-8 states (orange in bar charts).

**Why a mean rank can be below 1, like 0.88.** Most trials score 0 (perfect). A few miss by a little or a lot, and the average of many 0s and a few 3s or 50s is a small decimal. This also means **a handful of bad trials can drag a mean way up**. That is why we also show **top-1 accuracy** (how often the answer is exactly right) and medians, which are much less sensitive to a few disasters.

---

### Table 1: Sanity checks (does our setup match the paper's?)

| What | Ours | Paper |
|---|---|---|
| Audio | 100.49 h, 28,538 recordings | 100 h |
| Frames (20 ms) | 18,087,600 | not reported |
| Spoken word tokens used for ranking | **334,008** (32,046 words) | **334,008** |
| Unambiguous nouns / verbs | 776 / 59 (53 after the paper's exclusions) | 753 / 52 |
| Probe | linear 768 → 32, 24,576 weights | same |
| Probe settings | margin 0.3759, learning rate 0.00108, weight decay 0.00607 | same |
| Probe score (mAP) at the best checkpoint | 0.832 | not reported |

**What it means.** Getting **exactly** 334,008 tokens is strong evidence that our data preparation matches the authors' to the word. We have a few more nouns and verbs than the paper because the authors also removed some by hand, and their hand-removal list was not released. Several of our extra verbs are ones they say they removed, such as *knows*, which sounds like *nose*.

---

### Figure 1: Training the word probe

![Figure 1: probe training](workshop/figures/fig01_probe_training.png)

**What it shows.** Two lines over training time (the x-axis is the training step):
- **Blue (left axis): eval loss**, the penalty on frames the probe never trained on. It falls quickly from 0.094 and levels off around 0.057. Lower is better.
- **Green (right axis): eval mAP**, how well words can be told apart from the probe's output. It rises from **0.74 to 0.83**. Higher is better.
- The dotted line marks the **best checkpoint, step 3,200** (lowest eval loss 0.0564, mAP 0.832). The loss did not improve for the next 3 checks (3,400, 3,600, 3,800), so early stopping ended training there and reloaded step 3,200.

**What it means.** The probe learned what it was supposed to learn, and it learned it quickly. Both curves flatten, so more training would not change much. (mAP crept up to 0.836 at step 3,800, but the rule picks by loss; the difference is tiny.)

**The NaN bug: a lesson.** On our first try, the eval loss was **NaN** ("not a number") from the very first check. The eval set had 7,753 examples. Split into groups of 8, that leaves one lonely last example, and with nothing to compare it against, the loss formula divided zero by zero. Because NaN never counts as an "improvement", early stopping killed training after 7 minutes, and **nothing crashed**: the results just looked a bit worse. We fixed the formula to return 0 when there is nothing to compare. The lesson: always look at your training curve.

---

### Figure 2 (= the paper's Fig. 3): Noun -s vs. verb -s

![Figure 2: noun/verb transfer](workshop/figures/fig02_fig3_morphology.png)

**Question.** Does the "add -s" step learned from **nouns** work on **verbs**, and the other way round? For example, *daughter : daughters :: own : ?* takes a noun step and applies it to a verb.

**How to read it.** Rows are the source (NNS = plural nouns, VBZ = verbs with -s). Columns are the target. The diagonal cells (NNS→NNS, VBZ→VBZ) stay within one category. The off-diagonal cells cross categories.

| | Word: NNS target | Word: VBZ target | Wav2Vec: NNS target | Wav2Vec: VBZ target |
|---|---|---|---|---|
| **NNS source** | 0.88 (paper 1) | 5.4 (paper 10) | 7.9 (paper 6.7) | 58 (paper 68) |
| **VBZ source** | 3.7 (paper 2.7) | 6.3 (paper 7.9) | 25 (paper 29) | 38 (paper 37) |

**What it shows.**
- **Our numbers match the paper.** All 8 cells are within a factor of 2 of the paper's.
- **Word probe:** the noun step works on verbs (5.4) about as well as a verb step does (6.3). Crossing the grammar category barely hurts.
- **Raw Wav2Vec:** ranks are roughly 6–11× worse, and crossing categories hurts more: noun → verb is 58, vs. 38 for verb → verb.
- Nouns are easier targets than verbs in both models. The paper suggests this is because verbs have more forms that can be confused (*pleases* vs. *pleasing*).

**What it means.** After training only to recognize words, the probe uses **one shared "add -s" step for nouns and verbs**. It is not tracking grammar (plural vs. verb agreement); it is tracking the sound pattern "add a final hissing sound". That is the paper's first main claim, and we reproduced it.

> **Why these numbers differ slightly from `Project_Feasibility.md`.** That document reports 0.9 / 5.4 / 3.2 / 5.9 for the probe and 7.9 / 51 / 22 / 36 for wav2vec. The notebook removes the 6 verbs the paper also removes because they sound like other words (*knows ~ nose*, *meets ~ meats*, …). Same data; the paper's filter is applied here.

---

### Figure 3 (= the paper's Fig. 4): Does pronunciation matter? [s] / [z] / [ɪz]

![Figure 3: allomorph transfer](workshop/figures/fig03_fig4_allomorphs.png)

**Question.** Does the "add -s" step learned from [z] words (*dogs*) work on [s] words (*cats*) and [ɪz] words (*dishes*)? And across nouns and verbs?

**How to read it.** It is a 6 × 6 grid: Noun [s], Noun [z], Noun [ɪz], Verb [s], Verb [z], Verb [ɪz], as both source (rows) and target (columns). Don't try to read every number; **compare the overall brightness of the two panels**. The "Verb [ɪz]" row and column are extra cells the paper did not show ("–").

**What it shows.**
- **The Word panel is mostly light.** 18 of the 25 cells the paper reports are below rank 2. A step learned from one pronunciation works on the others.
- **The Wav2Vec panel has dark rows and columns.** For example, the "Verb [z]" source row is 35–61 everywhere, and noun → verb transfers are 11–52. Raw wav2vec cares a lot about the exact sound and category.
- **The probe has the same weak spots as the paper:** the **Verb [z] target column** (about 6–11) and the **Noun [ɪz] source row**.
- **Agreement with the paper:** 41 of 50 cells within 2× (22/25 for the probe, 19/25 for wav2vec).

**What it means.** The probe mostly ignores **how** the -s is pronounced: it has one fairly abstract "add -s" step. Raw wav2vec keeps much more detail about the specific pronunciation. That is the paper's second main claim, and it reproduced almost cell by cell for the probe. This is **exactly the test we then ran on the past tense** (Figure 6).

**Why do some wav2vec cells disagree with the paper more?** For example, Noun [s] → Noun [s] is 9.3 for us and 41 in the paper. Raw wav2vec means are driven by a few spectacular failures (a trial with rank 3,000 moves an average of 1,000 trials by 3). Which few trials fail depends on random sampling, so those means bounce around more.

#### Table: Median ranks for Figure 3

| src \ tgt | Noun [s] | Noun [z] | Noun [ɪz] | Verb [s] | Verb [z] | Verb [ɪz] |
|---|---|---|---|---|---|---|
| **Word** (all source rows) | 0 | 0 | 0 | 0–1 | 0–1 | 0–2 |
| **Wav2Vec**, noun sources | 0 | 0 | 0 | 0–2 | 2–4 | 3–9 |
| **Wav2Vec**, verb sources | 0 | 0 | 0 | 0 | 0 | 1–4 |

**What it means.** The **median** (the middle trial) is 0 in almost every cell: in most trials, both models get the exact word. The differences in the heatmaps come from how **often** and how **badly** they miss. The paper's own median figure (its Fig. 13) shows the same thing.

---

### Figure 4: How close is our reproduction, cell by cell?

![Figure 4: reproduction agreement](workshop/figures/fig04_reproduction_agreement.png)

**How to read it.** Each dot is one cell from Figure 2 or 3. Its position across is the paper's number; its position up is ours. Both axes are logarithmic.
- The **black diagonal line** = identical to the paper.
- The **grey band** = within a factor of 2 (half to double).
- Squares are Figure 2 cells, dots are Figure 3 cells; blue = Word, orange = Wav2Vec.

**What it shows.**
- Almost all dots hug the diagonal: **8 of 8** Figure 2 cells and **41 of 50** Figure 3 cells are inside the grey band.
- The typical ratio of ours to the paper's is **0.94 for the probe** and **0.89 for wav2vec**. That is close to 1, and if anything slightly better.
- The few dots outside the band are mostly **orange** (raw wav2vec), whose means depend on a few failed trials.

**What it means.** This one picture shows the reproduction worked. Any differences are the normal wobble from retraining a probe and randomly sampling trials. They do not change any conclusion.

---

### Figure 5: Real analogies vs. the random-pair control (the baseline)

![Figure 5: baselines](workshop/figures/fig05_baselines.png)

**Question.** How much better are real analogies than meaningless ones? For the **random-pair control**, the step comes from a random pair of words (*table → river*) and is applied to the same targets.

**How to read it.** The left panel shows mean rank (log scale; lower is better; error bars are 95% CIs). The right panel shows top-1 accuracy (higher is better). The small numbers under the bars are *trials / target words*. The dashed line on the left is the paper's stated chance level (94,306).

| Condition | Word mean rank | Word top-1 | Wav2Vec mean rank | Wav2Vec top-1 |
|---|---|---|---|---|
| NNS → NNS | 0.88 | 0.86 | 7.9 | 0.83 |
| VBZ → VBZ | 6.3 | 0.84 | 38 | 0.77 |
| NNS → VBZ | 5.4 | 0.75 | 58 | 0.34 |
| VBZ → NNS | 3.7 | 0.79 | 25 | 0.78 |
| random pair → NNS | 10,665 | 0.02 | 784 | 0.13 |
| random pair → VBZ | 11,097 | 0.02 | 926 | 0.10 |

**What it shows.**
- Real analogies are **hundreds to thousands of times better** than the random control. For the probe, exact-hit accuracy drops from 75–86% to 2%.
- The clearest model difference: when a noun step is applied to verbs (NNS → VBZ), wav2vec gets the exact word only **34%** of the time, against **75%** for the probe.
- The random control gives a much **lower** (better-looking) rank for raw wav2vec (about 800) than for the probe (about 11,000). That is because in raw wav2vec, *b − a + c* stays very close to *c*, and an inflected word sounds like its base. So even a meaningless step lands near the right word. The probe's space is more spread out by word identity, so a meaningless step really does land somewhere random.

**What it means.** The analogy results are real signal, not luck. Also, **our random control is not the paper's "chance level" of 94,306**. The paper says it computed its baseline the same way we did, so how it got 94,306 is an **open question**. We call ours the "random-pair control" and do not claim the two match.

---

### Table: False friends (the paper's Fig. 5; not in the notebook)

**Question.** Does the step transfer to and from words that only *sound* inflected, like *beside → besides* or *though → those*? These numbers come from `Project_Feasibility.md` §10.4. Each cell reads "ours (paper)".

| | Word: NNS | Word: NNS false friends | Wav2Vec: NNS | Wav2Vec: NNS false friends |
|---|---|---|---|---|
| **from NNS** | 0.9 (1) | 45.5 (5.3) | 7.9 (6.7) | 267.5 (47) |
| **from NNS false friends** | 10.6 (1.4) | 61.6 (8.4) | 3.5 (3) | 211.5 (76) |

| | Word: VBZ | Word: VBZ false friends | Wav2Vec: VBZ | Wav2Vec: VBZ false friends |
|---|---|---|---|---|
| **from VBZ** | 5.9 (7.9) | 24.2 (4.3) | 36.3 (37) | 182.4 (61) |
| **from VBZ false friends** | 26.1 (18) | 59.2 (8.7) | 39.2 (36) | 212.0 (66) |

**What it shows.** The **direction** matches the paper: false friends cost raw wav2vec far more (ranks about 180–270) than the probe (about 24–62). But our numbers are **3–8× larger** than the paper's.

**What it means.** This partly reproduces. The probe is fairly relaxed about whether a final -s is "real" grammar, which supports the paper's view that it tracks a sound pattern. The size difference most likely comes from a word-frequency filter (the WorldLex file) that the authors applied and we could not yet, because we have not downloaded that file. Without it, rare words and names with huge ranks stay in and inflate the averages. **Treat false-friend magnitudes as provisional.**

---

### Table: test-clean runs (the small pipeline check)

The probe was trained and tested on the small 5.4-hour `test-clean` set. Each cell shows our number, with the paper's held-out number (its Fig. 17) in the right-hand column.

| Cell | Word (ours) | Wav2Vec (ours) | Paper held-out: Word / Wav2Vec |
|---|---|---|---|
| NNS→NNS | 0.2 | 31.1 | 0.12 / 4.5 |
| NNS→VBZ | 11.6 | 536.8 | 6.6 / 92 |
| VBZ→NNS | 2.4 | 29.6 | 2 / 8.5 |
| VBZ→VBZ | 15.6 | 706.6 | 4.7 / 35 |

**What it means.** The probe's pattern matches the paper even on this small dataset. Raw wav2vec looks much worse than the paper here, probably because our test-clean word lists were built from test-clean alone (only 44 noun pairs and 33 verb pairs), not from the bigger train-clean-100 lists. This run was a **plumbing check**, not the paper's held-out experiment, which is still to do.

---

### The [ɪd] spelling check (a data-quality finding)

Before analysing the past tense, we checked how the aligner writes the *-ed* syllable in *wanted*. It turns out to use **two spellings**: `IH D` and `AH D`.

- 35% of [ɪd] past-tense tokens are spelled `AH D`.
- 64 of 230 [ɪd] verbs (28%) mostly use `AH D`.
- The authors' code only looked for `IH D`, so it **silently dropped more than a quarter of the [ɪd] verbs**.

We count both spellings as [ɪd] in all our summaries. Here is what that changes in the probe's past-tense grid (trials per cell and mean rank):

| Source → target | Trials, `IH D` only | Trials, both spellings | Mean rank, `IH D` only | Mean rank, both spellings |
|---|---|---|---|---|
| [d] → [ɪd] | 94 | 136 | 7.2 | 6.2 |
| [ɪd] → [d] | 93 | 133 | 12.8 | 10.7 |
| [ɪd] → [ɪd] | 32 | 67 | 0.38 | 0.85 |
| [ɪd] → [t] | 37 | 61 | 2.9 | **25.6** |
| [t] → [ɪd] | 32 | 53 | 3.2 | 2.7 |

**What it means.** Fixing the labels roughly doubles the [ɪd] data in some cells. It also shows how fragile a mean is: in the [ɪd] → [t] cell, adding 24 trials moved the mean from 2.9 to 25.6, because a few of the added trials missed badly. The [ɪd] sets for forced choice (4 and 9 items) and false friends (8 items) remain tiny, so those analyses are not yet possible for [ɪd]. A proper fix means regenerating the word lists and re-running the analogies.

---

### Figure 6 (new, not in the paper): Past tense across [t] / [d] / [ɪd]

![Figure 6: past-tense allomorph grid](workshop/figures/fig06_vbd_allomorph_grid.png)

**Question.** Does "add -ed" learned from one ending work on another? For example, *walk : walked* ([t]) :: *play : ?* ([d]). This is the past-tense version of Figure 3, using **regular verbs only**.

**How to read it.** Rows are the source ending; columns are the target ending. Each cell gives three numbers:
- the **mean rank**;
- **acc** = top-1 accuracy (how often the exact past form is the nearest word);
- **n** = trials / different target verbs.

| Word probe | → [t] | → [d] | → [ɪd] |
|---|---|---|---|
| **[t] →** | 0.04, acc 0.96 | 0.96, acc 0.82 | 2.7, acc 0.58 |
| **[d] →** | 0.5, acc 0.86 | 3.8, acc 0.85 | 6.2, acc 0.62 |
| **[ɪd] →** | 26, acc 0.66 | 11, acc 0.66 | 0.85, acc 0.87 |

| Raw Wav2Vec | → [t] | → [d] | → [ɪd] |
|---|---|---|---|
| **[t] →** | 0.19, acc 0.92 | 1.7, acc 0.67 | 4.0, acc 0.55 |
| **[d] →** | 34, acc 0.79 | 4.6, acc 0.75 | 7.9, acc 0.61 |
| **[ɪd] →** | 2.7, acc 0.61 | 2.1, acc 0.66 | 1.8, acc 0.82 |

**What it shows.**
- **Transfer works across all three endings.** In every cell, the probe picks the exact past form at least **58%** of the time, and up to **96%**.
- **The same ending is best:** the diagonal is 85–96% for the probe.
- **Transferring into [ɪd] is weakest** (58–62%). [ɪd] is the ending that adds a whole syllable, just as [ɪz] was the weak spot for -s.
- **The probe is at or above raw wav2vec in accuracy in all 9 cells**, but the gap is **smaller than it was for -s**, and the error bars overlap in most cells.
- **Look at accuracy, not just colour.** The darkest Word cell, [ɪd] → [t] with mean 26, still has 66% exact hits. Its mean is pulled up by a few bad trials (see the [ɪd] check above).

**What it means.** This is a first answer to our sub-question 1. **The regular past tense behaves like -s**: a largely shared "add -ed" step, mildly sensitive to pronunciation. It is a **pilot**: one probe, one layer, and 26–235 trials per cell. So we state it cautiously.

---

### Figure 7 (new): Regular vs. irregular past tense, compared with -s

![Figure 7: regular vs irregular](workshop/figures/fig07_regular_vs_irregular.png)

**Question.** How do regular and irregular past tenses compare with -s? Here the source and target are in the **same class**: regular → regular (*walk : walked :: play : ?*) or irregular → irregular (*sing : sang :: take : ?*).

**How to read it.** The left panel shows mean rank (log scale; lower is better). The right panel shows top-1 accuracy (higher is better). The small numbers are *trials / target verbs*.

| Condition | Word mean rank | Word top-1 | Wav2Vec mean rank | Wav2Vec top-1 |
|---|---|---|---|---|
| Past regular (pipeline's "regular" label) | 87 | 0.76 | 52 | 0.69 |
| Past regular (only verbs our auto-check also calls regular) | **5.1** | **0.77** | 7.1 | 0.71 |
| Past irregular | 1,833 | 0.22 | 1,672 | 0.16 |
| Past irregular (no modals, *be*, homographs) | 1,480 | 0.25 | 1,477 | 0.19 |
| Plural NNS | 0.75 | 0.90 | 1.4 | 0.80 |
| 3sg VBZ | 1.5 | 0.87 | 3.0 | 0.78 |
| Random-pair control | 10,881 | 0.02 | 855 | 0.12 |

**What it shows.**
- **Clean regular past ≈ -s.** Using only verbs that are truly regular, the probe hits the exact past form **77%** of the time, against 87–90% for -s. The mean rank is 5.1.
- **Labels matter a lot.** With the pipeline's automatic "regular" flag, the mean is **87**, not 5. Nearly all of the difference comes from a handful of verbs the pipeline wrongly calls regular because they change their stem: *bathe → bathed* (the vowel changes), *deal → dealt*, *mean → meant*. The worst 1% of trials carry 95% of the total rank.
- **Irregular → irregular mostly fails:** about **22% (probe)** and **16% (wav2vec)** exact hits, with mean ranks around 1,500–1,800. That is between real inflections and the random control. Removing modals (*could*, *should*), *be* and homographs barely helps.

**What it means.**
- The **irregular failure is expected under every theory.** *sing → sang* and *take → took* are different sound changes, so no account predicts one shared step between them. This is not evidence for dual-route or single-route yet.
- The **informative tests are still to come:**
  - **regular ↔ irregular** transfer (does *walk → walked* help predict *sang*?);
  - transfer **within a sound-change family** (*sing : sang :: ring : ?*).

  These need a new analogy run of about 3 hours.
- Before any theoretical claim, the verbs need a **careful hand classification** (Nigel's spreadsheets): regular [t]/[d]/[ɪd], semi-regular, devoicing (*send → sent*), vowel change (*sing → sang*), suppletive (*go → went*).

---

### Table: The worst target verbs (past tense)

The notebook lists the 15 worst and 15 best target verbs for each model and ending. Some of the worst:

| Model | Ending | Verb | Mean rank | Sounds like |
|---|---|---|---|---|
| Wav2Vec | [t] | tap → tapped | 3,129 | |
| Word | [d] | nose → nosed | 829 | |
| Wav2Vec | [d] | row → rowed | 728 | *road, rode, Rhode* |
| Word | [t] | peep → peeped | 327 | |
| Word | [ɪd] | side → sided | 186 | |
| Word | [d] | row → rowed | 173 | *road, rode* |
| Word | [t] | miss → missed | 50.5 | *mist* |
| Wav2Vec | [ɪd] | sight → sighted | 27 | *cited* |

**What it means.**
- Several bad cases are **homophones**: the correct answer *rowed* sounds exactly like *road*, so the model lands on the "wrong" word that sounds identical. The ranking does not forgive this, so those ranks are inflated for reasons that have nothing to do with grammar.
- Many of the worst verbs are rare, with only 1–3 trials each, so one bad trial dominates.
- This is the list Nigel can scan for odd items, and it raises a design question for the group: **exclude homophones, or count the sound-alike as correct?**

---

## 7. What it all means

**1. The measuring tool works and we can trust it.** We got the paper's code running end to end, matched its word set exactly (334,008 tokens), and reproduced its two main figures closely (8/8 and 41/50 cells within 2×). Anything we find for the past tense is measured with a tool that gives the paper's answer on -s.

**2. The paper's main claim holds.** A tiny linear probe, trained only to tell spoken words apart, ends up with **one shared "add -s" step** that works across nouns and verbs, and across [s]/[z]/[ɪz]. Raw wav2vec keeps more detail about category and pronunciation. In plain words: learning to recognize words pushes the model to treat -s as an abstract sound pattern, "add a hissing sound that matches in voicing", rather than as grammar.

**3. First look at the past tense: regular verbs look like -s.** The regular past tense transfers across [t]/[d]/[ɪd] with 58–96% exact hits for the probe, and clean regular verbs (77%) come close to -s (87–90%). [ɪd] is the weak spot, as [ɪz] was for -s.

**4. Irregular verbs don't share a step with each other, as every theory expects.** The interesting comparisons (regular vs. irregular, and within sound families) need one more run of about 3 hours.

**5. Labels matter as much as the model.** Two labelling problems changed our past-tense numbers:
- the aligner's two spellings of [ɪd] (the original code dropped 28% of [ɪd] verbs);
- the automatic "regular" flag letting in *dealt*, *meant* and *bathed* (moving the mean from 5 to 87).

So the next step is a careful hand classification of the verbs.

**6. Means can mislead; report accuracy and confidence intervals too.** A few failed trials can dominate a mean rank, especially for raw wav2vec. We report top-1 accuracy and per-verb confidence intervals alongside means.

**7. Costs are known and manageable.** One layer takes 8–9 hours end to end, uses 52 GB of hidden states, and needs about 115 GB of disk overall. Adding more layers or models is about 3.5 hours each, plus disk.

**Decisions this suggests for the proposal** (drafted in the notebook's Section 6, for the group to edit):
- The scope is the **past tense**; a newer speech model is a stretch goal at most.
- The verb set uses **hand-annotated classes**.
- The main measures are the **cosine between per-verb past-tense arrows** plus analogy rank, reported with top-1 accuracy and confidence intervals.
- The main generalization test is **held-out verbs** (leave one verb out).
- **Out of scope** (or stated limitations): participles (*sung*) and no-change verbs (*put*, *hit*).

---

## 8. What is not done yet, and open questions

**Not done yet.**
- **All 12 layers** (the paper's Fig. 1). About 620 GB and 2 days of computing. A few layers (e.g. 1, 4, 8, 11) would be a cheaper sample.
- **Regular ↔ irregular transfer.** The key new past-tense experiment. It needs a new analogy run of about 3 hours for both models.
- **Forced choice** (the paper's Fig. 6) and the **regression** (Fig. 7 / Table 7). These need the authors' summary notebook adapted, plus the WorldLex word-frequency file.
- **Held-out evaluation:** a probe trained on train-clean-100, tested on test-clean.
- **Offset-vector (cosine) and held-out-verb pilot.** Planned in the demo plan, but not yet in the notebook.
- **Fixing the [ɪd] label** in the code and regenerating the word lists.
- **A second probe training run** (a different random seed), to see how much results wobble.

**Open questions** (some are on the workshop "advice" slide).
- How did the paper get its **chance level of 94,306**, when the same method gives us about 11,000 (probe) and about 800 (wav2vec)?
- **Homophones** (*rowed ~ road*): exclude them, or count the sound-alike as correct?
- **Irregular verbs:** group them by sound-change family (*sing, ring, drink*), or test pair by pair?
- **Is layer 8 enough** for a past-tense study, or should we sample other layers?
- **No noun/verb ambiguity control for the past tense** yet. The flag exists only for -s.
- Spoken *walked* mixes **past, participle and adjective** uses ("he walked", "he has walked", "a walked dog"). The tagger's fine-grained tags are not stored yet.

---

## 9. Where everything lives

| What | Where |
|---|---|
| This guide | `Feasibility_Results_Simplified.md` (project files) |
| The explainer diagrams in this guide (and the script that draws them) | `explainer_figures/` (project files) |
| The feasibility figures (Figs 1–7) | `workshop/figures/` (project files) and `outputs/feasibility/figures/` (repo) |
| Full technical record (every fix, run and number) | `Project_Feasibility.md` (project files and repo) |
| Plan for the feasibility demo | `plans/feasibility_demo_plan.md` (project files) |
| Workshop slide brief | `workshop/feasibility_workshop_slide_brief.md` (project files) |
| The feasibility notebook | `notebooks/feasibility/feasibility_demo.ipynb` (repo) |
| Result tables (CSV) | `outputs/feasibility/tables/` (repo) |
| Data checks ([ɪd] spelling, inventory) | `outputs/feasibility/checks/` (repo) |
| Verb spreadsheets for Nigel | `outputs/feasibility/theory/` (repo) |
| The paper | `Project-Research-Paper_Morpho.pdf` (project files) |
| Our research question | `Research_Question_Group3-1.pdf` (project files) |
