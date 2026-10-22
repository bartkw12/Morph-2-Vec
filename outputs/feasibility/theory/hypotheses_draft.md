# Hypotheses draft: past tense in a word-recognition probe

A draft for the theory review to refine. It feeds the proposal section "hypothesis and what each outcome would mean".

What we can measure (see `README_annotation.md`):
- **Offset similarity.** Is the change *walk → walked* the same direction in the model's space as *play → played*, *want → wanted*, *keep → kept*, *sing → sang*?
- **Transfer.** Does the change learned from one verb predict another verb's past form?
- **Held-out verbs.** Does that work for verbs the average was not computed from?

| Account | Regular [t]/[d]/[ɪd] | Semi-regular (kept, slept) | Irregular (sang, took) | -ed false friends (bold/bowl) |
|---|---|---|---|---|
| Dual-route (Pinker) | one shared offset across allomorphs | partly shared | word-specific, no transfer | should not share the offset |
| Single-route (Rumelhart & McClelland) | shared, graded by phonology | intermediate | clusters by sound-change family (sing/ring/drink) | partly shared |
| Surface phonology (Gauthier et al. finding for -s) | shared | shares the regular [t] offset | no transfer | shares the offset |

Open points:
- Is "shares the regular [t] offset" the right prediction for semi-regulars under the surface-phonology account, or should *kept* pattern with [t] only in its final segment?
- Which sound-change families have enough verbs in our data to test the single-route clustering prediction? (See the `sound_change_family` column.)
- What should we predict for devoicing verbs (*send/sent, build/built*)?
- What would count as "partly shared", in numbers? For example, cosine above the random-pair control but below the regular-regular level.
