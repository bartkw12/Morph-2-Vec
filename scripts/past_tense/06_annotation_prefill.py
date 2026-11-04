"""
Past-tense step 6 (optional): suggested annotations for the 151 non-regular and `check` pairs, so the hand
annotation becomes a review rather than a first pass. Writes outputs/past_tense/annotation_prefill.csv; the
annotation spreadsheet (outputs/feasibility/theory/vbd_pairs_for_annotation.xlsx) is NOT edited.

Classes follow outputs/feasibility/theory/README_annotation.md (regular_t / regular_d / regular_id / semi_regular /
devoicing / ablaut / suppletive / other / exclude); families use its `sing-sang` style. The suggestions were
written by an AI coding assistant from the aligner's pronunciations and must be checked by a person.

    python scripts/past_tense/06_annotation_prefill.py
"""

from datetime import datetime
from io import StringIO
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt_common as P  # noqa: E402

C = P.C

# base | past | suggested class | family | reason
SUGGESTIONS = """
keep|kept|semi_regular|keep-kept|IY -> EH plus [t]
make|made|other|make-made|final K replaced by [d]; authors excluded; ~ maid
tell|told|semi_regular|tell-told|EH -> OW plus [d]; ~ tolled
have|had|other|have-had|final V replaced by [d]
catch|caught|semi_regular|bring-brought|rhyme replaced by -aught [ɔt]; 'other' is also defensible; ~ cot
lose|lost|semi_regular|lose-lost|UW -> AO, Z -> S, plus [t]
bring|brought|semi_regular|bring-brought|rhyme replaced by -ought [ɔt]; 'other' is also defensible
leave|left|semi_regular|leave-left|IY -> EH, V -> F, plus [t]; 'left' is also the direction (homograph)
think|thought|semi_regular|bring-brought|rhyme replaced by -ought [ɔt]; 'thought' is also a noun
feel|felt|semi_regular|keep-kept|IY -> EH plus [t]; 'felt' is also the fabric
do|did|other|do-did|UW -> IH plus [d]; authors excluded
seek|sought|semi_regular|bring-brought|rhyme replaced by -ought [ɔt]; ~ sot
sell|sold|semi_regular|tell-told|EH -> OW plus [d]
buy|bought|semi_regular|bring-brought|rhyme replaced by -ought [ɔt]
sleep|slept|semi_regular|keep-kept|IY -> EH plus [t]
teach|taught|semi_regular|bring-brought|rhyme replaced by -aught [ɔt]
flee|fled|semi_regular|flee-fled|IY -> EH plus [d]
sweep|swept|semi_regular|keep-kept|IY -> EH plus [t]
weep|wept|semi_regular|keep-kept|IY -> EH plus [t]
creep|crept|semi_regular|keep-kept|IY -> EH plus [t]
dwell|dwelt|other|burn-burnt|adds [t] after a voiced consonant, no vowel change
send|sent|devoicing|send-sent|final D -> T; ~ cent, scent
spend|spent|devoicing|send-sent|final D -> T
build|built|devoicing|send-sent|final D -> T
bend|bent|devoicing|send-sent|final D -> T
lend|lent|devoicing|send-sent|final D -> T
be|were|suppletive|be-was|different root; ~ we're
be|was|suppletive|be-was|different root
go|went|suppletive|go-went|different root
speak|spoke|ablaut|speak-spoke|IY -> OW; 'spoke' is also a noun
meet|met|ablaut|meet-met|IY -> EH; the final T belongs to the base
run|ran|ablaut|run-ran|AH -> AE
shake|shook|ablaut|take-took|EY -> UH
get|got|ablaut|get-got|EH -> AA
fall|fell|ablaut|fall-fell|AO -> EH
take|took|ablaut|take-took|EY -> UH
strike|struck|ablaut|strike-struck|AY -> AH
understand|understood|ablaut|stand-stood|AE N D -> UH D (the N is lost)
shoot|shot|ablaut|shoot-shot|UW -> AA; 'shot' is also a noun
sit|sat|ablaut|sing-sang|IH -> AE
stand|stood|ablaut|stand-stood|AE N D -> UH D (the N is lost)
lead|led|ablaut|meet-met|IY -> EH
break|broke|ablaut|break-broke|EY -> OW; 'broke' is also an adjective
find|found|ablaut|find-found|AY -> AW
know|knew|ablaut|know-knew|OW -> UW; ~ new
bite|bit|exclude||'bit' is mostly the noun ('a bit'); otherwise ablaut, bite-bit
begin|began|ablaut|sing-sang|IH -> AE
draw|drew|ablaut|draw-drew|AO -> UW
give|gave|ablaut|give-gave|IH -> EY
grow|grew|ablaut|know-knew|OW -> UW
see|saw|ablaut|see-saw|IY -> AO; 'saw' is also the tool
hold|held|ablaut|hold-held|OW -> EH
come|came|ablaut|come-came|AH -> EY
become|became|ablaut|come-came|AH -> EY
throw|threw|ablaut|know-knew|OW -> UW; ~ through
bind|bound|ablaut|find-found|AY -> AW; 'bound' is also an adjective ('bound to')
wear|wore|ablaut|bear-bore|EH R -> AO R; ~ war
write|wrote|ablaut|write-wrote|AY -> OW
fly|flew|ablaut|fly-flew|AY -> UW
bear|bore|ablaut|bear-bore|EH R -> AO R; 'bear' and 'bore' are also other words; ~ boar
drive|drove|ablaut|write-wrote|AY -> OW
arise|arose|ablaut|write-wrote|AY -> OW
ride|rode|ablaut|write-wrote|AY -> OW; ~ road, rowed
eat|ate|ablaut|eat-ate|IY -> EY; ~ eight
fight|fought|ablaut|fight-fought|AY -> AO; spelled like the -ought verbs, but nothing is added
forget|forgot|ablaut|get-got|EH -> AA
win|won|ablaut|cling-clung|IH -> AH; ~ one
behold|beheld|ablaut|hold-held|OW -> EH
sink|sank|ablaut|sing-sang|IH -> AE
shine|shone|ablaut|write-wrote|AY -> OW; ~ shown
bid|bade|ablaut|give-gave|IH -> EY
sing|sang|ablaut|sing-sang|IH -> AE
feed|fed|ablaut|meet-met|IY -> EH
crow|crew|exclude||'crew' is almost always the noun; crow -> crew is archaic
ring|rang|ablaut|sing-sang|IH -> AE
hide|hid|ablaut|bite-bit|AY -> IH; authors excluded
withdraw|withdrew|ablaut|draw-drew|AO -> UW
choose|chose|ablaut|choose-chose|UW -> OW
swing|swung|ablaut|cling-clung|IH -> AH
stick|stuck|ablaut|cling-clung|IH -> AH; 'stuck' is also an adjective
awake|awoke|ablaut|break-broke|EY -> OW
blow|blew|ablaut|know-knew|OW -> UW; ~ blue
drink|drank|ablaut|sing-sang|IH -> AE
tear|tore|ablaut|bear-bore|EH R -> AO R
swear|swore|ablaut|bear-bore|EH R -> AO R
breed|bred|ablaut|meet-met|IY -> EH; ~ bread
steal|stole|ablaut|speak-spoke|IY -> OW; 'stole' is also a noun
swim|swam|ablaut|sing-sang|IH -> AE
dig|dug|ablaut|cling-clung|IH -> AH
cling|clung|ablaut|cling-clung|IH -> AH
speed|sped|ablaut|meet-met|IY -> EH
shrink|shrank|ablaut|sing-sang|IH -> AE
mistake|mistook|ablaut|take-took|EY -> UH
misunderstand|misunderstood|ablaut|stand-stood|AE N D -> UH D (the N is lost)
slay|slew|ablaut|slay-slew|EY -> UW; 'slew' is also a noun
slide|slid|ablaut|bite-bit|AY -> IH
wake|woke|ablaut|break-broke|EY -> OW
overtake|overtook|ablaut|take-took|EY -> UH
overcome|overcame|ablaut|come-came|AH -> EY
dive|dove|exclude||the aligner's [dʌv] is the bird 'dove', not the past [doʊv]
undertake|undertook|ablaut|take-took|EY -> UH
lie|lay|exclude||'lay' is also the present of 'lay'
shall|should|exclude||modal; past use is marginal
will|would|exclude||modal; past use is marginal; ~ wood
can|could|exclude||modal; past use is marginal
rise|rose|exclude||'rose' is also the flower; ~ rows
may|might|exclude||modal; past use is marginal; ~ mite
use|used|regular_d||regular; the base pronunciation is the noun 'use' [juːs] (verb [juːz]); 'used to' is common
hear|heard|semi_regular|hear-heard|IY R -> ER plus [d]; ~ herd
lay|laid|regular_d||phonologically regular (lay + [d]); only the spelling is irregular
close|closed|regular_d||regular; the base pronunciation is the adjective 'close' [kloʊs]
mean|meant|semi_regular|keep-kept|IY -> EH plus [t]
say|said|semi_regular|flee-fled|EY -> EH plus [d]
learn|learned|regular_d||regular; the base also has 'learnt'
hang|hung|ablaut|cling-clung|AE -> AH; 'hanged' is the execution sense
pay|paid|regular_d||phonologically regular (pay + [d]); only the spelling is irregular
require|required|regular_d||regular; the vowel difference is an aligner variant
spring|sprang|ablaut|sing-sang|IH -> AE; the base also has 'sprung'
wind|wound|exclude||'wound' is also the injury
inquire|inquired|regular_d||regular; the vowel difference is an aligner variant
light|lighted|regular_id||regular; the base also has 'lit'
wick|wicked|exclude||'wicked' is the adjective, not a past of 'wick'
examine|examined|regular_d||regular; the vowel difference is an aligner variant
hesitate|hesitated|regular_id||regular; the vowel difference is an aligner variant
burn|burned|regular_d||regular; the base also has 'burnt'
admire|admired|regular_d||regular; the vowel difference is an aligner variant
rouse|roused|regular_d||regular; base pronounced with [s], past with [z]
dream|dreamed|regular_d||regular; authors excluded (two pasts)
light|lit|ablaut|bite-bit|AY -> IH
limit|limited|regular_id||regular; the vowel difference is an aligner variant
practice|practiced|regular_t||regular; the aligner's past pronunciation [præktaɪzd] looks wrong, please check
learn|learnt|other|burn-burnt|adds [t] after a voiced consonant, no vowel change
hang|hanged|regular_d||regular (the execution sense)
rag|ragged|exclude||'ragged' is the adjective
crook|crooked|exclude||'crooked' is the adjective
refresh|refreshed|regular_t||regular; the vowel difference is an aligner variant
bath|bathed|exclude||'bathed' is the past of 'bathe' [beɪð], not of 'bath'
spoil|spoiled|regular_d||regular; the base also has 'spoilt'
ornament|ornamented|regular_id||regular; the vowel difference is an aligner variant
regain|regained|regular_d||regular; the vowel difference is an aligner variant
deal|dealt|semi_regular|keep-kept|IY -> EH plus [t]
fortify|fortified|regular_d||regular; the vowel difference is an aligner variant
esteem|esteemed|regular_d||regular; the vowel difference is an aligner variant
spring|sprung|exclude||the participle, not the simple past
leap|leapt|semi_regular|keep-kept|IY -> EH plus [t]; authors excluded
dream|dreamt|semi_regular|keep-kept|IY -> EH plus [t]; authors excluded
spoil|spoilt|other|burn-burnt|adds [t] after a voiced consonant, no vowel change
comprehend|comprehended|regular_id||regular; the vowel difference is an aligner variant
burn|burnt|other|burn-burnt|adds [t] after a voiced consonant, no vowel change
mature|matured|regular_d||regular; the vowel difference is an aligner variant
merit|merited|regular_id||regular; the vowel difference is an aligner variant
"""

CLASSES = {"regular_t", "regular_d", "regular_id", "semi_regular", "devoicing", "ablaut", "suppletive", "other", "exclude"}


def main():
    start = datetime.now()
    s = pd.read_csv(StringIO(SUGGESTIONS.strip()), sep="|", header=None, keep_default_na=False,
                    names=["base", "inflected", "suggested_class", "sound_change_family", "reason"])
    assert set(s.suggested_class) <= CLASSES, set(s.suggested_class) - CLASSES
    assert not s.duplicated(["base", "inflected"]).any()

    v = pd.read_csv(P.CLASSES_CSV, keep_default_na=False)
    todo = v[~v.auto_class.str.startswith("regular_")]
    m = todo.merge(s, on=["base", "inflected"], how="outer", indicator=True, validate="1:1")
    assert (m._merge == "both").all(), m.loc[m._merge != "both", ["base", "inflected", "_merge"]]
    m["suggested_exclude"] = (m.suggested_class == "exclude").map({True: "yes", False: ""})
    m = m.rename(columns={"inflected": "past"})[
        ["base", "past", "auto_class", "auto_reason", "authors_excluded", "base_pron", "past_pron", "past_homophones",
         "past_tokens", "past_corpus_count", "suggested_class", "sound_change_family", "suggested_exclude", "reason"]]
    m.insert(len(m.columns), "suggested_by", "AI coding assistant; to be checked by a person")
    m.to_csv(P.OUT / "annotation_prefill.csv", index=False, encoding="utf-8-sig")

    print(pd.crosstab(m.auto_class, m.suggested_class, margins=True).to_string())
    print("\nfamilies:", m.sound_change_family.replace("", pd.NA).dropna().value_counts().to_dict())
    P.log_run("python scripts/past_tense/06_annotation_prefill.py", start, datetime.now(),
              f"annotation_prefill.csv ({len(m)} rows; xlsx not edited)")


if __name__ == "__main__":
    main()
