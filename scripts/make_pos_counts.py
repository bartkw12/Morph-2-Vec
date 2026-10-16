"""
Count part-of-speech tags for each word type in preprocessed LibriSpeech splits.

Produces `data/pos_counts.pkl` (word -> Counter of spaCy coarse POS tags), used by
`notebooks/analogy/prepare_inputs.ipynb` to exclude noun/verb-ambiguous items
(paper Appendix A, which tags with spaCy `en_core_web_trf`).

Words are taken from the alignment word tiers (`word_detail.utterance`), the same
labels used throughout the analysis, and tagged as pre-tokenized sentences so that
every alignment word receives exactly one tag.

    python scripts/make_pos_counts.py outputs/preprocessed_data/librispeech-train-clean-100
"""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
import pickle

import datasets
import spacy
from spacy.tokens import Doc
from tqdm.auto import tqdm


def iter_utterances(dataset: datasets.Dataset):
    for item in dataset:
        # drop silence / empty-word intervals
        words = [word for word in item["word_detail"]["utterance"] if word]
        if words:
            yield words


def main(args):
    if args.gpu:
        spacy.prefer_gpu()
    nlp = spacy.load(args.model, exclude=["parser", "ner", "lemmatizer"])

    pos_counts: defaultdict[str, Counter] = defaultdict(Counter)
    for dataset_path in args.dataset_paths:
        dataset = datasets.load_from_disk(dataset_path).select_columns(["word_detail"])
        docs = (Doc(nlp.vocab, words=words) for words in iter_utterances(dataset))
        for doc in tqdm(nlp.pipe(docs, batch_size=args.batch_size), total=len(dataset),
                        desc=f"Tagging {Path(dataset_path).name}"):
            for token in doc:
                pos_counts[token.text][token.pos_] += 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(pos_counts, f)
    print(f"Wrote POS counts for {len(pos_counts)} word types to {args.output}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("dataset_paths", nargs="+", help="preprocessed dataset directories")
    p.add_argument("-o", "--output", default="data/pos_counts.pkl")
    p.add_argument("--model", default="en_core_web_trf")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gpu", action="store_true", help="use GPU if spaCy can (requires cupy)")
    main(p.parse_args())
