"""
Utilities for working with transformer base models.
"""

from hydra.utils import instantiate
import transformers


def prepare_feature_extractor(config) -> transformers.SequenceFeatureExtractor:
    """
    Prepare the feature extractor used to pad batches of preprocessed waveforms.

    Inputs are already normalized during preprocessing, so the feature extractor
    is only responsible for padding / attention masks. This works for any
    raw-waveform model (wav2vec2, HuBERT, WavLM, ...).
    """
    feature_extractor = instantiate(config.feature_extractor,
                                    padding_value=0.0,
                                    do_normalize=True,
                                    return_attention_mask=False)
    return feature_extractor
