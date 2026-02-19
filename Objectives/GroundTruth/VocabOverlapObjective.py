import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Objectives.base import BaseObjective
from Datastructures.dataclass import ObjectiveContext

STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()


def _stem_word_set(words: set[str]) -> set[str]:
    return {STEMMER.stem(w) for w in words}


class VocabOverlapObjective(BaseObjective):
    """
    Calculates the percentage of Ground Truth content words that 'survived' in the ASR output.

    Pre-processing pipeline (applied to both GT and ASR):
      1. Lowercase + strip punctuation
      2. Remove stopwords (function words that appear regardless of distortion)
      3. Porter-stem each word so morphological variants count as the same token
         e.g. "bananas" == "banana", "smoothed" == "smooth", "sliding" == "slide"

    Formula: Intersection(stem(GT_content), stem(ASR_content)) / len(stem(GT_content))

    Range:
        0.0 = SUCCESS (No original content word stems found in ASR output)
        1.0 = FAILURE (All original content word stems found)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Pre-compute stemmed Ground Truth content word set
        clean_gt = re.sub(r'[^\w\s]', '', self.text_gt.lower())
        content_words_gt = set(clean_gt.split()) - STOPWORDS
        self.gt_words_set = _stem_word_set(content_words_gt)

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: ObjectiveContext) -> list[float]:
        asr_texts = context.asr_texts
        scores = []

        # Avoid division by zero if GT has no content words after stopword removal
        if not self.gt_words_set:
            return [0.0] * len(asr_texts)

        for asr_text in asr_texts:
            # 1. Handle Invalid/Empty Outputs
            if not asr_text:
                # Empty transcription = 0 GT words survived = success
                scores.append(0.0)
                continue

            # 2. Clean and stem ASR text
            clean_asr = re.sub(r'[^\w\s]', '', asr_text.lower())
            content_words_asr = set(clean_asr.split()) - STOPWORDS
            asr_words_set = _stem_word_set(content_words_asr)

            # 3. Calculate Intersection on stemmed sets
            intersection = self.gt_words_set.intersection(asr_words_set)

            # 4. Recall: how many GT stems survived in the ASR output
            ratio = len(intersection) / len(self.gt_words_set)

            scores.append(min(ratio, 1.0))

        return scores