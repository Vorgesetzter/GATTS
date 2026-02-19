import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Objectives.base import BaseObjective
from Datastructures.dataclass import ObjectiveContext

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def _lemmatize_word(word: str) -> str:
    # Try adjective POS first (collapses -er/-est: "smoothest" → "smooth")
    # then noun POS (collapses plurals: "bananas" → "banana").
    # If neither changes the word, return it as-is.
    for pos in ('a', 'v', 'n', 'r'):
        lemma = LEMMATIZER.lemmatize(word, pos=pos)
        if lemma != word:
            return lemma
    return word


def _lemmatize_word_set(words: set[str]) -> set[str]:
    return {_lemmatize_word(w) for w in words}


class VocabOverlapObjective(BaseObjective):
    """
    Calculates the percentage of Ground Truth content words that 'survived' in the ASR output.

    Pre-processing pipeline (applied to both GT and ASR):
      1. Lowercase + strip punctuation
      2. Remove stopwords (function words that appear regardless of distortion)
      3. WordNet-lemmatize: try adjective POS then noun POS to normalize
         degree variants ("smoothest" → "smooth") and plurals ("bananas" → "banana")
         while leaving verb tense ("slid" vs "sliding") as distinct tokens.

    Formula: Intersection(lemma(GT_content), lemma(ASR_content)) / len(lemma(GT_content))

    Range:
        0.0 = SUCCESS (No original content word lemmas found in ASR output)
        1.0 = FAILURE (All original content word lemmas found)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Pre-compute lemmatized Ground Truth content word set
        clean_gt = re.sub(r'[^\w\s]', '', self.text_gt.lower())
        content_words_gt = set(clean_gt.split()) - STOPWORDS
        self.gt_words_set = _lemmatize_word_set(content_words_gt)

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

            # 2. Clean and lemmatize ASR text
            clean_asr = re.sub(r'[^\w\s]', '', asr_text.lower())
            content_words_asr = set(clean_asr.split()) - STOPWORDS
            asr_words_set = _lemmatize_word_set(content_words_asr)

            # 3. Intersection on lemmatized sets
            intersection = self.gt_words_set.intersection(asr_words_set)

            # 4. Recall: how many GT lemmas survived in the ASR output
            ratio = len(intersection) / len(self.gt_words_set)

            scores.append(min(ratio, 1.0))

        return scores