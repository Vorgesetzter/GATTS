import re
from nltk.corpus import stopwords
from Objectives.base import BaseObjective
from Datastructures.dataclass import ObjectiveContext

STOPWORDS = set(stopwords.words('english'))


class VocabOverlapObjective(BaseObjective):
    """
    Calculates the percentage of Ground Truth content words that 'survived' in the ASR output.
    Stopwords (e.g. "the", "on", "a") are excluded from both GT and ASR sets before comparison,
    since they appear in almost any transcription regardless of distortion and carry no semantic content.
    This is immune to 'padding' cheats (adding 100 random words to dilute the score).

    Formula: Intersection(GT_content, ASR_content) / len(GT_content)

    Range:
        0.0 = SUCCESS (No original content words found)
        1.0 = FAILURE (All original content words found)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 1. Pre-compute Ground Truth Content Word Set
        # Remove punctuation and stopwords so only semantically meaningful words remain
        clean_gt = re.sub(r'[^\w\s]', '', self.text_gt.lower())
        self.gt_words_set = set(clean_gt.split()) - STOPWORDS

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

            # 2. Clean ASR Text
            clean_asr = re.sub(r'[^\w\s]', '', asr_text.lower())
            asr_words_set = set(clean_asr.split()) - STOPWORDS

            # 3. Calculate Intersection (Shared Words)
            intersection = self.gt_words_set.intersection(asr_words_set)

            # 4. Calculate Recall (Shared / GT_Length)
            # This ignores how long the ASR text is (immune to dilution).
            ratio = len(intersection) / len(self.gt_words_set)

            # Clamp just in case, though it shouldn't exceed 1.0 mathematically
            scores.append(min(ratio, 1.0))

        return scores