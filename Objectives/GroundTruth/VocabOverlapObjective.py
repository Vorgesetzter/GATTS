import re
from Objectives.base import BaseObjective
from Datastructures.dataclass import ObjectiveContext


class VocabOverlapObjective(BaseObjective):
    """
    Calculates the percentage of Ground Truth words that 'survived' in the ASR output.
    This is immune to 'padding' cheats (adding 100 random words to dilute the score).

    Formula: Intersection(GT, ASR) / len(GT)

    Range:
        0.0 = SUCCESS (No original words found)
        1.0 = FAILURE (All original words found)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 1. Pre-compute Ground Truth Word Set
        # We remove punctuation so "cat." matches "cat"
        clean_gt = re.sub(r'[^\w\s]', '', self.text_gt.lower())
        self.gt_words_set = set(clean_gt.split())

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: ObjectiveContext) -> list[float]:
        asr_texts = context.asr_texts
        scores = []

        # Avoid division by zero if GT is empty (unlikely)
        if not self.gt_words_set:
            return [0.0] * len(asr_texts)

        for asr_text in asr_texts:
            # 1. Handle Invalid/Empty Outputs
            if not asr_text:
                # If output is empty, technically 0 words survived (0.0).
                # However, for optimization, you usually want valid text.
                # If you want to penalize silence, return 1.0 here.
                # If you purely want to know "did words survive?", return 0.0.
                scores.append(0.0)
                continue

            # 2. Clean ASR Text
            clean_asr = re.sub(r'[^\w\s]', '', asr_text.lower())
            asr_words_set = set(clean_asr.split())

            # 3. Calculate Intersection (Shared Words)
            intersection = self.gt_words_set.intersection(asr_words_set)

            # 4. Calculate Recall (Shared / GT_Length)
            # This ignores how long the ASR text is (immune to dilution).
            ratio = len(intersection) / len(self.gt_words_set)

            # Clamp just in case, though it shouldn't exceed 1.0 mathematically
            scores.append(min(ratio, 1.0))

        return scores