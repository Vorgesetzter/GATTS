import jiwer
from Objectives.base import BaseObjective
from Datastructures.dataclass import ModelData, ModelEmbeddingData, ObjectiveContext


class WerTargetObjective(BaseObjective):
    """
    Word Error Rate between ASR text and target text.

    WER = (Substitutions + Deletions + Insertions) / Number_of_reference_words
    Values: usually (0, 1), rarely > 1
    0 = perfect match, 1 = 100% of words wrong

    Lower is better (we want ASR output to match target).
    Output is normalized to (0, 1) where:
         0 = 100% similarity / matches target (good for attack)
         1 = 0% similarity / different from target (bad for attack)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Lazy load WER transformations if not already loaded
        if self.model_data.wer_transformations is None:
            self.model_data.wer_transformations = jiwer.Compose([
                jiwer.RemoveEmptyStrings(),
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ReduceToListOfListOfWords(),
            ])

        self.wer_transformations = self.model_data.wer_transformations

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: ObjectiveContext) -> list[float]:
        """
        Batched WER calculation.
        Returns list of scores in range (0, 1) where 0 = matches target (good), 1 = different (bad).
        """
        asr_texts = context.asr_texts

        scores = []
        for asr_text in asr_texts:
            # Skip empty/invalid texts
            if not asr_text or len(asr_text) < 2:
                scores.append(1.0)  # Penalize invalid
                continue

            raw_wer = jiwer.wer(
                self.text_target,
                asr_text,
                reference_transform=self.wer_transformations,
                hypothesis_transform=self.wer_transformations,
            )

            # Normalize to (0, 1): raw_wer 0 -> 0 (100% similar), raw_wer 1+ -> 1 (0% similar)
            val = min(2.0, float(raw_wer))
            scores.append(val)

        return scores
