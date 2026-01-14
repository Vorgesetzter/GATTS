import torch
from Objectives.base import BaseObjective
from Datastructures.dataclass import ModelData, StepContext, AudioData, EmbeddingData
from Datastructures.enum import FitnessObjective


class L1Objective(BaseObjective):
    """
    L1 Norm of the interpolation vector.

    L1 = mean(|IV|) [Average absolute value of interpolation vector]
    Values: (0, 1)
    0 = only GT, 1 = only Target

    Lower is better (we want to stay close to GT).
    """
    objective_type = FitnessObjective.L1

    def __init__(
        self,
        config,
        model_data: ModelData,
        device: str = None,
        embedding_data: EmbeddingData = None,
        audio_data: AudioData = None
    ):
        super().__init__(config, model_data, device, embedding_data, audio_data)

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: StepContext, audio_data: AudioData) -> list[float]:
        iv = context.interpolation_vector

        if iv.dim() == 1:
            iv = iv.unsqueeze(0)

        # L1 = mean(|IV|) per sample
        l1_values = iv.abs().mean(dim=1)

        return l1_values.cpu().tolist()
