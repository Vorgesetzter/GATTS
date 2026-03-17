from abc import ABC, abstractmethod
from typing import Optional
import torch

from Datastructures.enum import AttackMode

from Datastructures.dataclass import ModelData, ModelEmbeddingData, ObjectiveContext


class BaseObjective(ABC):
    """
    Abstract base class for all fitness objectives.
    """

    def __init__(
        self,
        model_data: ModelData,
        device: str = None,
        embedding_data: ModelEmbeddingData = None,
        text_gt: Optional[str] = None,
        text_target: Optional[str] = None,
        mode: Optional["AttackMode"] = None,
        audio_gt: Optional[torch.Tensor] = None,
    ):
        self.model_data = model_data
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_data = embedding_data
        self.text_gt = text_gt
        self.text_target = text_target
        self.mode = mode
        self.audio_gt = audio_gt

    @property
    def name(self):
        """Returns the class name (e.g., 'PesqObjective') for logging."""
        return self.__class__.__name__

    @property
    def supports_batching(self) -> bool:
        """
        Override this to True if your _calculate_logic can handle a batch.
        Default is False (Safe Mode).
        """
        return False

    def calculate_score(self, context: ObjectiveContext) -> list[float]:
        """
        Public API. ALWAYS returns a list of floats, even if batch_size=1.

        Args:
            context: ObjectiveContext containing audio_mixed_batch, asr_texts,
                     interpolation_vectors, and optional mel_batch
        """
        batch_size = len(context)

        # --- PATH A: Batch Optimized (GPU Models) ---
        if self.supports_batching:
            try:
                return self._calculate_logic(context)
            except Exception as e:
                print(f"Error in {self.name} (Batch Mode): {e}")
                return [1.0] * batch_size

        # --- PATH B: Single Item Loop (CPU/Legacy Models) ---
        else:
            scores = []
            for i in range(batch_size):
                try:
                    single_ctx = context.get_item(i)

                    # Run safety checks on single item
                    if not single_ctx.asr_texts or len(single_ctx.asr_texts) < 2 or single_ctx.audio_mixed_batch.numel() == 0:
                        scores.append(1.0)
                        continue

                    val = self._calculate_logic(single_ctx)
                    scores.append(float(val))

                except Exception as e:
                    print(f"Error in {self.name} index {i}: {e}")
                    scores.append(1.0)

            return scores

    @abstractmethod
    def _calculate_logic(self, context: ObjectiveContext):
        """
        The specific math for this objective.

        Args:
            context: ObjectiveContext containing:
                - audio_mixed_batch: [Time] for single / [Batch, Time] for batch
                - asr_texts: single string / list of strings
                - interpolation_vectors: [Dim] for single / [Batch, Dim] for batch
                - mel_batch: Optional mel spectrogram tensor
        """
        pass
