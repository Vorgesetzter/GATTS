from abc import ABC, abstractmethod
from typing import Optional
import torch

from ..data.enum import AttackMode
from ..data.dataclass import ModelData, ModelEmbeddingData, ObjectiveContext


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

        If supports_batching is False, loops over the context item-by-item and
        collects results so that _calculate_logic only ever sees a single item.
        """
        if self.supports_batching:
            try:
                return self._calculate_logic(context)
            except Exception as e:
                print(f"Error in {self.name} (Batch Mode): {e}")
                return [1.0] * len(context)
        else:
            results = []
            for i in range(len(context)):
                try:
                    result = self._calculate_logic(context.get_item(i))
                    results.append(float(result))
                except Exception as e:
                    print(f"Error in {self.name} (Item {i}): {e}")
                    results.append(1.0)
            return results


    @abstractmethod
    def _calculate_logic(self, context: ObjectiveContext) -> list[float]:
        pass
