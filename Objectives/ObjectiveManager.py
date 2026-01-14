"""
ObjectiveManager - Centralized management of fitness objectives.

This class handles:
1. Initialization of active objectives (with lazy model loading)
2. Batch evaluation of all objectives
3. Score collection and organization
"""

# Naturalness objectives
from Objectives.Naturalness.PhonemeCountObjective import PhonemeCountObjective
from Objectives.Naturalness.UtmosObjective import UtmosObjective
from Objectives.Naturalness.PPLObjective import PPLObjective
from Objectives.Naturalness.PESQObjective import PesqObjective

# InterpolationVector objectives
from Objectives.InterpolationVector.L1Objective import L1Objective
from Objectives.InterpolationVector.L2Objective import L2Objective

# Target objectives
from Objectives.Target.WerTargetObjective import WerTargetObjective
from Objectives.Target.SbertTargetObjective import SbertTargetObjective
from Objectives.Target.TextEmbTargetObjective import TextEmbTargetObjective
from Objectives.Target.WhisperProbObjective import WhisperProbObjective
from Objectives.Target.Wav2VecDifferentObjective import Wav2VecDifferentObjective
from Objectives.Target.Wav2VecAsrObjective import Wav2VecAsrObjective

# GroundTruth objectives
from Objectives.GroundTruth.WerGtObjective import WerGtObjective
from Objectives.GroundTruth.SbertGtObjective import SbertGtObjective
from Objectives.GroundTruth.TextEmbGtObjective import TextEmbGtObjective
from Objectives.GroundTruth.Wav2VecSimilarObjective import Wav2VecSimilarObjective

from typing import Optional
from Datastructures.dataclass import ModelData, ConfigData, AudioData, EmbeddingData, StepContext
from Datastructures.enum import FitnessObjective
from Objectives.base import BaseObjective

class ObjectiveManager:
    """
    Manages initialization and evaluation of all fitness objectives.

    Usage:
        # Initialize once at start
        manager = ObjectiveManager(config_data, model_data, device, embedding_data, audio_data)

        # Evaluate on each batch
        scores = manager.evaluate_batch(context, audio_data)
    """

    def __init__(
        self,
        config: ConfigData,
        model_data: ModelData,
        device: str,
        embedding_data: Optional[EmbeddingData] = None,
        audio_data: Optional[AudioData] = None
    ):
        self.config = config
        self.model_data = model_data
        self.device = device
        self.embedding_data = embedding_data or EmbeddingData()
        self.audio_data = audio_data

        # Initialize objectives dict
        self.objectives: dict[FitnessObjective, BaseObjective] = {}

    def initialize_objectives(self):
        """
        Initialize only the active objectives.
        Each objective handles its own model loading and embedding computation in __init__.
        """

        for obj_enum in self.config.active_objectives:
            try:
                objective_cls = BaseObjective.get_class(obj_enum)

                # Initializes Objective, fills embedding_data if necessary
                objective = objective_cls(
                    self.config,
                    self.model_data,
                    self.device,
                    self.embedding_data,
                    self.audio_data
                )

                self.objectives[obj_enum] = objective
                print(f"Initialized {obj_enum.name} (batching={objective.supports_batching})")

            except ValueError:
                raise ValueError(f"Objective {obj_enum.name} not found in registry")
            except Exception as e:
                # FIX: Include the original error message 'e' so you can debug it!
                raise ValueError(f"Failed to initialize {obj_enum.name}. Error: {e}") from e

    def evaluate_batch(
            self,
            context: StepContext,
            audio_data: AudioData
    ) -> dict[FitnessObjective, list[float]]:
        """
        Evaluate all objectives on a batch of samples.
        """
        scores: dict[FitnessObjective, list[float]] = {}

        for obj_enum, objective in self.objectives.items():
            try:
                batch_scores = objective.calculate_score(context, audio_data)
                scores[obj_enum] = batch_scores
            except Exception as e:
                print(f"[ERROR] {obj_enum.name} evaluation failed: {e}")

                # FIX: Use a specific list inside context to get the size
                # (Standard dataclasses don't support len(context))
                batch_size = len(context.clean_text)

                # Return worst score (1.0) for all samples in batch
                scores[obj_enum] = [1.0] * batch_size

        return scores
    def evaluate_single(
        self,
        context: StepContext,
        audio_data: AudioData,
        index: int = 0
    ) -> dict[FitnessObjective, float]:
        """
        Evaluate all objectives on a single sample.

        This is a convenience method that extracts a single sample from batch results.

        Args:
            context: StepContext (can be batch or single)
            audio_data: AudioData with reference audio
            index: Index of sample to extract (default 0)

        Returns:
            Dictionary mapping FitnessObjective -> single score
        """
        batch_scores = self.evaluate_batch(context, audio_data)
        return {obj: scores[index] for obj, scores in batch_scores.items()}

    @property
    def active_objectives(self) -> list[FitnessObjective]:
        """Returns list of active objectives in order."""
        return list(self.objectives.keys())

    def get_objective_instance(self, obj_enum: FitnessObjective) -> Optional[BaseObjective]:
        """Get a specific objective instance."""
        return self.objectives.get(obj_enum)
