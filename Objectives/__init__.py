"""
Objectives Module - Modular fitness objective implementations.

Usage:
    from Datastructures.enum import FitnessObjective

    # Create an objective instance
    objective = get_objective(
        FitnessObjective.UTMOS,
        config=config_data,
        model_data=model_data,
        device='cuda',
        embedding_data=embedding_data
    )

    # Calculate scores for a batch
    scores = objective.calculate_score(context, audio_data)
"""

from Objectives.base import BaseObjective

__all__ = [
    "BaseObjective",
]
