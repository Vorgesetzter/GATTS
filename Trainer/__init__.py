"""
Trainer module - Class-based training infrastructure for adversarial TTS.

Classes:
    - EnvironmentLoader: Handles environment initialization and model loading
    - AdversarialTrainer: Main optimization loop (returns OptimizationResult)
    - OptimizationResult: Dataclass containing optimization results
    - RunLogger: Handles logging and result persistence (called separately)
    - GraphPlotter: Generates visualization graphs
"""

from Trainer.EnvironmentLoader import EnvironmentLoader
from Trainer.AdversarialTrainer import AdversarialTrainer, OptimizationResult
from Trainer.RunLogger import RunLogger
from Trainer.GraphPlotter import GraphPlotter

__all__ = [
    "EnvironmentLoader",
    "AdversarialTrainer",
    "OptimizationResult",
    "RunLogger",
    "GraphPlotter",
]
