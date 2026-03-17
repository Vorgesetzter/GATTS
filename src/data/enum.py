from enum import Enum

class AttackMode(Enum):
    TARGETED = "targeted"
    NOISE_UNTARGETED = "noise-untargeted"
    UNTARGETED = "untargeted"
    ZERO_UNTARGETED = "zero-untargeted"
    NEGATION_UNTARGETED = "negation-untargeted"
