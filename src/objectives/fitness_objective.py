from enum import Enum

# Naturalness objectives
from .naturalness.utmos_objective import UtmosObjective
from .naturalness.ppl_objective import PPLObjective
from .naturalness.pesq_objective import PesqObjective
from .naturalness.visqol_objective import VisqolObjective

# InterpolationVector objectives
from .interpolation_vector.l1_objective import L1Objective
from .interpolation_vector.l2_objective import L2Objective

# Target objectives
from .target.wer_target_objective import WerTargetObjective
from .target.mer_target_objective import MerTargetObjective
from .target.per_target_objective import PerTargetObjective
from .target.sbert_target_objective import SbertTargetObjective
from .target.text_emb_target_objective import TextEmbTargetObjective
from .target.whisper_prob_target_objective import WhisperProbTargetObjective
from .target.wav2vec_different_objective import Wav2VecDifferentObjective
from .target.wav2vec_asr_objective import Wav2VecAsrObjective

# GroundTruth objectives
from .ground_truth.wer_gt_objective import WerGtObjective
from .ground_truth.mer_gt_objective import MerGtObjective
from .ground_truth.per_gt_objective import PerGtObjective
from .ground_truth.sbert_gt_objective import SbertGtObjective
from .ground_truth.text_emb_gt_objective import TextEmbGtObjective
from .ground_truth.wav2vec_similar_objective import Wav2VecSimilarObjective
from .ground_truth.set_overlap_objective import SetOverlapObjective
from .ground_truth.whisper_prob_gt_objective import WhisperProbGtObjective

class FitnessObjective(Enum):
    # ==== Increase Naturalness ====
    UTMOS = UtmosObjective
    PPL = PPLObjective
    PESQ = PesqObjective
    VISQOL = VisqolObjective

    # ==== Interpolation Vector Restrictions ====
    L1 = L1Objective
    L2 = L2Objective

    # ==== Optimize Text Towards Target ====
    WER_TARGET = WerTargetObjective
    MER_TARGET = MerTargetObjective
    PER_TARGET = PerTargetObjective
    SBERT_TARGET = SbertTargetObjective
    TEXT_EMB_TARGET = TextEmbTargetObjective
    WHISPER_PROB_TARGET = WhisperProbTargetObjective

    # ==== Optimize Text Away From Ground-Truth ====
    WER_GT = WerGtObjective
    MER_GT = MerGtObjective
    PER_GT = PerGtObjective
    SBERT_GT = SbertGtObjective
    TEXT_EMB_GT = TextEmbGtObjective
    SET_OVERLAP = SetOverlapObjective
    WHISPER_PROB_GT = WhisperProbGtObjective

    # ==== Optimize Audio Similarity ====
    WAV2VEC_SIMILAR = Wav2VecSimilarObjective
    WAV2VEC_DIFFERENT = Wav2VecDifferentObjective
    WAV2VEC_ASR = Wav2VecAsrObjective
