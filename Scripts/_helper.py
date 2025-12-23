import torch
from torch import Tensor

from enum import Enum, auto

class AttackMode(Enum):
    TARGETED = "targeted"
    NOISE_UNTARGETED = "noise-untargeted"
    UNTARGETED = "untargeted"

class FitnessObjective(Enum):

    # ==== Increase Naturalness ====
    PHONEME_COUNT = auto()
    AVG_LOGPROB = auto()
    UTMOS = auto()
    PPL = auto()
    PESQ = auto()

    # ==== Interpolation Vector Restrictions ====
    L1 = auto()
    L2 = auto()

    # ==== Optimize Text Towards Target ====
    WER_TARGET = auto()
    SBERT_TARGET = auto()
    TEXT_EMB_TARGET = auto()

    # ==== Optimize Text Away From Ground-Truth ====
    WER_GT = auto()
    SBERT_GT = auto()
    TEXT_EMB_GT = auto()

    # ==== Optimize Audio Similarity ====
    WAV2VEC_SIMILAR = auto()
    WAV2VEC_DIFFERENT = auto()
    WAV2VEC_ASR = auto()


def length_to_mask(lengths: Tensor) -> Tensor:
    mask = torch.arange(lengths.max())  # Creates a Vector [0,1,2,3,...,x], where x = biggest value in lengths
    mask = mask.unsqueeze(0)  # Creates a Matrix [1,x] from Vector [x]
    mask = mask.expand(lengths.shape[0], -1)  # Expands the matrix from [1,x] to [y,x], where y = number of elements in lengths
    mask = mask.type_as(lengths)  # Assign mask the same type as lengths
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))  # gt = greater than, compares each value from lengths to a row of values in mask; unsqueeze = splits vector lengths into vectors of size 1
    return mask  # returns a mask of shape (batch_size, max_length) where mask[i, j] = 1 if j < lengths[i] and mask[i, j] = 0 otherwise.

def _pad_with_pattern(tensor: Tensor, amount: int, pattern: list[int]) -> Tensor:

    padding = torch.as_tensor(pattern, device=tensor.device, dtype=tensor.dtype)[torch.arange(amount, device=tensor.device) % len(pattern)]

    for _ in range(tensor.dim() - 1):
        padding = padding.unsqueeze(0)
    padding = padding.expand(*tensor.shape[:-1], amount)

    return torch.cat([tensor, padding], dim=-1)

def addNumbersPattern(a: Tensor, b: Tensor, pattern: list[int]) -> tuple[Tensor, Tensor]:

    len_a = a.size(-1)
    len_b = b.size(-1)

    # If equal: nothing to do
    if len_a == len_b:
        return a, b

    # Determine which tensor needs padding
    if len_a < len_b:
        a = _pad_with_pattern(a, len_b - len_a, pattern)
    else:
        b = _pad_with_pattern(b, len_a - len_b, pattern)

    return a, b

import torch

def _extend_to_size(x: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Extends the last dimension of x to `target_size` by *repeating elements in order*.

    Each position j in the original last dimension is repeated either
    floor(target_size / a) or ceil(target_size / a) times, with the
    first `target_size % a` positions getting the extra repeat.

    Example (a = 4, target_size = 8):
        [0.5, 0.9, 0.134, 0.542]
        -> [0.5, 0.5, 0.9, 0.9, 0.134, 0.134, 0.542, 0.542]

    Example (a = 5, target_size = 8):
        [1, 2, 3, 4, 5]
        -> [1, 1, 2, 2, 3, 3, 4, 5]

    x: (p, a)
    returns: (p, target_size)
    """
    p, a = x.shape

    # If already large enough, just crop
    if a >= target_size:
        return x[:, :target_size]

    # How many repeats per original position?
    base = target_size // a          # minimum repeats for each index
    rem = target_size % a            # first `rem` indices get one extra repeat


    # repeats[i] = how often index i should appear
    repeats = torch.full((a,), base, device=x.device, dtype=torch.long)
    if rem > 0:
        repeats[:rem] += 1


    # Build index pattern like:
    # a=4, target=8 -> base=2, rem=0, repeats=[2,2,2,2]
    # indices = [0,0,1,1,2,2,3,3]
    idx = torch.arange(a, device=x.device).repeat_interleave(repeats)

    # Safety: idx should be exactly target_size long
    assert idx.numel() == target_size, f"idx has length {idx.numel()}, expected {target_size}"

    # Apply same index pattern to all rows
    return_vector = x[:, idx]
    return return_vector

def adjustInterpolationVector(IV: Tensor, matrix: Tensor, subspace_optimization: bool) -> Tensor:

    # Matrix Multiplication, since IV not Scalar Value
    if IV.shape[1] != 1:
        if subspace_optimization:
            IV = IV @ matrix
        else:
            IV = _extend_to_size(IV, 512)

    IV = IV.permute(1, 0).unsqueeze(0)

    return IV

