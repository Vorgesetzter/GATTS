from ..base_objective import BaseObjective
from ...data.dataclass import ObjectiveContext


class L1Objective(BaseObjective):
    """
    L1 Norm of the interpolation vector.

    L1 = mean(|IV|) [Average absolute value of interpolation vector]
    Values: (0, 1)
    0 = only GT, 1 = only Target

    Lower is better (we want to stay close to GT).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: ObjectiveContext) -> list[float]:
        iv = context.interpolation_vectors

        if iv.dim() == 1:
            iv = iv.unsqueeze(0)

        # L1 = mean(|IV|) per sample
        # Flatten all non-batch dimensions to handle both 2D [batch, dim] and 3D [batch, dim1, dim2]
        l1_values = iv.abs().flatten(start_dim=1).mean(dim=1)

        return l1_values.cpu().tolist()
