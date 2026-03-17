from ..base_objective import BaseObjective
from ...data.dataclass import ObjectiveContext


class L2Objective(BaseObjective):
    """
    L2 Norm of the interpolation vector.

    L2 = sqrt(mean(IV^2)) [RMS - punishes larger values more]
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

        # L2 = sqrt(mean(IV^2)) per sample
        # Flatten all non-batch dimensions to handle both 2D [batch, dim] and 3D [batch, dim1, dim2]
        l2_values = (iv ** 2).flatten(start_dim=1).mean(dim=1).sqrt()

        return l2_values.cpu().tolist()
