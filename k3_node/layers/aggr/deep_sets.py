from typing import Optional
from .base import Aggregation


class DeepSetsAggregation(Aggregation):
    r"""Performs Deep Sets aggregation in which the elements to aggregate are
    first transformed by a Multi-Layer Perceptron (MLP)
    :math:`\phi_{\mathbf{\Theta}}`, summed, and then transformed by another MLP
    :math:`\rho_{\mathbf{\Theta}}`.
    """

    def __init__(
        self,
        local_nn: Optional[any] = None,
        global_nn: Optional[any] = None,
        local_mlp: Optional[any] = None,
        global_mlp: Optional[any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.local_nn = local_nn if local_nn is not None else local_mlp
        self.global_nn = global_nn if global_nn is not None else global_mlp

    def reset_parameters(self):
        if self.local_nn is not None and hasattr(self.local_nn, "reset_parameters"):
            self.local_nn.reset_parameters()
        if self.global_nn is not None and hasattr(self.global_nn, "reset_parameters"):
            self.global_nn.reset_parameters()

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        if self.local_nn is not None:
            x = self.local_nn(x)

        x = self.reduce(x, index=index, ptr=ptr, dim_size=dim_size, dim=dim, reduce="sum")

        if self.global_nn is not None:
            x = self.global_nn(x)

        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(local_nn={self.local_nn}, global_nn={self.global_nn})"

