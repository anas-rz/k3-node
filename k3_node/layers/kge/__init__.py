r"""Knowledge Graph Embedding (KGE) package."""

from .base import KGEModel
from .transe import TransE
from .complex import ComplEx
from .distmult import DistMult
from .rotate import RotatE
from .loader import KGTripletLoader

__all__ = classes = [
    "KGEModel",
    "TransE",
    "ComplEx",
    "DistMult",
    "RotatE",
    "KGTripletLoader",
]
