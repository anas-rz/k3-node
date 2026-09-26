"""Domain-specific applications built on K3-Node.

Submodules:
- `chemistry`: Molecular graphs, quantum property prediction, SMILES processing, Uni-Mol, AttentiveFP, SchNet, DimeNet.
- `bio`: Macromolecular structures, protein-protein interactions, docking, UniMolDocking.
- `materials`: Crystal graph neural networks, periodic boundaries, CHGNet, M3GNet, MEGNet, TensorNet, SO3Net, QET.
"""

from k3_node.applications import bio
from k3_node.applications import chemistry
from k3_node.applications import materials

__all__ = [
    "bio",
    "chemistry",
    "materials",
]
