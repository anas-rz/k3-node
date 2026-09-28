import os
from typing import Callable, Optional

import numpy as np

from k3_node.data import Data, InMemoryDataset


class MeshCorrespondence(InMemoryDataset):
    r"""A small shape correspondence dataset in the spirit of FAUST: randomly deformed copies of one
    :class:`GeometricShapes` mesh (the monkey head by default). Every copy is smoothly bent and
    slightly rotated; the task is to recognize every vertex, so ``y`` holds the vertex indices.

    The graphs hold ``pos`` and mesh faces ``face``; use :class:`~k3_node.transforms.FaceToEdge`
    to connect the vertices. As in FAUST, there are 80 training and 20 test meshes by default.

    Args:
        root (str): Directory where GeometricShapes is (or will be) downloaded.
        train (bool, optional): Build the training meshes if ``True``, else the test meshes.
        num_meshes (int, optional): Number of meshes. (default: ``80`` / ``20``)
        shape (str, optional): The GeometricShapes mesh to deform. (default: ``"3d_monkey"``)
        transform (callable, optional): A function applied to each mesh when it is accessed.
        pre_transform (callable, optional): A function applied to each mesh once, when built.
        seed (int, optional): Random seed. (default: ``0``)
    """

    def __init__(self, root: str, train: bool = True, num_meshes: Optional[int] = None, shape: str = "3d_monkey",
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, seed: int = 0):
        super().__init__(None, transform)
        from k3_node.datasets.geometric_shapes import GeometricShapes
        from k3_node.transforms.spatial import _np

        shapes = GeometricShapes(root, train=True)
        names = sorted(os.listdir(shapes.raw_dir))
        base = next(shapes[i] for i in range(len(shapes)) if names[int(_np(shapes[i].y)[0])] == shape)
        pos, face = _np(base.pos).astype(np.float64), _np(base.face)
        pos = pos / np.abs(pos).max()

        rng = np.random.default_rng(seed + (0 if train else 1))
        graphs = []
        for _ in range(num_meshes or (80 if train else 20)):
            w = rng.normal(size=(3, 3)) * 1.5
            bent = pos + 0.12 * np.sin(pos @ w + rng.uniform(0, 2 * np.pi, size=3))  # smooth deformation
            angle = np.deg2rad(rng.uniform(-15, 15))
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            data = Data(pos=(bent @ rot.T).astype(np.float32), face=face.copy(),
                        y=np.arange(len(pos), dtype=np.int64))
            graphs.append(pre_transform(data) if pre_transform is not None else data)
        self.data, self.slices = self.collate(graphs)
