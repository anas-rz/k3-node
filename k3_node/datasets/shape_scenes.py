from typing import Callable, Optional

import numpy as np

from k3_node.data import Data, InMemoryDataset


class ShapeScenes(InMemoryDataset):
    r"""Small point cloud scenes for semantic segmentation, built from :class:`GeometricShapes`
    meshes (a compact stand-in for ShapeNet part segmentation).

    Every scene contains ``shapes_per_scene`` objects, each a cube, sphere, cone or torus that is
    randomly scaled, rotated and placed apart from the others. ``num_points`` points are sampled on
    their surfaces: ``pos`` holds the positions (scaled to the unit ball), ``x`` the surface
    normals, and ``y`` the type of the object each point lies on (4 classes). Training scenes use
    the training meshes, test scenes the test meshes; the scenes are generated with a fixed seed.

    Args:
        root (str): Directory where GeometricShapes is (or will be) downloaded.
        train (bool, optional): Build training scenes if ``True``, else test scenes.
        num_scenes (int, optional): Number of scenes. (default: ``200`` for training, ``50`` for test)
        shapes_per_scene (int, optional): Objects per scene. (default: ``3``)
        num_points (int, optional): Points per scene. (default: ``1024``)
        transform (callable, optional): A function applied to each scene when it is accessed.
        seed (int, optional): Random seed. (default: ``0``)
    """

    categories = ["3d_cube", "3d_sphere", "3d_cone", "3d_torus"]

    def __init__(self, root: str, train: bool = True, num_scenes: Optional[int] = None, shapes_per_scene: int = 3,
                 num_points: int = 1024, transform: Optional[Callable] = None, seed: int = 0):
        super().__init__(None, transform)
        from k3_node.datasets.geometric_shapes import GeometricShapes
        from k3_node.transforms.spatial import _np

        shapes = GeometricShapes(root, train=train)
        names = sorted(__import__("os").listdir(shapes.raw_dir))
        meshes = {}
        for i in range(len(shapes)):
            data = shapes[i]
            name = names[int(_np(data.y)[0])]
            if name in self.categories:
                meshes[self.categories.index(name)] = (_np(data.pos).astype(np.float64), _np(data.face))

        rng = np.random.default_rng(seed + (0 if train else 1))
        num_scenes = num_scenes or (200 if train else 50)
        self.data, self.slices = self.collate(
            [self._scene(rng, meshes, shapes_per_scene, num_points) for _ in range(num_scenes)])

    @staticmethod
    def _sample(rng, pos, face, num):
        a, b, c = pos[face[0]], pos[face[1]], pos[face[2]]
        cross = np.cross(b - a, c - a)
        area = np.linalg.norm(cross, axis=1)
        tri = rng.choice(len(area), size=num, p=area / area.sum())
        u, v = rng.random((2, num))
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        points = a[tri] + u[:, None] * (b - a)[tri] + v[:, None] * (c - a)[tri]
        normals = cross[tri] / np.maximum(area[tri, None], 1e-12)
        return points, normals

    @staticmethod
    def _rotation(rng):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    def _scene(self, rng, meshes, shapes_per_scene, num_points):
        counts = np.full(shapes_per_scene, num_points // shapes_per_scene)
        counts[: num_points - counts.sum()] += 1
        angle = rng.random() * 2 * np.pi
        positions, normals, labels = [], [], []
        for i in range(shapes_per_scene):
            label = int(rng.integers(len(self.categories)))
            pos, face = meshes[label]
            points, normal = self._sample(rng, pos, face, counts[i])
            points = points / np.abs(points).max()
            rot = self._rotation(rng)
            theta = angle + 2 * np.pi * i / shapes_per_scene  # objects around a circle, apart
            offset = 2.5 * np.array([np.cos(theta), np.sin(theta), 0.0])
            positions.append(points @ rot.T * rng.uniform(0.6, 1.0) + offset)
            normals.append(normal @ rot.T)
            labels.append(np.full(counts[i], label))
        pos = np.concatenate(positions)
        pos = pos - pos.mean(axis=0)
        pos = pos / np.linalg.norm(pos, axis=1).max()
        return Data(pos=pos.astype(np.float32), x=np.concatenate(normals).astype(np.float32),
                    y=np.concatenate(labels).astype(np.int64))
