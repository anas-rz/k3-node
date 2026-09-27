"""Spatial transforms on every backend: inputs may be NumPy arrays or backend tensors."""
import numpy as np
import pytest
from keras import ops

import k3_node.transforms as T
from k3_node.data import Data


def _data(as_tensor):
    conv = ops.convert_to_tensor if as_tensor else np.asarray
    pos = conv(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32"))
    edge_index = conv(np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype="int64"))
    face = conv(np.array([[0], [1], [2]], dtype="int64"))
    return Data(pos=pos, edge_index=edge_index, face=face, num_nodes=3)


@pytest.mark.parametrize("as_tensor", [False, True])
def test_spatial_transforms_all_backends(as_tensor):
    data = _data(as_tensor)
    np_ = ops.convert_to_numpy

    assert tuple(T.Distance(norm=False)(data).edge_attr.shape) == (4, 1)
    cart = T.Cartesian(norm=False)(data).edge_attr
    assert tuple(cart.shape) == (4, 3)
    np.testing.assert_allclose(np_(cart)[0], [-1.0, 0.0, 0.0])
    assert str(np_(cart).dtype) == "float32"
    assert tuple(T.Cartesian()(T.Distance()(data)).edge_attr.shape) == (4, 4)  # cat=True appends
    assert tuple(T.LocalCartesian()(data).edge_attr.shape) == (4, 3)

    data_2d = Data(pos=data.pos[:, :2], edge_index=data.edge_index)
    assert tuple(T.Polar()(data_2d).edge_attr.shape) == (4, 2)
    assert tuple(T.Spherical()(data).edge_attr.shape) == (4, 3)

    np.testing.assert_allclose(np_(T.Center()(data).pos).mean(0), np.zeros(3), atol=1e-6)
    assert np.abs(np_(T.NormalizeScale()(data).pos)).max() <= 1.0
    assert tuple(T.NormalizeRotation(sort=True)(data).pos.shape) == (3, 3)

    assert tuple(T.RandomJitter(0.1)(data).pos.shape) == (3, 3)
    np.testing.assert_allclose(np_(T.RandomFlip(axis=0, p=1.0)(data).pos)[:, 0], -np_(data.pos)[:, 0])
    np.testing.assert_allclose(np_(T.RandomScale((1.5, 1.5))(data).pos), np_(data.pos) * 1.5)
    rot = T.RandomRotate(degrees=(90, 90), axis=2)(data).pos
    np.testing.assert_allclose(np.linalg.norm(np_(rot), axis=1), np.linalg.norm(np_(data.pos), axis=1), atol=1e-6)
    assert tuple(T.RandomShear(0.1)(data).pos.shape) == (3, 3)

    f2e = T.FaceToEdge()(data)
    assert f2e.face is None and tuple(f2e.edge_index.shape) == (2, 6)

    mesh = T.GenerateMeshNormals()(data)
    np.testing.assert_allclose(np.abs(np_(mesh.norm)[:, 2]), np.ones(3), atol=1e-6)
    ppf = T.PointPairFeatures()(Data(pos=data.pos, norm=mesh.norm, edge_index=data.edge_index))
    assert tuple(ppf.edge_attr.shape) == (4, 4)

    sampled = T.SamplePoints(16, include_normals=True)(data)
    assert tuple(sampled.pos.shape) == (16, 3) and tuple(sampled.normal.shape) == (16, 3)
    assert np.all(np.abs(np_(sampled.pos)[:, 2]) < 1e-6)  # all samples lie on the triangle's plane

    fixed = T.FixedPoints(2)(data)
    assert tuple(fixed.pos.shape) == (2, 3) and fixed.num_nodes == 2

    grid = T.GridSampling(size=0.5)(Data(pos=data.pos))
    assert grid.num_nodes == 3

    square = Data(pos=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype="float32"))
    assert tuple(T.Delaunay()(square).face.shape) == (3, 2)
