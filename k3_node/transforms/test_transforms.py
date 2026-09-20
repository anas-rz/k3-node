import math
import numpy as np
import pytest

try:
    import torch
except ImportError:
    torch = None

if torch is None:
    pytest.skip("PyTorch is required for transforms unit tests", allow_module_level=True)

import k3_node.transforms as T
from k3_node.data import Data, HeteroData


def test_compose():
    data = Data(x=torch.ones((4, 2)), edge_index=torch.tensor([[0, 1], [1, 2]]))
    transform = T.Compose([
        T.Constant(value=2.0, cat=False),
        T.AddSelfLoops(),
    ])
    out = transform(data)
    assert out.x.shape == (4, 1)
    assert out.x[0, 0].item() == 2.0
    assert out.edge_index.shape[1] == 2 + 4


def test_compose_filters():
    data1 = Data(num_nodes=5)
    data2 = Data(num_nodes=2)
    filt = T.ComposeFilters([lambda d: d.num_nodes > 3])
    assert filt(data1) is True
    assert filt(data2) is False


def test_general_transforms():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    # Constant
    d_const = T.Constant(value=5.0, cat=True)(data)
    assert d_const.x.shape == (2, 3)

    # NormalizeFeatures
    d_norm = T.NormalizeFeatures()(data)
    assert torch.allclose(d_norm.x.sum(dim=-1), torch.tensor([1.0, 1.0]))

    # SVDFeatureReduction
    d_svd = T.SVDFeatureReduction(out_channels=1)(data)
    assert d_svd.x.shape == (2, 1)

    # IndexToMask & MaskToIndex
    data_idx = Data(num_nodes=4, train_index=torch.tensor([0, 2]))
    d_mask = T.IndexToMask(attrs='train_index')(data_idx)
    assert hasattr(d_mask, 'train_mask')
    assert d_mask.train_mask.tolist() == [True, False, True, False]
    d_idx2 = T.MaskToIndex(attrs='train_mask')(d_mask)
    assert hasattr(d_idx2, 'train_index')
    assert d_idx2.train_index.tolist() == [0, 2]

    # Pad
    d_pad = T.Pad(max_num_nodes=5)(data)
    assert d_pad.x.shape == (5, 2)
    assert d_pad.num_nodes == 5

    # RandomNodeSplit
    data_nodes = Data(num_nodes=10, y=torch.tensor([0, 1] * 5))
    d_split = T.RandomNodeSplit(split='train_rest', num_val=2, num_test=2)(data_nodes)
    assert d_split.train_mask.sum() == 6
    assert d_split.val_mask.sum() == 2
    assert d_split.test_mask.sum() == 2

    # RemoveTrainingClasses
    data_cls = Data(y=torch.tensor([0, 1, 2, 0, 1, 2]))
    d_rem = T.RemoveTrainingClasses(classes=[0])(data_cls)
    assert (d_rem.y == 0).sum() == 0

    # ToSparseTensor
    d_adj = T.ToSparseTensor()(data)
    assert hasattr(d_adj, 'adj_t')


def test_graph_transforms():
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
    x = torch.randn(3, 4)
    data = Data(x=x, edge_index=edge_index, num_nodes=3)

    # ToUndirected
    d_undir = T.ToUndirected()(data)
    assert d_undir.is_undirected()

    # OneHotDegree
    d_ohd = T.OneHotDegree(max_degree=3)(data)
    assert d_ohd.x.shape[0] == 3

    # TargetIndegree
    d_tid = T.TargetIndegree()(data)
    assert d_tid.edge_attr is not None

    # LocalDegreeProfile
    d_ldp = T.LocalDegreeProfile()(data)
    assert d_ldp.x.shape == (3, 4 + 5)

    # AddSelfLoops & RemoveSelfLoops
    d_sl = T.AddSelfLoops()(data)
    assert d_sl.edge_index.shape[1] == 3 + 3
    d_rem_sl = T.RemoveSelfLoops()(d_sl)
    assert d_rem_sl.edge_index.shape[1] == 3

    # RemoveIsolatedNodes
    data_iso = Data(edge_index=torch.tensor([[0], [1]]), num_nodes=3, x=torch.randn(3, 2))
    d_no_iso = T.RemoveIsolatedNodes()(data_iso)
    assert d_no_iso.num_nodes == 2

    # RemoveDuplicatedEdges
    data_dup = Data(edge_index=torch.tensor([[0, 0], [1, 1]]), num_nodes=2)
    d_no_dup = T.RemoveDuplicatedEdges()(data_dup)
    assert d_no_dup.edge_index.shape[1] == 1

    # KNNGraph & RadiusGraph
    data_pos = Data(pos=torch.tensor([[0.0, 0.0], [0.1, 0.0], [2.0, 2.0]]))
    d_knn = T.KNNGraph(k=1)(data_pos)
    assert d_knn.edge_index is not None
    d_rad = T.RadiusGraph(r=0.5)(data_pos)
    assert d_rad.edge_index is not None

    # TwoHop
    d_2hop = T.TwoHop()(data)
    assert d_2hop.edge_index.shape[1] >= 3

    # LineGraph
    d_line = T.LineGraph()(data)
    assert d_line.num_nodes == 3

    # LaplacianLambdaMax
    d_lap = T.LaplacianLambdaMax()(data)
    assert hasattr(d_lap, 'lambda_max')

    # GDC
    d_gdc = T.GDC(self_loop_weight=1.0, normalization_in='sym', normalization_out='sym',
                  diffusion_kwargs=dict(method='ppr', alpha=0.15))(data)
    assert d_gdc.edge_index is not None

    # SIGN
    d_sign = T.SIGN(K=2)(data)
    assert hasattr(d_sign, 'x1') and hasattr(d_sign, 'x2')

    # GCNNorm
    d_gcn = T.GCNNorm()(data)
    assert d_gcn.edge_weight is not None

    # VirtualNode
    d_vn = T.VirtualNode()(data)
    assert d_vn.num_nodes == 4

    # AddLaplacianEigenvectorPE
    d_lpe = T.AddLaplacianEigenvectorPE(k=2)(data)
    assert hasattr(d_lpe, 'laplacian_eigenvector_pe')

    # AddRandomWalkPE
    d_rwpe = T.AddRandomWalkPE(walk_length=4)(data)
    assert hasattr(d_rwpe, 'random_walk_pe')

    # FeaturePropagation
    d_fp = T.FeaturePropagation(missing_mask=torch.tensor([True, False, False]), num_iterations=5)(data)
    assert d_fp.x.shape == (3, 4)

    # HalfHop
    d_hh = T.HalfHop(alpha=0.5)(data)
    assert d_hh.num_nodes > 3


def test_spatial_transforms():
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    face = torch.tensor([[0], [1], [2]], dtype=torch.long)
    data = Data(pos=pos, edge_index=edge_index, face=face, num_nodes=3)

    # Distance
    d_dist = T.Distance(norm=False)(data)
    assert d_dist.edge_attr.shape == (4, 1)

    # Cartesian & LocalCartesian
    d_cart = T.Cartesian(norm=False)(data)
    assert d_cart.edge_attr.shape == (4, 3)
    d_lcart = T.LocalCartesian(norm=False)(data)
    assert d_lcart.edge_attr.shape == (4, 3)

    # Polar & Spherical
    data_2d = Data(pos=pos[:, :2], edge_index=edge_index)
    d_pol = T.Polar(norm=False)(data_2d)
    assert d_pol.edge_attr.shape == (4, 2)
    d_sph = T.Spherical(norm=False)(data)
    assert d_sph.edge_attr.shape == (4, 3)

    # Center, NormalizeScale, NormalizeRotation
    d_cen = T.Center()(data)
    assert torch.allclose(d_cen.pos.mean(dim=0), torch.zeros(3), atol=1e-5)

    d_nscale = T.NormalizeScale()(data)
    assert d_nscale.pos.abs().max() <= 1.0

    d_nrot = T.NormalizeRotation()(data)
    assert d_nrot.pos.shape == (3, 3)

    # RandomJitter, RandomFlip, RandomScale, RandomRotate, RandomShear
    d_jit = T.RandomJitter(translate=0.1)(data)
    assert d_jit.pos.shape == (3, 3)

    d_flip = T.RandomFlip(axis=0, p=1.0)(data)
    assert torch.allclose(d_flip.pos[:, 0], -data.pos[:, 0])

    d_scale = T.RandomScale(scales=(1.5, 1.5))(data)
    assert torch.allclose(d_scale.pos, data.pos * 1.5)

    d_rot = T.RandomRotate(degrees=90, axis=0)(data)
    assert d_rot.pos.shape == (3, 3)

    d_shear = T.RandomShear(shear=0.1)(data)
    assert d_shear.pos.shape == (3, 3)

    # FaceToEdge
    d_f2e = T.FaceToEdge(remove_faces=True)(data)
    assert d_f2e.face is None
    assert d_f2e.edge_index is not None

    # GenerateMeshNormals
    d_mesh = T.GenerateMeshNormals()(data)
    assert hasattr(d_mesh, 'norm')
    assert d_mesh.norm.shape == (3, 3)

    # PointPairFeatures
    data_ppf = Data(pos=pos, norm=d_mesh.norm, edge_index=edge_index)
    d_ppf = T.PointPairFeatures()(data_ppf)
    assert d_ppf.edge_attr.shape == (4, 4)

    # Delaunay
    data_pts = Data(pos=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    d_del = T.Delaunay()(data_pts)
    assert hasattr(d_del, 'face')

    # FixedPoints
    d_fixed = T.FixedPoints(num=2)(data)
    assert d_fixed.pos.shape[0] == 2
    assert d_fixed.num_nodes == 2

    # GridSampling
    data_cloud = Data(pos=pos)
    d_grid = T.GridSampling(size=0.5)(data_cloud)
    assert d_grid.pos is not None
    assert d_grid.num_nodes > 0
