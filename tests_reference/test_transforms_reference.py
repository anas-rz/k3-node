import numpy as np
import pytest
import torch
import torch_geometric.data as pyg_data
import torch_geometric.transforms as PyGT

import k3_node.data as k3_data
import k3_node.transforms as K3T


def _to_np(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)


def test_reference_general_transforms_parity():
    # Constant
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    pyg_d = pyg_data.Data(x=x.clone(), edge_index=ei.clone())
    k3_d = k3_data.Data(x=x.clone(), edge_index=ei.clone())

    pyg_out = PyGT.Constant(value=7.0, cat=True)(pyg_d)
    k3_out = K3T.Constant(value=7.0, cat=True)(k3_d)
    assert np.allclose(_to_np(pyg_out.x), _to_np(k3_out.x))

    # NormalizeFeatures
    pyg_d2 = pyg_data.Data(x=torch.tensor([[1.0, 3.0], [2.0, 2.0]]), edge_index=ei.clone())
    k3_d2 = k3_data.Data(x=torch.tensor([[1.0, 3.0], [2.0, 2.0]]), edge_index=ei.clone())
    pyg_out2 = PyGT.NormalizeFeatures()(pyg_d2)
    k3_out2 = K3T.NormalizeFeatures()(k3_d2)
    assert np.allclose(_to_np(pyg_out2.x), _to_np(k3_out2.x))

    # IndexToMask & MaskToIndex
    pyg_d3 = pyg_data.Data(num_nodes=5, train_index=torch.tensor([1, 3]))
    k3_d3 = k3_data.Data(num_nodes=5, train_index=torch.tensor([1, 3]))
    pyg_out3 = PyGT.IndexToMask(attrs='train_index')(pyg_d3)
    k3_out3 = K3T.IndexToMask(attrs='train_index')(k3_d3)
    assert np.array_equal(_to_np(pyg_out3.train_mask), _to_np(k3_out3.train_mask))

    pyg_out3b = PyGT.MaskToIndex(attrs='train_mask')(pyg_out3)
    k3_out3b = K3T.MaskToIndex(attrs='train_mask')(k3_out3)
    assert np.array_equal(_to_np(pyg_out3b.train_index), _to_np(k3_out3b.train_index))

    # Pad
    pyg_d4 = pyg_data.Data(x=x.clone(), edge_index=ei.clone())
    k3_d4 = k3_data.Data(x=x.clone(), edge_index=ei.clone())
    pyg_out4 = PyGT.Pad(max_num_nodes=4)(pyg_d4)
    k3_out4 = K3T.Pad(max_num_nodes=4)(k3_d4)
    assert np.allclose(_to_np(pyg_out4.x), _to_np(k3_out4.x))
    assert np.allclose(_to_np(pyg_out4.edge_index), _to_np(k3_out4.edge_index))


def test_reference_graph_transforms_parity():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    pyg_d = pyg_data.Data(x=x.clone(), edge_index=ei.clone(), num_nodes=3)
    k3_d = k3_data.Data(x=x.clone(), edge_index=ei.clone(), num_nodes=3)

    # ToUndirected
    pyg_undir = PyGT.ToUndirected()(pyg_d)
    k3_undir = K3T.ToUndirected()(k3_d)
    assert np.allclose(
        np.sort(_to_np(pyg_undir.edge_index), axis=1),
        np.sort(_to_np(k3_undir.edge_index), axis=1),
    )

    # OneHotDegree
    pyg_ohd = PyGT.OneHotDegree(max_degree=3)(pyg_d)
    k3_ohd = K3T.OneHotDegree(max_degree=3)(k3_d)
    assert np.allclose(_to_np(pyg_ohd.x), _to_np(k3_ohd.x))

    # AddSelfLoops & RemoveSelfLoops
    pyg_sl = PyGT.AddSelfLoops()(pyg_d)
    k3_sl = K3T.AddSelfLoops()(k3_d)
    assert np.allclose(
        np.sort(_to_np(pyg_sl.edge_index), axis=1),
        np.sort(_to_np(k3_sl.edge_index), axis=1),
    )

    pyg_rsl = PyGT.RemoveSelfLoops()(pyg_sl)
    k3_rsl = K3T.RemoveSelfLoops()(k3_sl)
    assert np.allclose(
        np.sort(_to_np(pyg_rsl.edge_index), axis=1),
        np.sort(_to_np(k3_rsl.edge_index), axis=1),
    )

    # RemoveIsolatedNodes
    ei_iso = torch.tensor([[0], [1]], dtype=torch.long)
    x_iso = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    pyg_iso = pyg_data.Data(x=x_iso.clone(), edge_index=ei_iso.clone(), num_nodes=3)
    k3_iso = k3_data.Data(x=x_iso.clone(), edge_index=ei_iso.clone(), num_nodes=3)
    pyg_no_iso = PyGT.RemoveIsolatedNodes()(pyg_iso)
    k3_no_iso = K3T.RemoveIsolatedNodes()(k3_iso)
    assert pyg_no_iso.num_nodes == k3_no_iso.num_nodes
    assert np.allclose(_to_np(pyg_no_iso.x), _to_np(k3_no_iso.x))

    # RemoveDuplicatedEdges
    ei_dup = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.long)
    pyg_dup = pyg_data.Data(edge_index=ei_dup.clone(), num_nodes=2)
    k3_dup = k3_data.Data(edge_index=ei_dup.clone(), num_nodes=2)
    pyg_no_dup = PyGT.RemoveDuplicatedEdges()(pyg_dup)
    k3_no_dup = K3T.RemoveDuplicatedEdges()(k3_dup)
    assert pyg_no_dup.edge_index.shape[1] == k3_no_dup.edge_index.shape[1]

    # TwoHop
    pyg_2hop = PyGT.TwoHop()(pyg_d)
    k3_2hop = K3T.TwoHop()(k3_d)
    assert np.allclose(
        np.sort(_to_np(pyg_2hop.edge_index), axis=1),
        np.sort(_to_np(k3_2hop.edge_index), axis=1),
    )

    # LineGraph
    pyg_lg = PyGT.LineGraph()(pyg_d)
    k3_lg = K3T.LineGraph()(k3_d)
    assert pyg_lg.num_nodes == k3_lg.num_nodes
    assert pyg_lg.edge_index.shape == k3_lg.edge_index.shape

    # GCNNorm
    pyg_gcn = PyGT.GCNNorm()(pyg_d)
    k3_gcn = K3T.GCNNorm()(k3_d)
    assert np.allclose(_to_np(pyg_gcn.edge_weight), _to_np(k3_gcn.edge_weight), atol=1e-5)

    # AddRandomWalkPE
    pyg_rwpe = PyGT.AddRandomWalkPE(walk_length=3)(pyg_d)
    k3_rwpe = K3T.AddRandomWalkPE(walk_length=3)(k3_d)
    assert np.allclose(_to_np(pyg_rwpe.random_walk_pe), _to_np(k3_rwpe.random_walk_pe), atol=1e-5)

    # LaplacianLambdaMax
    pyg_lap = PyGT.LaplacianLambdaMax()(pyg_d)
    k3_lap = K3T.LaplacianLambdaMax()(k3_d)
    assert np.isclose(pyg_lap.lambda_max, k3_lap.lambda_max, atol=1e-5)

    # VirtualNode
    pyg_vn = PyGT.VirtualNode()(pyg_d)
    k3_vn = K3T.VirtualNode()(k3_d)
    assert pyg_vn.num_nodes == k3_vn.num_nodes
    assert pyg_vn.edge_index.shape == k3_vn.edge_index.shape


def test_reference_spatial_transforms_parity():
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    face = torch.tensor([[0], [1], [2]], dtype=torch.long)

    pyg_d = pyg_data.Data(pos=pos.clone(), edge_index=ei.clone(), face=face.clone(), num_nodes=3)
    k3_d = k3_data.Data(pos=pos.clone(), edge_index=ei.clone(), face=face.clone(), num_nodes=3)

    # Distance
    pyg_dist = PyGT.Distance(norm=False)(pyg_d)
    k3_dist = K3T.Distance(norm=False)(k3_d)
    assert np.allclose(_to_np(pyg_dist.edge_attr), _to_np(k3_dist.edge_attr), atol=1e-5)

    # Cartesian & LocalCartesian
    pyg_cart = PyGT.Cartesian(norm=False)(pyg_d)
    k3_cart = K3T.Cartesian(norm=False)(k3_d)
    assert np.allclose(_to_np(pyg_cart.edge_attr), _to_np(k3_cart.edge_attr), atol=1e-5)

    pyg_lcart = PyGT.LocalCartesian(norm=False)(pyg_d)
    k3_lcart = K3T.LocalCartesian(norm=False)(k3_d)
    assert np.allclose(_to_np(pyg_lcart.edge_attr), _to_np(k3_lcart.edge_attr), atol=1e-5)

    # Polar
    pyg_2d = pyg_data.Data(pos=pos[:, :2].clone(), edge_index=ei.clone())
    k3_2d = k3_data.Data(pos=pos[:, :2].clone(), edge_index=ei.clone())
    pyg_pol = PyGT.Polar(norm=False)(pyg_2d)
    k3_pol = K3T.Polar(norm=False)(k3_2d)
    assert np.allclose(_to_np(pyg_pol.edge_attr), _to_np(k3_pol.edge_attr), atol=1e-5)

    # Spherical
    pyg_sph = PyGT.Spherical(norm=False)(pyg_d)
    k3_sph = K3T.Spherical(norm=False)(k3_d)
    assert np.allclose(_to_np(pyg_sph.edge_attr), _to_np(k3_sph.edge_attr), atol=1e-5)

    # Center & NormalizeScale
    pyg_cen = PyGT.Center()(pyg_d)
    k3_cen = K3T.Center()(k3_d)
    assert np.allclose(_to_np(pyg_cen.pos), _to_np(k3_cen.pos), atol=1e-5)

    pyg_ns = PyGT.NormalizeScale()(pyg_d)
    k3_ns = K3T.NormalizeScale()(k3_d)
    assert np.allclose(_to_np(pyg_ns.pos), _to_np(k3_ns.pos), atol=1e-5)

    # LinearTransformation
    mat = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [2.0, 0.0, 1.0]], dtype=torch.float32)
    pyg_lt = PyGT.LinearTransformation(mat)(pyg_d)
    k3_lt = K3T.LinearTransformation(mat)(k3_d)
    assert np.allclose(_to_np(pyg_lt.pos), _to_np(k3_lt.pos), atol=1e-5)

    # FaceToEdge
    pyg_f2e = PyGT.FaceToEdge(remove_faces=True)(pyg_d)
    k3_f2e = K3T.FaceToEdge(remove_faces=True)(k3_d)
    assert np.allclose(
        np.sort(_to_np(pyg_f2e.edge_index), axis=1),
        np.sort(_to_np(k3_f2e.edge_index), axis=1),
    )

    # GenerateMeshNormals
    pyg_norm = PyGT.GenerateMeshNormals()(pyg_d)
    k3_norm = K3T.GenerateMeshNormals()(k3_d)
    assert np.allclose(_to_np(pyg_norm.norm), _to_np(k3_norm.norm), atol=1e-5)

    # PointPairFeatures
    pyg_ppf_in = pyg_data.Data(pos=pos.clone(), norm=pyg_norm.norm.clone(), edge_index=ei.clone())
    k3_ppf_in = k3_data.Data(pos=pos.clone(), norm=k3_norm.norm.clone(), edge_index=ei.clone())
    pyg_ppf = PyGT.PointPairFeatures()(pyg_ppf_in)
    k3_ppf = K3T.PointPairFeatures()(k3_ppf_in)
    assert np.allclose(_to_np(pyg_ppf.edge_attr), _to_np(k3_ppf.edge_attr), atol=1e-5)

    # Delaunay
    pts = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    pyg_del_in = pyg_data.Data(pos=pts.clone())
    k3_del_in = k3_data.Data(pos=pts.clone())
    pyg_del = PyGT.Delaunay()(pyg_del_in)
    k3_del = K3T.Delaunay()(k3_del_in)
    assert np.allclose(
        np.sort(_to_np(pyg_del.face), axis=0),
        np.sort(_to_np(k3_del.face), axis=0),
    )
