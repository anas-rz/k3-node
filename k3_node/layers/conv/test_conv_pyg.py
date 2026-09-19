import pytest
import numpy as np
import keras
from keras import ops

import k3_node.layers.conv as conv


def _make_graph(num_nodes=5, in_channels=8):
    x = ops.ones((num_nodes, in_channels), dtype="float32")
    edge_index = ops.convert_to_tensor(
        [[0, 1, 2, 3, 4, 1], [1, 2, 3, 4, 0, 0]], dtype="int64"
    )
    return x, edge_index


def test_simple_conv():
    x, edge_index = _make_graph()
    layer = conv.SimpleConv(aggr="mean")
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 8)


def test_gcn_conv():
    x, edge_index = _make_graph()
    layer = conv.GCNConv(8, 16)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_cheb_conv():
    x, edge_index = _make_graph()
    layer = conv.ChebConv(8, 16, K=3)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_sage_conv():
    x, edge_index = _make_graph()
    layer = conv.SAGEConv(8, 16)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)

    # Bipartite
    layer_bip = conv.SAGEConv((8, 8), 16)
    out_bip = layer_bip((x, x), edge_index)
    assert ops.shape(out_bip) == (5, 16)


def test_graph_conv():
    x, edge_index = _make_graph()
    layer = conv.GraphConv(8, 16)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_gated_graph_conv():
    x, edge_index = _make_graph()
    layer = conv.GatedGraphConv(out_channels=8, num_layers=2)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 8)


def test_res_gated_graph_conv():
    x, edge_index = _make_graph()
    layer = conv.ResGatedGraphConv(8, 16)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_gat_conv():
    x, edge_index = _make_graph()
    layer = conv.GATConv(8, 16, heads=2, concat=True)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 32)


def test_gatv2_conv():
    x, edge_index = _make_graph()
    layer = conv.GATv2Conv(8, 16, heads=2, concat=False)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_transformer_conv():
    x, edge_index = _make_graph()
    layer = conv.TransformerConv(8, 16, heads=2)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 32)


def test_agnn_conv():
    x, edge_index = _make_graph()
    layer = conv.AGNNConv()
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 8)


def test_tag_conv():
    x, edge_index = _make_graph()
    layer = conv.TAGConv(8, 16, K=2)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_gin_conv():
    x, edge_index = _make_graph()
    mlp = keras.Sequential([
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(16),
    ])
    layer = conv.GINConv(mlp)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_gine_conv():
    x, edge_index = _make_graph()
    mlp = keras.Sequential([keras.layers.Dense(16)])
    layer = conv.GINEConv(mlp, edge_dim=4)
    edge_attr = ops.ones((6, 4))
    out = layer(x, edge_index, edge_attr=edge_attr)
    assert ops.shape(out) == (5, 16)


def test_arma_conv():
    x, edge_index = _make_graph()
    layer = conv.ARMAConv(16, num_stacks=2, num_layers=2)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_sg_conv():
    x, edge_index = _make_graph()
    layer = conv.SGConv(8, 16, K=2)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_ssg_conv():
    x, edge_index = _make_graph()
    layer = conv.SSGConv(8, 16, alpha=0.5, K=2)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_appnp():
    x, edge_index = _make_graph()
    layer = conv.APPNP(K=3, alpha=0.1)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 8)


def test_mf_conv():
    x, edge_index = _make_graph()
    layer = conv.MFConv(8, 16)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_rgcn_conv():
    x, edge_index = _make_graph()
    edge_type = ops.convert_to_tensor([0, 1, 0, 1, 0, 1], dtype="int64")
    layer = conv.RGCNConv(8, 16, num_relations=2)
    out = layer(x, edge_index, edge_type=edge_type)
    assert ops.shape(out) == (5, 16)


def test_rgat_conv():
    x, edge_index = _make_graph()
    edge_type = ops.convert_to_tensor([0, 1, 0, 1, 0, 1], dtype="int64")
    layer = conv.RGATConv(8, 16, num_relations=2, heads=2)
    out = layer(x, edge_index, edge_type=edge_type)
    assert ops.shape(out) == (5, 32)


def test_signed_conv():
    x, edge_index = _make_graph()
    layer = conv.SignedConv(8, 16, first_aggr=True)
    pos_edge_index = edge_index[:, :3]
    neg_edge_index = edge_index[:, 3:]
    out = layer(x, pos_edge_index, neg_edge_index)
    assert ops.shape(out) == (5, 32)


def test_dir_gnn_conv():
    x, edge_index = _make_graph()
    base_conv = conv.GCNConv(8, 16)
    layer = conv.DirGNNConv(base_conv)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_antisymmetric_conv():
    x, edge_index = _make_graph()
    layer = conv.AntiSymmetricConv(8)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 8)


def test_mixhop_conv():
    x, edge_index = _make_graph()
    layer = conv.MixHopConv(8, 16, powers=[0, 1, 2])
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 48)


def test_pdn_conv():
    x, edge_index = _make_graph()
    edge_attr = ops.ones((6, 4))
    layer = conv.PDNConv(8, 16, edge_dim=4, hidden_channels=8)
    out = layer(x, edge_index, edge_attr=edge_attr)
    assert ops.shape(out) == (5, 16)


def test_fa_conv():
    x, edge_index = _make_graph()
    x_0 = x
    layer = conv.FAConv(8, eps=0.1)
    out = layer(x, x_0, edge_index)
    assert ops.shape(out) == (5, 8)


def test_film_conv():
    x, edge_index = _make_graph()
    layer = conv.FiLMConv(8, 16)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_supergat_conv():
    x, edge_index = _make_graph()
    layer = conv.SuperGATConv(8, 16, heads=2, attention_type="MX")
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 32)


def test_eg_conv():
    x, edge_index = _make_graph()
    layer = conv.EGConv(8, 16, aggregators=["sum", "mean"])
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_pan_conv():
    x, edge_index = _make_graph()
    layer = conv.PANConv(8, 16, filter_size=2)
    out, M = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)
    assert ops.shape(M) == (5, 5)


def test_gen_conv():
    x, edge_index = _make_graph()
    layer = conv.GENConv(8, 16, aggr="softmax")
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_pna_conv():
    x, edge_index = _make_graph()
    deg = ops.convert_to_tensor([1, 2, 1, 1, 1], dtype="float32")
    layer = conv.PNAConv(8, 16, aggregators=["mean", "max"], scalers=["identity"], deg=deg)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_le_conv():
    x, edge_index = _make_graph()
    layer = conv.LEConv(8, 16)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_cluster_gcn_conv():
    x, edge_index = _make_graph()
    layer = conv.ClusterGCNConv(8, 16)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_gcn2_conv():
    x, edge_index = _make_graph()
    x_0 = x
    layer = conv.GCN2Conv(channels=8, alpha=0.1)
    out = layer(x, x_0, edge_index)
    assert ops.shape(out) == (5, 8)


def test_lg_conv():
    x, edge_index = _make_graph()
    layer = conv.LGConv()
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 8)


def test_nn_conv_and_ec_conv():
    x, edge_index = _make_graph()
    nn = keras.Sequential([keras.layers.Dense(8 * 16)])
    layer = conv.NNConv(8, 16, nn=nn)
    edge_attr = ops.ones((6, 4))
    out = layer(x, edge_index, edge_attr=edge_attr)
    assert ops.shape(out) == (5, 16)
    # Check ECConv alias
    assert conv.ECConv is conv.NNConv


def test_cg_conv():
    x, edge_index = _make_graph()
    layer = conv.CGConv(8)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 8)


def test_edge_conv_and_dynamic():
    x, edge_index = _make_graph()
    nn = keras.Sequential([keras.layers.Dense(16)])
    layer = conv.EdgeConv(nn)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)

    dyn_layer = conv.DynamicEdgeConv(nn, k=2)
    out_dyn = dyn_layer(x)
    assert ops.shape(out_dyn) == (5, 16)


def test_general_conv():
    x, edge_index = _make_graph()
    layer = conv.GeneralConv(8, 16)
    out = layer(x, edge_index)
    assert ops.shape(out) == (5, 16)


def test_point_cloud_convs():
    x, edge_index = _make_graph()
    pos = ops.ones((5, 3))

    # PointNetConv
    local_nn = keras.Sequential([keras.layers.Dense(16)])
    pnet = conv.PointNetConv(local_nn=local_nn)
    out_pnet = pnet(x, pos, edge_index)
    assert ops.shape(out_pnet) == (5, 16)

    # PointConv
    pconv = conv.PointConv(local_nn=local_nn)
    out_pconv = pconv(x, pos, edge_index)
    assert ops.shape(out_pconv) == (5, 16)

    # PointTransformerConv
    pt_conv = conv.PointTransformerConv(in_channels=8, out_channels=16, dim=3)
    out_pt = pt_conv(x, pos, edge_index)
    assert ops.shape(out_pt) == (5, 16)

    # PointGNNConv
    mlp_h = keras.Sequential([keras.layers.Dense(3)])
    mlp_f = keras.Sequential([keras.layers.Dense(8)])
    mlp_g = keras.Sequential([keras.layers.Dense(8)])
    pgnn = conv.PointGNNConv(mlp_h=mlp_h, mlp_f=mlp_f, mlp_g=mlp_g)
    out_pgnn = pgnn(x, pos, edge_index)
    assert ops.shape(out_pgnn) == (5, 8)

    # PPFConv
    normal = ops.ones((5, 3))
    ppf_nn = keras.Sequential([keras.layers.Dense(16)])
    ppf = conv.PPFConv(local_nn=ppf_nn)
    out_ppf = ppf(x, pos, normal, edge_index)
    assert ops.shape(out_ppf) == (5, 16)

    # FeaStConv
    feast = conv.FeaStConv(8, 16, heads=2)
    out_feast = feast(x, edge_index)
    assert ops.shape(out_feast) == (5, 16)

    # GMMConv
    gmm = conv.GMMConv(8, 16, dim=2, kernel_size=2)
    pseudo = ops.ones((6, 2))
    out_gmm = gmm(x, edge_index, edge_attr=pseudo)
    assert ops.shape(out_gmm) == (5, 16)

    # GravNetConv
    grav = conv.GravNetConv(in_channels=8, out_channels=16, space_dimensions=3, propagate_dimensions=4, k=2)
    out_grav = grav(x)
    assert ops.shape(out_grav) == (5, 16)


def test_meshcnn_conv():
    conv_layer = conv.MeshCNNConv(4, 8)
    x = ops.ones((4, 4))
    edge_index = ops.convert_to_tensor([
        [1, 2, 3, 0, 0, 2, 3, 1, 0, 1, 3, 2, 0, 1, 2, 3],
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    ], dtype="int64")
    out = conv_layer(x, edge_index)
    assert ops.shape(out) == (4, 8)


def test_x_conv():
    conv_layer = conv.XConv(in_channels=8, out_channels=16, dim=3, kernel_size=4)
    x = ops.ones((10, 8))
    pos = ops.ones((10, 3))
    out = conv_layer(x, pos)
    assert ops.shape(out) == (10, 16)


def test_spline_conv():
    conv_layer = conv.SplineConv(in_channels=4, out_channels=8, dim=2, kernel_size=[3, 3])
    x = ops.ones((5, 4))
    edge_index = ops.convert_to_tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype="int64")
    edge_attr = ops.convert_to_tensor([[0.2, 0.5], [0.8, 0.1], [0.5, 0.9], [0.0, 1.0]], dtype="float32")
    out = conv_layer(x, edge_index, edge_attr=edge_attr)
    assert ops.shape(out) == (5, 8)


def test_hetero_conv():
    conv_layer = conv.HeteroConv({
        ("user", "likes", "item"): conv.SAGEConv((4, 4), 8),
        ("item", "liked_by", "user"): conv.SAGEConv((4, 4), 8),
    }, aggr="sum")
    x_dict = {"user": ops.ones((3, 4)), "item": ops.ones((5, 4))}
    edge_index_dict = {
        ("user", "likes", "item"): ops.convert_to_tensor([[0, 1, 2], [1, 2, 3]], dtype="int64"),
        ("item", "liked_by", "user"): ops.convert_to_tensor([[1, 2, 3], [0, 1, 2]], dtype="int64"),
    }
    out = conv_layer(x_dict, edge_index_dict)
    assert ops.shape(out["user"]) == (3, 8)
    assert ops.shape(out["item"]) == (5, 8)


def test_hgt_conv():
    metadata = (["author", "paper"], [("author", "writes", "paper"), ("paper", "written_by", "author")])
    conv_layer = conv.HGTConv(in_channels={"author": 8, "paper": 8}, out_channels=8, metadata=metadata, heads=2)
    x_dict = {"author": ops.ones((3, 8)), "paper": ops.ones((4, 8))}
    edge_index_dict = {
        ("author", "writes", "paper"): ops.convert_to_tensor([[0, 1, 2], [1, 2, 3]], dtype="int64"),
        ("paper", "written_by", "author"): ops.convert_to_tensor([[1, 2, 3], [0, 1, 2]], dtype="int64"),
    }
    out = conv_layer(x_dict, edge_index_dict)
    assert ops.shape(out["author"]) == (3, 8)
    assert ops.shape(out["paper"]) == (4, 8)


def test_han_conv():
    metadata = (["author", "paper"], [("author", "writes", "paper"), ("paper", "written_by", "author")])
    conv_layer = conv.HANConv(in_channels={"author": 8, "paper": 8}, out_channels=8, metadata=metadata, heads=2)
    x_dict = {"author": ops.ones((3, 8)), "paper": ops.ones((4, 8))}
    edge_index_dict = {
        ("author", "writes", "paper"): ops.convert_to_tensor([[0, 1, 2], [1, 2, 3]], dtype="int64"),
        ("paper", "written_by", "author"): ops.convert_to_tensor([[1, 2, 3], [0, 1, 2]], dtype="int64"),
    }
    out = conv_layer(x_dict, edge_index_dict)
    assert ops.shape(out["author"]) == (3, 8)
    assert ops.shape(out["paper"]) == (4, 8)


def test_heat_conv():
    conv_layer = conv.HEATConv(
        in_channels=8, out_channels=16, num_node_types=2, num_edge_types=3,
        edge_type_emb_dim=4, edge_dim=5, edge_attr_emb_dim=6, heads=2
    )
    x = ops.ones((4, 8))
    edge_index = ops.convert_to_tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype="int64")
    node_type = ops.convert_to_tensor([0, 1, 0, 1], dtype="int64")
    edge_type = ops.convert_to_tensor([0, 1, 2, 0], dtype="int64")
    edge_attr = ops.ones((4, 5))
    out = conv_layer(x, edge_index, node_type, edge_type, edge_attr)
    assert ops.shape(out) == (4, 32)


def test_hypergraph_conv():
    conv_layer = conv.HypergraphConv(8, 16, use_attention=True, heads=2)
    x = ops.ones((4, 8))
    hyperedge_index = ops.convert_to_tensor([[0, 1, 2, 1, 2, 3], [0, 0, 0, 1, 1, 1]], dtype="int64")
    hyperedge_attr = ops.ones((2, 8))
    out = conv_layer(x, hyperedge_index, hyperedge_attr=hyperedge_attr)
    assert ops.shape(out) == (4, 32)


def test_dna_conv():
    conv_layer = conv.DNAConv(channels=8, heads=2, groups=2)
    x = ops.ones((4, 3, 8))
    edge_index = ops.convert_to_tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype="int64")
    out = conv_layer(x, edge_index)
    assert ops.shape(out) == (4, 8)


def test_wl_conv():
    layer = conv.WLConv()
    x = ops.convert_to_tensor([0, 1, 0, 1], dtype="int64")
    edge_index = ops.convert_to_tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype="int64")
    out = layer(x, edge_index)
    assert ops.shape(out) == (4,)
    hist = layer.histogram(out)
    assert ops.shape(hist)[0] == 1

    layer_c = conv.WLConvContinuous()
    xc = ops.ones((4, 8))
    out_c = layer_c(xc, edge_index)
    assert ops.shape(out_c) == (4, 8)


def test_gps_conv():
    conv_layer = conv.GPSConv(channels=16, conv=conv.GCNConv(16, 16), heads=2)
    x = ops.ones((6, 16))
    edge_index = ops.convert_to_tensor([[0, 1, 2, 3, 4], [1, 2, 0, 4, 5]], dtype="int64")
    batch = ops.convert_to_tensor([0, 0, 0, 1, 1, 1], dtype="int64")
    out = conv_layer(x, edge_index, batch=batch)
    assert ops.shape(out) == (6, 16)


def test_cugraph_compatibility():
    x, edge_index = _make_graph()
    gat = conv.CuGraphGATConv(8, 16, heads=2)
    out_gat = gat(x, edge_index)
    assert ops.shape(out_gat) == (5, 32)

    sage = conv.CuGraphSAGEConv(8, 16)
    out_sage = sage(x, edge_index)
    assert ops.shape(out_sage) == (5, 16)

    rgcn = conv.CuGraphRGCNConv(8, 16, num_relations=2)
    edge_type = ops.convert_to_tensor([0, 1, 0, 1, 0, 1], dtype="int64")
    out_rgcn = rgcn(x, edge_index, edge_type=edge_type)
    assert ops.shape(out_rgcn) == (5, 16)
