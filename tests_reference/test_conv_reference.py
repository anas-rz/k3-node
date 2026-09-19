"""
Sanity-check reference parity tests against PyTorch Geometric.

These tests verify that k3-node convolution layers produce numerically
equivalent outputs to torch_geometric.nn.conv reference implementations.
These tests are intended for local validation and documentation.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import pytest
import torch
from keras import ops

import torch_geometric.nn.conv as pyg_conv
import k3_node.layers.conv as k3_conv


def test_reference_gcn_conv():
    torch.manual_seed(42)
    conv_pyg = pyg_conv.GCNConv(8, 16)
    conv_k3 = k3_conv.GCNConv(8, 16)
    conv_k3.build((None, 8))
    conv_k3.lin.kernel.assign(conv_pyg.lin.weight.detach().t())
    conv_k3.bias.assign(conv_pyg.bias.detach())

    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    out_pyg = conv_pyg(x, edge_index)
    out_k3 = conv_k3(x, edge_index)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-6


def test_reference_sage_conv():
    torch.manual_seed(42)
    conv_pyg = pyg_conv.SAGEConv((8, 8), 16)
    conv_k3 = k3_conv.SAGEConv((8, 8), 16)
    conv_k3.build([(None, 8), (None, 8)])
    conv_k3.lin_l.kernel.assign(conv_pyg.lin_l.weight.detach().t())
    conv_k3.lin_r.kernel.assign(conv_pyg.lin_r.weight.detach().t())
    conv_k3.lin_l.bias.assign(conv_pyg.lin_l.bias.detach())

    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    out_pyg = conv_pyg(x, edge_index)
    out_k3 = conv_k3(x, edge_index)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-6


def test_reference_gat_conv():
    torch.manual_seed(42)
    conv_pyg = pyg_conv.GATConv(8, 16, heads=2, concat=True)
    conv_k3 = k3_conv.GATConv(8, 16, heads=2, concat=True)
    conv_k3.build((None, 8))
    conv_k3.lin.kernel.assign(conv_pyg.lin.weight.detach().t())
    conv_k3.att_src.assign(conv_pyg.att_src.detach())
    conv_k3.att_dst.assign(conv_pyg.att_dst.detach())
    conv_k3.bias.assign(conv_pyg.bias.detach())

    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    out_pyg = conv_pyg(x, edge_index)
    out_k3 = conv_k3(x, edge_index)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-6


def test_reference_gin_conv():
    torch.manual_seed(42)
    nn_pyg = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16))
    conv_pyg = pyg_conv.GINConv(nn_pyg, train_eps=True)

    nn_k3 = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 16)
    )
    # Copy weights
    nn_k3[0].weight.data.copy_(nn_pyg[0].weight.data)
    nn_k3[0].bias.data.copy_(nn_pyg[0].bias.data)
    nn_k3[2].weight.data.copy_(nn_pyg[2].weight.data)
    nn_k3[2].bias.data.copy_(nn_pyg[2].bias.data)

    conv_k3 = k3_conv.GINConv(nn_k3, train_eps=True)
    conv_k3.eps.assign(conv_pyg.eps.detach())

    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    out_pyg = conv_pyg(x, edge_index)
    out_k3 = conv_k3(x, edge_index)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-6


def test_reference_cheb_conv():
    torch.manual_seed(42)
    conv_pyg = pyg_conv.ChebConv(8, 16, K=3)
    conv_k3 = k3_conv.ChebConv(8, 16, K=3)
    conv_k3.build((None, 8))
    conv_k3.lins[0].kernel.assign(conv_pyg.lins[0].weight.detach().t())
    conv_k3.lins[1].kernel.assign(conv_pyg.lins[1].weight.detach().t())
    conv_k3.lins[2].kernel.assign(conv_pyg.lins[2].weight.detach().t())
    conv_k3.bias.assign(conv_pyg.bias.detach())

    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    out_pyg = conv_pyg(x, edge_index)
    out_k3 = conv_k3(x, edge_index)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-5


def test_reference_meshcnn_conv():
    torch.manual_seed(42)
    conv_pyg = pyg_conv.MeshCNNConv(4, 8)
    conv_k3 = k3_conv.MeshCNNConv(4, 8)
    conv_k3.build((None, 4))
    for i in range(5):
        conv_k3.kernels[i].kernel.assign(conv_pyg.kernels[i].weight.detach().t())
        conv_k3.kernels[i].bias.assign(conv_pyg.kernels[i].bias.detach())

    x = torch.randn(4, 4)
    edge_index = torch.tensor([
        [1, 2, 3, 0, 0, 2, 3, 1, 0, 1, 3, 2, 0, 1, 2, 3],
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    ], dtype=torch.long)

    out_pyg = conv_pyg(x, edge_index)
    out_k3 = conv_k3(x, edge_index)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-6


def test_reference_hetero_conv():
    torch.manual_seed(42)
    pyg_conv_layer = pyg_conv.HeteroConv({
        ("user", "likes", "item"): pyg_conv.SAGEConv((4, 4), 8),
        ("item", "liked_by", "user"): pyg_conv.SAGEConv((4, 4), 8),
    }, aggr="sum")

    k3_conv_layer = k3_conv.HeteroConv({
        ("user", "likes", "item"): k3_conv.SAGEConv((4, 4), 8),
        ("item", "liked_by", "user"): k3_conv.SAGEConv((4, 4), 8),
    }, aggr="sum")

    for et in [("user", "likes", "item"), ("item", "liked_by", "user")]:
        s_pyg = pyg_conv_layer.convs[et]
        s_k3 = k3_conv_layer.convs[et]
        s_k3.build([(None, 4), (None, 4)])
        s_k3.lin_l.kernel.assign(s_pyg.lin_l.weight.detach().t())
        s_k3.lin_r.kernel.assign(s_pyg.lin_r.weight.detach().t())
        s_k3.lin_l.bias.assign(s_pyg.lin_l.bias.detach())

    x_dict = {"user": torch.randn(3, 4), "item": torch.randn(5, 4)}
    edge_index_dict = {
        ("user", "likes", "item"): torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        ("item", "liked_by", "user"): torch.tensor([[1, 2, 3], [0, 1, 2]], dtype=torch.long),
    }

    out_pyg = pyg_conv_layer(x_dict, edge_index_dict)
    out_k3 = k3_conv_layer(x_dict, edge_index_dict)
    for k in out_pyg:
        diff = (out_pyg[k] - out_k3[k]).abs().max().item()
        assert diff < 1e-6


def test_reference_hypergraph_conv():
    torch.manual_seed(42)
    pyg_conv_layer = pyg_conv.HypergraphConv(4, 8, use_attention=True, heads=2, concat=True)
    k3_conv_layer = k3_conv.HypergraphConv(4, 8, use_attention=True, heads=2, concat=True)
    k3_conv_layer.build()
    k3_conv_layer.lin.kernel.assign(pyg_conv_layer.lin.weight.detach().t())
    k3_conv_layer.att.assign(pyg_conv_layer.att.detach())
    k3_conv_layer.bias.assign(pyg_conv_layer.bias.detach())

    x = torch.randn(4, 4)
    hyperedge_index = torch.tensor([[0, 1, 2, 1, 2, 3], [0, 0, 0, 1, 1, 1]], dtype=torch.long)
    hyperedge_attr = torch.randn(2, 4)

    out_pyg = pyg_conv_layer(x, hyperedge_index, hyperedge_attr=hyperedge_attr)
    out_k3 = k3_conv_layer(x, hyperedge_index, hyperedge_attr=hyperedge_attr)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-6


def test_reference_dna_conv():
    torch.manual_seed(42)
    pyg_conv_layer = pyg_conv.DNAConv(channels=8, heads=2, groups=2)
    k3_conv_layer = k3_conv.DNAConv(channels=8, heads=2, groups=2)
    k3_conv_layer.build()

    k3_conv_layer.multi_head.lin_q.weight.assign(pyg_conv_layer.multi_head.lin_q.weight.detach())
    k3_conv_layer.multi_head.lin_q.bias.assign(pyg_conv_layer.multi_head.lin_q.bias.detach())
    k3_conv_layer.multi_head.lin_k.weight.assign(pyg_conv_layer.multi_head.lin_k.weight.detach())
    k3_conv_layer.multi_head.lin_k.bias.assign(pyg_conv_layer.multi_head.lin_k.bias.detach())
    k3_conv_layer.multi_head.lin_v.weight.assign(pyg_conv_layer.multi_head.lin_v.weight.detach())
    k3_conv_layer.multi_head.lin_v.bias.assign(pyg_conv_layer.multi_head.lin_v.bias.detach())

    x = torch.randn(4, 3, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    out_pyg = pyg_conv_layer(x, edge_index)
    out_k3 = k3_conv_layer(x, edge_index)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-6


def test_reference_wl_conv():
    pyg_wl = pyg_conv.WLConv()
    k3_wl = k3_conv.WLConv()
    x = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    out_pyg = pyg_wl(x, edge_index)
    out_k3 = k3_wl(x, edge_index)
    assert (out_pyg - out_k3).abs().max().item() == 0

    pyg_wlc = pyg_conv.WLConvContinuous()
    k3_wlc = k3_conv.WLConvContinuous()
    xc = torch.randn(4, 8)
    edge_weight = torch.tensor([1.0, 0.5, 0.8, 1.2])
    out_pyg_c = pyg_wlc(xc, edge_index, edge_weight=edge_weight)
    out_k3_c = k3_wlc(xc, edge_index, edge_weight=edge_weight)
    diff = (out_pyg_c - out_k3_c).abs().max().item()
    assert diff < 1e-6


def test_reference_han_conv():
    metadata = (["author", "paper"], [("author", "writes", "paper"), ("paper", "written_by", "author")])
    torch.manual_seed(42)
    conv_pyg = pyg_conv.HANConv(in_channels={"author": 16, "paper": 16}, out_channels=16, metadata=metadata, heads=2)
    conv_k3 = k3_conv.HANConv(in_channels={"author": 16, "paper": 16}, out_channels=16, metadata=metadata, heads=2)
    conv_k3.build()

    for nt in ["author", "paper"]:
        conv_k3.proj[nt].kernel.assign(conv_pyg.proj[nt].weight.detach().t())
        conv_k3.proj[nt].bias.assign(conv_pyg.proj[nt].bias.detach())

    conv_k3.k_lin.kernel.assign(conv_pyg.k_lin.weight.detach().t())
    conv_k3.k_lin.bias.assign(conv_pyg.k_lin.bias.detach())
    conv_k3.q.assign(conv_pyg.q.detach())

    for et in metadata[1]:
        key = "__".join(et)
        conv_k3.lin_src[key].assign(conv_pyg.lin_src[key].detach())
        conv_k3.lin_dst[key].assign(conv_pyg.lin_dst[key].detach())

    x_dict = {"author": torch.randn(4, 16), "paper": torch.randn(6, 16)}
    edge_index_dict = {
        ("author", "writes", "paper"): torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        ("paper", "written_by", "author"): torch.tensor([[1, 2, 3, 4], [0, 1, 2, 3]], dtype=torch.long),
    }

    out_pyg = conv_pyg(x_dict, edge_index_dict)
    out_k3 = conv_k3(x_dict, edge_index_dict)
    for nt in metadata[0]:
        diff = (out_pyg[nt] - out_k3[nt]).abs().max().item()
        assert diff < 1e-5


def test_reference_heat_conv():
    torch.manual_seed(42)
    pyg_conv_layer = pyg_conv.HEATConv(
        in_channels=8, out_channels=16, num_node_types=2, num_edge_types=3,
        edge_type_emb_dim=4, edge_dim=5, edge_attr_emb_dim=6, heads=2, concat=True
    )
    k3_conv_layer = k3_conv.HEATConv(
        in_channels=8, out_channels=16, num_node_types=2, num_edge_types=3,
        edge_type_emb_dim=4, edge_dim=5, edge_attr_emb_dim=6, heads=2, concat=True
    )
    k3_conv_layer.build()

    k3_conv_layer.hetero_lin.weight.assign(pyg_conv_layer.hetero_lin.weight.detach())
    k3_conv_layer.hetero_lin.bias.assign(pyg_conv_layer.hetero_lin.bias.detach())
    k3_conv_layer.edge_type_emb.embeddings.assign(pyg_conv_layer.edge_type_emb.weight.detach())
    k3_conv_layer.edge_attr_emb.kernel.assign(pyg_conv_layer.edge_attr_emb.weight.detach().t())
    k3_conv_layer.att.kernel.assign(pyg_conv_layer.att.weight.detach().t())
    k3_conv_layer.lin.kernel.assign(pyg_conv_layer.lin.weight.detach().t())
    k3_conv_layer.lin.bias.assign(pyg_conv_layer.lin.bias.detach())

    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    node_type = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    edge_attr = torch.randn(4, 5)

    out_pyg = pyg_conv_layer(x, edge_index, node_type, edge_type, edge_attr)
    out_k3 = k3_conv_layer(x, edge_index, node_type, edge_type, edge_attr)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-5


def test_reference_hgt_conv():
    metadata = (["author", "paper"], [("author", "writes", "paper"), ("paper", "written_by", "author")])
    torch.manual_seed(42)
    conv_pyg = pyg_conv.HGTConv(in_channels={"author": 16, "paper": 16}, out_channels=16, metadata=metadata, heads=2)
    conv_k3 = k3_conv.HGTConv(in_channels={"author": 16, "paper": 16}, out_channels=16, metadata=metadata, heads=2)
    conv_k3.build()

    for nt in ["author", "paper"]:
        conv_k3.kqv_lin.lins[nt].weight.assign(conv_pyg.kqv_lin.lins[nt].weight.detach().t())
        conv_k3.kqv_lin.lins[nt].bias.assign(conv_pyg.kqv_lin.lins[nt].bias.detach())
        conv_k3.out_lin.lins[nt].weight.assign(conv_pyg.out_lin.lins[nt].weight.detach().t())
        conv_k3.out_lin.lins[nt].bias.assign(conv_pyg.out_lin.lins[nt].bias.detach())
        conv_k3.skip[nt].assign(conv_pyg.skip[nt].detach())

    conv_k3.k_rel.weight.assign(conv_pyg.k_rel.weight.detach())
    conv_k3.v_rel.weight.assign(conv_pyg.v_rel.weight.detach())

    for et in metadata[1]:
        key = "__".join(et)
        conv_k3.p_rel[key].assign(conv_pyg.p_rel[key].detach())

    x_dict = {"author": torch.randn(4, 16), "paper": torch.randn(6, 16)}
    edge_index_dict = {
        ("author", "writes", "paper"): torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        ("paper", "written_by", "author"): torch.tensor([[1, 2, 3, 4], [0, 1, 2, 3]], dtype=torch.long),
    }

    out_pyg = conv_pyg(x_dict, edge_index_dict)
    out_k3 = conv_k3(x_dict, edge_index_dict)
    for nt in metadata[0]:
        diff = (out_pyg[nt] - out_k3[nt]).abs().max().item()
        assert diff < 1e-4
