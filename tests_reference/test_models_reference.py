"""
Sanity-check reference parity tests against PyTorch Geometric.

These tests verify that k3-node's ports of `torch_geometric.nn.models`
produce numerically equivalent outputs to their PyG counterparts. These
tests are intended for local validation and documentation, and are not run
as part of the default GitHub Actions test suites.
"""

import os
import pytest
os.environ["KERAS_BACKEND"] = "torch"
import math
import numpy as np
import torch
import keras
from keras import ops

from torch_geometric.nn.models import MLP as PyGMLP
from torch_geometric.nn.models import ARLinkPredictor as PyGARLinkPredictor
from torch_geometric.nn.models import GAE as PyGGAE
from torch_geometric.nn.models import DeepGraphInfomax as PyGDeepGraphInfomax
from torch_geometric.nn import DeepGCNLayer as PyGDeepGCNLayer
from torch_geometric.nn import GCNConv as PyGGCNConv
from torch_geometric.nn import LayerNorm as PyGLayerNorm
from torch_geometric.nn.models.attentive_fp import GATEConv as PyGGATEConv
from torch_geometric.nn.models.attentive_fp import AttentiveFP as PyGAttentiveFP
from torch_geometric.nn import JumpingKnowledge as PyGJK
from torch_geometric.nn import HeteroJumpingKnowledge as PyGHeteroJK
from torch_geometric.nn.models import MaskLabel as PyGMaskLabel
from torch_geometric.nn.models import MetaLayer as PyGMetaLayer
from torch_geometric.nn.models import PMLP as PyGPMLP
from torch_geometric.nn.models import Polynormer as PyGPolynormer
from torch_geometric.nn.models import (
    GCN as PyGGCN,
    GraphSAGE as PyGGraphSAGE,
    LabelPropagation as PyGLabelPropagation,
    CorrectAndSmooth as PyGCorrectAndSmooth,
    LightGCN as PyGLightGCN,
    RECT_L as PyGRECT_L,
    GroupAddRev as PyGGroupAddRev,
    MetaPath2Vec as PyGMetaPath2Vec,
)
from torch_geometric.nn.attention import SGFormerAttention as PyGSGFormerAttention
from torch_geometric.nn.models.tgn import (
    TimeEncoder as PyGTimeEncoder,
    IdentityMessage as PyGIdentityMessage,
)
from torch_geometric.nn.models.schnet import (
    ShiftedSoftplus as PyGShiftedSoftplus,
    GaussianSmearing as PyGGaussianSmearing,
)
from torch_geometric.nn.models.dimenet import (
    Envelope as PyGEnvelope,
    BesselBasisLayer as PyGBesselBasisLayer,
)
from torch_geometric.nn.models.visnet import (
    CosineCutoff as PyGCosineCutoff,
    Sphere as PyGSphere,
)
from torch_geometric.nn.models.gpse import (
    GPSENodeEncoder as PyGGPSENodeEncoder,
)

from k3_node.models import MLP, ARLinkPredictor, GAE, DeepGraphInfomax, DeepGCNLayer, AttentiveFP
from k3_node.models import (
    JumpingKnowledge, HeteroJumpingKnowledge,
    MaskLabel, MetaLayer, PMLP, Polynormer,
    GCN, GraphSAGE, LabelPropagation, CorrectAndSmooth, LightGCN, RECT_L,
    GroupAddRev, MetaPath2Vec,
    TimeEncoder, IdentityMessage,
    ShiftedSoftplus, GaussianSmearing,
    DimeNet, DimeNetPlusPlus, BesselBasisLayer,
    GPSE, GPSENodeEncoder,
    ViSNet, LPFormer,
    GraphMAE2, sce_loss, load_graphmae2_weights,
)
from k3_node.models.dimenet import Envelope
from k3_node.models.visnet import CosineCutoff, Sphere
from k3_node.layers.attention import SGFormerAttention
from k3_node.models.attentive_fp import GATEConv
from k3_node.layers.conv import GCNConv
from k3_node.layers.norm import LayerNorm



def _copy_gru(pyg_cell, k3_cell, units):
    def reorder(w):
        r, z, n = w[:units], w[units:2 * units], w[2 * units:3 * units]
        return torch.cat([z, r, n], dim=0)

    kernel = reorder(pyg_cell.weight_ih).t().contiguous()
    recurrent_kernel = reorder(pyg_cell.weight_hh).t().contiguous()
    bias_ih = reorder(pyg_cell.bias_ih.unsqueeze(-1)).squeeze(-1)
    bias_hh = reorder(pyg_cell.bias_hh.unsqueeze(-1)).squeeze(-1)
    bias = torch.stack([bias_ih, bias_hh], dim=0)

    k3_cell.kernel.assign(kernel.detach())
    k3_cell.recurrent_kernel.assign(recurrent_kernel.detach())
    k3_cell.bias.assign(bias.detach())


def _copy_gat(pyg_conv, k3_conv):
    k3_conv.att_src.assign(pyg_conv.att_src.detach())
    k3_conv.att_dst.assign(pyg_conv.att_dst.detach())
    k3_conv.bias.assign(pyg_conv.bias.detach())
    k3_conv.lin.kernel.assign(pyg_conv.lin.weight.detach().t())


def test_reference_mlp():
    torch.manual_seed(42)
    pyg = PyGMLP([16, 32, 32, 8], norm=None, plain_last=True)
    k3 = MLP([16, 32, 32, 8], norm=None, plain_last=True)
    for pl, kl in zip(pyg.lins, k3.lins):
        kl.kernel.assign(pl.weight.detach().t())
        kl.bias.assign(pl.bias.detach())

    x = torch.randn(4, 16)
    out_pyg = pyg(x)
    out_k3 = k3(x)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-5


def test_reference_ar_link_predictor():
    torch.manual_seed(42)
    pyg = PyGARLinkPredictor(in_channels=16, hidden_channels=32, num_layers=2)
    k3 = ARLinkPredictor(in_channels=16, hidden_channels=32, num_layers=2)
    for pl, kl in zip(pyg.lins, k3.lins):
        kl.kernel.assign(pl.weight.detach().t())
        kl.bias.assign(pl.bias.detach())
    k3.lin_attract.kernel.assign(pyg.lin_attract.weight.detach().t())
    k3.lin_attract.bias.assign(pyg.lin_attract.bias.detach())
    k3.lin_repel.kernel.assign(pyg.lin_repel.weight.detach().t())
    k3.lin_repel.bias.assign(pyg.lin_repel.bias.detach())

    x = torch.randn(6, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    out_pyg = pyg(x, edge_index)
    out_k3 = k3(x, edge_index)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-5


def test_reference_gae():
    # `encoder=lambda x: x` makes this an exact (weight-free) check.
    pyg = PyGGAE(encoder=lambda x: x)
    k3 = GAE(encoder=lambda x: x)

    x = torch.tensor([[1.0, -1.0], [1.0, 2.0], [2.0, 1.0]])
    z_pyg = pyg.encode(x)
    z_k3 = k3.encode(x)
    assert np.allclose(z_pyg.numpy(), ops.convert_to_numpy(z_k3), atol=1e-6)

    adj_pyg = pyg.decoder.forward_all(z_pyg)
    adj_k3 = k3.decoder.forward_all(z_k3)
    assert np.allclose(adj_pyg.detach().numpy(), ops.convert_to_numpy(adj_k3), atol=1e-6)

    edge_index = torch.tensor([[0, 1], [1, 2]])
    val_pyg = pyg.decode(z_pyg, edge_index)
    val_k3 = k3.decode(z_k3, edge_index)
    assert np.allclose(val_pyg.detach().numpy(), ops.convert_to_numpy(val_k3), atol=1e-6)


def test_reference_deep_graph_infomax():
    torch.manual_seed(42)
    pyg = PyGDeepGraphInfomax(
        hidden_channels=16,
        encoder=lambda x: x,
        summary=lambda z, *args: z.mean(dim=0),
        corruption=lambda x: x + 1,
    )
    k3 = DeepGraphInfomax(
        hidden_channels=16,
        encoder=lambda x: x,
        summary=lambda z, *args: ops.mean(z, axis=0),
        corruption=lambda x: x + 1,
    )
    k3.weight.assign(pyg.weight.detach())

    x = torch.randn(10, 16)
    pos_z_pyg, neg_z_pyg, summary_pyg = pyg(x)
    pos_z_k3, neg_z_k3, summary_k3 = k3(x)

    assert np.allclose(pos_z_pyg.numpy(), ops.convert_to_numpy(pos_z_k3), atol=1e-6)
    assert np.allclose(neg_z_pyg.numpy(), ops.convert_to_numpy(neg_z_k3), atol=1e-6)
    assert np.allclose(summary_pyg.numpy(), ops.convert_to_numpy(summary_k3), atol=1e-6)

    loss_pyg = pyg.loss(pos_z_pyg, neg_z_pyg, summary_pyg)
    loss_k3 = k3.loss(pos_z_k3, neg_z_k3, summary_k3)
    assert abs(loss_pyg.item() - float(ops.convert_to_numpy(loss_k3))) < 1e-5


def test_reference_deepgcn_layer():
    torch.manual_seed(42)
    conv_pyg = PyGGCNConv(8, 8)
    norm_pyg = PyGLayerNorm(8)
    layer_pyg = PyGDeepGCNLayer(conv_pyg, norm_pyg, torch.nn.ReLU(), block="res+")

    conv_k3 = GCNConv(8, 8)
    conv_k3.build((None, 8))
    conv_k3.lin.kernel.assign(conv_pyg.lin.weight.detach().t())
    conv_k3.bias.assign(conv_pyg.bias.detach())
    norm_k3 = LayerNorm(8)
    norm_k3.weight.assign(norm_pyg.weight.detach())
    norm_k3.bias.assign(norm_pyg.bias.detach())
    layer_k3 = DeepGCNLayer(conv_k3, norm_k3, keras.layers.ReLU(), block="res+")

    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 4]])

    layer_pyg.eval()
    out_pyg = layer_pyg(x, edge_index)
    out_k3 = layer_k3(x, edge_index, training=False)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-4


def test_reference_gate_conv():
    torch.manual_seed(0)
    pyg = PyGGATEConv(16, 16, edge_dim=3, dropout=0.0)
    k3 = GATEConv(16, 16, edge_dim=3, dropout=0.0)

    k3.att_l.assign(pyg.att_l.detach())
    k3.att_r.assign(pyg.att_r.detach())
    k3.bias.assign(pyg.bias.detach())
    k3.lin1.kernel.assign(pyg.lin1.weight.detach().t())
    k3.lin2.kernel.assign(pyg.lin2.weight.detach().t())

    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_attr = torch.randn(edge_index.size(1), 3)

    out_pyg = pyg(x, edge_index, edge_attr)
    out_k3 = k3(x, edge_index, edge_attr)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-5


def test_reference_attentive_fp():
    torch.manual_seed(0)
    hidden = 16
    pyg = PyGAttentiveFP(8, hidden, 4, edge_dim=3, num_layers=2, num_timesteps=2)
    k3 = AttentiveFP(8, hidden, 4, edge_dim=3, num_layers=2, num_timesteps=2)

    k3.lin1.kernel.assign(pyg.lin1.weight.detach().t())
    k3.lin1.bias.assign(pyg.lin1.bias.detach())

    k3.gate_conv.att_l.assign(pyg.gate_conv.att_l.detach())
    k3.gate_conv.att_r.assign(pyg.gate_conv.att_r.detach())
    k3.gate_conv.bias.assign(pyg.gate_conv.bias.detach())
    k3.gate_conv.lin1.kernel.assign(pyg.gate_conv.lin1.weight.detach().t())
    k3.gate_conv.lin2.kernel.assign(pyg.gate_conv.lin2.weight.detach().t())
    _copy_gru(pyg.gru, k3.gru, hidden)

    for conv_pyg, conv_k3 in zip(pyg.atom_convs, k3.atom_convs):
        _copy_gat(conv_pyg, conv_k3)
    for gru_pyg, gru_k3 in zip(pyg.atom_grus, k3.atom_grus):
        _copy_gru(gru_pyg, gru_k3, hidden)

    _copy_gat(pyg.mol_conv, k3.mol_conv)
    _copy_gru(pyg.mol_gru, k3.mol_gru, hidden)

    k3.lin2.kernel.assign(pyg.lin2.weight.detach().t())
    k3.lin2.bias.assign(pyg.lin2.bias.detach())

    x = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 1, 3, 4], [1, 0, 4, 3]])
    edge_attr = torch.randn(edge_index.size(1), 3)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])

    pyg.eval()
    out_pyg = pyg(x, edge_index, edge_attr, batch)
    out_k3 = k3(x, edge_index, edge_attr, batch, training=False)

    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-3


def test_reference_jumping_knowledge():
    torch.manual_seed(42)
    num_nodes, channels, num_layers = 10, 8, 4
    xs_torch = [torch.randn(num_nodes, channels) for _ in range(num_layers)]

    # cat mode
    pyg_cat = PyGJK('cat')
    k3_cat = JumpingKnowledge('cat')
    out_pyg = pyg_cat(xs_torch)
    out_k3 = k3_cat(xs_torch)
    assert np.allclose(out_pyg.detach().numpy(), ops.convert_to_numpy(out_k3), atol=1e-6)

    # max mode
    pyg_max = PyGJK('max')
    k3_max = JumpingKnowledge('max')
    out_pyg = pyg_max(xs_torch)
    out_k3 = k3_max(xs_torch)
    assert np.allclose(out_pyg.detach().numpy(), ops.convert_to_numpy(out_k3), atol=1e-6)


def test_reference_hetero_jumping_knowledge():
    torch.manual_seed(42)
    num_nodes, channels, num_layers = 10, 8, 4
    types = ["author", "paper"]
    xs_dict_torch = {
        key: [torch.randn(num_nodes, channels) for _ in range(num_layers)]
        for key in types
    }

    # cat mode
    pyg = PyGHeteroJK(types, mode='cat')
    k3 = HeteroJumpingKnowledge(types, mode='cat')
    out_pyg = pyg(xs_dict_torch)
    out_k3 = k3(xs_dict_torch)
    for k in types:
        assert np.allclose(out_pyg[k].detach().numpy(), ops.convert_to_numpy(out_k3[k]), atol=1e-6)

    # max mode
    pyg = PyGHeteroJK(types, mode='max')
    k3 = HeteroJumpingKnowledge(types, mode='max')
    out_pyg = pyg(xs_dict_torch)
    out_k3 = k3(xs_dict_torch)
    for k in types:
        assert np.allclose(out_pyg[k].detach().numpy(), ops.convert_to_numpy(out_k3[k]), atol=1e-6)


def test_reference_mask_label():
    torch.manual_seed(42)
    pyg_add = PyGMaskLabel(num_classes=5, out_channels=10, method="add")
    k3_add = MaskLabel(num_classes=5, out_channels=10, method="add")
    k3_add.build((None, 10))
    k3_add.emb.weights[0].assign(pyg_add.emb.weight.detach())

    x = torch.randn(6, 10)
    y = torch.tensor([0, 1, 4, 2, 3, 1])
    mask = torch.tensor([True, False, True, False, True, False])

    out_pyg = pyg_add(x, y, mask)
    out_k3 = k3_add(x, y, mask)
    assert np.allclose(out_pyg.detach().numpy(), ops.convert_to_numpy(out_k3), atol=1e-6)

    pyg_cat = PyGMaskLabel(num_classes=5, out_channels=10, method="concat")
    k3_cat = MaskLabel(num_classes=5, out_channels=10, method="concat")
    k3_cat.build((None, 10))
    k3_cat.emb.weights[0].assign(pyg_cat.emb.weight.detach())

    out_pyg = pyg_cat(x, y, mask)
    out_k3 = k3_cat(x, y, mask)
    assert np.allclose(out_pyg.detach().numpy(), ops.convert_to_numpy(out_k3), atol=1e-6)


def test_reference_meta_layer():
    torch.manual_seed(42)

    class PyGEdgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2 * 4 + 3 + 2, 5)

        def forward(self, src, dst, edge_attr, u, batch):
            out = torch.cat([src, dst, edge_attr, u[batch]], 1)
            return self.lin(out)

    class K3EdgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2 * 4 + 3 + 2, 5)

        def forward(self, src, dst, edge_attr, u, batch):
            out = torch.cat([src, dst, edge_attr, u[batch]], 1)
            return self.lin(out)

    pyg_edge = PyGEdgeModel()
    k3_edge = K3EdgeModel()
    k3_edge.lin.weight.data.copy_(pyg_edge.lin.weight.data)
    k3_edge.lin.bias.data.copy_(pyg_edge.lin.bias.data)

    pyg_op = PyGMetaLayer(edge_model=pyg_edge)
    k3_op = MetaLayer(edge_model=k3_edge)

    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_attr = torch.randn(3, 3)
    u = torch.randn(2, 2)
    batch = torch.tensor([0, 0, 0, 1, 1])

    x_pyg, e_pyg, u_pyg = pyg_op(x, edge_index, edge_attr, u, batch)
    x_k3, e_k3, u_k3 = k3_op(x, edge_index, edge_attr, u, batch)

    assert np.allclose(e_pyg.detach().numpy(), ops.convert_to_numpy(e_k3), atol=1e-5)


def test_reference_pmlp():
    torch.manual_seed(42)
    in_c, hidden_c, out_c, num_layers = 8, 16, 4, 3
    pyg = PyGPMLP(in_c, hidden_c, out_c, num_layers=num_layers, dropout=0.0, norm=False, bias=True)
    k3 = PMLP(in_c, hidden_c, out_c, num_layers=num_layers, dropout=0.0, norm=False, bias=True)

    for i in range(num_layers):
        k3._weights_list[i].assign(pyg.lins[i].weight.detach().t())
        k3._biases_list[i].assign(pyg.lins[i].bias.detach())

    x = torch.randn(6, in_c)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])

    # Train mode
    pyg.train()
    k3.training = True
    out_pyg = pyg(x)
    out_k3 = k3(x, training=True)
    assert np.allclose(out_pyg.detach().numpy(), ops.convert_to_numpy(out_k3), atol=1e-5)

    # Eval mode
    pyg.eval()
    k3.training = False
    out_pyg = pyg(x, edge_index)
    out_k3 = k3(x, edge_index, training=False)
    diff = np.max(np.abs(out_pyg.detach().numpy() - ops.convert_to_numpy(out_k3)))
    assert diff < 1e-4


def test_reference_polynormer():
    torch.manual_seed(42)
    in_c, hidden_c, out_c = 16, 16, 8
    pyg = PyGPolynormer(
        in_c, hidden_c, out_c,
        local_layers=1, global_layers=1,
        in_dropout=0.0, dropout=0.0, global_dropout=0.0,
        pre_ln=False, post_bn=False, qk_shared=True,
    )
    k3 = Polynormer(
        in_c, hidden_c, out_c,
        local_layers=1, global_layers=1,
        in_dropout=0.0, dropout=0.0, global_dropout=0.0,
        pre_ln=False, post_bn=False, qk_shared=True,
    )

    x = torch.randn(6, in_c)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    batch = torch.tensor([0, 0, 0, 1, 1, 1])

    # Build all paths by running forward pass
    k3._global = False
    _ = k3(x, edge_index, batch)
    k3._global = True
    _ = k3(x, edge_index, batch)

    # Copy local weights
    k3.local_convs[0].lin.kernel.assign(pyg.local_convs[0].lin.weight.detach().t())
    k3.local_convs[0].bias.assign(pyg.local_convs[0].bias.detach())
    k3.h_lins[0].kernel.assign(pyg.h_lins[0].weight.detach().t())
    k3.h_lins[0].bias.assign(pyg.h_lins[0].bias.detach())
    k3.lins[0].kernel.assign(pyg.lins[0].weight.detach().t())
    k3.lins[0].bias.assign(pyg.lins[0].bias.detach())
    k3.lns[0].weights[0].assign(pyg.lns[0].weight.detach())
    k3.lns[0].weights[1].assign(pyg.lns[0].bias.detach())
    k3.pred_local.kernel.assign(pyg.pred_local.weight.detach().t())
    k3.pred_local.bias.assign(pyg.pred_local.bias.detach())

    # Eval local
    pyg.eval()
    pyg._global = False
    k3._global = False
    out_pyg = pyg(x, edge_index, batch).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3(x, edge_index, batch, training=False))
    diff = np.max(np.abs(out_pyg - out_k3))
    assert diff < 1e-4

    # Copy global weights
    k3.ln.weights[0].assign(pyg.ln.weight.detach())
    k3.ln.weights[1].assign(pyg.ln.bias.detach())
    k3.pred_global.kernel.assign(pyg.pred_global.weight.detach().t())
    k3.pred_global.bias.assign(pyg.pred_global.bias.detach())

    pyg_attn = pyg.global_attn[0]
    k3_attn = k3.global_attn[0]
    k3_attn.h_lins.kernel.assign(pyg_attn.h_lins.weight.detach().t())
    k3_attn.h_lins.bias.assign(pyg_attn.h_lins.bias.detach())
    k3_attn.k.kernel.assign(pyg_attn.k.weight.detach().t())
    k3_attn.v.kernel.assign(pyg_attn.v.weight.detach().t())
    k3_attn.lns.weights[0].assign(pyg_attn.lns.weight.detach())
    k3_attn.lns.weights[1].assign(pyg_attn.lns.bias.detach())
    k3_attn.lin_out.kernel.assign(pyg_attn.lin_out.weight.detach().t())
    k3_attn.lin_out.bias.assign(pyg_attn.lin_out.bias.detach())

    # Eval global
    pyg._global = True
    k3._global = True
    out_pyg_g = pyg(x, edge_index, batch).detach().numpy()
    out_k3_g = ops.convert_to_numpy(k3(x, edge_index, batch, training=False))
    diff_g = np.max(np.abs(out_pyg_g - out_k3_g))
    assert diff_g < 1e-3


def test_reference_gcn():
    torch.manual_seed(42)
    pyg_gcn = PyGGCN(8, 16, num_layers=2, out_channels=4, dropout=0.0)
    k3_gcn = GCN(8, 16, num_layers=2, out_channels=4, dropout=0.0)
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    _ = k3_gcn(x, edge_index)
    for i in range(2):
        k3_gcn.convs[i].lin.kernel.assign(pyg_gcn.convs[i].lin.weight.detach().t())
        if pyg_gcn.convs[i].bias is not None:
            k3_gcn.convs[i].bias.assign(pyg_gcn.convs[i].bias.detach())

    pyg_gcn.eval()
    out_pyg = pyg_gcn(x, edge_index).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_gcn(x, edge_index, training=False))
    assert np.allclose(out_pyg, out_k3, atol=1e-4)


def test_reference_graph_sage():
    torch.manual_seed(42)
    pyg_sage = PyGGraphSAGE(8, 16, num_layers=2, out_channels=4, dropout=0.0)
    k3_sage = GraphSAGE(8, 16, num_layers=2, out_channels=4, dropout=0.0)
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    _ = k3_sage(x, edge_index)
    for i in range(2):
        k3_sage.convs[i].lin_l.kernel.assign(pyg_sage.convs[i].lin_l.weight.detach().t())
        k3_sage.convs[i].lin_r.kernel.assign(pyg_sage.convs[i].lin_r.weight.detach().t())
        if pyg_sage.convs[i].lin_l.bias is not None:
            k3_sage.convs[i].lin_l.bias.assign(pyg_sage.convs[i].lin_l.bias.detach())

    pyg_sage.eval()
    out_pyg = pyg_sage(x, edge_index).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_sage(x, edge_index, training=False))
    assert np.allclose(out_pyg, out_k3, atol=1e-4)


def test_reference_label_prop():
    torch.manual_seed(42)
    pyg_lp = PyGLabelPropagation(num_layers=2, alpha=0.5)
    k3_lp = LabelPropagation(num_layers=2, alpha=0.5)
    y = torch.tensor([1, 0, 0, 2])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    out_pyg = pyg_lp(y, edge_index).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_lp(y, edge_index))
    assert np.allclose(out_pyg, out_k3, atol=1e-5)


def test_reference_correct_and_smooth():
    torch.manual_seed(42)
    pyg_cs = PyGCorrectAndSmooth(2, 0.5, 2, 0.5, autoscale=False, scale=1.0)
    k3_cs = CorrectAndSmooth(2, 0.5, 2, 0.5, autoscale=False, scale=1.0)
    y_soft = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    y_true = torch.tensor([1, 0, 1, 0])
    mask = torch.tensor([True, False, True, False])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    out_pyg = pyg_cs.correct(y_soft, y_true[mask], mask, edge_index).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_cs.correct(y_soft, y_true[mask], mask, edge_index))
    assert np.allclose(out_pyg, out_k3, atol=1e-5)


def test_reference_lightgcn():
    torch.manual_seed(42)
    pyg_lg = PyGLightGCN(10, 8, num_layers=2)
    k3_lg = LightGCN(10, 8, num_layers=2)
    k3_lg.build((None,))
    k3_lg.embedding.weights[0].assign(pyg_lg.embedding.weight.detach())

    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    labels = torch.tensor([[0, 1], [2, 3]])
    out_pyg = pyg_lg(edges, labels).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_lg(edges, labels))
    assert np.allclose(out_pyg, out_k3, atol=1e-4)


def test_reference_rect():
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    pyg_rect = PyGRECT_L(8, 16)
    k3_rect = RECT_L(8, 16)
    _ = k3_rect(x, edge_index)

    k3_rect.conv.lin.kernel.assign(pyg_rect.conv.lin.weight.detach().t())
    k3_rect.conv.bias.assign(pyg_rect.conv.bias.detach())
    k3_rect.lin.kernel.assign(pyg_rect.lin.weight.detach().t())
    k3_rect.lin.bias.assign(pyg_rect.lin.bias.detach())

    pyg_rect.eval()
    out_pyg = pyg_rect(x, edge_index).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_rect(x, edge_index, training=False))
    assert np.allclose(out_pyg, out_k3, atol=1e-4)


def test_reference_sgformer_attention():
    torch.manual_seed(42)
    attn_pyg = PyGSGFormerAttention(16, heads=2, head_channels=8)
    attn_k3 = SGFormerAttention(16, heads=2, head_channels=8)

    x = torch.randn(2, 4, 16)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])

    _ = attn_k3(x, mask)
    attn_k3.q.kernel.assign(attn_pyg.q.weight.detach().t())
    attn_k3.k.kernel.assign(attn_pyg.k.weight.detach().t())
    attn_k3.v.kernel.assign(attn_pyg.v.weight.detach().t())

    out_pyg = attn_pyg(x, mask).detach().numpy()
    out_k3 = ops.convert_to_numpy(attn_k3(x, mask))
    assert np.allclose(out_pyg, out_k3, atol=1e-5)


def test_reference_group_add_rev():
    torch.manual_seed(42)
    conv1_pyg = PyGGCNConv(8, 8)
    conv2_pyg = PyGGCNConv(8, 8)
    rev_pyg = PyGGroupAddRev(torch.nn.ModuleList([conv1_pyg, conv2_pyg]), disable=True)

    conv1_k3 = GCNConv(8, 8)
    conv2_k3 = GCNConv(8, 8)
    rev_k3 = GroupAddRev([conv1_k3, conv2_k3])

    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])

    _ = rev_k3(x, edge_index=edge_index)
    conv1_k3.lin.kernel.assign(conv1_pyg.lin.weight.detach().t())
    conv1_k3.bias.assign(conv1_pyg.bias.detach())
    conv2_k3.lin.kernel.assign(conv2_pyg.lin.weight.detach().t())
    conv2_k3.bias.assign(conv2_pyg.bias.detach())

    out_pyg = rev_pyg(x, edge_index).detach().numpy()
    out_k3 = ops.convert_to_numpy(rev_k3(x, edge_index=edge_index))
    assert np.allclose(out_pyg, out_k3, atol=1e-5)

    inv_pyg = rev_pyg._inverse(torch.from_numpy(out_pyg), edge_index).detach().numpy()
    inv_k3 = ops.convert_to_numpy(rev_k3.inverse(out_k3, edge_index=edge_index))
    assert np.allclose(inv_pyg, inv_k3, atol=1e-5)


def test_reference_metapath2vec():
    torch.manual_seed(42)
    edge_index_dict = {
        ("a", "w", "p"): torch.tensor([[0, 1], [0, 1]]),
        ("p", "b", "a"): torch.tensor([[0, 1], [0, 1]]),
    }
    metapath = [("a", "w", "p"), ("p", "b", "a")]

    pyg_m2v = PyGMetaPath2Vec(edge_index_dict, 16, metapath, 2, 2)
    k3_m2v = MetaPath2Vec(edge_index_dict, 16, metapath, 2, 2)

    _ = k3_m2v("a")
    k3_m2v.embedding.embeddings.assign(pyg_m2v.embedding.weight.detach())

    pos_rw = torch.tensor([[0, 1], [1, 0]])
    neg_rw = torch.tensor([[0, 2], [1, 2]])

    loss_pyg = pyg_m2v.loss(pos_rw, neg_rw).item()
    loss_k3 = float(ops.convert_to_numpy(k3_m2v.loss(pos_rw, neg_rw)))
    assert abs(loss_pyg - loss_k3) < 1e-5


def test_reference_tgn_helpers():
    torch.manual_seed(42)
    enc_pyg = PyGTimeEncoder(16)
    enc_k3 = TimeEncoder(16)

    t = torch.tensor([1.0, 2.5, 3.7])
    _ = enc_k3(t)
    enc_k3.lin.kernel.assign(enc_pyg.lin.weight.detach().t())
    enc_k3.lin.bias.assign(enc_pyg.lin.bias.detach())

    out_pyg = enc_pyg(t).detach().numpy()
    out_k3 = ops.convert_to_numpy(enc_k3(t))
    assert np.allclose(out_pyg, out_k3, atol=1e-5)

    msg_pyg = PyGIdentityMessage(8, 16, 16)
    msg_k3 = IdentityMessage(8, 16, 16)
    z_src = torch.randn(3, 16)
    z_dst = torch.randn(3, 16)
    raw_msg = torch.randn(3, 8)
    t_enc = torch.randn(3, 16)

    m_pyg = msg_pyg(z_src, z_dst, raw_msg, t_enc).detach().numpy()
    m_k3 = ops.convert_to_numpy(msg_k3(z_src, z_dst, raw_msg, t_enc))
    assert np.allclose(m_pyg, m_k3, atol=1e-5)


def test_reference_schnet_components():
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    out_pyg = PyGShiftedSoftplus()(x).numpy()
    out_k3 = ops.convert_to_numpy(ShiftedSoftplus()(x))
    assert np.allclose(out_pyg, out_k3, atol=1e-6)

    d = torch.tensor([0.5, 1.5, 3.0])
    out_pyg = PyGGaussianSmearing(0.0, 5.0, 10)(d).numpy()
    out_k3 = ops.convert_to_numpy(GaussianSmearing(0.0, 5.0, 10)(d))
    assert np.allclose(out_pyg, out_k3, atol=1e-5)


def test_reference_dimenet_components():
    d = torch.tensor([0.5, 1.5, 3.0])
    env_pyg = PyGEnvelope(5)(d / 5.0).numpy()
    env_k3 = ops.convert_to_numpy(Envelope(5)(d / 5.0))
    assert np.allclose(env_pyg, env_k3, atol=1e-6)

    bessel_pyg = PyGBesselBasisLayer(6, 5.0)(d).detach().numpy()
    bessel_k3 = ops.convert_to_numpy(BesselBasisLayer(6, 5.0)(d))
    assert np.allclose(bessel_pyg, bessel_k3, atol=1e-5)


def test_reference_visnet_components():
    d = torch.tensor([0.5, 1.5, 3.0, 5.0, 6.0])
    cutoff_pyg = PyGCosineCutoff(5.0)(d).numpy()
    cutoff_k3 = ops.convert_to_numpy(CosineCutoff(5.0)(d))
    assert np.allclose(cutoff_pyg, cutoff_k3, atol=1e-6)

    torch.manual_seed(42)
    v = torch.randn(5, 3)
    sh_pyg = PyGSphere(lmax=2)(v).numpy()
    sh_k3 = ops.convert_to_numpy(Sphere(lmax=2)(v))
    assert np.allclose(sh_pyg, sh_k3, atol=1e-5)


def test_reference_gpse_encoder():
    torch.manual_seed(42)
    pyg_enc = PyGGPSENodeEncoder(
        dim_emb=32,
        dim_pe_in=16,
        dim_pe_out=8,
        dim_in=12,
        expand_x=True,
        norm_type=None,
        model_type="linear",
        dropout_be=0.0,
        dropout_ae=0.0,
    )
    k3_enc = GPSENodeEncoder(
        dim_emb=32,
        dim_pe_in=16,
        dim_pe_out=8,
        dim_in=12,
        expand_x=True,
        norm_type=None,
        model_type="linear",
        dropout_be=0.0,
        dropout_ae=0.0,
    )

    x = torch.randn(4, 12)
    pe = torch.randn(4, 16)
    _ = k3_enc(x, pe)

    k3_enc.linear_x.kernel.assign(pyg_enc.linear_x.weight.detach().t())
    k3_enc.linear_x.bias.assign(pyg_enc.linear_x.bias.detach())
    k3_enc.pe_encoder.kernel.assign(pyg_enc.pe_encoder.weight.detach().t())
    k3_enc.pe_encoder.bias.assign(pyg_enc.pe_encoder.bias.detach())

    out_pyg = pyg_enc(x, pe).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_enc(x, pe))
    assert np.allclose(out_pyg, out_k3, atol=1e-5)


def test_reference_graphmae2_components():
    import torch.nn as nn
    import torch.nn.functional as F

    # 1. SCE loss reference comparison
    def pt_sce_loss(x, y, alpha=3.0):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        return ((1.0 - (x * y).sum(dim=-1)) ** alpha).mean()

    torch.manual_seed(42)
    x = torch.randn(10, 32)
    y = torch.randn(10, 32)
    l_pt = pt_sce_loss(x, y, alpha=2.5).item()
    l_k3 = float(ops.convert_to_numpy(sce_loss(x, y, alpha=2.5)))
    assert abs(l_pt - l_k3) < 1e-5

    # 2. GAT layer mathematical reference comparison
    class RefGATConv(nn.Module):
        def __init__(self, in_feats, out_feats, num_heads, residual=True, norm=None, activation=None):
            super().__init__()
            self.fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
            self.attn_l = nn.Parameter(torch.randn(1, num_heads, out_feats))
            self.attn_r = nn.Parameter(torch.randn(1, num_heads, out_feats))
            self.bias = nn.Parameter(torch.zeros(num_heads * out_feats))
            self.num_heads = num_heads
            self.out_feats = out_feats
            self.res_fc = (
                nn.Linear(in_feats, num_heads * out_feats, bias=False)
                if residual and in_feats != num_heads * out_feats
                else None
            )
            self.norm = nn.LayerNorm(num_heads * out_feats, eps=1e-5) if norm == "layernorm" else None
            self.activation = nn.PReLU(1) if activation == "prelu" else None

        def forward(self, x, edge_index):
            N = x.shape[0]
            feat = self.fc(x).view(N, self.num_heads, self.out_feats)
            el = (feat * self.attn_l).sum(dim=-1, keepdim=True)
            er = (feat * self.attn_r).sum(dim=-1, keepdim=True)
            row, col = edge_index[0], edge_index[1]
            e = F.leaky_relu(el[row] + er[col], negative_slope=0.2)
            from torch_geometric.utils import softmax as pt_softmax
            a = pt_softmax(e, col, num_nodes=N)
            msg = a * feat[row]
            rst = torch.zeros(N, self.num_heads, self.out_feats, device=x.device)
            rst.index_add_(0, col, msg)
            rst = rst + self.bias.view(1, self.num_heads, self.out_feats)
            if self.res_fc is not None:
                rst = rst + self.res_fc(x).view(N, self.num_heads, self.out_feats)
            rst = rst.flatten(1)
            if self.norm is not None:
                rst = self.norm(rst)
            if self.activation is not None:
                rst = self.activation(rst)
            return rst

    from k3_node.models.graphmae2 import GraphMAE2GATConv

    in_feats, out_feats, num_heads = 16, 8, 4
    ref_gat = RefGATConv(in_feats, out_feats, num_heads, residual=True, norm="layernorm", activation="prelu")
    k3_gat = GraphMAE2GATConv(in_feats, out_feats, num_heads, residual=True, norm="layernorm", activation="prelu")
    k3_gat.build((None, in_feats))

    k3_gat.fc.kernel.assign(ref_gat.fc.weight.detach().t())
    k3_gat.attn_l.assign(ref_gat.attn_l.detach())
    k3_gat.attn_r.assign(ref_gat.attn_r.detach())
    k3_gat.bias.assign(ref_gat.bias.detach())
    if ref_gat.res_fc is not None:
        k3_gat.res_fc.kernel.assign(ref_gat.res_fc.weight.detach().t())
    k3_gat.norm.gamma.assign(ref_gat.norm.weight.detach())
    k3_gat.norm.beta.assign(ref_gat.norm.bias.detach())
    k3_gat.activation.alpha.assign(ref_gat.activation.weight.detach())

    x_test = torch.randn(6, in_feats)
    edge_idx = torch.tensor([[0, 1, 2, 3, 4, 0, 5], [1, 2, 3, 4, 0, 2, 0]])

    out_ref = ref_gat(x_test, edge_idx).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_gat(x_test, edge_idx))
    assert np.allclose(out_ref, out_k3, atol=1e-5)


def test_reference_graphmae2_checkpoint():
    ckpt_path = "GraphMAE2-main/GraphMAE2_checkpoints/gat_gat_1024_4_ogbn-arxiv_0.5_1024_checkpoint.pt"
    if not os.path.exists(ckpt_path):
        import pytest
        pytest.skip(f"Checkpoint {ckpt_path} not found")

    model = GraphMAE2(
        in_dim=128,
        num_hidden=1024,
        num_layers=4,
        num_dec_layers=1,
        nhead=8,
        nhead_out=1,
        activation="prelu",
        norm="layernorm",
        residual=True,
    )
    load_graphmae2_weights(model, ckpt_path)

    torch.manual_seed(42)
    x = torch.randn(6, 128)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 0, 2]])

    emb = ops.convert_to_numpy(model.embed(x, edge_index))
    assert emb.shape == (6, 1024)
    assert not np.isnan(emb).any()

    loss = float(ops.convert_to_numpy(model.loss(x, edge_index)))
    assert loss > 0.0


# =========================================================================
# Graphormer & Graphormer-3D Reference Parity Tests
# =========================================================================


def test_reference_graphormer_node_feature():
    torch.manual_seed(42)
    num_heads, num_atoms, num_in_deg, num_out_deg, hidden_dim, n_layers = 4, 16, 10, 10, 32, 2

    # Reference PyTorch implementation from Graphormer-main
    class RefGraphNodeFeature(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.atom_encoder = torch.nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
            self.in_degree_encoder = torch.nn.Embedding(num_in_deg, hidden_dim, padding_idx=0)
            self.out_degree_encoder = torch.nn.Embedding(num_out_deg, hidden_dim, padding_idx=0)
            self.graph_token = torch.nn.Embedding(1, hidden_dim)

        def forward(self, batched_data):
            x, in_degree, out_degree = batched_data["x"], batched_data["in_degree"], batched_data["out_degree"]
            n_graph, n_node = x.size()[:2]
            node_feature = self.atom_encoder(x).sum(dim=-2)
            node_feature = node_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
            graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
            return torch.cat([graph_token_feature, node_feature], dim=1)

    ref_gnf = RefGraphNodeFeature()
    ref_gnf.eval()

    from k3_node.models.graphormer import GraphNodeFeature

    k3_gnf = GraphNodeFeature(
        num_atoms=num_atoms,
        num_in_degree=num_in_deg,
        num_out_degree=num_out_deg,
        hidden_dim=hidden_dim,
    )
    k3_gnf.build(None)

    k3_gnf.atom_encoder.weights[0].assign(ref_gnf.atom_encoder.weight.detach())
    k3_gnf.in_degree_encoder.weights[0].assign(ref_gnf.in_degree_encoder.weight.detach())
    k3_gnf.out_degree_encoder.weights[0].assign(ref_gnf.out_degree_encoder.weight.detach())
    k3_gnf.graph_token.weights[0].assign(ref_gnf.graph_token.weight.detach())

    bsz, n_node = 2, 4
    x = torch.randint(1, num_atoms, (bsz, n_node, 2))
    in_degree = torch.randint(0, num_in_deg, (bsz, n_node))
    out_degree = torch.randint(0, num_out_deg, (bsz, n_node))

    batched_data = {"x": x, "in_degree": in_degree, "out_degree": out_degree}
    out_ref = ref_gnf(batched_data).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_gnf(x, in_degree, out_degree))

    assert np.allclose(out_ref, out_k3, atol=1e-5)


def test_reference_graphormer_attn_bias():
    torch.manual_seed(42)
    num_heads, num_atoms, num_edges, num_spatial, num_edge_dis = 2, 8, 4, 8, 4

    class RefGraphAttnBias(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = num_heads
            self.multi_hop_max_dist = 20
            self.edge_encoder = torch.nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
            self.edge_dis_encoder = torch.nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
            self.spatial_pos_encoder = torch.nn.Embedding(num_spatial, num_heads, padding_idx=0)
            self.graph_token_virtual_distance = torch.nn.Embedding(1, num_heads)

        def forward(self, batched_data):
            attn_bias, spatial_pos, x = batched_data["attn_bias"], batched_data["spatial_pos"], batched_data["x"]
            edge_input = batched_data["edge_input"]
            n_graph, n_node = x.size()[:2]
            graph_attn_bias = attn_bias.clone().unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
            t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
            graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_dist, :, :],
            )
            edge_input = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
            graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)
            return graph_attn_bias

    ref_gab = RefGraphAttnBias()
    ref_gab.eval()

    from k3_node.models.graphormer import GraphAttnBias

    k3_gab = GraphAttnBias(
        num_heads=num_heads,
        num_atoms=num_atoms,
        num_edges=num_edges,
        num_spatial=num_spatial,
        num_edge_dis=num_edge_dis,
        edge_type="multi_hop",
    )
    k3_gab.build(None)

    k3_gab.edge_encoder.weights[0].assign(ref_gab.edge_encoder.weight.detach())
    k3_gab.edge_dis_encoder.weights[0].assign(ref_gab.edge_dis_encoder.weight.detach())
    k3_gab.spatial_pos_encoder.weights[0].assign(ref_gab.spatial_pos_encoder.weight.detach())
    k3_gab.graph_token_virtual_distance.weights[0].assign(ref_gab.graph_token_virtual_distance.weight.detach())

    bsz, n_node = 2, 3
    attn_bias = torch.zeros(bsz, n_node + 1, n_node + 1)
    spatial_pos = torch.randint(0, num_spatial, (bsz, n_node, n_node))
    x = torch.zeros(bsz, n_node, 1, dtype=torch.long)
    edge_input = torch.randint(0, num_edges, (bsz, n_node, n_node, 2, 2))

    data = {"attn_bias": attn_bias, "spatial_pos": spatial_pos, "x": x, "edge_input": edge_input}
    out_ref = ref_gab(data).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_gab(attn_bias, spatial_pos, x, edge_input))

    assert np.allclose(out_ref, out_k3, atol=1e-5)


def test_reference_graphormer_attention():
    torch.manual_seed(42)
    embed_dim, num_heads = 16, 2
    head_dim = embed_dim // num_heads

    class RefMultiheadAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
            self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
            self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
            self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
            self.scaling = head_dim ** -0.5

        def forward(self, x, attn_bias=None, padding_mask=None):
            bsz, tgt_len, _ = x.size()
            q = (self.q_proj(x) * self.scaling).view(bsz, tgt_len, num_heads, head_dim).permute(0, 2, 1, 3)
            k = self.k_proj(x).view(bsz, tgt_len, num_heads, head_dim).permute(0, 2, 1, 3)
            v = self.v_proj(x).view(bsz, tgt_len, num_heads, head_dim).permute(0, 2, 1, 3)
            attn_weights = torch.matmul(q, k.transpose(-1, -2))
            if attn_bias is not None:
                attn_weights = attn_weights + attn_bias
            if padding_mask is not None:
                attn_weights = attn_weights.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), -1e9)
            attn_probs = torch.softmax(attn_weights, dim=-1)
            attn = torch.matmul(attn_probs, v).permute(0, 2, 1, 3).contiguous().view(bsz, tgt_len, embed_dim)
            return self.out_proj(attn)

    ref_mha = RefMultiheadAttention()
    ref_mha.eval()

    from k3_node.models.graphormer import GraphormerMultiheadAttention

    k3_mha = GraphormerMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
    k3_mha.build(None)

    k3_mha.q_proj.kernel.assign(ref_mha.q_proj.weight.detach().t())
    k3_mha.q_proj.bias.assign(ref_mha.q_proj.bias.detach())
    k3_mha.k_proj.kernel.assign(ref_mha.k_proj.weight.detach().t())
    k3_mha.k_proj.bias.assign(ref_mha.k_proj.bias.detach())
    k3_mha.v_proj.kernel.assign(ref_mha.v_proj.weight.detach().t())
    k3_mha.v_proj.bias.assign(ref_mha.v_proj.bias.detach())
    k3_mha.out_proj.kernel.assign(ref_mha.out_proj.weight.detach().t())
    k3_mha.out_proj.bias.assign(ref_mha.out_proj.bias.detach())

    bsz, seq_len = 2, 4
    x = torch.randn(bsz, seq_len, embed_dim)
    attn_bias = torch.randn(bsz, num_heads, seq_len, seq_len)
    pad_mask = torch.tensor([[False, False, False, True], [False, False, True, True]])

    out_ref = ref_mha(x, attn_bias=attn_bias, padding_mask=pad_mask).detach().numpy()
    out_k3, _ = k3_mha(x, attn_bias=attn_bias, key_padding_mask=pad_mask, training=False)
    out_k3 = ops.convert_to_numpy(out_k3)

    assert np.allclose(out_ref, out_k3, atol=1e-5)


def test_reference_graphormer_gaussian_layer():
    torch.manual_seed(42)
    num_kernel, edge_types = 16, 8

    class RefGaussianLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.means = torch.nn.Embedding(1, num_kernel)
            self.stds = torch.nn.Embedding(1, num_kernel)
            self.mul = torch.nn.Embedding(edge_types, 1)
            self.bias = torch.nn.Embedding(edge_types, 1)

        def forward(self, x, edges):
            mul = self.mul(edges)
            bias = self.bias(edges)
            x = mul * x.unsqueeze(-1) + bias
            mean = self.means.weight.float().view(-1)
            std = self.stds.weight.float().view(-1).abs() + 1e-5
            diff = (x - mean) / std
            a = math.sqrt(2 * math.pi)
            return torch.exp(-0.5 * (diff ** 2)) / (a * std)

    ref_gbf = RefGaussianLayer()
    ref_gbf.eval()

    from k3_node.models.graphormer_3d import GaussianLayer

    k3_gbf = GaussianLayer(num_kernel=num_kernel, edge_types=edge_types)
    k3_gbf.build(None)

    k3_gbf.means.weights[0].assign(ref_gbf.means.weight.detach())
    k3_gbf.stds.weights[0].assign(ref_gbf.stds.weight.detach())
    k3_gbf.mul.weights[0].assign(ref_gbf.mul.weight.detach())
    k3_gbf.bias.weights[0].assign(ref_gbf.bias.weight.detach())

    bsz, n_node = 2, 3
    dist = torch.rand(bsz, n_node, n_node)
    edges = torch.randint(0, edge_types, (bsz, n_node, n_node))

    out_ref = ref_gbf(dist, edges).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_gbf(dist, edges))

    assert np.allclose(out_ref, out_k3, atol=1e-5)


def test_reference_graphormer3d_node_head():
    torch.manual_seed(42)
    embed_dim, num_heads = 16, 2
    head_dim = embed_dim // num_heads

    class RefNodeTaskHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
            self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
            self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
            self.scaling = head_dim ** -0.5
            self.force_proj1 = torch.nn.Linear(embed_dim, 1)
            self.force_proj2 = torch.nn.Linear(embed_dim, 1)
            self.force_proj3 = torch.nn.Linear(embed_dim, 1)

        def forward(self, query, attn_bias, delta_pos):
            bsz, n_node, _ = query.size()
            q = (self.q_proj(query).view(bsz, n_node, num_heads, -1).transpose(1, 2) * self.scaling)
            k = self.k_proj(query).view(bsz, n_node, num_heads, -1).transpose(1, 2)
            v = self.v_proj(query).view(bsz, n_node, num_heads, -1).transpose(1, 2)
            attn = q @ k.transpose(-1, -2)
            attn_probs = torch.softmax(attn.view(-1, n_node, n_node) + attn_bias, dim=-1).view(bsz, num_heads, n_node, n_node)
            rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1)
            rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
            x = rot_attn_probs @ v.unsqueeze(2)
            x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
            f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
            f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)
            f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)
            return torch.cat([f1, f2, f3], dim=-1).float()

    ref_head = RefNodeTaskHead()
    ref_head.eval()

    from k3_node.models.graphormer_3d import NodeTaskHead

    k3_head = NodeTaskHead(embed_dim=embed_dim, num_heads=num_heads)
    k3_head.build(None)

    k3_head.q_proj.kernel.assign(ref_head.q_proj.weight.detach().t())
    k3_head.q_proj.bias.assign(ref_head.q_proj.bias.detach())
    k3_head.k_proj.kernel.assign(ref_head.k_proj.weight.detach().t())
    k3_head.k_proj.bias.assign(ref_head.k_proj.bias.detach())
    k3_head.v_proj.kernel.assign(ref_head.v_proj.weight.detach().t())
    k3_head.v_proj.bias.assign(ref_head.v_proj.bias.detach())

    k3_head.force_proj1.kernel.assign(ref_head.force_proj1.weight.detach().t())
    k3_head.force_proj1.bias.assign(ref_head.force_proj1.bias.detach())
    k3_head.force_proj2.kernel.assign(ref_head.force_proj2.weight.detach().t())
    k3_head.force_proj2.bias.assign(ref_head.force_proj2.bias.detach())
    k3_head.force_proj3.kernel.assign(ref_head.force_proj3.weight.detach().t())
    k3_head.force_proj3.bias.assign(ref_head.force_proj3.bias.detach())

    bsz, n_node = 2, 3
    query = torch.randn(bsz, n_node, embed_dim)
    attn_bias = torch.randn(bsz * num_heads, n_node, n_node)
    delta_pos = torch.randn(bsz, n_node, n_node, 3)

    out_ref = ref_head(query, attn_bias, delta_pos).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_head(query, attn_bias, delta_pos))

    assert np.allclose(out_ref, out_k3, atol=1e-5)


def test_reference_graphormer_full():
    torch.manual_seed(42)
    num_atoms, num_in_deg, num_out_deg, num_edges, num_spatial, num_edge_dis = 8, 4, 4, 4, 4, 2
    embed_dim, ffn_dim, num_heads, num_layers, num_classes = 16, 32, 2, 1, 1

    from k3_node.models import Graphormer

    model = Graphormer(
        num_atoms=num_atoms,
        num_in_degree=num_in_deg,
        num_out_degree=num_out_deg,
        num_edges=num_edges,
        num_spatial=num_spatial,
        num_edge_dis=num_edge_dis,
        num_encoder_layers=num_layers,
        embedding_dim=embed_dim,
        ffn_embedding_dim=ffn_dim,
        num_attention_heads=num_heads,
        num_classes=num_classes,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )
    model.build(None)

    bsz, n_node = 2, 3
    x = torch.randint(1, num_atoms, (bsz, n_node, 1))
    in_deg = torch.randint(0, num_in_deg, (bsz, n_node))
    out_deg = torch.randint(0, num_out_deg, (bsz, n_node))
    attn_bias = torch.zeros(bsz, n_node + 1, n_node + 1)
    spatial_pos = torch.randint(0, num_spatial, (bsz, n_node, n_node))
    edge_input = torch.randint(0, num_edges, (bsz, n_node, n_node, 2, 1))

    out_k3 = ops.convert_to_numpy(model(
        x=x, in_degree=in_deg, out_degree=out_deg, attn_bias=attn_bias, spatial_pos=spatial_pos, edge_input=edge_input
    ))
    assert out_k3.shape == (bsz, num_classes)
    assert not np.isnan(out_k3).any()


def test_reference_graphormer3d_full():
    torch.manual_seed(42)
    layers, blocks, embed_dim, ffn_dim, heads, num_kernel, atom_types = 1, 1, 16, 32, 2, 8, 8

    from k3_node.models import Graphormer3D

    model = Graphormer3D(
        layers=layers,
        blocks=blocks,
        embed_dim=embed_dim,
        ffn_embed_dim=ffn_dim,
        attention_heads=heads,
        num_kernel=num_kernel,
        atom_types=atom_types,
        dropout=0.0,
        attention_dropout=0.0,
    )
    model.build(None)

    bsz, n_node = 2, 4
    atoms = torch.tensor([[1, 2, 3, 0], [2, 3, 0, 0]], dtype=torch.long)
    tags = torch.tensor([[1, 2, 2, 0], [1, 2, 0, 0]], dtype=torch.long)
    pos = torch.randn(bsz, n_node, 3)

    energy, forces = model(atoms, tags, pos)
    energy_np = ops.convert_to_numpy(energy)
    forces_np = ops.convert_to_numpy(forces)

    assert energy_np.shape == (bsz,)
    assert forces_np.shape == (bsz, n_node, 3)
    assert not np.isnan(energy_np).any()
    assert not np.isnan(forces_np).any()


def test_reference_gps_model():
    import sys
    import os
    import torch_geometric.utils

    class TorchScatterShim:
        @staticmethod
        def scatter(src, index, dim=0, out=None, dim_size=None, reduce='sum'):
            return torch_geometric.utils.scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)
        @staticmethod
        def scatter_add(src, index, dim=0, out=None, dim_size=None):
            return torch_geometric.utils.scatter(src, index, dim=dim, dim_size=dim_size, reduce='sum')
        @staticmethod
        def scatter_max(src, index, dim=0, out=None, dim_size=None):
            return torch_geometric.utils.scatter(src, index, dim=dim, dim_size=dim_size, reduce='max')
    sys.modules['torch_scatter'] = TorchScatterShim

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    graphgps_dir = os.path.join(base_dir, "GraphGPS")
    if not os.path.exists(graphgps_dir):
        pytest.skip("GraphGPS repository not found")
    sys.path.insert(0, graphgps_dir)

    try:
        import ogb.utils.features
    except ImportError:
        import types
        ogb = types.ModuleType("ogb")
        ogb.__path__ = []
        ogb.utils = types.ModuleType("ogb.utils")
        ogb.utils.features = types.ModuleType("ogb.utils.features")
        ogb.graphproppred = types.ModuleType("ogb.graphproppred")
        ogb.graphproppred.PygGraphPropPredDataset = object
        sys.modules["ogb"] = ogb
        sys.modules["ogb.utils"] = ogb.utils
        sys.modules["ogb.utils.features"] = ogb.utils.features
        sys.modules["ogb.graphproppred"] = ogb.graphproppred
    ogb.utils.features.get_atom_feature_dims = lambda: [119, 4, 12, 12, 10, 6, 6, 2, 2]
    ogb.utils.features.get_bond_feature_dims = lambda: [5, 6, 2]

    from torch_geometric.data import Batch
    from torch_geometric.graphgym.config import cfg, set_cfg
    from graphgps.network.gps_model import GPSModel as PyTGPSModel
    from k3_node.models.gps_model import GPSModel as K3GPSModel, load_gps_weights, download_gps_checkpoint

    ckpt_path = download_gps_checkpoint("pcqm4m-GPS+RWSE.deep")
    if not os.path.exists(ckpt_path):
        return

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(ckpt_path))), "config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "config.yaml")

    set_cfg(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_path)
    cfg.dataset.node_encoder = True
    cfg.dataset.node_encoder_name = 'Atom+RWSE'
    cfg.dataset.edge_encoder = True
    cfg.dataset.edge_encoder_name = 'Bond'
    cfg.posenc_RWSE.kernel.times = list(range(1, 17))

    pyt_model = PyTGPSModel(dim_in=1, dim_out=1)
    pyt_model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    pyt_model.load_state_dict(ckpt["model_state"])

    k3_model = K3GPSModel(
        dim_in=256,
        dim_out=1,
        num_layers=16,
        dim_hidden=256,
        num_heads=8,
        local_gnn_type="CustomGatedGCN",
        act="gelu",
        dropout=0.0,
        attn_dropout=0.0,
        batch_norm=True,
        layer_norm=False,
        node_encoder_type="Atom+RWSE",
        edge_encoder_type="Bond",
        rwse_num_steps=16,
        rwse_dim_pe=20,
        graph_pooling="mean",
        head_layers=2,
    )

    x = torch.tensor([[6, 0, 4, 4, 3, 2, 2, 0, 0],
                      [8, 0, 2, 2, 2, 1, 1, 0, 0],
                      [6, 0, 3, 3, 3, 2, 2, 0, 0]], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor([[0, 0, 0],
                              [0, 0, 0],
                              [1, 0, 0],
                              [1, 0, 0]], dtype=torch.long)
    pestat_RWSE = torch.ones((3, 16), dtype=torch.float)
    batch_idx = torch.tensor([0, 0, 0], dtype=torch.long)

    # Build and load K3 model
    _ = k3_model(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_index.numpy()),
        edge_attr=ops.convert_to_tensor(edge_attr.numpy()),
        pestat_RWSE=ops.convert_to_tensor(pestat_RWSE.numpy()),
        batch=ops.convert_to_tensor(batch_idx.numpy()),
        training=False,
    )
    load_gps_weights(k3_model, ckpt_path)

    # PyTorch reference forward pass
    pyt_data = Batch(
        x=x.clone(),
        edge_index=edge_index.clone(),
        edge_attr=edge_attr.clone(),
        pestat_RWSE=pestat_RWSE.clone(),
        batch=batch_idx.clone(),
        y=torch.zeros((1, 1)),
    )
    with torch.no_grad():
        pyt_pred, _ = pyt_model(pyt_data)
        pyt_val = pyt_pred.item()

    # Keras 3 forward pass
    k3_pred = k3_model(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_index.numpy()),
        edge_attr=ops.convert_to_tensor(edge_attr.numpy()),
        pestat_RWSE=ops.convert_to_tensor(pestat_RWSE.numpy()),
        batch=ops.convert_to_tensor(batch_idx.numpy()),
        training=False,
    )
    k3_val = float(ops.convert_to_numpy(k3_pred)[0, 0])

    np.testing.assert_allclose(k3_val, pyt_val, rtol=1e-3, atol=1e-3)


def test_reference_grover():
    import sys
    import os
    import os.path as osp
    from unittest.mock import MagicMock

    if "rdkit" not in sys.modules:
        sys.modules["rdkit"] = MagicMock()
        sys.modules["rdkit.Chem"] = MagicMock()
        sys.modules["rdkit.Chem.rdchem"] = MagicMock()
        sys.modules["rdkit.DataStructs"] = MagicMock()

    grover_path = osp.join(osp.dirname(osp.dirname(__file__)), "grover")
    if not osp.exists(grover_path):
        pytest.skip("grover repository not found")
    if grover_path not in sys.path:
        sys.path.insert(0, grover_path)

    from grover.model.models import GROVEREmbedding
    from k3_node.models.grover import GROVER, load_grover_weights

    ckpt_path = "/tmp/grover_download_test/grover_base.pt"
    if not os.path.exists(ckpt_path):
        cache_path = osp.expanduser("~/.cache/k3_node/grover/grover_base.pt")
        if os.path.exists(cache_path):
            ckpt_path = cache_path
        else:
            pytest.skip("GROVER base checkpoint not found at /tmp or cache.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    args.cuda = False
    args.dropout = 0.0

    # 1. PyTorch reference model
    pt_model = GROVEREmbedding(args)
    state_dict = ckpt["state_dict"]
    new_state_dict = {
        k[len("grover.") :] if k.startswith("grover.") else k: v for k, v in state_dict.items()
    }
    pt_model.load_state_dict(new_state_dict, strict=True)
    pt_model.eval()

    # 2. K3 GROVER model
    k3_model = GROVER(
        hidden_size=args.hidden_size,
        num_attn_head=args.num_attn_head,
        depth=args.depth,
        num_mt_block=args.num_mt_block,
        node_fdim=151,
        edge_fdim=165,
        dropout=0.0,
        activation=args.activation,
        atom_emb_output="both",
        bias=args.bias,
    )
    load_grover_weights(k3_model, ckpt_path)

    # 3. Create test molecular batch
    torch.manual_seed(42)
    np.random.seed(42)

    num_atoms = 5
    num_bonds = 7
    atom_fdim = 151
    bond_fdim = 165

    f_atoms_np = np.random.randn(num_atoms, atom_fdim).astype(np.float32)
    f_atoms_np[0] = 0.0
    f_bonds_np = np.random.randn(num_bonds, bond_fdim).astype(np.float32)
    f_bonds_np[0] = 0.0

    a2b_np = np.array([[0, 0], [2, 0], [1, 4], [3, 0], [0, 0]], dtype=np.int64)
    b2a_np = np.array([0, 1, 2, 2, 3, 1, 3], dtype=np.int64)
    b2revb_np = np.array([0, 2, 1, 4, 3, 6, 5], dtype=np.int64)
    a_scope_np = np.array([[1, 2], [3, 2]], dtype=np.int64)
    b_scope_np = np.array([[1, 3], [4, 3]], dtype=np.int64)
    a2a_np = b2a_np[a2b_np]

    batch_pt = (
        torch.from_numpy(f_atoms_np),
        torch.from_numpy(f_bonds_np),
        torch.from_numpy(a2b_np),
        torch.from_numpy(b2a_np),
        torch.from_numpy(b2revb_np),
        torch.from_numpy(a_scope_np),
        torch.from_numpy(b_scope_np),
        torch.from_numpy(a2a_np),
    )

    batch_k3 = (
        ops.convert_to_tensor(f_atoms_np),
        ops.convert_to_tensor(f_bonds_np),
        ops.convert_to_tensor(a2b_np),
        ops.convert_to_tensor(b2a_np),
        ops.convert_to_tensor(b2revb_np),
        ops.convert_to_tensor(a_scope_np),
        ops.convert_to_tensor(b_scope_np),
        ops.convert_to_tensor(a2a_np),
    )

    with torch.no_grad():
        pt_out = pt_model(batch_pt)

    k3_out = k3_model(batch_k3, training=False)

    for key in ["atom_from_atom", "atom_from_bond", "bond_from_atom", "bond_from_bond"]:
        pt_arr = pt_out[key].detach().cpu().numpy()
        k3_arr = ops.convert_to_numpy(k3_out[key])
        np.testing.assert_allclose(k3_arr, pt_arr, rtol=1e-4, atol=1e-4)


def test_reference_mole_bert():
    import sys
    import os
    import os.path as osp
    from unittest.mock import MagicMock

    if "torch_scatter" not in sys.modules:
        sys.modules["torch_scatter"] = MagicMock()

    mole_bert_path = osp.join(osp.dirname(osp.dirname(__file__)), "Mole-BERT")
    if not osp.exists(mole_bert_path):
        pytest.skip("Mole-BERT repository not found")
    if mole_bert_path not in sys.path:
        sys.path.insert(0, mole_bert_path)

    from model import GNN as PyTGNN
    from k3_node.models.mole_bert import MoleBERTGNN, load_mole_bert_weights, download_mole_bert_checkpoint

    ckpt_path = osp.join(mole_bert_path, "model_gin", "Mole-BERT.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = download_mole_bert_checkpoint()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 1. PyTorch reference model
    pt_model = PyTGNN(num_layer=5, emb_dim=300, JK="last", drop_ratio=0.0, gnn_type="gin")
    pt_model.load_state_dict(ckpt)
    pt_model.eval()

    # 2. Keras 3 MoleBERTGNN
    k3_model = MoleBERTGNN(num_layer=5, emb_dim=300, JK="last", drop_ratio=0.0)
    k3_model.build(None)
    load_mole_bert_weights(k3_model, ckpt_path)

    # 3. Create test graph
    torch.manual_seed(42)
    np.random.seed(42)

    num_nodes = 6
    num_edges = 10
    x_np = np.stack(
        [
            np.random.randint(0, 119, size=num_nodes),
            np.random.randint(0, 3, size=num_nodes),
        ],
        axis=1,
    ).astype(np.int64)
    src = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5], dtype=np.int64)
    dst = np.array([1, 0, 2, 1, 3, 2, 4, 3, 5, 4], dtype=np.int64)
    edge_index_np = np.stack([src, dst], axis=0)
    edge_attr_np = np.stack(
        [
            np.random.randint(0, 5, size=num_edges),
            np.random.randint(0, 3, size=num_edges),
        ],
        axis=1,
    ).astype(np.int64)

    with torch.no_grad():
        pt_out = pt_model(
            torch.from_numpy(x_np),
            torch.from_numpy(edge_index_np),
            torch.from_numpy(edge_attr_np),
        ).numpy()

    k3_out = ops.convert_to_numpy(
        k3_model(
            ops.convert_to_tensor(x_np),
            ops.convert_to_tensor(edge_index_np),
            ops.convert_to_tensor(edge_attr_np),
            training=False,
        )
    )

    np.testing.assert_allclose(k3_out, pt_out, rtol=1e-4, atol=1e-4)


# ==============================================================================
# Uni-Mol Reference Parity Tests
# ==============================================================================

def test_reference_unimol_gaussian_layer():
    from k3_node.models import UniMolGaussianLayer

    def gaussian(x, mean, std):
        pi = 3.14159
        a = (2 * pi) ** 0.5
        return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

    class PyTGaussianLayer(torch.nn.Module):
        def __init__(self, K=32, edge_types=64):
            super().__init__()
            self.K = K
            self.means = torch.nn.Embedding(1, K)
            self.stds = torch.nn.Embedding(1, K)
            self.mul = torch.nn.Embedding(edge_types, 1)
            self.bias = torch.nn.Embedding(edge_types, 1)

        def forward(self, x, edge_type):
            mul = self.mul(edge_type).type_as(x)
            bias = self.bias(edge_type).type_as(x)
            x = mul * x.unsqueeze(-1) + bias
            x = x.expand(-1, -1, -1, self.K)
            mean = self.means.weight.float().view(-1)
            std = self.stds.weight.float().view(-1).abs() + 1e-5
            return gaussian(x.float(), mean, std).type_as(self.means.weight)

    torch.manual_seed(42)
    np.random.seed(42)

    K = 32
    edge_types = 64
    pt_layer = PyTGaussianLayer(K=K, edge_types=edge_types)
    pt_layer.eval()

    k3_layer = UniMolGaussianLayer(num_kernel=K, edge_types=edge_types)
    k3_layer.build(None)

    # Copy weights
    k3_layer.means.embeddings.assign(ops.convert_to_tensor(pt_layer.means.weight.detach().numpy()))
    k3_layer.stds.embeddings.assign(ops.convert_to_tensor(pt_layer.stds.weight.detach().numpy()))
    k3_layer.mul.embeddings.assign(ops.convert_to_tensor(pt_layer.mul.weight.detach().numpy()))
    k3_layer.bias.embeddings.assign(ops.convert_to_tensor(pt_layer.bias.weight.detach().numpy()))

    bsz, seq_len = 2, 5
    dist_np = np.random.uniform(0.5, 5.0, (bsz, seq_len, seq_len)).astype(np.float32)
    et_np = np.random.randint(0, edge_types, (bsz, seq_len, seq_len)).astype(np.int64)

    with torch.no_grad():
        pt_out = pt_layer(torch.from_numpy(dist_np), torch.from_numpy(et_np)).numpy()

    k3_out = ops.convert_to_numpy(
        k3_layer(ops.convert_to_tensor(dist_np), ops.convert_to_tensor(et_np, dtype="int32"))
    )

    np.testing.assert_allclose(k3_out, pt_out, rtol=1e-5, atol=1e-5)


def test_reference_unimol_self_multihead_attention():
    import importlib.util
    import os.path as osp

    spec = importlib.util.spec_from_file_location(
        "ref_transformers",
        osp.join(osp.dirname(osp.dirname(__file__)), "Uni-Mol", "unimol_tools", "unimol_tools", "models", "transformers.py"),
    )
    ref_tf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_tf)

    from k3_node.layers.attention import SelfMultiheadAttentionWithPair

    torch.manual_seed(42)
    np.random.seed(42)

    embed_dim = 32
    num_heads = 4
    bsz = 2
    seq_len = 6

    pt_layer = ref_tf.SelfMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
    pt_layer.eval()

    k3_layer = SelfMultiheadAttentionWithPair(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
    k3_layer.build(None)

    # Copy weights
    k3_layer.in_proj.kernel.assign(ops.convert_to_tensor(pt_layer.in_proj.weight.t().detach().numpy()))
    k3_layer.in_proj.bias.assign(ops.convert_to_tensor(pt_layer.in_proj.bias.detach().numpy()))
    k3_layer.out_proj.kernel.assign(ops.convert_to_tensor(pt_layer.out_proj.weight.t().detach().numpy()))
    k3_layer.out_proj.bias.assign(ops.convert_to_tensor(pt_layer.out_proj.bias.detach().numpy()))

    x_np = np.random.randn(bsz, seq_len, embed_dim).astype(np.float32)
    attn_bias_np = np.random.randn(bsz * num_heads, seq_len, seq_len).astype(np.float32)

    with torch.no_grad():
        pt_out = pt_layer(
            torch.from_numpy(x_np),
            attn_bias=torch.from_numpy(attn_bias_np),
        ).numpy()

    k3_out = ops.convert_to_numpy(
        k3_layer(
            ops.convert_to_tensor(x_np),
            attn_bias=ops.convert_to_tensor(attn_bias_np),
            training=False,
        )
    )

    np.testing.assert_allclose(k3_out, pt_out, rtol=1e-4, atol=1e-4)


def test_reference_unimol_transformer_encoder_layer():
    import importlib.util
    import os.path as osp

    spec = importlib.util.spec_from_file_location(
        "ref_transformers",
        osp.join(osp.dirname(osp.dirname(__file__)), "Uni-Mol", "unimol_tools", "unimol_tools", "models", "transformers.py"),
    )
    ref_tf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_tf)

    from k3_node.layers.attention import TransformerEncoderLayerWithPair

    torch.manual_seed(42)
    np.random.seed(42)

    embed_dim = 32
    ffn_dim = 64
    num_heads = 4
    bsz = 2
    seq_len = 5

    pt_layer = ref_tf.TransformerEncoderLayer(
        embed_dim=embed_dim,
        ffn_embed_dim=ffn_dim,
        attention_heads=num_heads,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        activation_fn="gelu",
        post_ln=False,
    )
    pt_layer.eval()

    k3_layer = TransformerEncoderLayerWithPair(
        embed_dim=embed_dim,
        ffn_embed_dim=ffn_dim,
        attention_heads=num_heads,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        activation_fn="gelu",
        post_ln=False,
    )
    k3_layer.build(None)

    # Copy weights
    k3_layer.self_attn.in_proj.kernel.assign(ops.convert_to_tensor(pt_layer.self_attn.in_proj.weight.t().detach().numpy()))
    k3_layer.self_attn.in_proj.bias.assign(ops.convert_to_tensor(pt_layer.self_attn.in_proj.bias.detach().numpy()))
    k3_layer.self_attn.out_proj.kernel.assign(ops.convert_to_tensor(pt_layer.self_attn.out_proj.weight.t().detach().numpy()))
    k3_layer.self_attn.out_proj.bias.assign(ops.convert_to_tensor(pt_layer.self_attn.out_proj.bias.detach().numpy()))
    k3_layer.self_attn_layer_norm.gamma.assign(ops.convert_to_tensor(pt_layer.self_attn_layer_norm.weight.detach().numpy()))
    k3_layer.self_attn_layer_norm.beta.assign(ops.convert_to_tensor(pt_layer.self_attn_layer_norm.bias.detach().numpy()))

    k3_layer.fc1.kernel.assign(ops.convert_to_tensor(pt_layer.fc1.weight.t().detach().numpy()))
    k3_layer.fc1.bias.assign(ops.convert_to_tensor(pt_layer.fc1.bias.detach().numpy()))
    k3_layer.fc2.kernel.assign(ops.convert_to_tensor(pt_layer.fc2.weight.t().detach().numpy()))
    k3_layer.fc2.bias.assign(ops.convert_to_tensor(pt_layer.fc2.bias.detach().numpy()))
    k3_layer.final_layer_norm.gamma.assign(ops.convert_to_tensor(pt_layer.final_layer_norm.weight.detach().numpy()))
    k3_layer.final_layer_norm.beta.assign(ops.convert_to_tensor(pt_layer.final_layer_norm.bias.detach().numpy()))

    x_np = np.random.randn(bsz, seq_len, embed_dim).astype(np.float32)
    attn_bias_np = np.random.randn(bsz * num_heads, seq_len, seq_len).astype(np.float32)

    with torch.no_grad():
        pt_out = pt_layer(
            torch.from_numpy(x_np),
            attn_bias=torch.from_numpy(attn_bias_np),
        ).numpy()

    k3_out = ops.convert_to_numpy(
        k3_layer(
            ops.convert_to_tensor(x_np),
            attn_bias=ops.convert_to_tensor(attn_bias_np),
            training=False,
        )
    )

    np.testing.assert_allclose(k3_out, pt_out, rtol=1e-4, atol=1e-4)


def test_reference_unimol2_triangle_multiplication():
    from k3_node.layers.attention import TriangleMultiplication

    torch.manual_seed(42)
    np.random.seed(42)

    class PyTTriangleMultiplication(torch.nn.Module):
        def __init__(self, pair_dim, hidden_dim, mode="outgoing"):
            super().__init__()
            self.mode = mode
            self.norm = torch.nn.LayerNorm(pair_dim, eps=1e-5)
            self.proj_a = torch.nn.Linear(pair_dim, hidden_dim, bias=False)
            self.proj_b = torch.nn.Linear(pair_dim, hidden_dim, bias=False)
            self.gate_a = torch.nn.Linear(pair_dim, hidden_dim, bias=True)
            self.gate_b = torch.nn.Linear(pair_dim, hidden_dim, bias=True)
            self.gate_out = torch.nn.Linear(pair_dim, pair_dim, bias=True)
            self.proj_out = torch.nn.Linear(hidden_dim, pair_dim, bias=True)
            self.norm_out = torch.nn.LayerNorm(hidden_dim, eps=1e-5)

        def forward(self, pair):
            residual = pair
            x = self.norm(pair)
            a = self.proj_a(x) * torch.sigmoid(self.gate_a(x))
            b = self.proj_b(x) * torch.sigmoid(self.gate_b(x))
            if self.mode == "outgoing":
                out = torch.einsum("bikc,bjkc->bijc", a, b)
            else:
                out = torch.einsum("bkic,bkjc->bijc", a, b)
            out = self.norm_out(out)
            out = self.proj_out(out) * torch.sigmoid(self.gate_out(pair))
            return residual + out

    pair_dim = 16
    hidden_dim = 8
    bsz = 2
    seq_len = 4

    for mode in ["outgoing", "incoming"]:
        pt_layer = PyTTriangleMultiplication(pair_dim=pair_dim, hidden_dim=hidden_dim, mode=mode)
        pt_layer.eval()

        k3_layer = TriangleMultiplication(pair_dim=pair_dim, hidden_dim=hidden_dim, mode=mode)
        k3_layer.build(None)

        # Copy weights
        k3_layer.norm.gamma.assign(ops.convert_to_tensor(pt_layer.norm.weight.detach().numpy()))
        k3_layer.norm.beta.assign(ops.convert_to_tensor(pt_layer.norm.bias.detach().numpy()))

        k3_layer.proj_a.kernel.assign(ops.convert_to_tensor(pt_layer.proj_a.weight.t().detach().numpy()))
        k3_layer.proj_b.kernel.assign(ops.convert_to_tensor(pt_layer.proj_b.weight.t().detach().numpy()))
        k3_layer.gate_a.kernel.assign(ops.convert_to_tensor(pt_layer.gate_a.weight.t().detach().numpy()))
        k3_layer.gate_a.bias.assign(ops.convert_to_tensor(pt_layer.gate_a.bias.detach().numpy()))
        k3_layer.gate_b.kernel.assign(ops.convert_to_tensor(pt_layer.gate_b.weight.t().detach().numpy()))
        k3_layer.gate_b.bias.assign(ops.convert_to_tensor(pt_layer.gate_b.bias.detach().numpy()))

        k3_layer.gate_out.kernel.assign(ops.convert_to_tensor(pt_layer.gate_out.weight.t().detach().numpy()))
        k3_layer.gate_out.bias.assign(ops.convert_to_tensor(pt_layer.gate_out.bias.detach().numpy()))
        k3_layer.proj_out.kernel.assign(ops.convert_to_tensor(pt_layer.proj_out.weight.t().detach().numpy()))
        k3_layer.proj_out.bias.assign(ops.convert_to_tensor(pt_layer.proj_out.bias.detach().numpy()))
        k3_layer.norm_out.gamma.assign(ops.convert_to_tensor(pt_layer.norm_out.weight.detach().numpy()))
        k3_layer.norm_out.beta.assign(ops.convert_to_tensor(pt_layer.norm_out.bias.detach().numpy()))

        pair_np = np.random.randn(bsz, seq_len, seq_len, pair_dim).astype(np.float32)

        with torch.no_grad():
            pt_out = pt_layer(torch.from_numpy(pair_np)).numpy()

        k3_out = ops.convert_to_numpy(
            k3_layer(ops.convert_to_tensor(pair_np), training=False)
        )

        np.testing.assert_allclose(k3_out, pt_out, rtol=1e-4, atol=1e-4)


def test_reference_unimol2_outer_product():
    from k3_node.layers.attention import OuterProduct

    torch.manual_seed(42)
    np.random.seed(42)

    class PyTOuterProduct(torch.nn.Module):
        def __init__(self, embed_dim, pair_dim, hidden_dim=32):
            super().__init__()
            self.norm = torch.nn.LayerNorm(embed_dim, eps=1e-5)
            self.proj_left = torch.nn.Linear(embed_dim, hidden_dim, bias=True)
            self.proj_right = torch.nn.Linear(embed_dim, hidden_dim, bias=True)
            self.proj_out = torch.nn.Linear(hidden_dim, pair_dim, bias=True)

        def forward(self, x):
            x_norm = self.norm(x)
            left = self.proj_left(x_norm)
            right = self.proj_right(x_norm)
            prod = left.unsqueeze(2) * right.unsqueeze(1)
            return self.proj_out(prod)

    embed_dim = 32
    pair_dim = 16
    hidden_dim = 8
    bsz = 2
    seq_len = 5

    pt_layer = PyTOuterProduct(embed_dim=embed_dim, pair_dim=pair_dim, hidden_dim=hidden_dim)
    pt_layer.eval()

    k3_layer = OuterProduct(embed_dim=embed_dim, pair_dim=pair_dim, hidden_dim=hidden_dim)
    k3_layer.build(None)

    # Copy weights
    k3_layer.norm.gamma.assign(ops.convert_to_tensor(pt_layer.norm.weight.detach().numpy()))
    k3_layer.norm.beta.assign(ops.convert_to_tensor(pt_layer.norm.bias.detach().numpy()))

    k3_layer.proj_left.kernel.assign(ops.convert_to_tensor(pt_layer.proj_left.weight.t().detach().numpy()))
    k3_layer.proj_left.bias.assign(ops.convert_to_tensor(pt_layer.proj_left.bias.detach().numpy()))
    k3_layer.proj_right.kernel.assign(ops.convert_to_tensor(pt_layer.proj_right.weight.t().detach().numpy()))
    k3_layer.proj_right.bias.assign(ops.convert_to_tensor(pt_layer.proj_right.bias.detach().numpy()))

    k3_layer.proj_out.kernel.assign(ops.convert_to_tensor(pt_layer.proj_out.weight.t().detach().numpy()))
    k3_layer.proj_out.bias.assign(ops.convert_to_tensor(pt_layer.proj_out.bias.detach().numpy()))

    x_np = np.random.randn(bsz, seq_len, embed_dim).astype(np.float32)

    with torch.no_grad():
        pt_out = pt_layer(torch.from_numpy(x_np)).numpy()

    k3_out = ops.convert_to_numpy(
        k3_layer(ops.convert_to_tensor(x_np), training=False)
    )
    np.testing.assert_allclose(k3_out, pt_out, rtol=1e-4, atol=1e-4)
