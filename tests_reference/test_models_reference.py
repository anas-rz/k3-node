"""
Sanity-check reference parity tests against PyTorch Geometric.

These tests verify that k3-node's ports of `torch_geometric.nn.models`
produce numerically equivalent outputs to their PyG counterparts. These
tests are intended for local validation and documentation, and are not run
as part of the default GitHub Actions test suites.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
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

from k3_node.models import MLP, ARLinkPredictor, GAE, DeepGraphInfomax, DeepGCNLayer, AttentiveFP
from k3_node.models import (
    JumpingKnowledge, HeteroJumpingKnowledge,
    MaskLabel, MetaLayer, PMLP, Polynormer,
)
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

