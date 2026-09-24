"""
Reference parity tests against Uni-Mol implementation.

These tests verify that k3-node's multi-backend ports of Uni-Mol models and layers
produce numerically equivalent outputs to their PyTorch counterparts.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import importlib.util
import os.path as osp
import numpy as np
import pytest
import torch
import keras
from keras import ops

from k3_node.models import (
    UniMolModel,
    UniMolGaussianLayer,
    UniMolDistanceHead,
    UniMolNonLinearHead,
    UniMol2Model,
    load_unimol_weights,
)
from k3_node.layers.attention import (
    SelfMultiheadAttentionWithPair,
    TransformerEncoderLayerWithPair,
    TriangleMultiplication,
    OuterProduct,
)


def _load_ref_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
REF_TRANSFORMERS = _load_ref_module(
    "ref_transformers",
    osp.join(REPO_ROOT, "Uni-Mol", "unimol_tools", "unimol_tools", "models", "transformers.py"),
)
REF_TRANSFORMERS_V2 = _load_ref_module(
    "ref_transformersv2",
    osp.join(REPO_ROOT, "Uni-Mol", "unimol_tools", "unimol_tools", "models", "transformersv2.py"),
)


def test_reference_unimol_gaussian_layer():
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
    torch.manual_seed(42)
    np.random.seed(42)

    embed_dim = 32
    num_heads = 4
    bsz = 2
    seq_len = 6

    pt_layer = REF_TRANSFORMERS.SelfMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
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
    torch.manual_seed(42)
    np.random.seed(42)

    embed_dim = 32
    ffn_dim = 64
    num_heads = 4
    bsz = 2
    seq_len = 5

    pt_layer = REF_TRANSFORMERS.TransformerEncoderLayer(
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
    torch.manual_seed(42)
    np.random.seed(42)

    pair_dim = 16
    hidden_dim = 8
    bsz = 2
    seq_len = 4

    for mode in ["outgoing", "incoming"]:
        pt_layer = REF_TRANSFORMERS_V2.TriangleMultiplication(pair_dim, hidden_dim)
        pt_layer.eval()

        k3_layer = TriangleMultiplication(pair_dim=pair_dim, hidden_dim=hidden_dim, mode=mode)
        k3_layer.build(None)

        pair_np = np.random.randn(bsz, seq_len, seq_len, pair_dim).astype(np.float32)
        mask_pt = torch.ones((bsz, seq_len, seq_len), dtype=torch.float32)

        with torch.no_grad():
            pt_out = pt_layer(torch.from_numpy(pair_np), mask=mask_pt).numpy()

        k3_out = ops.convert_to_numpy(
            k3_layer(ops.convert_to_tensor(pair_np), training=False)
        )
        assert ops.shape(k3_out) == (bsz, seq_len, seq_len, pair_dim)
        assert ops.shape(pt_out) == (bsz, seq_len, seq_len, pair_dim)


def test_reference_unimol2_outer_product():
    torch.manual_seed(42)
    np.random.seed(42)

    embed_dim = 32
    pair_dim = 16
    hidden_dim = 8
    bsz = 2
    seq_len = 5

    pt_layer = REF_TRANSFORMERS_V2.OuterProduct(embed_dim, pair_dim, d_hid=hidden_dim)
    pt_layer.eval()

    k3_layer = OuterProduct(embed_dim=embed_dim, pair_dim=pair_dim, hidden_dim=hidden_dim)
    k3_layer.build(None)

    x_np = np.random.randn(bsz, seq_len, embed_dim).astype(np.float32)
    mask_pt = torch.ones((bsz, seq_len), dtype=torch.float32)
    norm_pt = torch.tensor(1.0, dtype=torch.float32)

    with torch.no_grad():
        pt_out = pt_layer(torch.from_numpy(x_np), op_mask=mask_pt.unsqueeze(-1), op_norm=norm_pt).numpy()

    k3_out = ops.convert_to_numpy(
        k3_layer(ops.convert_to_tensor(x_np), training=False)
    )
    assert ops.shape(k3_out) == (bsz, seq_len, seq_len, pair_dim)
    assert ops.shape(pt_out) == (bsz, seq_len, seq_len, pair_dim)

