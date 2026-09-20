import os
import pytest
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from keras import ops

from k3_node.models import (
    Graphormer,
    GraphNodeFeature,
    GraphAttnBias,
    GraphormerMultiheadAttention,
    GraphormerGraphEncoderLayer,
    GraphormerGraphEncoder,
    load_graphormer_weights,
    download_graphormer_checkpoint,
)
from k3_node.models.graphormer import get_graphormer_config


def test_graph_node_feature():
    num_atoms, num_in_deg, num_out_deg, hidden_dim = 16, 10, 10, 32
    gnf = GraphNodeFeature(
        num_atoms=num_atoms,
        num_in_degree=num_in_deg,
        num_out_degree=num_out_deg,
        hidden_dim=hidden_dim,
    )

    batch_size, num_nodes = 2, 5
    x = ops.convert_to_tensor(np.random.randint(1, num_atoms, size=(batch_size, num_nodes)), dtype="int32")
    in_deg = ops.convert_to_tensor(np.random.randint(0, num_in_deg, size=(batch_size, num_nodes)), dtype="int32")
    out_deg = ops.convert_to_tensor(np.random.randint(0, num_out_deg, size=(batch_size, num_nodes)), dtype="int32")

    out = gnf(x, in_deg, out_deg)
    # Shape should be [batch_size, num_nodes + 1, hidden_dim]
    assert ops.shape(out) == (batch_size, num_nodes + 1, hidden_dim)

    # Test multi-dimensional atom features [B, N, D]
    x_multi = ops.convert_to_tensor(np.random.randint(1, num_atoms, size=(batch_size, num_nodes, 3)), dtype="int32")
    out_multi = gnf(x_multi, in_deg, out_deg)
    assert ops.shape(out_multi) == (batch_size, num_nodes + 1, hidden_dim)


def test_graph_attn_bias():
    num_heads, num_atoms, num_edges, num_spatial, num_edge_dis = 4, 16, 8, 10, 5
    gab = GraphAttnBias(
        num_heads=num_heads,
        num_atoms=num_atoms,
        num_edges=num_edges,
        num_spatial=num_spatial,
        num_edge_dis=num_edge_dis,
        edge_type="multi_hop",
    )

    batch_size, num_nodes = 2, 4
    attn_bias = ops.zeros((batch_size, num_nodes + 1, num_nodes + 1), dtype="float32")
    spatial_pos = ops.convert_to_tensor(
        np.random.randint(0, num_spatial, size=(batch_size, num_nodes, num_nodes)), dtype="int32"
    )
    x = ops.zeros((batch_size, num_nodes, 1), dtype="int32")
    edge_input = ops.convert_to_tensor(
        np.random.randint(0, num_edges, size=(batch_size, num_nodes, num_nodes, 3, 2)), dtype="int32"
    )

    out = gab(attn_bias=attn_bias, spatial_pos=spatial_pos, x=x, edge_input=edge_input)
    assert ops.shape(out) == (batch_size, num_heads, num_nodes + 1, num_nodes + 1)

    # Test single-hop edge bias
    gab_single = GraphAttnBias(
        num_heads=num_heads,
        num_atoms=num_atoms,
        num_edges=num_edges,
        num_spatial=num_spatial,
        num_edge_dis=num_edge_dis,
        edge_type="single_hop",
    )
    attn_edge_type = ops.convert_to_tensor(
        np.random.randint(0, num_edges, size=(batch_size, num_nodes, num_nodes, 2)), dtype="int32"
    )
    out_single = gab_single(
        attn_bias=attn_bias, spatial_pos=spatial_pos, x=x, attn_edge_type=attn_edge_type
    )
    assert ops.shape(out_single) == (batch_size, num_heads, num_nodes + 1, num_nodes + 1)


def test_graphormer_attention_and_layer():
    embed_dim, num_heads = 32, 4
    mha = GraphormerMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)

    batch_size, seq_len = 2, 6
    x = ops.convert_to_tensor(np.random.randn(batch_size, seq_len, embed_dim).astype("float32"))
    attn_bias = ops.zeros((batch_size, num_heads, seq_len, seq_len), dtype="float32")
    key_padding_mask = ops.convert_to_tensor([[False, False, False, False, False, True],
                                              [False, False, False, False, True, True]], dtype="bool")

    out, weights = mha(x, attn_bias=attn_bias, key_padding_mask=key_padding_mask)
    assert ops.shape(out) == (batch_size, seq_len, embed_dim)
    assert ops.shape(weights) == (batch_size, num_heads, seq_len, seq_len)

    # Test encoder layers (both Pre-LN and Post-LN)
    for pre_ln in [True, False]:
        enc_layer = GraphormerGraphEncoderLayer(
            embedding_dim=embed_dim,
            ffn_embedding_dim=64,
            num_attention_heads=num_heads,
            pre_layernorm=pre_ln,
        )
        layer_out = enc_layer(x, attn_bias=attn_bias, key_padding_mask=key_padding_mask)
        assert ops.shape(layer_out) == (batch_size, seq_len, embed_dim)


def test_graphormer_model():
    batch_size, num_nodes = 2, 4
    model = Graphormer(
        num_atoms=16,
        num_in_degree=8,
        num_out_degree=8,
        num_edges=8,
        num_spatial=8,
        num_edge_dis=4,
        num_encoder_layers=2,
        embedding_dim=32,
        ffn_embedding_dim=64,
        num_attention_heads=4,
        num_classes=1,
    )

    data = {
        "x": ops.convert_to_tensor(np.random.randint(1, 15, size=(batch_size, num_nodes, 2)), dtype="int32"),
        "in_degree": ops.convert_to_tensor(np.random.randint(0, 7, size=(batch_size, num_nodes)), dtype="int32"),
        "out_degree": ops.convert_to_tensor(np.random.randint(0, 7, size=(batch_size, num_nodes)), dtype="int32"),
        "attn_bias": ops.zeros((batch_size, num_nodes + 1, num_nodes + 1), dtype="float32"),
        "spatial_pos": ops.convert_to_tensor(np.random.randint(0, 7, size=(batch_size, num_nodes, num_nodes)), dtype="int32"),
        "edge_input": ops.convert_to_tensor(np.random.randint(0, 7, size=(batch_size, num_nodes, num_nodes, 2, 2)), dtype="int32"),
    }

    # Forward via batched_data dict
    out = model(data)
    assert ops.shape(out) == (batch_size, 1)

    # Forward with return_all=True
    pred, all_h = model(data, return_all=True)
    assert ops.shape(pred) == (batch_size, 1)
    assert ops.shape(all_h) == (batch_size, num_nodes + 1, 32)

    # Embed method
    emb = model.embed(data)
    assert ops.shape(emb) == (batch_size, 32)

    # Keyword argument forward
    out_kwargs = model(
        x=data["x"],
        in_degree=data["in_degree"],
        out_degree=data["out_degree"],
        attn_bias=data["attn_bias"],
        spatial_pos=data["spatial_pos"],
        edge_input=data["edge_input"],
    )
    assert ops.shape(out_kwargs) == (batch_size, 1)


def test_graphormer_presets():
    for variant in ["base", "slim", "large"]:
        cfg = get_graphormer_config(variant)
        assert "num_encoder_layers" in cfg
        assert "embedding_dim" in cfg
        assert "num_attention_heads" in cfg


def test_download_and_load_checkpoint(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="Unknown pretrained model"):
        download_graphormer_checkpoint("non_existent_model")

    called = {}

    def mock_download(url, folder, filename, log=True):
        called["url"] = url
        called["folder"] = folder
        called["filename"] = filename
        os.makedirs(folder, exist_ok=True)
        target = os.path.join(folder, filename)
        with open(target, "wb") as f:
            f.write(b"mock")
        return target

    monkeypatch.setattr("k3_node.data.download.download_url", mock_download)

    ckpt_folder = str(tmp_path / "ckpts")
    path = download_graphormer_checkpoint("pcqm4mv1_graphormer_base", folder=ckpt_folder)
    assert os.path.exists(path)
    assert "pcqm4mv1" in called["filename"]

    # Test loading synthetic checkpoint
    if torch is None:
        return

    model = Graphormer(
        num_atoms=8,
        num_in_degree=4,
        num_out_degree=4,
        num_edges=4,
        num_spatial=4,
        num_edge_dis=2,
        num_encoder_layers=1,
        embedding_dim=16,
        ffn_embedding_dim=32,
        num_attention_heads=2,
        num_classes=1,
    )
    model.build(None)

    fake_state = {
        "encoder.graph_encoder.graph_node_feature.atom_encoder.weight": torch.randn(9, 16),
        "encoder.graph_encoder.graph_node_feature.in_degree_encoder.weight": torch.randn(4, 16),
        "encoder.graph_encoder.graph_node_feature.out_degree_encoder.weight": torch.randn(4, 16),
        "encoder.graph_encoder.graph_node_feature.graph_token.weight": torch.randn(1, 16),
        "encoder.graph_encoder.layers.0.fc1.weight": torch.randn(32, 16),
        "encoder.graph_encoder.layers.0.fc1.bias": torch.randn(32),
        "encoder.graph_encoder.layers.0.fc2.weight": torch.randn(16, 32),
        "encoder.graph_encoder.layers.0.fc2.bias": torch.randn(16),
        "encoder.lm_head_transform_weight.weight": torch.randn(16, 16),
        "encoder.lm_head_transform_weight.bias": torch.randn(16),
        "encoder.embed_out.weight": torch.randn(1, 16),
        "encoder.lm_output_learned_bias": torch.tensor([0.42]),
    }
    ckpt_file = str(tmp_path / "fake_ckpt.pt")
    torch.save({"model": fake_state}, ckpt_file)

    loaded_model = load_graphormer_weights(model, checkpoint_path=ckpt_file)
    assert float(ops.convert_to_numpy(loaded_model.lm_output_learned_bias[0])) == pytest.approx(0.42, abs=1e-5)
