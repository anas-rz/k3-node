import os
import pytest
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from keras import ops

from k3_node.models import (
    Graphormer3D,
    GaussianLayer,
    RBF,
    Graphormer3DEncoderLayer,
    NodeTaskHead,
    load_graphormer3d_weights,
    download_graphormer3d_checkpoint,
)


def test_gaussian_layer():
    num_kernel, edge_types = 32, 16
    gbf = GaussianLayer(num_kernel=num_kernel, edge_types=edge_types)

    batch_size, num_nodes = 2, 4
    dist = ops.convert_to_tensor(np.abs(np.random.randn(batch_size, num_nodes, num_nodes)).astype("float32"))
    edges = ops.convert_to_tensor(np.random.randint(0, edge_types, size=(batch_size, num_nodes, num_nodes)), dtype="int32")

    out = gbf(dist, edges)
    assert ops.shape(out) == (batch_size, num_nodes, num_nodes, num_kernel)
    assert np.all(ops.convert_to_numpy(out) >= 0.0)


def test_rbf_layer():
    num_kernel, edge_types = 32, 16
    rbf = RBF(num_kernel=num_kernel, edge_types=edge_types)

    batch_size, num_nodes = 2, 4
    dist = ops.convert_to_tensor(np.abs(np.random.randn(batch_size, num_nodes, num_nodes)).astype("float32"))
    edges = ops.convert_to_tensor(np.random.randint(0, edge_types, size=(batch_size, num_nodes, num_nodes)), dtype="int32")

    out = rbf(dist, edges)
    assert ops.shape(out) == (batch_size, num_nodes, num_nodes, num_kernel)
    assert np.all(ops.convert_to_numpy(out) >= 0.0)


def test_node_task_head():
    embed_dim, num_heads = 32, 4
    head = NodeTaskHead(embed_dim=embed_dim, num_heads=num_heads)

    batch_size, num_nodes = 2, 5
    query = ops.convert_to_tensor(np.random.randn(batch_size, num_nodes, embed_dim).astype("float32"))
    attn_bias = ops.zeros((batch_size * num_heads, num_nodes, num_nodes), dtype="float32")
    delta_pos = ops.convert_to_tensor(np.random.randn(batch_size, num_nodes, num_nodes, 3).astype("float32"))

    forces = head(query, attn_bias, delta_pos)
    assert ops.shape(forces) == (batch_size, num_nodes, 3)


def test_graphormer3d_encoder_layer():
    embed_dim, num_heads = 32, 4
    layer = Graphormer3DEncoderLayer(
        embedding_dim=embed_dim,
        ffn_embedding_dim=64,
        num_attention_heads=num_heads,
    )

    batch_size, num_nodes = 2, 4
    x = ops.convert_to_tensor(np.random.randn(batch_size, num_nodes, embed_dim).astype("float32"))
    attn_bias = ops.zeros((batch_size * num_heads, num_nodes, num_nodes), dtype="float32")

    out = layer(x, attn_bias=attn_bias)
    assert ops.shape(out) == (batch_size, num_nodes, embed_dim)


def test_graphormer3d_model():
    batch_size, num_nodes = 2, 5
    model = Graphormer3D(
        layers=2,
        blocks=2,
        embed_dim=32,
        ffn_embed_dim=64,
        attention_heads=4,
        num_kernel=16,
        atom_types=16,
    )

    atoms = ops.convert_to_tensor([[1, 2, 3, 4, 0], [2, 3, 4, 0, 0]], dtype="int32")
    tags = ops.convert_to_tensor([[1, 1, 2, 2, 0], [1, 2, 2, 0, 0]], dtype="int32")
    pos = ops.convert_to_tensor(np.random.randn(batch_size, num_nodes, 3).astype("float32"))

    energy, forces = model(atoms, tags, pos)
    assert ops.shape(energy) == (batch_size,)
    assert ops.shape(forces) == (batch_size, num_nodes, 3)


def test_graphormer3d_checkpoint_download_and_load(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="Unknown pretrained 3D model"):
        download_graphormer3d_checkpoint("invalid_3d_name")

    called = {}

    def mock_download(url, folder, filename, log=True):
        called["url"] = url
        called["folder"] = folder
        called["filename"] = filename
        os.makedirs(folder, exist_ok=True)
        target = os.path.join(folder, filename)
        with open(target, "wb") as f:
            f.write(b"mock_3d")
        return target

    monkeypatch.setattr("k3_node.data.download.download_url", mock_download)

    ckpt_folder = str(tmp_path / "ckpts3d")
    path = download_graphormer3d_checkpoint("oc20is2re_graphormer3d_base", folder=ckpt_folder)
    assert os.path.exists(path)
    assert "oc20is2re" in called["filename"]

    # Test loading synthetic checkpoint
    if torch is None:
        return

    model = Graphormer3D(
        layers=1,
        blocks=1,
        embed_dim=16,
        ffn_embed_dim=32,
        attention_heads=2,
        num_kernel=8,
        atom_types=8,
    )
    model.build(None)

    fake_state = {
        "encoder.atom_encoder.weight": torch.randn(8, 16),
        "encoder.tag_encoder.weight": torch.randn(3, 16),
        "encoder.gbf.means.weight": torch.randn(1, 8),
        "encoder.gbf.stds.weight": torch.randn(1, 8),
        "encoder.gbf.mul.weight": torch.randn(64, 1),
        "encoder.gbf.bias.weight": torch.randn(64, 1),
        "encoder.bias_proj.layer1.weight": torch.randn(8, 8),
        "encoder.bias_proj.layer1.bias": torch.randn(8),
        "encoder.bias_proj.layer2.weight": torch.randn(2, 8),
        "encoder.bias_proj.layer2.bias": torch.randn(2),
        "encoder.edge_proj.weight": torch.randn(16, 8),
        "encoder.edge_proj.bias": torch.randn(16),
        "encoder.layers.0.self_attn.in_proj.weight": torch.randn(48, 16),
        "encoder.layers.0.self_attn.in_proj.bias": torch.randn(48),
        "encoder.layers.0.self_attn.out_proj.weight": torch.randn(16, 16),
        "encoder.layers.0.self_attn.out_proj.bias": torch.randn(16),
        "encoder.engergy_proj.layer1.weight": torch.randn(16, 16),
        "encoder.engergy_proj.layer1.bias": torch.randn(16),
        "encoder.engergy_proj.layer2.weight": torch.randn(1, 16),
        "encoder.engergy_proj.layer2.bias": torch.randn(1),
        "encoder.energe_agg_factor.weight": torch.randn(3, 1),
    }
    ckpt_file = str(tmp_path / "fake_3d.pt")
    torch.save({"model": fake_state}, ckpt_file)

    loaded_model = load_graphormer3d_weights(model, checkpoint_path=ckpt_file)
    assert loaded_model.atom_encoder.embeddings.shape == (8, 16)

