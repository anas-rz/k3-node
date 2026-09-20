import os
import pytest
import numpy as np
from keras import ops

from k3_node.models import (
    GraphMAE2,
    sce_loss,
    load_graphmae2_weights,
    download_graphmae2_checkpoint,
)


def test_sce_loss():
    # Identical vectors -> loss should be 0
    x = ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    loss_identical = sce_loss(x, x, alpha=2.0)
    assert float(ops.convert_to_numpy(loss_identical)) < 1e-6

    # Orthogonal vectors -> cos_sim = 0, (1 - 0)^2 = 1
    y = ops.convert_to_tensor([[0.0, 1.0], [1.0, 0.0]], dtype="float32")
    loss_ortho = sce_loss(x, y, alpha=2.0)
    assert abs(float(ops.convert_to_numpy(loss_ortho)) - 1.0) < 1e-5

    # Opposing vectors -> cos_sim = -1, (1 - (-1))^2 = 4
    z = ops.convert_to_tensor([[-1.0, 0.0], [0.0, -1.0]], dtype="float32")
    loss_opp = sce_loss(x, z, alpha=2.0)
    assert abs(float(ops.convert_to_numpy(loss_opp)) - 4.0) < 1e-5


def test_graphmae2_basic():
    in_dim = 16
    num_hidden = 32
    num_nodes = 8

    x = ops.convert_to_tensor(np.random.randn(num_nodes, in_dim).astype("float32"))
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 2],
        [1, 2, 3, 4, 5, 6, 7, 0, 2, 0],
    ], dtype="int32")

    model = GraphMAE2(
        in_dim=in_dim,
        num_hidden=num_hidden,
        num_layers=2,
        num_dec_layers=1,
        num_remasking=2,
        nhead=4,
        nhead_out=1,
        activation="prelu",
        norm="layernorm",
        residual=True,
    )

    expected_repr = f"GraphMAE2(in_dim={in_dim}, num_hidden={num_hidden}, num_layers=2, num_dec_layers=1, nhead=4)"
    assert str(model) == expected_repr

    # Forward pass / embed
    emb = model(x, edge_index)
    assert ops.shape(emb) == (num_nodes, num_hidden)

    emb2 = model.embed(x, edge_index)
    assert ops.shape(emb2) == (num_nodes, num_hidden)

    # Loss computation
    loss_val = model.loss(x, edge_index)
    assert float(ops.convert_to_numpy(loss_val)) >= 0.0

    # EMA update
    initial_teacher_w = float(ops.convert_to_numpy(ops.mean(model.encoder_ema.gat_layers[0].fc.kernel)))
    # Modify student weights
    model.encoder.gat_layers[0].fc.kernel.assign(model.encoder.gat_layers[0].fc.kernel + 1.0)
    model.ema_update(momentum=0.5)
    updated_teacher_w = float(ops.convert_to_numpy(ops.mean(model.encoder_ema.gat_layers[0].fc.kernel)))
    assert updated_teacher_w != initial_teacher_w


@pytest.mark.parametrize("remask_method", ["random", "fixed"])
def test_graphmae2_remask_methods(remask_method):
    in_dim = 8
    num_hidden = 16
    num_nodes = 6

    x = ops.convert_to_tensor(np.random.randn(num_nodes, in_dim).astype("float32"))
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 0],
    ], dtype="int32")

    model = GraphMAE2(
        in_dim=in_dim,
        num_hidden=num_hidden,
        num_layers=2,
        num_dec_layers=1,
        num_remasking=2,
        nhead=2,
        nhead_out=1,
        remask_method=remask_method,
        loss_fn="sce",
    )

    loss_val = model.loss(x, edge_index)
    assert float(ops.convert_to_numpy(loss_val)) >= 0.0


def test_load_graphmae2_checkpoint():
    ckpt_path = "GraphMAE2-main/GraphMAE2_checkpoints/gat_gat_1024_4_ogbn-arxiv_0.5_1024_checkpoint.pt"
    if not os.path.exists(ckpt_path):
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

    # Verify forward pass with loaded weights
    num_nodes = 5
    x = ops.convert_to_tensor(np.random.randn(num_nodes, 128).astype("float32"))
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3, 4, 0],
        [1, 2, 3, 4, 0, 2],
    ], dtype="int32")

    out = model(x, edge_index)
    assert ops.shape(out) == (num_nodes, 1024)

    # Compute loss with loaded weights
    loss = model.loss(x, edge_index)
    assert float(ops.convert_to_numpy(loss)) >= 0.0


def test_load_graphmae2_checkpoint_products():
    ckpt_path = "GraphMAE2-main/GraphMAE2_checkpoints/gat_gat_1024_4_ogbn-products_0.5_1024_checkpoint.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Checkpoint {ckpt_path} not found")

    model = GraphMAE2(
        in_dim=100,
        num_hidden=1024,
        num_layers=4,
        num_dec_layers=1,
        nhead=4,
        nhead_out=1,
        activation="prelu",
        norm="layernorm",
        residual=True,
    )

    model.load_weights_from_checkpoint(ckpt_path)

    num_nodes = 4
    x = ops.convert_to_tensor(np.random.randn(num_nodes, 100).astype("float32"))
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 0],
    ], dtype="int32")

    out = model(x, edge_index)
    assert ops.shape(out) == (num_nodes, 1024)


def test_download_graphmae2_checkpoint(monkeypatch, tmp_path):
    # Test invalid dataset
    with pytest.raises(ValueError, match="Unknown dataset"):
        download_graphmae2_checkpoint("unknown_dataset")

    # Test existing file returns path directly without download
    local_dir = "GraphMAE2-main/GraphMAE2_checkpoints"
    if os.path.exists(local_dir):
        path = download_graphmae2_checkpoint("ogbn-arxiv", folder=local_dir)
        assert os.path.exists(path)
        assert "ogbn-arxiv" in path

    # Test download invocation using monkeypatched download_google_url
    called = {}

    def mock_download(id, folder, filename, log=True):
        called["id"] = id
        called["folder"] = folder
        called["filename"] = filename
        os.makedirs(folder, exist_ok=True)
        target = os.path.join(folder, filename)
        with open(target, "wb") as f:
            f.write(b"mock_content")
        return target

    monkeypatch.setattr("k3_node.models.graphmae2.download_google_url", mock_download)

    fake_folder = str(tmp_path / "ckpts")
    res_path = download_graphmae2_checkpoint("mag-scholar-f", folder=fake_folder)
    assert os.path.exists(res_path)
    assert called["id"] == "1KpQk_OKbbo4qTLQYZ84pAJDy1sh4oZv2"
    assert "mag-scholar-f" in called["filename"]


def test_graphmae2_from_pretrained():
    local_dir = "GraphMAE2-main/GraphMAE2_checkpoints"
    if not os.path.exists(local_dir):
        pytest.skip(f"Checkpoints directory {local_dir} not found")

    model = GraphMAE2.from_pretrained("ogbn-arxiv", folder=local_dir)
    assert model.in_dim == 128
    assert model.num_hidden == 1024
    assert model.nhead == 8

    num_nodes = 3
    x = ops.convert_to_tensor(np.random.randn(num_nodes, 128).astype("float32"))
    edge_index = ops.convert_to_tensor([[0, 1, 2], [1, 2, 0]], dtype="int32")
    out = model(x, edge_index)
    assert ops.shape(out) == (num_nodes, 1024)

    # Test products config
    model_prod = GraphMAE2.from_pretrained("ogbn-products", folder=local_dir)
    assert model_prod.in_dim == 100
    assert model_prod.num_hidden == 1024
    assert model_prod.nhead == 4

