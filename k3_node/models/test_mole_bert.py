import os
import numpy as np
import pytest
import keras.ops as ops
try:
    import torch
except ImportError:
    torch = None

from k3_node.models.mole_bert import (
    MoleBERT,
    MoleBERTGNN,
    MoleBERTGINConv,
    load_mole_bert_weights,
    download_mole_bert_checkpoint,
)


def _make_dummy_graph(num_nodes=5, num_edges=8, emb_dim=300):
    np.random.seed(42)
    x = np.stack(
        [
            np.random.randint(0, 119, size=num_nodes),
            np.random.randint(0, 3, size=num_nodes),
        ],
        axis=1,
    )
    src = np.random.randint(0, num_nodes, size=num_edges)
    dst = np.random.randint(0, num_nodes, size=num_edges)
    edge_index = np.stack([src, dst], axis=0)
    edge_attr = np.stack(
        [
            np.random.randint(0, 5, size=num_edges),
            np.random.randint(0, 3, size=num_edges),
        ],
        axis=1,
    )
    batch = np.array([0, 0, 0, 1, 1], dtype=np.int64)

    return (
        ops.convert_to_tensor(x, dtype="int64"),
        ops.convert_to_tensor(edge_index, dtype="int64"),
        ops.convert_to_tensor(edge_attr, dtype="int64"),
        ops.convert_to_tensor(batch, dtype="int64"),
    )


def test_mole_bert_gin_conv():
    emb_dim = 64
    conv = MoleBERTGINConv(emb_dim=emb_dim)
    conv.build(None)

    num_nodes = 4
    x = ops.convert_to_tensor(np.random.randn(num_nodes, emb_dim).astype(np.float32))
    edge_index = ops.convert_to_tensor(
        np.array([[0, 1, 1, 2], [1, 0, 2, 1]]), dtype="int64"
    )
    edge_attr = ops.convert_to_tensor(
        np.array([[0, 0], [0, 0], [1, 0], [1, 0]]), dtype="int64"
    )

    out = conv(x, edge_index, edge_attr)
    assert ops.shape(out) == (num_nodes, emb_dim)


def test_mole_bert_gnn_jk_modes():
    x, edge_index, edge_attr, _ = _make_dummy_graph(num_nodes=5, num_edges=8, emb_dim=32)
    emb_dim = 32

    # 1. JK = 'last'
    gnn_last = MoleBERTGNN(num_layer=3, emb_dim=emb_dim, JK="last")
    gnn_last.build(None)
    out_last = gnn_last(x, edge_index, edge_attr)
    assert ops.shape(out_last) == (5, emb_dim)

    # 2. JK = 'concat'
    gnn_concat = MoleBERTGNN(num_layer=3, emb_dim=emb_dim, JK="concat")
    gnn_concat.build(None)
    out_concat = gnn_concat(x, edge_index, edge_attr)
    assert ops.shape(out_concat) == (5, 4 * emb_dim)

    # 3. JK = 'sum'
    gnn_sum = MoleBERTGNN(num_layer=3, emb_dim=emb_dim, JK="sum")
    gnn_sum.build(None)
    out_sum = gnn_sum(x, edge_index, edge_attr)
    assert ops.shape(out_sum) == (5, emb_dim)

    # 4. JK = 'max'
    gnn_max = MoleBERTGNN(num_layer=3, emb_dim=emb_dim, JK="max")
    gnn_max.build(None)
    out_max = gnn_max(x, edge_index, edge_attr)
    assert ops.shape(out_max) == (5, emb_dim)


def test_mole_bert_pooling_and_prediction():
    x, edge_index, edge_attr, batch = _make_dummy_graph(num_nodes=5, num_edges=8, emb_dim=32)
    emb_dim = 32

    # Multi-task graph classification (e.g. ClinTox: 2 tasks, Tox21: 12 tasks)
    for pooling in ["mean", "sum", "max"]:
        model = MoleBERT(
            num_layer=3,
            emb_dim=emb_dim,
            num_tasks=2,
            JK="last",
            graph_pooling=pooling,
        )
        model.build(None)

        logits, node_rep = model((x, edge_index, edge_attr, batch))
        assert ops.shape(logits) == (2, 2)
        assert ops.shape(node_rep) == (5, emb_dim)


def test_mole_bert_synthetic_checkpoint_load(tmp_path):
    if torch is None:
        pytest.skip("PyTorch is required for checkpoint loading test")
    emb_dim = 16
    num_layer = 3
    model = MoleBERT(num_layer=num_layer, emb_dim=emb_dim, num_tasks=1)
    model.build(None)

    state_dict = {
        "x_embedding1.weight": torch.randn(120, emb_dim),
        "x_embedding2.weight": torch.randn(3, emb_dim),
    }
    for l in range(num_layer):
        state_dict[f"gnns.{l}.edge_embedding1.weight"] = torch.randn(6, emb_dim)
        state_dict[f"gnns.{l}.edge_embedding2.weight"] = torch.randn(3, emb_dim)
        state_dict[f"gnns.{l}.mlp.0.weight"] = torch.randn(2 * emb_dim, emb_dim)
        state_dict[f"gnns.{l}.mlp.0.bias"] = torch.zeros(2 * emb_dim)
        state_dict[f"gnns.{l}.mlp.2.weight"] = torch.randn(emb_dim, 2 * emb_dim)
        state_dict[f"gnns.{l}.mlp.2.bias"] = torch.zeros(emb_dim)
        state_dict[f"batch_norms.{l}.weight"] = torch.ones(emb_dim)
        state_dict[f"batch_norms.{l}.bias"] = torch.zeros(emb_dim)
        state_dict[f"batch_norms.{l}.running_mean"] = torch.zeros(emb_dim)
        state_dict[f"batch_norms.{l}.running_var"] = torch.ones(emb_dim)

    ckpt_file = str(tmp_path / "synthetic_mole_bert.pth")
    torch.save(state_dict, ckpt_file)

    load_mole_bert_weights(model, ckpt_file)

    x, edge_index, edge_attr, batch = _make_dummy_graph(num_nodes=5, num_edges=8, emb_dim=emb_dim)
    logits, node_rep = model((x, edge_index, edge_attr, batch))
    assert ops.shape(logits) == (2, 1)
    assert ops.shape(node_rep) == (5, emb_dim)


def test_mole_bert_official_checkpoint_load():
    ckpt_path = "Mole-BERT/model_gin/Mole-BERT.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = download_mole_bert_checkpoint()

    model = MoleBERT(num_layer=5, emb_dim=300, num_tasks=1)
    model.build(None)
    load_mole_bert_weights(model, ckpt_path)

    x, edge_index, edge_attr, batch = _make_dummy_graph(num_nodes=5, num_edges=8, emb_dim=300)
    logits, node_rep = model((x, edge_index, edge_attr, batch))
    assert ops.shape(logits) == (2, 1)
    assert ops.shape(node_rep) == (5, 300)

