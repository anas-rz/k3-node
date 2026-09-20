import os
import tempfile
import numpy as np
import pytest
import keras
from keras import ops

from k3_node.models.gps_model import (
    AtomEncoder,
    BondEncoder,
    RWSEEncoder,
    CustomGatedGCN,
    GPSLayer,
    SANGraphHead,
    GPSModel,
    load_gps_weights,
    download_gps_checkpoint,
)


def test_atom_and_bond_encoders():
    # 9 features for atoms, 3 for bonds
    x = ops.convert_to_tensor(np.array([[6, 0, 4, 4, 3, 2, 2, 0, 0],
                                        [8, 0, 2, 2, 2, 1, 1, 0, 0],
                                        [1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32))
    atom_enc = AtomEncoder(emb_dim=64)
    x_emb = atom_enc(x)
    assert ops.shape(x_emb) == (3, 64)

    edge_attr = ops.convert_to_tensor(np.array([[0, 0, 0],
                                                [1, 0, 0]], dtype=np.int32))
    bond_enc = BondEncoder(emb_dim=64)
    e_emb = bond_enc(edge_attr)
    assert ops.shape(e_emb) == (2, 64)


def test_rwse_encoder():
    pestat = ops.convert_to_tensor(np.ones((4, 16), dtype=np.float32))
    rwse = RWSEEncoder(num_rw_steps=16, pe_dim=20)
    pe_out = rwse(pestat, training=False)
    assert ops.shape(pe_out) == (4, 20)


def test_custom_gated_gcn():
    x = ops.convert_to_tensor(np.random.randn(4, 32).astype(np.float32))
    e = ops.convert_to_tensor(np.random.randn(6, 32).astype(np.float32))
    edge_index = ops.convert_to_tensor(np.array([[0, 1, 1, 2, 2, 3],
                                                 [1, 0, 2, 1, 3, 2]], dtype=np.int32))

    layer = CustomGatedGCN(in_dim=32, out_dim=32, dropout=0.0, residual=True, act="gelu")
    x_out, e_out = layer(x, edge_index, e, training=False)

    assert ops.shape(x_out) == (4, 32)
    assert ops.shape(e_out) == (6, 32)


def test_gps_layer():
    dim_h = 32
    num_heads = 4
    x = ops.convert_to_tensor(np.random.randn(5, dim_h).astype(np.float32))
    e = ops.convert_to_tensor(np.random.randn(6, dim_h).astype(np.float32))
    edge_index = ops.convert_to_tensor(np.array([[0, 1, 1, 2, 3, 4],
                                                 [1, 0, 2, 1, 4, 3]], dtype=np.int32))
    # Two graphs: graph 0 has 3 nodes, graph 1 has 2 nodes
    batch = ops.convert_to_tensor(np.array([0, 0, 0, 1, 1], dtype=np.int32))

    layer = GPSLayer(
        dim_h=dim_h,
        local_gnn_type="CustomGatedGCN",
        global_model_type="Transformer",
        num_heads=num_heads,
        act="gelu",
        dropout=0.0,
        attn_dropout=0.0,
        batch_norm=True,
    )

    x_out, e_out = layer(x, edge_index, e, batch=batch, training=False)
    assert ops.shape(x_out) == (5, dim_h)
    assert ops.shape(e_out) == (6, dim_h)


def test_san_graph_head():
    x = ops.convert_to_tensor(np.random.randn(6, 64).astype(np.float32))
    batch = ops.convert_to_tensor(np.array([0, 0, 0, 1, 1, 1], dtype=np.int32))

    head = SANGraphHead(dim_in=64, dim_out=1, L=2, act="gelu", pooling="mean")
    pred = head(x, batch=batch, training=False)
    assert ops.shape(pred) == (2, 1)

    head_sum = SANGraphHead(dim_in=64, dim_out=2, L=1, act="relu", pooling="sum")
    pred_sum = head_sum(x, batch=batch, training=False)
    assert ops.shape(pred_sum) == (2, 2)


def test_gps_model_forward():
    model = GPSModel(
        dim_in=32,
        dim_out=1,
        num_layers=2,
        dim_hidden=32,
        num_heads=4,
        local_gnn_type="CustomGatedGCN",
        act="gelu",
        dropout=0.1,
        attn_dropout=0.1,
        batch_norm=True,
        node_encoder_type="Atom+RWSE",
        edge_encoder_type="Bond",
        rwse_num_steps=16,
        rwse_dim_pe=8,
        graph_pooling="mean",
        head_layers=2,
    )

    x = ops.convert_to_tensor(np.array([[6, 0, 4, 4, 3, 2, 2, 0, 0],
                                        [8, 0, 2, 2, 2, 1, 1, 0, 0],
                                        [6, 0, 3, 3, 3, 2, 2, 0, 0]], dtype=np.int32))
    edge_index = ops.convert_to_tensor(np.array([[0, 1, 1, 2],
                                                 [1, 0, 2, 1]], dtype=np.int32))
    edge_attr = ops.convert_to_tensor(np.array([[0, 0, 0],
                                                [0, 0, 0],
                                                [1, 0, 0],
                                                [1, 0, 0]], dtype=np.int32))
    pestat = ops.convert_to_tensor(np.ones((3, 16), dtype=np.float32))
    batch = ops.convert_to_tensor(np.array([0, 0, 0], dtype=np.int32))

    pred_eval = model(x, edge_index, edge_attr=edge_attr, pestat_RWSE=pestat, batch=batch, training=False)
    pred_train = model(x, edge_index, edge_attr=edge_attr, pestat_RWSE=pestat, batch=batch, training=True)

    assert ops.shape(pred_eval) == (1, 1)
    assert ops.shape(pred_train) == (1, 1)


def test_gps_model_load_weights_synthetic():
    torch = pytest.importorskip("torch")

    dim_h = 32
    rwse_pe = 8
    model = GPSModel(
        dim_in=dim_h,
        dim_out=1,
        num_layers=1,
        dim_hidden=dim_h,
        num_heads=2,
        local_gnn_type="CustomGatedGCN",
        act="gelu",
        dropout=0.0,
        attn_dropout=0.0,
        batch_norm=True,
        node_encoder_type="Atom+RWSE",
        edge_encoder_type="Bond",
        rwse_num_steps=16,
        rwse_dim_pe=rwse_pe,
        graph_pooling="mean",
        head_layers=1,
    )
    model.build(None)

    # Synthetic PyTorch state dict
    state_dict = {}
    for i in range(9):
        dim_val = [119, 4, 12, 12, 10, 6, 6, 2, 2][i]
        state_dict[f"encoder.node_encoder.encoder1.atom_embedding_list.{i}.weight"] = torch.randn(dim_val, dim_h - rwse_pe)
    for i in range(3):
        dim_val = [5, 6, 2][i]
        state_dict[f"encoder.edge_encoder.bond_embedding_list.{i}.weight"] = torch.randn(dim_val, dim_h)

    p_rwse = "encoder.node_encoder.encoder2"
    state_dict[f"{p_rwse}.raw_norm.weight"] = torch.ones(16)
    state_dict[f"{p_rwse}.raw_norm.bias"] = torch.zeros(16)
    state_dict[f"{p_rwse}.raw_norm.running_mean"] = torch.zeros(16)
    state_dict[f"{p_rwse}.raw_norm.running_var"] = torch.ones(16)
    state_dict[f"{p_rwse}.pe_encoder.weight"] = torch.randn(rwse_pe, 16)
    state_dict[f"{p_rwse}.pe_encoder.bias"] = torch.zeros(rwse_pe)

    p0 = "layers.0"
    for proj in ["A", "B", "C", "D", "E"]:
        state_dict[f"{p0}.local_model.{proj}.weight"] = torch.randn(dim_h, dim_h)
        state_dict[f"{p0}.local_model.{proj}.bias"] = torch.zeros(dim_h)
    for bn in ["bn_node_x", "bn_edge_e"]:
        state_dict[f"{p0}.local_model.{bn}.weight"] = torch.ones(dim_h)
        state_dict[f"{p0}.local_model.{bn}.bias"] = torch.zeros(dim_h)
        state_dict[f"{p0}.local_model.{bn}.running_mean"] = torch.zeros(dim_h)
        state_dict[f"{p0}.local_model.{bn}.running_var"] = torch.ones(dim_h)

    state_dict[f"{p0}.norm1_local.weight"] = torch.ones(dim_h)
    state_dict[f"{p0}.norm1_local.bias"] = torch.zeros(dim_h)
    state_dict[f"{p0}.norm1_local.running_mean"] = torch.zeros(dim_h)
    state_dict[f"{p0}.norm1_local.running_var"] = torch.ones(dim_h)

    state_dict[f"{p0}.self_attn.in_proj_weight"] = torch.randn(3 * dim_h, dim_h)
    state_dict[f"{p0}.self_attn.in_proj_bias"] = torch.zeros(3 * dim_h)
    state_dict[f"{p0}.self_attn.out_proj.weight"] = torch.randn(dim_h, dim_h)
    state_dict[f"{p0}.self_attn.out_proj.bias"] = torch.zeros(dim_h)

    state_dict[f"{p0}.norm1_attn.weight"] = torch.ones(dim_h)
    state_dict[f"{p0}.norm1_attn.bias"] = torch.zeros(dim_h)
    state_dict[f"{p0}.norm1_attn.running_mean"] = torch.zeros(dim_h)
    state_dict[f"{p0}.norm1_attn.running_var"] = torch.ones(dim_h)

    state_dict[f"{p0}.ff_linear1.weight"] = torch.randn(dim_h * 2, dim_h)
    state_dict[f"{p0}.ff_linear1.bias"] = torch.zeros(dim_h * 2)
    state_dict[f"{p0}.ff_linear2.weight"] = torch.randn(dim_h, dim_h * 2)
    state_dict[f"{p0}.ff_linear2.bias"] = torch.zeros(dim_h)

    state_dict[f"{p0}.norm2.weight"] = torch.ones(dim_h)
    state_dict[f"{p0}.norm2.bias"] = torch.zeros(dim_h)
    state_dict[f"{p0}.norm2.running_mean"] = torch.zeros(dim_h)
    state_dict[f"{p0}.norm2.running_var"] = torch.ones(dim_h)

    state_dict["post_mp.FC_layers.0.weight"] = torch.randn(dim_h // 2, dim_h)
    state_dict["post_mp.FC_layers.0.bias"] = torch.zeros(dim_h // 2)
    state_dict["post_mp.FC_layers.1.weight"] = torch.randn(1, dim_h // 2)
    state_dict["post_mp.FC_layers.1.bias"] = torch.zeros(1)

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        torch.save({"model_state": state_dict}, f.name)
        ckpt_path = f.name

    try:
        load_gps_weights(model, ckpt_path)
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


def test_gps_model_pretrained_checkpoint_if_available():
    torch = pytest.importorskip("torch")
    ckpt_path = download_gps_checkpoint("pcqm4m-GPS+RWSE.deep")
    if os.path.exists(ckpt_path):
        model = GPSModel(
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
        x = ops.convert_to_tensor(np.array([[6, 0, 4, 4, 3, 2, 2, 0, 0],
                                            [8, 0, 2, 2, 2, 1, 1, 0, 0],
                                            [6, 0, 3, 3, 3, 2, 2, 0, 0]], dtype=np.int32))
        edge_index = ops.convert_to_tensor(np.array([[0, 1, 1, 2],
                                                     [1, 0, 2, 1]], dtype=np.int32))
        edge_attr = ops.convert_to_tensor(np.array([[0, 0, 0],
                                                    [0, 0, 0],
                                                    [1, 0, 0],
                                                    [1, 0, 0]], dtype=np.int32))
        pestat = ops.convert_to_tensor(np.ones((3, 16), dtype=np.float32))
        batch = ops.convert_to_tensor(np.array([0, 0, 0], dtype=np.int32))

        # Build and load
        _ = model(x, edge_index, edge_attr=edge_attr, pestat_RWSE=pestat, batch=batch, training=False)
        load_gps_weights(model, ckpt_path)

        out = model(x, edge_index, edge_attr=edge_attr, pestat_RWSE=pestat, batch=batch, training=False)
        val = float(ops.convert_to_numpy(out)[0, 0])
        # PyTorch reference output was ~15.3017
        np.testing.assert_allclose(val, 15.3017, rtol=1e-3, atol=1e-3)

