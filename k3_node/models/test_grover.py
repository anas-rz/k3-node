import os
import pytest
import numpy as np
import keras
from keras import ops
try:
    import torch
except ImportError:
    torch = None

from k3_node.models.grover import (
    GROVER,
    GroverPReLU,
    MPNEncoder,
    MTBlock,
    Readout,
    GTransEncoder,
    load_grover_weights,
    download_grover_checkpoint,
)


def _make_dummy_batch(num_atoms=5, num_bonds=7, atom_fdim=151, bond_fdim=165, edge_fdim=None):
    if edge_fdim is not None:
        bond_fdim = edge_fdim
    np.random.seed(42)
    f_atoms = np.random.randn(num_atoms, atom_fdim).astype(np.float32)
    f_atoms[0] = 0.0
    f_bonds = np.random.randn(num_bonds, bond_fdim).astype(np.float32)
    f_bonds[0] = 0.0

    a2b = np.array([[0, 0], [2, 0], [1, 4], [3, 0], [0, 0]], dtype=np.int64)
    b2a = np.array([0, 1, 2, 2, 3, 1, 3], dtype=np.int64)
    b2revb = np.array([0, 2, 1, 4, 3, 6, 5], dtype=np.int64)
    a_scope = np.array([[1, 2], [3, 2]], dtype=np.int64)
    b_scope = np.array([[1, 3], [4, 3]], dtype=np.int64)
    a2a = b2a[a2b]

    return (
        ops.convert_to_tensor(f_atoms),
        ops.convert_to_tensor(f_bonds),
        ops.convert_to_tensor(a2b),
        ops.convert_to_tensor(b2a),
        ops.convert_to_tensor(b2revb),
        ops.convert_to_tensor(a_scope),
        ops.convert_to_tensor(b_scope),
        ops.convert_to_tensor(a2a),
    )


def test_grover_prelu():
    layer = GroverPReLU(init_val=0.25)
    x = ops.convert_to_tensor([-2.0, 0.0, 3.0], dtype="float32")
    y = layer(x)
    np.testing.assert_allclose(ops.convert_to_numpy(y), [-0.5, 0.0, 3.0], atol=1e-6)

    # Config test
    config = layer.get_config()
    assert config["init_val"] == 0.25


def test_mpn_encoder():
    hidden_size = 32
    depth = 3
    num_atoms = 4
    num_bonds = 4
    atom_dim = 16
    bond_dim = 20

    # 1. Atom messages
    encoder_atom = MPNEncoder(
        hidden_size=hidden_size,
        depth=depth,
        atom_messages=True,
        input_layer="fc",
        input_dim=atom_dim,
    )
    init_msg = ops.convert_to_tensor(np.random.randn(num_atoms, atom_dim).astype(np.float32))
    a2nei = ops.convert_to_tensor(np.array([[0, 0], [2, 0], [1, 3], [2, 0]]), dtype="int64")

    out = encoder_atom(
        init_messages=init_msg,
        init_attached_features=None,
        a2nei=a2nei,
        a2attached=None,
    )
    assert ops.shape(out) == (num_atoms, hidden_size)

    # 2. Bond messages (non-backtracking)
    encoder_bond = MPNEncoder(
        hidden_size=hidden_size,
        depth=depth,
        atom_messages=False,
        input_layer="fc",
        input_dim=bond_dim,
    )
    init_bond = ops.convert_to_tensor(np.random.randn(num_bonds, bond_dim).astype(np.float32))
    a2b = ops.convert_to_tensor(np.array([[0], [2], [1], [3]]), dtype="int64")
    b2a = ops.convert_to_tensor(np.array([0, 1, 2, 2]), dtype="int64")
    b2revb = ops.convert_to_tensor(np.array([0, 2, 1, 3]), dtype="int64")

    out_bond = encoder_bond(
        init_messages=init_bond,
        init_attached_features=None,
        a2nei=a2b,
        a2attached=None,
        b2a=b2a,
        b2revb=b2revb,
    )
    assert ops.shape(out_bond) == (num_bonds, hidden_size)


def test_mt_block():
    hidden_size = 32
    num_attn_head = 2
    depth = 3
    num_atoms = 5
    num_bonds = 7
    atom_dim = 16
    bond_dim = 20

    block = MTBlock(
        hidden_size=hidden_size,
        input_dim=bond_dim,
        num_attn_head=num_attn_head,
        depth=depth,
        atom_messages=False,
    )

    batch = _make_dummy_batch(num_atoms, num_bonds, atom_dim, bond_dim)
    f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch

    out_atoms, out_bonds = block(f_atoms, f_bonds, a2b, b2a, b2revb, a2a)
    assert ops.shape(out_bonds) == (num_bonds, hidden_size)


def test_readout():
    hidden_size = 32
    emb = ops.convert_to_tensor(np.random.randn(8, hidden_size).astype(np.float32))
    scope = ops.convert_to_tensor(np.array([[0, 3], [3, 4], [7, 1]]), dtype="int64")

    # Mean Readout
    readout_mean = Readout(rtype="mean", hidden_size=hidden_size)
    out_mean = readout_mean(emb, scope)
    assert ops.shape(out_mean) == (3, hidden_size)

    # Self-attention Readout
    readout_sa = Readout(rtype="self_attention", hidden_size=hidden_size, attn_hidden=16, attn_out=4)
    out_sa = readout_sa(emb, scope)
    assert ops.shape(out_sa) == (3, 4 * hidden_size)


def test_gtrans_encoder():
    hidden_size = 32
    batch = _make_dummy_batch(num_atoms=5, num_bonds=7, atom_fdim=151, edge_fdim=165)
    f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = batch

    # 'both'
    encoder = GTransEncoder(
        hidden_size=hidden_size,
        edge_fdim=165,
        node_fdim=151,
        num_mt_block=1,
        num_attn_head=2,
        depth=3,
        atom_emb_output="both",
    )
    res = encoder(f_atoms, f_bonds, a2b, b2a, b2revb, a2a)
    assert "atom_from_atom" in res
    assert "bond_from_bond" in res
    assert ops.shape(res["atom_from_atom"]) == (5, hidden_size)
    assert ops.shape(res["bond_from_bond"]) == (7, hidden_size)


def test_grover_model_and_fingerprint():
    hidden_size = 32
    model = GROVER(
        hidden_size=hidden_size,
        edge_fdim=165,
        node_fdim=151,
        num_mt_block=1,
        num_attn_head=2,
        depth=3,
        atom_emb_output="both",
        readout_type="mean",
    )

    batch = _make_dummy_batch(num_atoms=5, num_bonds=7, atom_fdim=151, edge_fdim=165)
    out = model(batch)
    assert ops.shape(out["atom_from_atom"]) == (5, hidden_size)

    # Fingerprint
    fp = model.get_fingerprint(batch, fingerprint_source="both")
    # 2 molecules in scope, 4 branches * 32 = 128
    assert ops.shape(fp) == (2, 4 * hidden_size)


def test_load_grover_weights_synthetic(tmp_path):
    if torch is None:
        pytest.skip("PyTorch is required for checkpoint loading test")
    hidden_size = 16
    num_attn_head = 2
    depth = 2
    node_fdim = 151
    edge_fdim = 165

    model = GROVER(
        hidden_size=hidden_size,
        edge_fdim=edge_fdim,
        node_fdim=node_fdim,
        num_mt_block=1,
        num_attn_head=num_attn_head,
        depth=depth,
        atom_emb_output="both",
    )
    model.build(None)

    # Create dummy torch state dict
    state_dict = {}
    for block_name, dim in [("edge_blocks", edge_fdim), ("node_blocks", node_fdim)]:
        prefix = f"grover.encoders.{block_name}.0"
        state_dict[f"{prefix}.W_i.weight"] = torch.randn(hidden_size, dim)
        state_dict[f"{prefix}.act_func.weight"] = torch.tensor([0.25])
        state_dict[f"{prefix}.layernorm.weight"] = torch.ones(hidden_size)
        state_dict[f"{prefix}.layernorm.bias"] = torch.zeros(hidden_size)
        state_dict[f"{prefix}.sublayer.norm.weight"] = torch.ones(hidden_size)
        state_dict[f"{prefix}.sublayer.norm.bias"] = torch.zeros(hidden_size)
        state_dict[f"{prefix}.W_o.weight"] = torch.randn(hidden_size, hidden_size * num_attn_head)
        for i in range(3):
            state_dict[f"{prefix}.attn.linear_layers.{i}.weight"] = torch.randn(hidden_size, hidden_size)
            state_dict[f"{prefix}.attn.linear_layers.{i}.bias"] = torch.zeros(hidden_size)
        state_dict[f"{prefix}.attn.output_linear.weight"] = torch.randn(hidden_size, hidden_size)
        for hi in range(num_attn_head):
            for mpn in ["mpn_q", "mpn_k", "mpn_v"]:
                state_dict[f"{prefix}.heads.{hi}.{mpn}.W_h.weight"] = torch.randn(hidden_size, hidden_size)
                state_dict[f"{prefix}.heads.{hi}.{mpn}.act_func.weight"] = torch.tensor([0.25])

    for ffn_key, in_dim in [
        ("ffn_atom_from_atom", hidden_size + node_fdim),
        ("ffn_atom_from_bond", hidden_size + node_fdim),
        ("ffn_bond_from_atom", hidden_size + edge_fdim),
        ("ffn_bond_from_bond", hidden_size + edge_fdim),
    ]:
        prefix = f"grover.encoders.{ffn_key}"
        state_dict[f"{prefix}.W_1.weight"] = torch.randn(hidden_size * 4, in_dim)
        state_dict[f"{prefix}.W_1.bias"] = torch.zeros(hidden_size * 4)
        state_dict[f"{prefix}.W_2.weight"] = torch.randn(hidden_size, hidden_size * 4)
        state_dict[f"{prefix}.W_2.bias"] = torch.zeros(hidden_size)
        state_dict[f"{prefix}.act_func.weight"] = torch.tensor([0.25])

    for sub in ["atom_from_atom", "atom_from_bond", "bond_from_atom", "bond_from_bond"]:
        state_dict[f"grover.encoders.{sub}_sublayer.norm.weight"] = torch.ones(hidden_size)
        state_dict[f"grover.encoders.{sub}_sublayer.norm.bias"] = torch.zeros(hidden_size)

    state_dict["grover.encoders.act_func_node.weight"] = torch.tensor([0.25])
    state_dict["grover.encoders.act_func_edge.weight"] = torch.tensor([0.25])

    save_path = str(tmp_path / "mock_grover.pt")
    torch.save({"state_dict": state_dict}, save_path)

    load_grover_weights(model, save_path)
    # verify forward pass succeeds
    batch = _make_dummy_batch(num_atoms=5, num_bonds=7, atom_fdim=node_fdim, edge_fdim=edge_fdim)
    out = model(batch)
    assert ops.shape(out["atom_from_atom"]) == (5, hidden_size)


def test_grover_official_pretrained_load():
    ckpt_path = "/tmp/grover_download_test/grover_base.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip("Official checkpoint not present at /tmp/grover_download_test/grover_base.pt")

    model = GROVER(
        hidden_size=800,
        edge_fdim=165,
        node_fdim=151,
        num_mt_block=1,
        num_attn_head=4,
        depth=6,
        atom_emb_output="both",
    )
    load_grover_weights(model, ckpt_path)

    batch = _make_dummy_batch(num_atoms=5, num_bonds=7, atom_fdim=151, edge_fdim=165)
    out = model(batch)
    assert ops.shape(out["atom_from_atom"]) == (5, 800)
    assert ops.shape(out["bond_from_bond"]) == (7, 800)
