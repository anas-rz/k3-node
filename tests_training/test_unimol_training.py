"""
Training verification tests for all Uni-Mol models.

Verifies that models can compute losses, calculate gradients, execute
training steps, and that training successfully reduces loss.
"""

import numpy as np
import pytest
import keras
from keras import ops

from k3_node.models import (
    UniMolModel,
    UniMolConfGenModel,
    UniMolDockingModel,
    UniMol2Model,
    UniMolPlusPCQModel,
    UniMolPlusOC20Model,
    DockingPoseModelV2,
)


def test_unimol_model_training():
    bsz = 2
    seq_len = 6
    vocab_size = 32
    embed_dim = 32
    heads = 4
    epochs = 10

    model = UniMolModel(
        output_dim=2,
        vocab_size=vocab_size,
        encoder_layers=2,
        encoder_embed_dim=embed_dim,
        encoder_ffn_embed_dim=64,
        encoder_attention_heads=heads,
        num_kernel=16,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss="mse")

    inputs = {
        "src_tokens": np.random.randint(1, vocab_size, (bsz, seq_len)).astype("int32"),
        "src_coord": np.random.randn(bsz, seq_len, 3).astype("float32"),
    }
    target = np.random.randn(bsz, 2).astype("float32")

    l0 = float(model.train_on_batch(inputs, target))
    l_last = l0
    for _ in range(epochs - 1):
        l_last = float(model.train_on_batch(inputs, target))

    assert l_last < l0, f"UniMolModel loss did not decrease: initial={l0:.4f}, final={l_last:.4f}"


def test_unimol_conf_gen_training():
    bsz = 2
    seq_len = 5
    vocab_size = 32
    embed_dim = 32
    heads = 4
    epochs = 10

    model = UniMolConfGenModel(
        vocab_size=vocab_size,
        encoder_layers=2,
        encoder_embed_dim=embed_dim,
        encoder_ffn_embed_dim=64,
        encoder_attention_heads=heads,
        num_kernel=16,
    )

    class TrainableWrapper(keras.Model):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def call(self, inputs, training=False):
            updated_coord, pred_dist = self.m(inputs, training=training)
            # Flatten outputs for joint loss
            c = ops.reshape(updated_coord, (bsz, -1))
            d = ops.reshape(pred_dist, (bsz, -1))
            return ops.concatenate([c, d], axis=-1)

    wrapper = TrainableWrapper(model)
    wrapper.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss="mse")

    inputs = {
        "src_tokens": np.random.randint(1, vocab_size, (bsz, seq_len)).astype("int32"),
        "src_coord": np.random.randn(bsz, seq_len, 3).astype("float32"),
    }
    target_coord = np.random.randn(bsz, seq_len * 3).astype("float32")
    target_dist = np.random.randn(bsz, seq_len * seq_len).astype("float32")
    target = np.concatenate([target_coord, target_dist], axis=-1)

    l0 = float(wrapper.train_on_batch(inputs, target))
    l_last = l0
    for _ in range(epochs - 1):
        l_last = float(wrapper.train_on_batch(inputs, target))

    assert l_last < l0, f"UniMolConfGenModel loss did not decrease: initial={l0:.4f}, final={l_last:.4f}"


def test_unimol_docking_training():
    bsz = 2
    seq_len = 5
    vocab_size = 32
    embed_dim = 32
    heads = 4
    epochs = 10

    model = UniMolDockingModel(
        vocab_size=vocab_size,
        encoder_layers=2,
        encoder_embed_dim=embed_dim,
        encoder_ffn_embed_dim=64,
        encoder_attention_heads=heads,
        num_kernel=16,
    )

    class TrainableWrapper(keras.Model):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def call(self, inputs, training=False):
            pose, dist = self.m(inputs, training=training)
            p = ops.reshape(pose, (bsz, -1))
            d = ops.reshape(dist, (bsz, -1))
            return ops.concatenate([p, d], axis=-1)

    wrapper = TrainableWrapper(model)
    wrapper.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss="mse")

    inputs = {
        "src_tokens": np.random.randint(1, vocab_size, (bsz, seq_len)).astype("int32"),
        "src_coord": np.random.randn(bsz, seq_len, 3).astype("float32"),
    }
    target_pose = np.random.randn(bsz, seq_len * 3).astype("float32")
    target_dist = np.random.randn(bsz, seq_len * seq_len).astype("float32")
    target = np.concatenate([target_pose, target_dist], axis=-1)

    l0 = float(wrapper.train_on_batch(inputs, target))
    l_last = l0
    for _ in range(epochs - 1):
        l_last = float(wrapper.train_on_batch(inputs, target))

    assert l_last < l0, f"UniMolDockingModel loss did not decrease: initial={l0:.4f}, final={l_last:.4f}"


def test_unimol2_model_training():
    bsz = 2
    seq_len = 5
    embed_dim = 32
    pair_dim = 16
    heads = 4
    epochs = 10

    model = UniMol2Model(
        model_size="84m",
        output_dim=2,
        num_encoder_layers=2,
        encoder_embed_dim=embed_dim,
        num_attention_heads=heads,
        pair_embed_dim=pair_dim,
        ffn_embedding_dim=64,
        pair_hidden_dim=8,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss="mse")

    inputs = {
        "atom_types": np.random.randint(0, 128, (bsz, seq_len)).astype("int32"),
        "coords": np.random.randn(bsz, seq_len, 3).astype("float32"),
    }
    target = np.random.randn(bsz, 2).astype("float32")

    l0 = float(model.train_on_batch(inputs, target))
    l_last = l0
    for _ in range(epochs - 1):
        l_last = float(model.train_on_batch(inputs, target))

    assert l_last < l0, f"UniMol2Model loss did not decrease: initial={l0:.4f}, final={l_last:.4f}"


def test_unimol_plus_pcq_training():
    bsz = 2
    seq_len = 5
    embed_dim = 32
    pair_dim = 16
    heads = 4
    epochs = 10

    model = UniMolPlusPCQModel(
        num_layers=2,
        embed_dim=embed_dim,
        pair_dim=pair_dim,
        num_heads=heads,
        output_dim=1,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss="mse")

    inputs = {
        "atom_types": np.random.randint(0, 64, (bsz, seq_len)).astype("int32"),
        "coords": np.random.randn(bsz, seq_len, 3).astype("float32"),
    }
    target = np.random.randn(bsz, 1).astype("float32")

    l0 = float(model.train_on_batch(inputs, target))
    l_last = l0
    for _ in range(epochs - 1):
        l_last = float(model.train_on_batch(inputs, target))

    assert l_last < l0, f"UniMolPlusPCQModel loss did not decrease: initial={l0:.4f}, final={l_last:.4f}"


def test_unimol_plus_oc20_training():
    bsz = 2
    seq_len = 5
    embed_dim = 32
    pair_dim = 16
    heads = 4
    epochs = 10

    model = UniMolPlusOC20Model(
        num_layers=2,
        embed_dim=embed_dim,
        pair_dim=pair_dim,
        num_heads=heads,
        output_dim=1,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss="mse")

    inputs = {
        "atom_types": np.random.randint(0, 64, (bsz, seq_len)).astype("int32"),
        "coords": np.random.randn(bsz, seq_len, 3).astype("float32"),
    }
    target = np.random.randn(bsz, 1).astype("float32")

    l0 = float(model.train_on_batch(inputs, target))
    l_last = l0
    for _ in range(epochs - 1):
        l_last = float(model.train_on_batch(inputs, target))

    assert l_last < l0, f"UniMolPlusOC20Model loss did not decrease: initial={l0:.4f}, final={l_last:.4f}"


def test_unimol_docking_v2_training():
    bsz = 2
    n_mol = 3
    n_pkt = 4
    embed_dim = 32
    pair_dim = 16
    heads = 4
    epochs = 10

    model = DockingPoseModelV2(
        mol_vocab_size=32,
        pocket_vocab_size=32,
        embed_dim=embed_dim,
        pair_dim=pair_dim,
        num_layers=2,
        num_heads=heads,
    )

    class TrainableWrapper(keras.Model):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def call(self, inputs, training=False):
            pred_mol_coords, pred_dist = self.m(inputs, training=training)
            c = ops.reshape(pred_mol_coords, (bsz, -1))
            d = ops.reshape(pred_dist, (bsz, -1))
            return ops.concatenate([c, d], axis=-1)

    wrapper = TrainableWrapper(model)
    wrapper.compile(optimizer=keras.optimizers.Adam(learning_rate=0.02), loss="mse")

    inputs = {
        "mol_tokens": np.random.randint(0, 32, (bsz, n_mol)).astype("int32"),
        "pocket_tokens": np.random.randint(0, 32, (bsz, n_pkt)).astype("int32"),
        "mol_coords": np.random.randn(bsz, n_mol, 3).astype("float32"),
        "pocket_coords": np.random.randn(bsz, n_pkt, 3).astype("float32"),
    }
    target_c = np.random.randn(bsz, n_mol * 3).astype("float32")
    target_d = np.random.randn(bsz, (n_mol + n_pkt) ** 2).astype("float32")
    target = np.concatenate([target_c, target_d], axis=-1)

    l0 = float(wrapper.train_on_batch(inputs, target))
    l_last = l0
    for _ in range(epochs - 1):
        l_last = float(wrapper.train_on_batch(inputs, target))

    assert l_last < l0, f"DockingPoseModelV2 loss did not decrease: initial={l0:.4f}, final={l_last:.4f}"

