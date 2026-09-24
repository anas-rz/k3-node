import os
import tempfile
import numpy as np
import pytest
import torch
import keras
from keras import ops

from k3_node.models import (
    UniMolModel,
    UniMolConfGenModel,
    UniMolDockingModel,
    UniMolGaussianLayer,
    UniMolNonLinearHead,
    UniMolDistanceHead,
    download_unimol_checkpoint,
    load_unimol_weights,
)


def test_unimol_gaussian_layer():
    num_kernel = 64
    edge_types = 128
    layer = UniMolGaussianLayer(num_kernel=num_kernel, edge_types=edge_types)

    bsz, seq_len = 2, 5
    dist = ops.convert_to_tensor(np.random.uniform(0.5, 5.0, (bsz, seq_len, seq_len)).astype("float32"))
    edge_type = ops.convert_to_tensor(np.random.randint(0, edge_types, (bsz, seq_len, seq_len)), dtype="int32")

    out = layer(dist, edge_type)
    assert ops.shape(out) == (bsz, seq_len, seq_len, num_kernel)


def test_unimol_nonlinear_head():
    head = UniMolNonLinearHead(input_dim=32, out_dim=16, activation_fn="gelu")
    x = ops.convert_to_tensor(np.random.randn(2, 5, 32).astype("float32"))
    out = head(x)
    assert ops.shape(out) == (2, 5, 16)


def test_unimol_distance_head():
    head = UniMolDistanceHead(heads=8, activation_fn="gelu")
    pair = ops.convert_to_tensor(np.random.randn(2, 6, 6, 8).astype("float32"))
    dist = head(pair)
    assert ops.shape(dist) == (2, 6, 6)
    # Check symmetry
    dist_np = ops.convert_to_numpy(dist)
    np.testing.assert_allclose(dist_np, np.transpose(dist_np, (0, 2, 1)), atol=1e-5)


def test_unimol_model_forward():
    bsz = 2
    seq_len = 8
    vocab_size = 64
    embed_dim = 64
    heads = 4
    layers_num = 2

    model = UniMolModel(
        output_dim=2,
        vocab_size=vocab_size,
        encoder_layers=layers_num,
        encoder_embed_dim=embed_dim,
        encoder_ffn_embed_dim=128,
        encoder_attention_heads=heads,
        num_kernel=32,
    )

    src_tokens = ops.convert_to_tensor(np.random.randint(1, vocab_size, (bsz, seq_len)), dtype="int32")
    src_coord = ops.convert_to_tensor(np.random.randn(bsz, seq_len, 3).astype("float32"))

    # Forward logits
    logits = model(src_tokens, src_coord=src_coord)
    assert ops.shape(logits) == (bsz, 2)

    # Return repr
    reprs = model(src_tokens, src_coord=src_coord, return_repr=True)
    assert "cls_repr" in reprs
    assert ops.shape(reprs["cls_repr"]) == (bsz, embed_dim)
    assert ops.shape(reprs["encoder_rep"]) == (bsz, seq_len, embed_dim)


def test_unimol_conf_gen_and_docking():
    bsz = 2
    seq_len = 6
    vocab_size = 64
    embed_dim = 64
    heads = 4

    conf_model = UniMolConfGenModel(
        vocab_size=vocab_size,
        encoder_layers=2,
        encoder_embed_dim=embed_dim,
        encoder_ffn_embed_dim=128,
        encoder_attention_heads=heads,
        num_kernel=32,
    )
    tokens = ops.convert_to_tensor(np.random.randint(1, vocab_size, (bsz, seq_len)), dtype="int32")
    coord = ops.convert_to_tensor(np.random.randn(bsz, seq_len, 3).astype("float32"))

    updated_coord, pred_dist = conf_model(tokens, src_coord=coord)
    assert ops.shape(updated_coord) == (bsz, seq_len, 3)
    assert ops.shape(pred_dist) == (bsz, seq_len, seq_len)

    docking_model = UniMolDockingModel(
        vocab_size=vocab_size,
        encoder_layers=2,
        encoder_embed_dim=embed_dim,
        encoder_ffn_embed_dim=128,
        encoder_attention_heads=heads,
        num_kernel=32,
    )
    pose, pred_dist2 = docking_model(tokens, src_coord=coord)
    assert ops.shape(pose) == (bsz, seq_len, 3)
    assert ops.shape(pred_dist2) == (bsz, seq_len, seq_len)


def test_unimol_weight_loading():
    vocab_size = 32
    embed_dim = 32
    heads = 4
    layers_num = 2

    model = UniMolModel(
        output_dim=2,
        vocab_size=vocab_size,
        encoder_layers=layers_num,
        encoder_embed_dim=embed_dim,
        encoder_ffn_embed_dim=64,
        encoder_attention_heads=heads,
        num_kernel=16,
    )
    model.build(None)

    # Create synthetic PyTorch state_dict
    state_dict = {
        "embed_tokens.weight": torch.randn(vocab_size, embed_dim),
        "gbf.means.weight": torch.randn(1, 16),
        "gbf.stds.weight": torch.randn(1, 16),
        "gbf.mul.weight": torch.randn(1024, 1),
        "gbf.bias.weight": torch.randn(1024, 1),
        "gbf_proj.linear1.weight": torch.randn(16, 16),
        "gbf_proj.linear1.bias": torch.randn(16),
        "gbf_proj.linear2.weight": torch.randn(heads, 16),
        "gbf_proj.linear2.bias": torch.randn(heads),
        "encoder.emb_layer_norm.weight": torch.ones(embed_dim),
        "encoder.emb_layer_norm.bias": torch.zeros(embed_dim),
        "classification_head.out_proj.weight": torch.randn(2, embed_dim),
        "classification_head.out_proj.bias": torch.randn(2),
    }

    for i in range(layers_num):
        prefix = f"encoder.layers.{i}"
        state_dict[f"{prefix}.self_attn.in_proj.weight"] = torch.randn(embed_dim * 3, embed_dim)
        state_dict[f"{prefix}.self_attn.in_proj.bias"] = torch.randn(embed_dim * 3)
        state_dict[f"{prefix}.self_attn.out_proj.weight"] = torch.randn(embed_dim, embed_dim)
        state_dict[f"{prefix}.self_attn.out_proj.bias"] = torch.randn(embed_dim)
        state_dict[f"{prefix}.self_attn_layer_norm.weight"] = torch.ones(embed_dim)
        state_dict[f"{prefix}.self_attn_layer_norm.bias"] = torch.zeros(embed_dim)
        state_dict[f"{prefix}.fc1.weight"] = torch.randn(64, embed_dim)
        state_dict[f"{prefix}.fc1.bias"] = torch.randn(64)
        state_dict[f"{prefix}.fc2.weight"] = torch.randn(embed_dim, 64)
        state_dict[f"{prefix}.fc2.bias"] = torch.randn(embed_dim)
        state_dict[f"{prefix}.final_layer_norm.weight"] = torch.ones(embed_dim)
        state_dict[f"{prefix}.final_layer_norm.bias"] = torch.zeros(embed_dim)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
        torch.save({"model": state_dict}, tmp_path)

    try:
        load_unimol_weights(model, checkpoint_path=tmp_path, download=False)
        # Check assigned weight matches
        embed_w = ops.convert_to_numpy(model.embed_tokens.embeddings)
        np.testing.assert_allclose(embed_w, state_dict["embed_tokens.weight"].numpy(), atol=1e-5)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

