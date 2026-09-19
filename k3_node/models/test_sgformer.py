from keras import ops, random
from k3_node.layers.attention import SGFormerAttention
from k3_node.models import SGFormer


def test_sgformer_attention():
    attn = SGFormerAttention(channels=32, heads=2, head_channels=16)
    x = random.normal((2, 4, 32))
    mask = ops.convert_to_tensor([[True, True, True, False], [True, True, False, False]])

    out = attn(x, mask=mask)
    assert out.shape == (2, 4, 16)


def test_sgformer():
    model = SGFormer(
        in_channels=16,
        hidden_channels=32,
        out_channels=7,
        trans_num_layers=1,
        trans_num_heads=1,
        gnn_num_layers=1,
        aggregate="add",
    )
    x = random.normal((6, 16))
    edge_index = ops.convert_to_tensor([[0, 1, 2, 3, 4], [1, 2, 0, 4, 5]], dtype="int64")

    out = model(x, edge_index)
    assert out.shape == (6, 7)

    # Test with batch
    batch = ops.convert_to_tensor([0, 0, 0, 1, 1, 1], dtype="int64")
    out_batch = model(x, edge_index, batch=batch)
    assert out_batch.shape == (6, 7)

    # Test aggregate='cat'
    model_cat = SGFormer(
        in_channels=16,
        hidden_channels=32,
        out_channels=7,
        trans_num_layers=1,
        trans_num_heads=1,
        gnn_num_layers=1,
        aggregate="cat",
    )
    out_cat = model_cat(x, edge_index)
    assert out_cat.shape == (6, 7)

