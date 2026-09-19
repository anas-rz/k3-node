import keras.ops as ops
from k3_node.models.gpse import GPSE, GPSENodeEncoder


def test_gpse_encoder():
    x = ops.ones((4, 16))
    pos_enc = ops.ones((4, 32))

    encoder = GPSENodeEncoder(
        dim_emb=64,
        dim_pe_in=32,
        dim_pe_out=16,
        dim_in=16,
        expand_x=True,
    )
    out = encoder(x, pos_enc)
    assert out.shape == (4, 64)


def test_gpse_model():
    x = ops.ones((4, 16))
    edge_index = ops.convert_to_tensor([[0, 1, 2, 3], [1, 2, 3, 0]])

    model = GPSE(
        dim_in=16,
        dim_inner=32,
        layers_pre_mp=1,
        layers_mp=2,
        layers_post_mp=1,
        use_repr=True,
    )
    out = model(x, edge_index)
    assert out.shape == (4, 32)

