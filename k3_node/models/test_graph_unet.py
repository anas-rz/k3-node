from keras import ops, random
from k3_node.models import GraphUNet


def test_graph_unet():
    model = GraphUNet(
        in_channels=16,
        hidden_channels=32,
        out_channels=8,
        depth=2,
        pool_ratios=0.5,
    )
    x = random.normal((6, 16))
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3, 4, 5, 0, 2],
        [1, 2, 0, 4, 5, 3, 3, 4],
    ], dtype="int64")

    out = model(x, edge_index)
    assert out.shape == (6, 8)

    # Test with batch
    batch = ops.convert_to_tensor([0, 0, 0, 1, 1, 1], dtype="int64")
    out_batch = model(x, edge_index, batch=batch)
    assert out_batch.shape == (6, 8)

