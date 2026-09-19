import keras
from keras import ops, random
from k3_node.models import SignedGCN


def test_signed_gcn():
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3, 0, 1, 4, 5],
        [1, 2, 3, 0, 4, 5, 2, 3],
    ], dtype="int64")

    pos_edge_index, neg_edge_index = SignedGCN.split_edges(edge_index, test_ratio=0.2)
    assert pos_edge_index.shape[0] == 2
    assert neg_edge_index.shape[0] == 2

    model = SignedGCN(in_channels=16, hidden_channels=32, num_layers=2)
    x = random.normal((6, 16))

    out = model(x, pos_edge_index, neg_edge_index)
    assert out.shape == (6, 32)

    # Test discriminate
    disc = model.discriminate(out, edge_index)
    assert disc.shape == (8, 3)

    # Test losses
    loss = model.loss(out, pos_edge_index, neg_edge_index)
    assert loss.shape == ()
