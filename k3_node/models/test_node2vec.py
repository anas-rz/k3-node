from keras import ops
from k3_node.models import Node2Vec


def test_node2vec():
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3, 0, 2],
        [1, 2, 3, 0, 2, 0],
    ], dtype="int64")

    model = Node2Vec(
        edge_index,
        embedding_dim=16,
        walk_length=4,
        context_size=3,
        walks_per_node=2,
        num_negative_samples=1,
    )

    # Test forward
    out = model()
    assert out.shape == (4, 16)

    batch = ops.convert_to_tensor([0, 1], dtype="int64")
    out_batch = model(batch)
    assert out_batch.shape == (2, 16)

    # Test pos and neg sampling
    pos_rw = model.pos_sample(batch)
    assert pos_rw.shape[1] == 3

    neg_rw = model.neg_sample(batch)
    assert neg_rw.shape[1] == 3

    # Test loss
    loss = model.loss(pos_rw, neg_rw)
    assert loss.shape == ()
    assert float(ops.convert_to_numpy(loss)) > 0

