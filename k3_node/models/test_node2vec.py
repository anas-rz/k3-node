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



def test_node2vec_fit_and_biased_walks():
    import numpy as np
    from k3_node.models import Node2Vec

    rng = np.random.default_rng(0)
    edge_index = rng.integers(0, 20, size=(2, 80))
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    import keras
    model = Node2Vec(edge_index, embedding_dim=8, walk_length=5, context_size=3, walks_per_node=2, num_nodes=20)
    model.compile(keras.optimizers.Adam(0.01))
    history = model.fit(epochs=2, batch_size=8, verbose=0)
    assert len(history["loss"]) == 2 and np.isfinite(history["loss"]).all()
    # With a tiny p the walk (almost) always steps back to where it came from
    model.p, model.q = 1e-6, 1.0
    walk = model._random_walk(int(edge_index[0, 0]))
    assert all(walk[i] == walk[i - 2] for i in range(2, len(walk)))
