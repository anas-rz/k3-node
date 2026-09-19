from keras import ops

from k3_node.layers.kge import TransE, DistMult, ComplEx, RotatE


def _run_model(cls):
    model = cls(num_nodes=10, num_relations=5, hidden_channels=32)
    assert str(model) == f"{cls.__name__}(10, num_relations=5, hidden_channels=32)"

    head_index = ops.convert_to_tensor([0, 2, 4, 6, 8])
    rel_type = ops.convert_to_tensor([0, 1, 2, 3, 4])
    tail_index = ops.convert_to_tensor([1, 3, 5, 7, 9])

    loader = model.loader(head_index, rel_type, tail_index, batch_size=5)
    for h, r, t in loader:
        out = model(h, r, t)
        assert ops.shape(out) == (5,)

        loss = model.loss(h, r, t)
        assert float(ops.convert_to_numpy(loss)) >= 0.0

        mean_rank, mrr, hits = model.test(h, r, t, batch_size=5, log=False)
        assert 0 <= mean_rank <= 10
        assert 0 < mrr <= 1
        assert hits == 1.0


def test_transe():
    _run_model(TransE)


def test_distmult():
    _run_model(DistMult)


def test_complex():
    _run_model(ComplEx)


def test_rotate():
    _run_model(RotatE)


def test_complex_scoring():
    model = ComplEx(num_nodes=5, num_relations=2, hidden_channels=1)

    model.node_emb.embeddings.assign(
        ops.convert_to_tensor([[2.0], [3.0], [5.0], [1.0], [2.0]], dtype="float32")
    )
    model.node_emb_im.embeddings.assign(
        ops.convert_to_tensor([[4.0], [1.0], [3.0], [1.0], [2.0]], dtype="float32")
    )
    model.rel_emb.embeddings.assign(ops.convert_to_tensor([[2.0], [3.0]], dtype="float32"))
    model.rel_emb_im.embeddings.assign(ops.convert_to_tensor([[3.0], [1.0]], dtype="float32"))

    score = model(
        ops.convert_to_tensor([1, 3]),
        ops.convert_to_tensor([1, 0]),
        ops.convert_to_tensor([2, 4]),
    )
    assert ops.convert_to_numpy(score).tolist() == [58.0, 8.0]
