from keras import ops
from k3_node.models import MetaPath2Vec


def test_metapath2vec():
    edge_index_dict = {
        ("author", "writes", "paper"): ops.convert_to_tensor([[0, 1, 1], [0, 0, 1]], dtype="int64"),
        ("paper", "written_by", "author"): ops.convert_to_tensor([[0, 0, 1], [0, 1, 1]], dtype="int64"),
    }
    metapath = [
        ("author", "writes", "paper"),
        ("paper", "written_by", "author"),
    ]

    model = MetaPath2Vec(
        edge_index_dict=edge_index_dict,
        embedding_dim=16,
        metapath=metapath,
        walk_length=2,
        context_size=2,
        walks_per_node=2,
    )

    # Test forward
    out_author = model("author")
    assert out_author.shape == (2, 16)

    out_paper = model("paper")
    assert out_paper.shape == (2, 16)

    batch = ops.convert_to_tensor([0], dtype="int64")
    out_batch = model("author", batch)
    assert out_batch.shape == (1, 16)

    # Test sampling
    pos_rw = model._pos_sample(batch)
    neg_rw = model._neg_sample(batch)
    assert pos_rw.shape[1] == 2
    assert neg_rw.shape[1] == 2

    # Test loss
    loss = model.loss(pos_rw, neg_rw)
    assert loss.shape == ()
    assert float(ops.convert_to_numpy(loss)) > 0

