import pytest
from keras import ops
import numpy as np

from k3_node.models import LightGCN


@pytest.mark.parametrize('embedding_dim', [16, 32])
def test_lightgcn(embedding_dim):
    num_nodes = 50
    edge_index = ops.convert_to_tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3, 4, 5, 6, 7, 0],
    ])
    edge_label_index = ops.convert_to_tensor([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ])

    model = LightGCN(num_nodes, embedding_dim, num_layers=2)
    assert str(model) == f'LightGCN({num_nodes}, {embedding_dim}, num_layers=2)'

    pred = model(edge_index, edge_label_index)
    assert ops.shape(pred) == (4,)

    prob = model.predict_link(edge_index, edge_label_index, prob=True)
    assert ops.shape(prob) == (4,)

    loss = model.recommendation_loss(
        pos_edge_rank=pred[:2],
        neg_edge_rank=pred[2:],
        lambda_reg=1e-4,
    )
    assert float(loss) > 0.0

    recs = model.recommend(edge_index, k=2)
    assert ops.shape(recs) == (num_nodes, 2)

