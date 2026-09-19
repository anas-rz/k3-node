from keras import ops

from k3_node.models import RECT_L


def test_rect():
    x = ops.ones((6, 8))
    y = ops.convert_to_tensor([1, 0, 0, 2, 1, 1])
    edge_index = ops.convert_to_tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    mask = ops.convert_to_tensor([True, False, True, False, True, False])

    model = RECT_L(8, 16)
    assert str(model) == 'RECT_L(8, 16)'

    out = model(x, edge_index)
    assert ops.shape(out) == (6, 8)

    embed_out = model.embed(x, edge_index)
    assert ops.shape(embed_out) == (6, 16)

    labels_out = model.get_semantic_labels(x, y, mask)
    assert ops.shape(labels_out) == (3, 8)

