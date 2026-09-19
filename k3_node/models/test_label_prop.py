from keras import ops

from k3_node.models import LabelPropagation


def test_label_prop():
    y = ops.convert_to_tensor([1, 0, 0, 2, 1, 1])
    edge_index = ops.convert_to_tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    mask = ops.convert_to_tensor([True, False, True, False, True, False])

    model = LabelPropagation(num_layers=2, alpha=0.5)
    assert str(model) == 'LabelPropagation(num_layers=2, alpha=0.5)'

    # Without mask:
    out = model(y, edge_index)
    assert ops.shape(out) == (6, 3)

    # With mask:
    out = model(y, edge_index, mask)
    assert ops.shape(out) == (6, 3)

    # With post_step:
    out = model(y, edge_index, mask, post_step=lambda y: ops.zeros_like(y))
    assert float(ops.sum(out)) == 0.0

