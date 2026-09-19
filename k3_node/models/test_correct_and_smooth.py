from keras import ops

from k3_node.models import CorrectAndSmooth


def test_correct_and_smooth():
    y_soft = ops.repeat(ops.convert_to_tensor([[0.1, 0.5, 0.4]]), 6, axis=0)
    y_true = ops.convert_to_tensor([1, 0, 0, 2, 1, 1])
    edge_index = ops.convert_to_tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    mask = ops.convert_to_tensor([True, False, True, False, True, False])

    model = CorrectAndSmooth(
        num_correction_layers=2,
        correction_alpha=0.5,
        num_smoothing_layers=2,
        smoothing_alpha=0.5,
    )
    assert str(model) == ('CorrectAndSmooth(\n'
                          '  correct: num_layers=2, alpha=0.5\n'
                          '  smooth:  num_layers=2, alpha=0.5\n'
                          '  autoscale=True, scale=1.0\n'
                          ')')

    out = model.correct(y_soft, y_true[mask], mask, edge_index)
    assert ops.shape(out) == (6, 3)

    out = model.smooth(y_soft, y_true[mask], mask, edge_index)
    assert ops.shape(out) == (6, 3)

    # Without autoscale:
    model_no_auto = CorrectAndSmooth(
        num_correction_layers=2,
        correction_alpha=0.5,
        num_smoothing_layers=2,
        smoothing_alpha=0.5,
        autoscale=False,
    )
    out = model_no_auto.correct(y_soft, y_true[mask], mask, edge_index)
    assert ops.shape(out) == (6, 3)

