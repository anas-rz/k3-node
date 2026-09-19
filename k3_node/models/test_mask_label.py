import numpy as np
from keras import ops

from k3_node.models import MaskLabel


def test_mask_label_add():
    model = MaskLabel(2, 10)
    assert str(model) == 'MaskLabel()'

    x = ops.ones((4, 10))
    y = ops.convert_to_tensor([1, 0, 1, 0])
    mask = ops.convert_to_tensor([False, False, True, True])

    out = model(x, y, mask)
    assert ops.shape(out) == (4, 10)

    # Non-masked nodes should have same features as input (plus zero embedding)
    out_np = ops.convert_to_numpy(out)
    x_np = ops.convert_to_numpy(x)
    mask_np = ops.convert_to_numpy(mask)
    # Non-masked embeddings are zero so output matches input exactly
    assert np.allclose(out_np[~mask_np], x_np[~mask_np])


def test_mask_label_concat():
    model = MaskLabel(2, 10, method='concat')

    x = ops.ones((4, 10))
    y = ops.convert_to_tensor([1, 0, 1, 0])
    mask = ops.convert_to_tensor([False, False, True, True])

    out = model(x, y, mask)
    assert ops.shape(out) == (4, 20)

    # First 10 features should be identical to x
    out_np = ops.convert_to_numpy(out)
    x_np = ops.convert_to_numpy(x)
    assert np.allclose(out_np[:, :10], x_np)


def test_mask_label_concat_non_masked_zero():
    """For concat method, non-masked nodes' label embedding part should be 0."""
    model = MaskLabel(3, 8, method='concat')

    x = ops.ones((6, 16))
    y = ops.convert_to_tensor([0, 1, 2, 0, 1, 2])
    mask = ops.convert_to_tensor([True, True, False, False, False, False])

    out = model(x, y, mask)
    assert ops.shape(out) == (6, 24)

    out_np = ops.convert_to_numpy(out)
    mask_np = ops.convert_to_numpy(mask)
    # Non-masked embedding portion should be all zeros
    assert np.allclose(out_np[~mask_np, 16:], 0.0)


def test_mask_label_invalid_method():
    import pytest
    with pytest.raises(ValueError, match="'method' must be either 'add' or 'concat'"):
        MaskLabel(2, 10, method='multiply')


def test_ratio_mask():
    mask = ops.convert_to_tensor([True, True, True, True, False, False, False, False])
    out = MaskLabel.ratio_mask(mask, 0.5)
    out_np = ops.convert_to_numpy(out)
    mask_np = ops.convert_to_numpy(mask)
    # True entries in original False positions should stay False
    assert out_np[~mask_np].sum() == 0
    # Number of remaining True entries should be <= original count
    assert out_np[:4].sum() <= 4


def test_ratio_mask_zero():
    """ratio=0 should mask all True entries."""
    mask = ops.convert_to_tensor([True, True, True, False])
    out = MaskLabel.ratio_mask(mask, 0.0)
    out_np = ops.convert_to_numpy(out)
    assert out_np.sum() == 0


def test_ratio_mask_one():
    """ratio=1.0 should keep all True entries."""
    mask = ops.convert_to_tensor([True, True, False, False])
    out = MaskLabel.ratio_mask(mask, 1.0)
    out_np = ops.convert_to_numpy(out)
    assert out_np.sum() == 2

