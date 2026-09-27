"""Keras data adapters for training on whole graphs with ``model.fit``."""
from typing import Optional, Sequence

import numpy as np
import keras
from keras import ops


class FullGraphDataset(keras.utils.PyDataset):
    r"""Feeds one whole graph to :meth:`keras.Model.fit` / ``evaluate`` / ``predict`` as a single batch.

    Passing node arrays together with ``edge_index`` directly to ``fit`` fails, because Keras
    slices every input along its first axis and ``edge_index`` has shape ``[2, num_edges]``.
    This dataset yields the full graph as one batch per epoch instead.

    Args:
        data: A :class:`~k3_node.data.Data` object.
        inputs (Sequence[str]): Attributes passed to the model, in order.
            (default: ``("x", "edge_index")``)
        target (str, optional): Attribute used as the target. (default: ``"y"``)
        mask (str, optional): Node mask attribute (e.g. ``"train_mask"``) used as sample weights.
            (default: :obj:`None`)
        as_dict (bool): Yield inputs as a dict keyed by attribute name instead of a tuple.
            (default: :obj:`False`)
        normalize_mask (bool): Scale the mask so the loss is the mean over the masked nodes (as in
            PyG) rather than Keras' sum divided by the number of all nodes. (default: :obj:`True`)

    Example:
        >>> model.fit(FullGraphDataset(data, mask="train_mask"), epochs=200)
        >>> model.evaluate(FullGraphDataset(data, mask="test_mask"))
    """

    def __init__(
        self,
        data,
        inputs: Sequence[str] = ("x", "edge_index"),
        target: Optional[str] = "y",
        mask: Optional[str] = None,
        as_dict: bool = False,
        normalize_mask: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        arrays = {}
        for name in inputs:
            value = getattr(data, name, None)
            if value is None:
                raise ValueError(f"Data object has no attribute '{name}'")
            arrays[name] = ops.convert_to_numpy(value)
        self._inputs = arrays if as_dict else tuple(arrays[name] for name in inputs)
        if not as_dict and len(self._inputs) == 1:
            self._inputs = self._inputs[0]

        self._target = None
        if target is not None and getattr(data, target, None) is not None:
            self._target = ops.convert_to_numpy(getattr(data, target))

        self._sample_weight = None
        if mask is not None:
            mask_value = getattr(data, mask, None)
            if mask_value is None:
                raise ValueError(f"Data object has no mask attribute '{mask}'")
            weight = ops.convert_to_numpy(mask_value).astype("float32")
            if normalize_mask and weight.sum() > 0:
                weight = weight * (weight.shape[0] / weight.sum())
            self._sample_weight = weight

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        if self._target is None:
            return (self._inputs,)
        if self._sample_weight is None:
            return self._inputs, self._target
        return self._inputs, self._target, self._sample_weight
