"""Makes graphs and graph loaders usable directly with ``model.fit`` / ``evaluate`` / ``predict``.

A model receives each batch as a :class:`GraphBatch`: a named tuple with one field per graph
attribute (``data.x``, ``data.edge_index``, ``data.batch``, ...), so its ``call`` reads like
PyG's ``forward``. The target (``y`` or ``edge_label``) and sample weights are passed to Keras
separately.
"""
import collections
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import keras
from keras import ops


class _GraphBatchMixin:
    __slots__ = ()

    @property
    def num_graphs(self) -> Optional[int]:
        """Number of graphs in the batch.

        It is a Python ``int`` whenever the batch shape is known, including inside compiled
        (``jax.jit`` / XLA) training steps, so it can be passed as ``size`` to the global pooling
        functions: ``global_add_pool(x, data.batch, data.num_graphs)``.
        """
        ptr = getattr(self, "ptr", None)
        if ptr is not None:
            n = ptr.shape[0]
            return None if n is None else int(n) - 1
        return 1 if getattr(self, "batch", None) is None else None

    @property
    def num_nodes(self) -> Optional[int]:
        x = getattr(self, "x", None)
        return None if x is None else x.shape[0]


_TYPES: Dict[Tuple[str, ...], type] = {}


def _graph_batch_type(fields: Tuple[str, ...]) -> type:
    if fields not in _TYPES:
        base = collections.namedtuple("GraphBatch", fields)
        _TYPES[fields] = type("GraphBatch", (base, _GraphBatchMixin), {"__slots__": ()})
    return _TYPES[fields]


def _as_array(value) -> Optional[np.ndarray]:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return None
    try:
        array = np.asarray(ops.convert_to_numpy(value))
    except Exception:
        return None
    if array.dtype.kind not in "biuf" or array.ndim == 0:  # skip strings (e.g. SMILES) and objects
        return None
    if array.dtype == np.float64:
        array = array.astype(np.float32)
    elif array.dtype == np.int64:
        array = array.astype(np.int32)
    return array


def to_keras_batch(
    data: Any,
    target: Optional[str] = None,
    mask: Optional[str] = None,
    normalize_mask: bool = True,
):
    r"""Converts a :class:`~k3_node.data.Data` / :class:`~k3_node.data.Batch` into Keras' format.

    Returns ``(inputs,)``, ``(inputs, y)`` or ``(inputs, y, sample_weight)``, where ``inputs`` is a
    :class:`GraphBatch` holding every array attribute except the target and ``*_mask`` attributes.

    Args:
        data: The graph (or mini-batch of graphs).
        target (str, optional): Attribute to predict. Defaults to ``"edge_label"`` if present
            (link prediction), else ``"y"``.
        mask (str, optional): Node mask attribute (e.g. ``"train_mask"``) used as sample weights.
        normalize_mask (bool): Scale sample weights so the loss is the mean over the weighted
            nodes (as in PyG), not Keras' sum divided by the number of all nodes.
    """
    attrs = data.to_dict() if hasattr(data, "to_dict") else dict(vars(data))
    if target is None:
        target = "edge_label" if attrs.get("edge_label") is not None else "y"

    fields = {}
    for key in sorted(attrs):
        if key == target or key.endswith("_mask"):
            continue
        array = _as_array(attrs[key])
        if array is not None:
            fields[key] = array
    for key in ("batch", "ptr"):  # Batch exposes these as properties
        value = getattr(data, key, None)
        if key not in fields and value is not None and _as_array(value) is not None:
            fields[key] = _as_array(value)
    inputs = _graph_batch_type(tuple(fields))(**fields)

    y = _as_array(attrs.get(target))
    if y is None:
        return (inputs,)

    weight = None
    if mask is not None:
        if attrs.get(mask) is None:
            raise ValueError(f"The graph has no mask attribute '{mask}'")
        weight = np.asarray(ops.convert_to_numpy(attrs[mask])).astype(np.float32)
    elif isinstance(attrs.get("batch_size"), int) and y.shape[0] == attrs.get("num_nodes", y.shape[0]):
        # NeighborLoader: only the first `batch_size` (seed) nodes are supervised
        weight = np.zeros(y.shape[0], dtype=np.float32)
        weight[: attrs["batch_size"]] = 1.0
    if weight is None:
        return inputs, y
    if normalize_mask and weight.sum() > 0:
        weight = weight * (weight.shape[0] / weight.sum())
    return inputs, y, weight


class KerasLoaderMixin(keras.utils.PyDataset):
    r"""Lets a graph loader be passed straight to ``model.fit`` / ``evaluate`` / ``predict``.

    Iterating the loader still yields :class:`~k3_node.data.Batch` objects; Keras instead reads
    batches through ``__getitem__`` in the format produced by :func:`to_keras_batch`.
    """

    # Class-level defaults stand in for PyDataset.__init__, which loader constructors don't call.
    _workers = 1
    _use_multiprocessing = False
    _max_queue_size = 10
    keras_target: Optional[str] = None
    keras_mask: Optional[str] = None

    def _keras_index_batches(self):
        if getattr(self, "_keras_batches", None) is None:
            batch_sampler = getattr(self, "batch_sampler", None)
            if batch_sampler is not None and getattr(self, "batch_size", None) is not None:
                self._keras_batches = list(iter(batch_sampler))
            else:
                self._keras_batches = False  # no random access: stream from the iterator
        return self._keras_batches

    def __getitem__(self, index):
        index_batches = self._keras_index_batches()
        if index_batches:
            batch = self.collate_fn([self.dataset[i] for i in index_batches[index]])
        else:
            if index == 0 or getattr(self, "_keras_iter", None) is None:
                self._keras_iter = iter(self)
            try:
                batch = next(self._keras_iter)
            except StopIteration:
                self._keras_iter = iter(self)
                batch = next(self._keras_iter)
        return to_keras_batch(batch, target=self.keras_target, mask=self.keras_mask)

    @property
    def num_batches(self):
        return len(self)

    def on_epoch_end(self):
        self._keras_batches = None  # reshuffle next epoch
        self._keras_iter = None


class FullGraphDataset(keras.utils.PyDataset):
    r"""Feeds one whole graph to ``model.fit`` / ``evaluate`` / ``predict`` as a single batch.

    Passing node arrays together with ``edge_index`` directly to ``fit`` fails, because Keras
    slices every input along its first axis and ``edge_index`` has shape ``[2, num_edges]``.
    This dataset yields the full graph as one batch per epoch instead. The model receives a
    :class:`GraphBatch` (``data.x``, ``data.edge_index``, ...).

    Args:
        data: A :class:`~k3_node.data.Data` object.
        mask (str, optional): Node mask attribute (e.g. ``"train_mask"``) whose nodes are used in
            the loss and in ``weighted_metrics``. (default: :obj:`None`, all nodes)
        target (str, optional): Attribute to predict. (default: ``"y"``)

    Example:
        ```python
        model.compile(optimizer="adam", loss=..., weighted_metrics=["accuracy"])
        model.fit(FullGraphDataset(data, mask="train_mask"), epochs=200)
        model.evaluate(FullGraphDataset(data, mask="test_mask"))
        ```
    """

    def __init__(self, data, mask: Optional[str] = None, target: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._batch = to_keras_batch(data, target=target, mask=mask)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self._batch


def loader_bases(base: type) -> tuple:
    """Base classes for a graph loader: its data-loader base plus :class:`KerasLoaderMixin`."""
    return (base, KerasLoaderMixin) if base is not object else (KerasLoaderMixin,)
