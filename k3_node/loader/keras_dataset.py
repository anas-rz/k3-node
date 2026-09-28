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


HeteroGraphBatch = collections.namedtuple("HeteroGraphBatch", ["x_dict", "edge_index_dict"])


def _hetero_keras_batch(data, target, mask, normalize_mask, node_type):
    x_dict = {k: _as_array(v) for k, v in data.collect("x").items()}
    edge_index_dict = {k: _as_array(v) for k, v in data.collect("edge_index").items()}
    inputs = HeteroGraphBatch(x_dict, edge_index_dict)
    if node_type is None:
        return (inputs,)
    store = data[node_type]
    y = _as_array(getattr(store, target or "y"))
    if mask is None:
        return inputs, y
    weight = np.asarray(ops.convert_to_numpy(getattr(store, mask))).astype(np.float32)
    if normalize_mask and weight.sum() > 0:
        weight = weight * (weight.shape[0] / weight.sum())
    return inputs, y, weight


def to_keras_batch(
    data: Any,
    target: Optional[str] = None,
    mask: Optional[str] = None,
    normalize_mask: bool = True,
    index: Optional[str] = None,
    node_type: Optional[str] = None,
):
    r"""Converts a :class:`~k3_node.data.Data` / :class:`~k3_node.data.Batch` into Keras' format.

    For a :class:`~k3_node.data.HeteroData` graph, ``inputs`` is a :class:`HeteroGraphBatch`
    (``x_dict`` and ``edge_index_dict``) and ``target`` / ``mask`` are read from ``node_type``.

    Returns ``(inputs,)``, ``(inputs, y)`` or ``(inputs, y, sample_weight)``, where ``inputs`` is a
    :class:`GraphBatch` holding every array attribute except the target and ``*_mask`` attributes.

    Args:
        data: The graph (or mini-batch of graphs).
        target (str, optional): Attribute to predict. Defaults to ``"edge_label"`` if present
            (link prediction), else ``"y"``.
        mask (str, optional): Node mask attribute (e.g. ``"train_mask"``) used as sample weights.
        normalize_mask (bool): Scale sample weights so the loss is the mean over the weighted
            nodes (as in PyG), not Keras' sum divided by the number of all nodes.
        index (str, optional): Node index attribute (e.g. ``"train_idx"``) for datasets that store
            splits as node indices, with ``target`` holding the labels of exactly those nodes
            (e.g. ``"train_y"``). The labels are placed at their nodes and only these nodes count.
    """
    if hasattr(data, "node_types") and hasattr(data, "edge_types"):
        return _hetero_keras_batch(data, target, mask, normalize_mask, node_type)
    attrs = data.to_dict() if hasattr(data, "to_dict") else dict(vars(data))
    if target is None:
        target = "edge_label" if attrs.get("edge_label") is not None else "y"

    fields = {}
    for key in sorted(attrs):
        if key == target or key.endswith(("_mask", "_idx")):
            continue
        array = _as_array(attrs[key])
        if array is not None:
            fields[key] = array
    for key in ("batch", "ptr"):  # Batch exposes these as properties
        value = getattr(data, key, None)
        if key not in fields and value is not None and _as_array(value) is not None:
            fields[key] = _as_array(value)
    if "batch" in fields and fields["batch"].shape[0] == 0:
        # Graphs without nodes (e.g. batches of events): an empty `batch` would come first and
        # make Keras weight the reported loss by 0.
        fields.pop("batch")
        fields.pop("ptr", None)
    inputs = _graph_batch_type(tuple(fields))(**fields)

    y = _as_array(attrs.get(target))
    if y is None:
        return (inputs,)

    weight = None
    if index is not None:
        if attrs.get(index) is None:
            raise ValueError(f"The graph has no index attribute '{index}'")
        idx = np.asarray(ops.convert_to_numpy(attrs[index])).astype(np.int64)
        num_nodes = attrs.get("num_nodes") or data.num_nodes
        y_full = np.zeros((num_nodes,) + y.shape[1:], dtype=y.dtype)
        y_full[idx] = y
        y, weight = y_full, np.zeros(num_nodes, dtype=np.float32)
        weight[idx] = 1.0
    elif mask is not None:
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

    def with_mask(self, mask: str):
        r"""Only the nodes in the node mask attribute ``mask`` (e.g. ``"train_mask"``) of each batch
        count in the loss and in ``weighted_metrics``. Returns the loader, for chaining."""
        self.keras_mask = mask
        return self

    def with_target(self, target: str):
        r"""Sets the attribute Keras predicts (default: ``"edge_label"`` if present, else ``"y"``).
        Returns the loader, for chaining."""
        self.keras_target = target
        return self

    @property
    def num_batches(self):
        return len(self)

    def on_epoch_end(self):
        self._keras_batches = None  # reshuffle next epoch
        self._keras_iter = None


def _patch_tf_signature():
    """On TensorFlow, Keras fixes every tensor size that is the same in the first few batches it
    reads (except the first axis). A sampling loader with fewer batches than that (e.g. a single
    validation batch) returns different subgraphs on every call, so Keras would fix sizes that
    change later. For K3-Node's loaders, the first batches are therefore read at least twice:
    sizes that vary between calls become variable, and deterministic batches keep static shapes.
    Falls back to Keras' behavior if its internals change."""
    try:
        from keras.src.trainers.data_adapters import data_adapter_utils
        from keras.src.trainers.data_adapters.py_dataset_adapter import PyDatasetAdapter
    except ImportError:
        return
    if getattr(PyDatasetAdapter, "_k3_node_patched", False):
        return
    original = PyDatasetAdapter.get_tf_dataset

    def get_tf_dataset(self):
        dataset = getattr(self, "py_dataset", None)
        if getattr(self, "_output_signature", "missing") is None and isinstance(dataset, KerasLoaderMixin):
            try:
                num_samples = max(data_adapter_utils.NUM_BATCHES_FOR_TENSOR_SPEC, 2)
                num_batches = dataset.num_batches or num_samples
                # e.g. 3 samples of a 1-batch loader read batch 0 three times
                batches = [self._standardize_batch(dataset[i % num_batches]) for i in range(num_samples)]
                self._output_signature = data_adapter_utils.get_tensor_spec(batches)
            except Exception:
                self._output_signature = None
        return original(self)

    PyDatasetAdapter.get_tf_dataset = get_tf_dataset
    PyDatasetAdapter._k3_node_patched = True


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
        index (str, optional): For splits stored as node indices: the index attribute (e.g.
            ``"train_idx"``), with ``target`` giving the labels of those nodes (e.g. ``"train_y"``).
        neg_sampling_ratio (float, optional): For link prediction: adds this many random
            non-edges per labeled edge (label 0), sampled anew every epoch.
        node_type (str, optional): For a heterogeneous graph: the node type whose ``target`` is
            predicted and whose ``mask`` selects the nodes. The model receives ``data.x_dict``
            and ``data.edge_index_dict``.

    Example:
        ```python
        model.compile(optimizer="adam", loss=..., weighted_metrics=["accuracy"])
        model.fit(FullGraphDataset(data, mask="train_mask"), epochs=200)
        model.evaluate(FullGraphDataset(data, mask="test_mask"))
        ```
    """

    def __init__(self, data, mask: Optional[str] = None, target: Optional[str] = None,
                 index: Optional[str] = None, neg_sampling_ratio: Optional[float] = None,
                 node_type: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._data, self._neg_sampling_ratio = data, neg_sampling_ratio
        self._kwargs = dict(target=target, mask=mask, index=index)
        if node_type is not None:
            self._kwargs["node_type"] = node_type
        self._batch = None if neg_sampling_ratio else to_keras_batch(data, **self._kwargs)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        if self._neg_sampling_ratio:  # fresh negative edges every epoch
            return to_keras_batch(add_negative_edges(self._data, self._neg_sampling_ratio), **self._kwargs)
        return self._batch


def add_negative_edges(data, ratio: float = 1.0):
    r"""Returns a copy of a link prediction graph with random non-edges added to its labeled edges.

    ``ratio`` negatives are sampled per labeled edge in ``edge_label_index`` (or per edge of
    ``edge_index`` if there are no labeled edges), avoiding the edges of ``edge_index``. They are
    appended to ``edge_label_index`` with label 0 in ``edge_label``.
    """
    import copy

    from k3_node.models.utils import negative_sampling

    pos = getattr(data, "edge_label_index", None)
    pos = np.asarray(ops.convert_to_numpy(data.edge_index if pos is None else pos))
    label = getattr(data, "edge_label", None)
    label = np.ones(pos.shape[1], np.float32) if label is None else np.asarray(ops.convert_to_numpy(label))
    neg = np.asarray(ops.convert_to_numpy(negative_sampling(
        data.edge_index, data.num_nodes, num_neg_samples=int(round(ratio * pos.shape[1])))))
    out = copy.copy(data)
    out.edge_label_index = np.concatenate([pos, neg.astype(pos.dtype)], axis=1)
    out.edge_label = np.concatenate([label, np.zeros(neg.shape[1], label.dtype)])
    return out


def loader_bases(base: type) -> tuple:
    """Base classes for a graph loader: its data-loader base plus :class:`KerasLoaderMixin`."""
    return (base, KerasLoaderMixin) if base is not object else (KerasLoaderMixin,)


if keras.config.backend() == "tensorflow":
    _patch_tf_signature()
