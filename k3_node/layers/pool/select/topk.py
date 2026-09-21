from typing import Callable, Optional, Union
from keras import initializers, layers, ops

from .base import Select, SelectOutput


from k3_node.layers.conv.utils import is_tracing


def topk(
    x,
    ratio: Optional[Union[float, int]],
    batch,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
):
    r"""Selects top-k items according to score and batch assignment."""
    if is_tracing(x) or is_tracing(batch):
        return ops.arange(ops.shape(x)[0], dtype="int32")

    batch = ops.cast(batch, dtype="int32")
    num_nodes = ops.shape(x)[0]

    num_graphs = ops.max(batch) + 1 if ops.shape(batch)[0] > 0 else 0
    try:
        num_graphs = int(num_graphs)
    except (TypeError, ValueError):
        pass

    if min_score is not None:
        scores_max = ops.segment_max(x, batch, num_segments=num_graphs)
        scores_max_expanded = ops.take(scores_max, batch, axis=0) - tol
        scores_min = ops.minimum(scores_max_expanded, min_score)
        mask = x > scores_min
        perm = ops.where(mask)
        if isinstance(perm, (tuple, list)):
            perm = perm[0]
        perm = ops.reshape(perm, (-1,))
        return ops.cast(perm, "int32")

    if ratio is not None:
        ones = ops.ones((num_nodes,), dtype="int32")
        num_nodes_per_graph = ops.segment_sum(ones, batch, num_segments=num_graphs)

        if ratio >= 1:
            k = ops.full(ops.shape(num_nodes_per_graph), int(ratio), dtype="int32")
        else:
            k = ops.cast(
                ops.ceil(ratio * ops.cast(num_nodes_per_graph, x.dtype)),
                dtype="int32",
            )

        # Composite key: sorts by batch ascending, then by score descending
        score_span = ops.max(x) - ops.min(x) + 1.0
        key = ops.cast(batch, x.dtype) * (score_span * 2.0) - x
        perm = ops.argsort(key)

        batch_sorted = ops.take(batch, perm, axis=0)
        # ptr for cumsum
        ptr = ops.concatenate(
            [ops.zeros((1,), dtype="int32"), ops.cumsum(num_nodes_per_graph)[:-1]],
            axis=0,
        )
        rank_in_graph = ops.arange(num_nodes, dtype="int32") - ops.take(ptr, batch_sorted, axis=0)
        mask = rank_in_graph < ops.take(k, batch_sorted, axis=0)
        valid_idx = ops.where(mask)
        if isinstance(valid_idx, (tuple, list)):
            valid_idx = valid_idx[0]
        valid_idx = ops.reshape(valid_idx, (-1,))
        return ops.cast(ops.take(perm, valid_idx, axis=0), "int32")

    raise ValueError("At least one of 'ratio' and 'min_score' must be specified.")


class SelectTopK(Select):
    r"""Selects the top-:math:`k` nodes with highest projection scores."""
    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
        min_score: Optional[float] = None,
        act: Union[str, Callable] = "tanh",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if ratio is None and min_score is None:
            raise ValueError(
                f"At least one of 'ratio' and 'min_score' must be specified in '{self.__class__.__name__}'"
            )

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.act_fn = act if callable(act) else layers.Activation(act)

        self.weight = self.add_weight(
            shape=(1, in_channels),
            initializer=initializers.RandomUniform(
                minval=-1.0 / (in_channels**0.5), maxval=1.0 / (in_channels**0.5)
            ),
            trainable=True,
            name="weight",
        )

    def reset_parameters(self):
        limit = 1.0 / (self.in_channels**0.5)
        init = initializers.RandomUniform(minval=-limit, maxval=limit)
        self.weight.assign(init(self.weight.shape, dtype=self.weight.dtype))

    def build(self, input_shape=None):
        if hasattr(self.act_fn, "built") and not self.act_fn.built:
            self.act_fn.build(input_shape)
        self.built = True

    def call(self, x, batch=None) -> SelectOutput:
        num_nodes = ops.shape(x)[0]
        if batch is None:
            batch = ops.zeros((num_nodes,), dtype="int32")
        else:
            batch = ops.cast(batch, dtype="int32")

        if len(ops.shape(x)) == 1:
            x = ops.expand_dims(x, axis=-1)

        score = ops.sum(x * self.weight, axis=-1)

        if is_tracing(x) or is_tracing(batch):
            node_index = ops.arange(num_nodes, dtype="int32")
            return SelectOutput(
                node_index=node_index,
                num_nodes=num_nodes,
                cluster_index=node_index,
                num_clusters=num_nodes,
                weight=score,
            )

        if self.min_score is None:
            norm_w = ops.sqrt(ops.sum(ops.power(self.weight, 2), axis=-1))
            score = self.act_fn(score / norm_w)
        else:
            # Graph-wise softmax
            num_graphs = ops.max(batch) + 1 if ops.shape(batch)[0] > 0 else 0
            try:
                num_graphs = int(num_graphs)
            except (TypeError, ValueError):
                pass
            score_max = ops.segment_max(score, batch, num_segments=num_graphs)
            score_max_exp = ops.take(score_max, batch, axis=0)
            exp_score = ops.exp(score - score_max_exp)
            exp_sum = ops.segment_sum(exp_score, batch, num_segments=num_graphs)
            exp_sum_exp = ops.take(exp_sum, batch, axis=0)
            score = exp_score / (exp_sum_exp + 1e-12)

        node_index = topk(score, self.ratio, batch, self.min_score)
        num_selected = ops.shape(node_index)[0]

        return SelectOutput(
            node_index=node_index,
            num_nodes=num_nodes,
            cluster_index=ops.arange(num_selected, dtype="int32"),
            num_clusters=num_selected,
            weight=ops.take(score, node_index, axis=0),
        )

    def __repr__(self) -> str:
        if self.min_score is None:
            arg = f"ratio={self.ratio}"
        else:
            arg = f"min_score={self.min_score}"
        return f"{self.__class__.__name__}({self.in_channels}, {arg})"
