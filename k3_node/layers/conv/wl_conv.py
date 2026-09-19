import numpy as np
import keras
from keras import ops
from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import scatter


class WLConv(keras.layers.Layer):
    r"""The Weisfeiler Lehman (WL) operator from the `"A Reduction of a Graph
    to a Canonical Form and an Algebra Arising During this Reduction"
    <https://www.iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf>`_ paper.

    Args:
        **kwargs: Additional layer arguments.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hashmap = {}

    def reset_parameters(self):
        self.hashmap = {}

    def call(self, x, edge_index):
        if len(ops.shape(x)) > 1:
            x = ops.argmax(x, axis=-1)

        x_np = ops.convert_to_numpy(x)
        edge_index_np = ops.convert_to_numpy(edge_index)

        num_nodes = len(x_np)
        row, col = edge_index_np[0], edge_index_np[1]

        # Group neighbors by target node col
        neighbors_dict = {i: [] for i in range(num_nodes)}
        for src, dst in zip(row, col):
            neighbors_dict[int(dst)].append(int(x_np[src]))

        out = []
        for i in range(num_nodes):
            node_color = int(x_np[i])
            sorted_neighs = sorted(neighbors_dict[i])
            key = hash((node_color, tuple(sorted_neighs)))
            if key not in self.hashmap:
                self.hashmap[key] = len(self.hashmap)
            out.append(self.hashmap[key])

        return ops.convert_to_tensor(np.array(out, dtype=np.int64))

    def histogram(self, x, batch=None, norm: bool = False):
        x_np = ops.convert_to_numpy(x)
        num_nodes = len(x_np)
        if batch is None:
            batch_np = np.zeros(num_nodes, dtype=np.int64)
        else:
            batch_np = ops.convert_to_numpy(batch).astype(np.int64)

        num_colors = len(self.hashmap)
        batch_size = int(np.max(batch_np)) + 1 if len(batch_np) > 0 else 1

        hist = np.zeros((batch_size, num_colors), dtype=np.float32)
        for b, c in zip(batch_np, x_np):
            hist[b, c] += 1.0

        if norm:
            norms = np.linalg.norm(hist, axis=-1, keepdims=True)
            hist = hist / np.maximum(norms, 1e-12)

        return ops.convert_to_tensor(hist)


class WLConvContinuous(MessagePassing):
    r"""The Weisfeiler Lehman operator from the `"Wasserstein
    Weisfeiler-Lehman Graph Kernels" <https://arxiv.org/abs/1906.01277>`_ paper.

    Args:
        **kwargs: Additional arguments of :class:`MessagePassing`.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

    def call(self, x, edge_index, edge_weight=None):
        if isinstance(x, (tuple, list)):
            x_src, x_dst = x[0], x[1]
        else:
            x_src = x_dst = x

        out = self.propagate(edge_index, x=(x_src, x_dst), edge_weight=edge_weight)

        dst_index = edge_index[1]
        if edge_weight is None:
            edge_weight = ops.ones((ops.shape(dst_index)[0],), dtype=out.dtype)

        num_nodes = ops.shape(out)[0]
        deg = scatter(edge_weight, dst_index, dim=0, dim_size=num_nodes, reduce="sum")
        deg_inv = ops.where(ops.equal(deg, 0), 0.0, 1.0 / deg)
        out = ops.expand_dims(deg_inv, axis=-1) * out

        if x_dst is not None:
            out = 0.5 * (x_dst + out)

        return out

    def message(self, x_j, edge_weight=None):
        if edge_weight is not None:
            return ops.expand_dims(edge_weight, axis=-1) * x_j
        return x_j

