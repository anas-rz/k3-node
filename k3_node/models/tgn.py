from typing import Callable, Dict, List, Optional, Tuple
import copy
import numpy as np
import keras
from keras import ops

from k3_node.layers.aggr import MeanAggregation


class TimeEncoder(keras.layers.Layer):
    """Layer ``TimeEncoder``.

    Example:
        ```python
        import numpy as np
        from k3_node.models import TimeEncoder

        t = np.array([1.0, 2.0, 3.0], dtype="float32")  # time differences
        print(tuple(TimeEncoder(out_channels=16)(t).shape))  # (3, 16)
        ```
    """
    def __init__(self, out_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.lin = keras.layers.Dense(out_channels)

    def reset_parameters(self):
        if self.lin.built:
            self.lin.kernel.assign(keras.initializers.GlorotUniform()(self.lin.kernel.shape))
            if self.lin.bias is not None:
                self.lin.bias.assign(ops.zeros(self.lin.bias.shape))

    def call(self, t):
        t = ops.reshape(ops.cast(t, "float32"), (-1, 1))
        return ops.cos(self.lin(t))


class IdentityMessage(keras.layers.Layer):
    """Layer ``IdentityMessage``.

    Example:
        ```python
        import numpy as np
        from k3_node.models import IdentityMessage, LastAggregator, TGNMemory

        memory = TGNMemory(
            num_nodes=5, raw_msg_dim=8, memory_dim=16, time_dim=16,
            message_module=IdentityMessage(raw_msg_dim=8, memory_dim=16, time_dim=16),
            aggregator_module=LastAggregator(),
        )
        src, dst = np.array([0, 1]), np.array([1, 2])  # two interaction events
        t = np.array([1.0, 2.0], dtype="float32")
        raw_msg = np.random.rand(2, 8).astype("float32")
        memory.update_state(src, dst, t, raw_msg)  # update the memory of the involved nodes

        mem, last_update = memory(np.array([0, 1, 2]))
        print(tuple(mem.shape), tuple(last_update.shape))  # (3, 16) (3,)
        ```
    """
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def call(self, z_src, z_dst, raw_msg, t_enc):
        return ops.concatenate([z_src, z_dst, raw_msg, t_enc], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "raw_msg_dim": self.raw_msg_dim,
            "memory_dim": self.memory_dim,
            "time_dim": self.time_dim,
        })
        return config



class LastAggregator(keras.layers.Layer):
    """Layer ``LastAggregator``.

    Example:
        ```python
        import numpy as np
        from k3_node.models import LastAggregator

        msg = np.random.rand(4, 8).astype("float32")  # 4 messages
        index = np.array([0, 0, 1, 1])  # destination node of each message
        t = np.array([1.0, 2.0, 1.0, 3.0], dtype="float32")  # message timestamps
        out = LastAggregator()(msg, index, t, dim_size=2)  # keeps the latest message per node
        print(tuple(out.shape))  # (2, 8)
        ```
    """
    def call(self, msg, index, t, dim_size: int):
        # Which message is the latest per node is decided on the host (it does not depend on the
        # model); the chosen messages are gathered with ops, so gradients reach them.
        from k3_node.ops.host import to_numpy

        t_np = np.asarray(to_numpy(t)).reshape(-1)
        index_np = np.asarray(to_numpy(index)).astype(np.int64).reshape(-1)
        latest = np.full(dim_size, -1, dtype=np.int64)
        order = np.lexsort((np.arange(len(t_np)), t_np))  # by time, ties: later message wins
        latest[index_np[order]] = order
        has_msg = latest >= 0
        if msg.shape[0] == 0:  # no stored messages yet
            return ops.zeros((dim_size, msg.shape[-1]), dtype=msg.dtype)
        out = ops.take(msg, np.maximum(latest, 0), axis=0)
        return out * ops.cast(ops.convert_to_tensor(has_msg[:, None]), out.dtype)


class MeanAggregator(keras.layers.Layer):
    """Layer ``MeanAggregator``.

    Example:
        ```python
        import numpy as np
        from k3_node.models import MeanAggregator

        msg = np.random.rand(4, 8).astype("float32")  # 4 messages
        index = np.array([0, 0, 1, 1])  # destination node of each message
        t = np.array([1.0, 2.0, 1.0, 3.0], dtype="float32")  # message timestamps
        out = MeanAggregator()(msg, index, t, dim_size=2)  # averages the messages per node
        print(tuple(out.shape))  # (2, 8)
        ```
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean_aggr = MeanAggregation()

    def build(self, input_shape=None):
        self.built = True

    def call(self, msg, index, t, dim_size: int):
        return self.mean_aggr(msg, index=index, dim_size=dim_size, dim=0)


class LastNeighborLoader:
    def __init__(self, num_nodes: int, size: int):
        self.num_nodes = num_nodes
        self.size = size
        self.reset_state()

    def reset_state(self):
        self.cur_e_id = 0
        self.neighbors = np.empty((self.num_nodes, self.size), dtype=np.int64)
        self.e_id = np.full((self.num_nodes, self.size), -1, dtype=np.int64)

    def __call__(self, n_id):
        n_id_np = ops.convert_to_numpy(n_id).astype(np.int64)
        nbrs = self.neighbors[n_id_np]
        nodes = np.repeat(n_id_np[:, None], self.size, axis=1)
        e_ids = self.e_id[n_id_np]

        mask = e_ids >= 0
        nbrs = nbrs[mask]
        nodes = nodes[mask]
        e_ids = e_ids[mask]

        unique_nodes = np.unique(np.concatenate([n_id_np, nbrs]))
        assoc = {node: i for i, node in enumerate(unique_nodes)}

        mapped_nbrs = np.array([assoc[x] for x in nbrs], dtype=np.int64)
        mapped_nodes = np.array([assoc[x] for x in nodes], dtype=np.int64)
        edge_index = np.stack([mapped_nbrs, mapped_nodes], axis=0) if len(mapped_nbrs) > 0 else np.empty((2, 0), dtype=np.int64)

        return (
            ops.convert_to_tensor(unique_nodes, dtype="int64"),
            ops.convert_to_tensor(edge_index, dtype="int64"),
            ops.convert_to_tensor(e_ids, dtype="int64"),
        )

    def insert(self, src, dst):
        src_np = ops.convert_to_numpy(src).astype(np.int64)
        dst_np = ops.convert_to_numpy(dst).astype(np.int64)

        neighbors = np.concatenate([src_np, dst_np], axis=0)
        nodes = np.concatenate([dst_np, src_np], axis=0)
        num_interactions = len(src_np)
        e_ids = np.repeat(np.arange(self.cur_e_id, self.cur_e_id + num_interactions, dtype=np.int64), 2)
        self.cur_e_id += num_interactions

        for node, nbr, eid in zip(nodes, neighbors, e_ids):
            # insert into row node
            cur_eids = self.e_id[node]
            cur_nbrs = self.neighbors[node]
            all_eids = np.concatenate([cur_eids, [eid]])
            all_nbrs = np.concatenate([cur_nbrs, [nbr]])
            top_idx = np.argsort(-all_eids)[: self.size]
            self.e_id[node] = all_eids[top_idx]
            self.neighbors[node] = all_nbrs[top_idx]


class TGNMemory(keras.layers.Layer):
    r"""The Temporal Graph Network (TGN) memory model from the
    `"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    <https://arxiv.org/abs/2006.10637>`_ paper.

    Args:
        num_nodes (int): The number of nodes to save memories for.
        raw_msg_dim (int): The raw message dimensionality.
        memory_dim (int): The hidden memory dimensionality.
        time_dim (int): The time encoding dimensionality.
        message_module (Callable): Function combining source and destination
            node memory, raw message, and time encoding.
        aggregator_module (Callable): Function aggregating messages to the
            same destination into a single representation.

    Example:
        ```python
        import numpy as np
        from k3_node.models import IdentityMessage, LastAggregator, TGNMemory

        memory = TGNMemory(
            num_nodes=5, raw_msg_dim=8, memory_dim=16, time_dim=16,
            message_module=IdentityMessage(raw_msg_dim=8, memory_dim=16, time_dim=16),
            aggregator_module=LastAggregator(),
        )
        src, dst = np.array([0, 1]), np.array([1, 2])  # two interaction events
        t = np.array([1.0, 2.0], dtype="float32")
        raw_msg = np.random.rand(2, 8).astype("float32")
        memory.update_state(src, dst, t, raw_msg)  # update the memory of the involved nodes

        mem, last_update = memory(np.array([0, 1, 2]))
        print(tuple(mem.shape), tuple(last_update.shape))  # (3, 16) (3,)
        ```
    """
    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        message_module: Callable,
        aggregator_module: Callable,
        msg_d_module: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        if msg_d_module is not None:
            self.msg_d_module = msg_d_module
        else:
            try:
                self.msg_d_module = message_module.__class__(**message_module.get_config())
            except Exception:
                self.msg_d_module = message_module
        self.aggr_module = aggregator_module

        self.time_enc = TimeEncoder(time_dim)
        self.gru = keras.layers.GRUCell(memory_dim)

        self.memory = self.add_weight(
            name="memory",
            shape=(num_nodes, memory_dim),
            initializer="zeros",
            trainable=False,
        )
        self.last_update = self.add_weight(
            name="last_update",
            shape=(num_nodes,),
            initializer="zeros",
            trainable=False,
            dtype="float32",
        )

        self.msg_s_store = {}
        self.msg_d_store = {}
        self._reset_message_store()

    def build(self, input_shape=None):
        self.built = True

    def reset_parameters(self):
        if hasattr(self.msg_s_module, "reset_parameters"):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, "reset_parameters"):
            self.msg_d_module.reset_parameters()
        if hasattr(self.aggr_module, "reset_parameters"):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.reset_state()

    def reset_state(self):
        r"""Starts again from an empty memory and message stores."""
        self.memory.assign(ops.zeros((self.num_nodes, self.memory_dim), dtype=self.memory.dtype))
        self.last_update.assign(ops.zeros((self.num_nodes,), dtype=self.last_update.dtype))
        self._reset_message_store()

    def _reset_message_store(self):
        empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32),
                 np.zeros((0, self.raw_msg_dim), np.float32))
        self.msg_s_store = {j: empty for j in range(self.num_nodes)}
        self.msg_d_store = {j: empty for j in range(self.num_nodes)}

    # Training mode, as in PyG: in training mode the forward pass computes the memory updated by
    # the stored messages (differentiable); switching to evaluation flushes all pending updates.
    training_mode = True

    def train(self, mode: bool = True):
        if self.training_mode and not mode:
            self._update_memory(np.arange(self.num_nodes))
            self._reset_message_store()
        self.training_mode = mode
        return self

    def eval(self):
        return self.train(False)

    def call(self, n_id):
        r"""Returns the memory and last update time of the nodes ``n_id``."""
        if self.training_mode:
            return self._get_updated_memory(np.asarray(ops.convert_to_numpy(n_id)).astype(np.int64))
        return ops.take(self.memory, n_id, axis=0), ops.take(self.last_update, n_id, axis=0)

    def update_state(self, src, dst, t, raw_msg):
        r"""Updates the memory with the new events ``(src, dst, t, raw_msg)``."""
        src, dst = (np.asarray(ops.convert_to_numpy(a)).astype(np.int64) for a in (src, dst))
        t = np.asarray(ops.convert_to_numpy(t)).astype(np.float32)
        raw_msg = np.asarray(ops.convert_to_numpy(raw_msg)).astype(np.float32)
        n_id = np.unique(np.concatenate([src, dst]))
        if self.training_mode:
            self._update_memory(n_id)
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        else:
            self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
            self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
            self._update_memory(n_id)

    def detach(self):
        r"""Kept for PyG compatibility: the stored memory is never differentiated."""

    @staticmethod
    def _update_msg_store(src, dst, t, raw_msg, store):
        # every node keeps the messages of the latest batch it took part in
        order = np.argsort(src, kind="stable")
        nodes, starts = np.unique(src[order], return_index=True)
        for node, idx in zip(nodes, np.split(order, starts[1:])):
            store[int(node)] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def _compute_msg(self, n_id, store, module):
        src, dst, t, raw = (np.concatenate(parts) for parts in zip(*[store[int(i)] for i in n_id]))
        t_rel = ops.convert_to_tensor(t) - ops.take(self.last_update, src, axis=0)
        t_enc = self.time_enc(t_rel)
        msg = module(ops.take(self.memory, src, axis=0), ops.take(self.memory, dst, axis=0),
                     ops.convert_to_tensor(raw), t_enc)
        return msg, t, src

    def _get_updated_memory(self, n_id):
        assoc = np.full(self.num_nodes, -1, dtype=np.int64)
        assoc[n_id] = np.arange(len(n_id))
        msg_s, t_s, src_s = self._compute_msg(n_id, self.msg_s_store, self.msg_s_module)
        msg_d, t_d, src_d = self._compute_msg(n_id, self.msg_d_store, self.msg_d_module)
        idx = np.concatenate([src_s, src_d])
        t = np.concatenate([t_s, t_d])
        aggr = self.aggr_module(ops.concatenate([msg_s, msg_d], axis=0), assoc[idx], t, dim_size=len(n_id))
        memory, _ = self.gru(aggr, [ops.take(self.memory, n_id, axis=0)])
        latest = np.full(self.num_nodes, -np.inf, dtype=np.float32)
        np.maximum.at(latest, idx, t)
        has = np.isfinite(latest[n_id])
        # as PyG: the last update time is the latest message time (0 for nodes without messages)
        last = np.where(has, latest[n_id], 0.0).astype(np.float32)
        return memory, ops.convert_to_tensor(last)

    def _update_memory(self, n_id):
        n_id = np.asarray(n_id).astype(np.int64)
        if len(n_id) == 0:
            return
        memory, last_update = self._get_updated_memory(n_id)
        memory_np = np.asarray(ops.convert_to_numpy(self.memory)).copy()
        memory_np[n_id] = np.asarray(ops.convert_to_numpy(ops.stop_gradient(memory)))
        self.memory.assign(memory_np)
        last_np = np.asarray(ops.convert_to_numpy(self.last_update)).copy()
        last_np[n_id] = np.asarray(ops.convert_to_numpy(last_update))
        self.last_update.assign(last_np)
