from typing import Callable, Dict, List, Optional, Tuple
import copy
import numpy as np
import keras
from keras import ops

from k3_node.layers.aggr import MeanAggregation


class TimeEncoder(keras.layers.Layer):
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
    def call(self, msg, index, t, dim_size: int):
        t_np = ops.convert_to_numpy(t)
        index_np = ops.convert_to_numpy(index).astype(np.int64)
        msg_np = ops.convert_to_numpy(msg)

        out_np = np.zeros((dim_size, msg_np.shape[-1]), dtype=msg_np.dtype)
        max_t = np.full((dim_size,), -1e18, dtype=t_np.dtype)
        for m, idx, ti in zip(msg_np, index_np, t_np):
            if ti > max_t[idx]:
                max_t[idx] = ti
                out_np[idx] = m
        return ops.convert_to_tensor(out_np, dtype=msg.dtype)


class MeanAggregator(keras.layers.Layer):
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
        self.memory.assign(ops.zeros((self.num_nodes, self.memory_dim), dtype=self.memory.dtype))
        self.last_update.assign(ops.zeros((self.num_nodes,), dtype=self.last_update.dtype))
        self._reset_message_store()

    def _reset_message_store(self):
        self.msg_s_store = {j: ([], [], [], []) for j in range(self.num_nodes)}
        self.msg_d_store = {j: ([], [], [], []) for j in range(self.num_nodes)}

    def call(self, n_id):
        mem = ops.take(self.memory, n_id, axis=0)
        last_up = ops.take(self.last_update, n_id, axis=0)
        return mem, last_up

    def update_state(self, src, dst, t, raw_msg):
        src_np = ops.convert_to_numpy(src).astype(np.int64)
        dst_np = ops.convert_to_numpy(dst).astype(np.int64)
        t_np = ops.convert_to_numpy(t).astype(np.float32)
        msg_np = ops.convert_to_numpy(raw_msg).astype(np.float32)

        # Update memory for interacting nodes
        n_id_unique = np.unique(np.concatenate([src_np, dst_np]))
        self._update_memory(n_id_unique)

        # Store messages
        for s, d, ti, m in zip(src_np, dst_np, t_np, msg_np):
            self.msg_s_store[s][0].append(s)
            self.msg_s_store[s][1].append(d)
            self.msg_s_store[s][2].append(ti)
            self.msg_s_store[s][3].append(m)

            self.msg_d_store[d][0].append(d)
            self.msg_d_store[d][1].append(s)
            self.msg_d_store[d][2].append(ti)
            self.msg_d_store[d][3].append(m)

    def _update_memory(self, n_id_np):
        if len(n_id_np) == 0:
            return

        # Compute messages for n_id
        all_msgs = []
        all_indices = []
        all_times = []

        assoc = {int(node): i for i, node in enumerate(n_id_np)}

        for store, module in [(self.msg_s_store, self.msg_s_module), (self.msg_d_store, self.msg_d_module)]:
            for node in n_id_np:
                s_list, d_list, t_list, m_list = store[int(node)]
                if len(s_list) > 0:
                    s_tensor = ops.convert_to_tensor(np.array(s_list, dtype=np.int64), dtype="int64")
                    d_tensor = ops.convert_to_tensor(np.array(d_list, dtype=np.int64), dtype="int64")
                    t_tensor = ops.convert_to_tensor(np.array(t_list, dtype=np.float32), dtype="float32")
                    m_tensor = ops.convert_to_tensor(np.array(m_list, dtype=np.float32), dtype="float32")

                    t_last = ops.take(self.last_update, s_tensor, axis=0)
                    t_rel = t_tensor - t_last
                    t_enc = self.time_enc(t_rel)

                    mem_s = ops.take(self.memory, s_tensor, axis=0)
                    mem_d = ops.take(self.memory, d_tensor, axis=0)
                    computed_msg = module(mem_s, mem_d, m_tensor, t_enc)

                    all_msgs.append(computed_msg)
                    target_assoc = np.array([assoc[int(x)] for x in s_list], dtype=np.int64)
                    all_indices.append(ops.convert_to_tensor(target_assoc, dtype="int64"))
                    all_times.append(t_tensor)

        if len(all_msgs) > 0:
            cat_msgs = ops.concatenate(all_msgs, axis=0)
            cat_indices = ops.concatenate(all_indices, axis=0)
            cat_times = ops.concatenate(all_times, axis=0)

            aggr = self.aggr_module(cat_msgs, cat_indices, cat_times, len(n_id_np))
            n_id_tensor = ops.convert_to_tensor(n_id_np, dtype="int64")
            cur_mem = ops.take(self.memory, n_id_tensor, axis=0)
            new_mem, _ = self.gru(aggr, [cur_mem])

            # Update memory tensor
            mem_np = ops.convert_to_numpy(self.memory)
            mem_np[n_id_np] = ops.convert_to_numpy(new_mem)
            self.memory.assign(ops.convert_to_tensor(mem_np, dtype=self.memory.dtype))

            # Update last_update tensor
            times_np = ops.convert_to_numpy(cat_times)
            indices_np = ops.convert_to_numpy(cat_indices)
            last_up_np = ops.convert_to_numpy(self.last_update)
            for idx, ti in zip(indices_np, times_np):
                node = n_id_np[idx]
                last_up_np[node] = max(last_up_np[node], ti)
            self.last_update.assign(ops.convert_to_tensor(last_up_np, dtype=self.last_update.dtype))

            # Clear processed messages
            for node in n_id_np:
                self.msg_s_store[int(node)] = ([], [], [], [])
                self.msg_d_store[int(node)] = ([], [], [], [])
