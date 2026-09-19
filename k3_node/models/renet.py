from typing import Callable, List, Optional, Tuple
import math
import numpy as np
import keras
from keras import ops

from k3_node.layers.aggr import MeanAggregation


class RENet(keras.Model):
    r"""The Recurrent Event Network model from the `"Recurrent Event Network
    for Reasoning over Temporal Knowledge Graphs"
    <https://arxiv.org/abs/1904.05530>`_ paper.

    Args:
        num_nodes (int): The number of nodes in the knowledge graph.
        num_rels (int): The number of relations in the knowledge graph.
        hidden_channels (int): Hidden size of node and relation embeddings.
        seq_len (int): The sequence length of past events.
        num_layers (int, optional): The number of recurrent layers.
            (default: :obj:`1`)
        dropout (float, optional): Dropout rate before final prediction.
            (default: :obj:`0.0`)
        bias (bool, optional): If set to :obj:`False`, all layers will not
            learn an additive bias. (default: :obj:`True`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_rels: int,
        hidden_channels: int,
        seq_len: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.hidden_channels = hidden_channels
        self.seq_len = seq_len
        self.dropout_rate = dropout
        self.num_layers = num_layers

        self.ent = self.add_weight(
            name="ent",
            shape=(num_nodes, hidden_channels),
            initializer=keras.initializers.GlorotUniform(),
        )
        self.rel = self.add_weight(
            name="rel",
            shape=(num_rels, hidden_channels),
            initializer=keras.initializers.GlorotUniform(),
        )

        self.sub_gru = keras.layers.GRU(
            hidden_channels,
            return_sequences=False,
            use_bias=bias,
        )
        self.obj_gru = keras.layers.GRU(
            hidden_channels,
            return_sequences=False,
            use_bias=bias,
        )

        self.sub_lin = keras.layers.Dense(num_nodes, use_bias=bias)
        self.obj_lin = keras.layers.Dense(num_nodes, use_bias=bias)
        self.drop = keras.layers.Dropout(dropout)
        self.mean_aggr = MeanAggregation()

    def reset_parameters(self):
        self.ent.assign(
            keras.initializers.GlorotUniform()(self.ent.shape)
        )
        self.rel.assign(
            keras.initializers.GlorotUniform()(self.rel.shape)
        )


    def call(
        self,
        sub,
        rel,
        obj,
        h_sub,
        h_sub_t,
        h_sub_batch,
        h_obj,
        h_obj_t,
        h_obj_batch,
        training=False,
    ):
        batch_size = ops.shape(sub)[0]
        seq_len = self.seq_len

        h_sub_t = h_sub_t + h_sub_batch * seq_len
        h_obj_t = h_obj_t + h_obj_batch * seq_len

        ent_h_sub = ops.take(self.ent, h_sub, axis=0)
        ent_h_obj = ops.take(self.ent, h_obj, axis=0)

        h_sub_scatter = self.mean_aggr(
            ent_h_sub, index=h_sub_t, dim_size=batch_size * seq_len, dim=0
        )
        h_sub = ops.reshape(h_sub_scatter, (batch_size, seq_len, -1))

        h_obj_scatter = self.mean_aggr(
            ent_h_obj, index=h_obj_t, dim_size=batch_size * seq_len, dim=0
        )
        h_obj = ops.reshape(h_obj_scatter, (batch_size, seq_len, -1))

        sub_emb = ops.take(self.ent, sub, axis=0)
        rel_emb = ops.take(self.rel, rel, axis=0)
        obj_emb = ops.take(self.ent, obj, axis=0)

        sub_rep = ops.repeat(ops.expand_dims(sub_emb, 1), seq_len, axis=1)
        rel_rep = ops.repeat(ops.expand_dims(rel_emb, 1), seq_len, axis=1)
        obj_rep = ops.repeat(ops.expand_dims(obj_emb, 1), seq_len, axis=1)

        gru_sub_in = ops.concatenate([sub_rep, h_sub, rel_rep], axis=-1)
        gru_obj_in = ops.concatenate([obj_rep, h_obj, rel_rep], axis=-1)

        h_sub = self.sub_gru(gru_sub_in, training=training)
        h_obj = self.obj_gru(gru_obj_in, training=training)

        h_sub = ops.concatenate([sub_emb, h_sub, rel_emb], axis=-1)
        h_obj = ops.concatenate([obj_emb, h_obj, rel_emb], axis=-1)

        h_sub = self.drop(h_sub, training=training)
        h_obj = self.drop(h_obj, training=training)

        log_prob_obj = ops.log_softmax(self.sub_lin(h_sub), axis=-1)
        log_prob_sub = ops.log_softmax(self.obj_lin(h_obj), axis=-1)

        return log_prob_obj, log_prob_sub

    def test(self, logits, y):
        r"""Given ground-truth :obj:`y`, computes Mean Reciprocal Rank (MRR)
        and Hits at 1/3/10.
        """
        logits_np = ops.convert_to_numpy(logits)
        y_np = ops.convert_to_numpy(y).reshape(-1, 1)

        perm = np.argsort(-logits_np, axis=1)
        mask = y_np == perm

        rows, cols = np.nonzero(mask)
        mrr = float(np.mean(1.0 / (cols + 1.0)))
        hits1 = float(np.sum(cols < 1) / len(y_np))
        hits3 = float(np.sum(cols < 3) / len(y_np))
        hits10 = float(np.sum(cols < 10) / len(y_np))

        return ops.convert_to_tensor([mrr, hits1, hits3, hits10], dtype="float32")
