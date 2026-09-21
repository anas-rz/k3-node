from typing import Optional, Union

import keras
from keras import ops
import numpy as np

from k3_node.layers.conv import LGConv


class BPRLoss:
    r"""The Bayesian Personalized Ranking (BPR) loss."""
    def __init__(self, lambda_reg: float = 0.0):
        self.lambda_reg = lambda_reg

    def __call__(self, positives, negatives, parameters=None):
        diff = positives - negatives
        # log(sigmoid(x)) = -log(1 + exp(-x)) or ops.log_sigmoid if available
        log_prob = ops.mean(-ops.softplus(-diff))

        regularization = 0.0
        if self.lambda_reg != 0.0 and parameters is not None:
            regularization = self.lambda_reg * ops.sum(ops.square(parameters))
            regularization = regularization / ops.cast(ops.shape(positives)[0], "float32")

        return -log_prob + regularization


class LightGCN(keras.Model):
    r"""The LightGCN model from the `"LightGCN: Simplifying and Powering
    Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper.

    Args:
        num_nodes (int): The number of nodes in the graph.
        embedding_dim (int): The dimensionality of node embeddings.
        num_layers (int): The number of :class:`LGConv` layers.
        alpha (float or Tensor, optional): The scalar or vector specifying
            the re-weighting coefficients for aggregating the final embedding.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        num_layers: int,
        alpha: Optional[Union[float, list]] = None,
        **kwargs,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        if alpha is None:
            alpha = [1.0 / (num_layers + 1)] * (num_layers + 1)
        elif isinstance(alpha, (int, float)):
            alpha = [float(alpha)] * (num_layers + 1)
        self.alpha_list = list(alpha)

        self.embedding = keras.layers.Embedding(
            num_nodes,
            embedding_dim,
            embeddings_initializer=keras.initializers.GlorotUniform(),
        )
        self.convs = [LGConv(**kwargs) for _ in range(num_layers)]

    def build(self, input_shape=None):
        self.embedding.build((None,))
        self.built = True

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.embedding.built:
            self.embedding.embeddings.assign(
                keras.initializers.GlorotUniform()(shape=(self.num_nodes, self.embedding_dim))
            )
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

    def get_embedding(self, edge_index, edge_weight=None):
        r"""Returns the embedding of nodes in the graph."""
        if not self.embedding.built:
            self.embedding.build((None,))
        # Embedding weights: shape [num_nodes, embedding_dim]
        x = self.embedding.weights[0]
        out = x * self.alpha_list[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            out = out + x * self.alpha_list[i + 1]

        return out

    def call(self, edge_index, edge_label_index=None, edge_weight=None):
        r"""Computes rankings for pairs of nodes."""
        if edge_label_index is None:
            edge_label_index = edge_index

        out = self.get_embedding(edge_index, edge_weight)

        out_src = ops.take(out, edge_label_index[0], axis=0)
        out_dst = ops.take(out, edge_label_index[1], axis=0)

        return ops.sum(out_src * out_dst, axis=-1)

    def predict_link(
        self,
        edge_index,
        edge_label_index=None,
        edge_weight=None,
        prob: bool = False,
    ):
        pred = ops.sigmoid(self(edge_index, edge_label_index, edge_weight))
        return pred if prob else ops.round(pred)

    def recommend(
        self,
        edge_index,
        edge_weight=None,
        src_index=None,
        dst_index=None,
        k: int = 1,
        sorted: bool = True,
    ):
        out = self.get_embedding(edge_index, edge_weight)
        out_src = ops.take(out, src_index, axis=0) if src_index is not None else out
        out_dst = ops.take(out, dst_index, axis=0) if dst_index is not None else out

        pred = out_src @ ops.transpose(out_dst)
        top_indices = ops.top_k(pred, k=k, sorted=sorted)[1]

        if dst_index is not None:
            top_indices = ops.take(dst_index, top_indices, axis=0)

        return top_indices

    def link_pred_loss(self, pred, edge_label):
        loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        return loss_fn(edge_label, pred)

    def recommendation_loss(
        self,
        pos_edge_rank,
        neg_edge_rank,
        node_id=None,
        lambda_reg: float = 1e-4,
    ):
        loss_fn = BPRLoss(lambda_reg)
        emb = self.embedding.weights[0]
        emb = emb if node_id is None else ops.take(emb, node_id, axis=0)
        return loss_fn(pos_edge_rank, neg_edge_rank, emb)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'{self.embedding_dim}, num_layers={self.num_layers})')

