"""GNN and KGE Subgraph Encoders for GraphRAG and KG-LLM augmentation."""

from typing import List, Literal, Optional, Tuple, Union

import keras
from keras import layers, ops

from k3_node.layers.conv import GCNConv, RGCNConv, SAGEConv
from k3_node.layers.kge import KGEModel, TransE
from k3_node.rag.subgraph import SubgraphResult


class RGCNSubGraphEncoder(keras.Model):
    r"""Relational Graph Convolutional Network (RGCN) encoder for multi-relational subgraphs.

    Processes subgraphs extracted from Knowledge Graphs with multiple relation types,
    updating entity representations through relational message passing and pooling them
    into a dense graph embedding.

    Args:
        in_channels: Dimensionality of input node features.
        hidden_channels: Hidden representation dimension.
        out_channels: Output graph embedding dimension.
        num_relations: Total number of relation types in the Knowledge Graph.
        num_layers: Number of RGCN message passing layers. (default: 2)
        num_bases: Optional number of basis decomposition components for relation weights.
        pooling: Readout pooling strategy across subgraph nodes:
            - `"mean"`: Global average over all subgraph nodes.
            - `"sum"`: Global sum over all subgraph nodes.
            - `"max"`: Global maximum over all subgraph nodes.
            - `"center"`: Pool only the retrieved center seed entities.
            - `"none"`: Return all node embeddings without pooling.
        dropout: Dropout rate applied between convolution layers. (default: 0.0)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_relations: int,
        num_layers: int = 2,
        num_bases: Optional[int] = None,
        pooling: Literal["mean", "sum", "max", "center", "none"] = "mean",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.pooling = pooling
        self.dropout_rate = dropout

        self.convs = []
        for i in range(num_layers):
            c_in = in_channels if i == 0 else hidden_channels
            c_out = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(
                RGCNConv(
                    in_channels=c_in,
                    out_channels=c_out,
                    num_relations=num_relations,
                    num_bases=num_bases,
                )
            )

        self.act = layers.Activation("relu")
        self.drop = layers.Dropout(dropout) if dropout > 0.0 else None

    def call(
        self,
        x: any,
        edge_index: any,
        edge_type: Optional[any] = None,
        center_nodes: Optional[any] = None,
        training: Optional[bool] = None,
    ) -> any:
        """Forward pass encoding the subgraph into a graph embedding.

        Args:
            x: Node feature tensor of shape `(num_nodes, in_channels)`.
            edge_index: Graph edge indices of shape `(2, num_edges)`.
            edge_type: 1D tensor of relation IDs for each edge `(num_edges,)`.
            center_nodes: Optional 1D tensor of seed entity indices in the subgraph.
            training: Whether running in training mode.

        Returns:
            Tensor of shape `(1, out_channels)` (or `(num_nodes, out_channels)` if `pooling="none"`).
        """
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_type=edge_type)
            if i < self.num_layers - 1:
                h = self.act(h)
                if self.drop is not None:
                    h = self.drop(h, training=training)

        if self.pooling == "none":
            return h

        if self.pooling == "center" and center_nodes is not None:
            center_nodes = ops.convert_to_tensor(center_nodes, dtype="int32")
            if ops.shape(center_nodes)[0] > 0:
                center_h = ops.take(h, center_nodes, axis=0)
                pooled = ops.mean(center_h, axis=0, keepdims=True)
                return pooled

        if self.pooling == "sum":
            return ops.sum(h, axis=0, keepdims=True)
        elif self.pooling == "max":
            return ops.max(h, axis=0, keepdims=True)
        else:  # "mean" or fallback
            return ops.mean(h, axis=0, keepdims=True)

    def encode_subgraph(self, subgraph: SubgraphResult, default_x_dim: Optional[int] = None) -> any:
        """Helper to encode a SubgraphResult directly."""
        x = subgraph.x
        if x is None:
            dim = default_x_dim if default_x_dim is not None else self.in_channels
            num_n = subgraph.num_nodes if subgraph.num_nodes > 0 else 1
            x = ops.ones((num_n, dim), dtype="float32")

        return self(
            x=x,
            edge_index=subgraph.edge_index,
            edge_type=subgraph.edge_type,
            center_nodes=subgraph.center_nodes,
        )


class TransEPrefixEncoder(keras.Model):
    r"""Knowledge Graph Embedding (KGE) prefix encoder using TransE representations.

    Uses pretrained or end-to-end entity and relation embeddings from TransE ($h + r \approx t$)
    to encode extracted knowledge subgraphs into a unified dense embedding vector.

    Args:
        num_nodes: Total number of entities in the knowledge graph.
        num_relations: Total number of relation types.
        embedding_dim: Dimension of entity and relation embeddings. (default: 64)
        out_channels: Output projected graph embedding dimension. (default: 128)
        kge_model: Optional pretrained `KGEModel` (e.g. `TransE`). If provided,
            its embeddings are reused.
        pooling: Readout pooling strategy across entities (`"mean"`, `"center"`, `"sum"`).
    """

    def __init__(
        self,
        num_nodes: Optional[int] = None,
        num_relations: Optional[int] = None,
        embedding_dim: int = 64,
        out_channels: int = 128,
        kge_model: Optional[KGEModel] = None,
        pooling: Literal["mean", "sum", "center"] = "mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.out_channels = out_channels
        self.pooling = pooling

        if kge_model is not None:
            self.node_emb = kge_model.node_emb
            self.rel_emb = kge_model.rel_emb
            self.embedding_dim = kge_model.hidden_channels
        else:
            if num_nodes is None or num_relations is None:
                raise ValueError("Must provide either `kge_model` or both `num_nodes` and `num_relations`.")
            self.node_emb = layers.Embedding(num_nodes, embedding_dim)
            self.rel_emb = layers.Embedding(num_relations, embedding_dim)
            self.node_emb.build((None,))
            self.rel_emb.build((None,))

        # Projection head combining entity representation and relational context
        self.proj = keras.Sequential(
            [
                layers.Dense(out_channels, activation="relu"),
                layers.LayerNormalization(),
                layers.Dense(out_channels),
            ]
        )

    def call(
        self,
        subgraph_nodes: any,
        edge_index: Optional[any] = None,
        edge_type: Optional[any] = None,
        center_nodes: Optional[any] = None,
    ) -> any:
        """Encode a set of subgraph entities and relations into a dense embedding.

        Args:
            subgraph_nodes: 1D tensor of original entity IDs in the subgraph `(num_nodes,)`.
            edge_index: Optional edge index tensor `(2, num_edges)`.
            edge_type: Optional relation type tensor `(num_edges,)`.
            center_nodes: Optional indices of seed entities within `subgraph_nodes`.

        Returns:
            Dense embedding tensor of shape `(1, out_channels)`.
        """
        subgraph_nodes = ops.convert_to_tensor(subgraph_nodes, dtype="int32")
        entity_embeddings = self.node_emb(subgraph_nodes)  # [num_nodes, embedding_dim]

        # Entity pooling
        if self.pooling == "center" and center_nodes is not None:
            center_idx = ops.convert_to_tensor(center_nodes, dtype="int32")
            if ops.shape(center_idx)[0] > 0:
                ent_h = ops.take(entity_embeddings, center_idx, axis=0)
                node_repr = ops.mean(ent_h, axis=0, keepdims=True)
            else:
                node_repr = ops.mean(entity_embeddings, axis=0, keepdims=True)
        elif self.pooling == "sum":
            node_repr = ops.sum(entity_embeddings, axis=0, keepdims=True)
        else:
            node_repr = ops.mean(entity_embeddings, axis=0, keepdims=True)

        # Relational translation context (h + r - t in TransE)
        if edge_index is not None and edge_type is not None:
            edge_index = ops.convert_to_tensor(edge_index, dtype="int32")
            edge_type = ops.convert_to_tensor(edge_type, dtype="int32")
            num_e = ops.shape(edge_index)[1]

            if num_e > 0:
                row, col = edge_index[0], edge_index[1]
                h_sub = ops.take(entity_embeddings, row, axis=0)
                t_sub = ops.take(entity_embeddings, col, axis=0)
                r_sub = self.rel_emb(edge_type)
                # TransE relation representation
                triple_context = ops.mean(h_sub + r_sub - t_sub, axis=0, keepdims=True)
                combined = ops.concatenate([node_repr, triple_context], axis=-1)
            else:
                combined = ops.concatenate([node_repr, ops.zeros_like(node_repr)], axis=-1)
        else:
            combined = ops.concatenate([node_repr, ops.zeros_like(node_repr)], axis=-1)

        return self.proj(combined)

    def encode_subgraph(self, subgraph: SubgraphResult) -> any:
        """Helper to encode a SubgraphResult directly."""
        nodes = subgraph.nodes if subgraph.nodes is not None else np.arange(subgraph.num_nodes)
        return self(
            subgraph_nodes=nodes,
            edge_index=subgraph.edge_index,
            edge_type=subgraph.edge_type,
            center_nodes=subgraph.center_nodes,
        )


class GNNSubGraphEncoder(keras.Model):
    r"""General-purpose GNN encoder (GCN / GraphSAGE) for homogeneous subgraphs.

    Args:
        in_channels: Input node feature dimensionality.
        hidden_channels: Hidden representation dimension.
        out_channels: Output graph embedding dimension.
        conv_type: Type of GNN convolution (`"gcn"` or `"sage"`). (default: `"gcn"`)
        num_layers: Number of convolution layers. (default: 2)
        pooling: Pooling strategy (`"mean"`, `"sum"`, `"max"`, `"center"`).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        conv_type: Literal["gcn", "sage"] = "gcn",
        num_layers: int = 2,
        pooling: Literal["mean", "sum", "max", "center"] = "mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.num_layers = num_layers

        ConvCls = GCNConv if conv_type.lower() == "gcn" else SAGEConv
        self.convs = []
        for i in range(num_layers):
            c_in = in_channels if i == 0 else hidden_channels
            c_out = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(ConvCls(c_in, c_out))

        self.act = layers.Activation("relu")

    def call(
        self,
        x: any,
        edge_index: any,
        center_nodes: Optional[any] = None,
    ) -> any:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < self.num_layers - 1:
                h = self.act(h)

        if self.pooling == "center" and center_nodes is not None:
            center_nodes = ops.convert_to_tensor(center_nodes, dtype="int32")
            if ops.shape(center_nodes)[0] > 0:
                return ops.mean(ops.take(h, center_nodes, axis=0), axis=0, keepdims=True)

        if self.pooling == "sum":
            return ops.sum(h, axis=0, keepdims=True)
        elif self.pooling == "max":
            return ops.max(h, axis=0, keepdims=True)
        else:
            return ops.mean(h, axis=0, keepdims=True)
