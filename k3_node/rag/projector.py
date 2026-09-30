"""Prefix Projectors and KG-LLM Connectors for prompt augmentation."""

from typing import Literal, Optional, Tuple, Union

import keras
from keras import layers, ops

from k3_node.rag.subgraph import SubgraphResult


class GraphPrefixProjector(keras.layers.Layer):
    r"""Projects graph/KGE representations into virtual prefix vectors for LLM prompt augmentation.

    Maps a graph embedding vector of shape `(batch_size, in_channels)` into `num_prefix_tokens`
    virtual token embeddings of dimension `llm_dim` (e.g. 4096 for Llama 3 / Mistral),
    suitable for prepending to text token embeddings in LLM forward passes.

    Args:
        in_channels: Input graph feature dimension from the GNN/KGE encoder.
        llm_dim: Embedding dimension of the target LLM (e.g., 4096 for Llama-3-8B / Mistral-7B,
            2048 for Gemma-2B). (default: 4096)
        num_prefix_tokens: Number of virtual prefix tokens to produce. (default: 8)
        projector_type: Architecture of the projection head:
            - `"mlp"`: 2-layer MLP with LayerNorm and GELU activation.
            - `"linear"`: Single linear transformation.
        hidden_dim: Optional hidden dimension for MLP projector. (default: `2 * in_channels`)
        dropout: Dropout probability. (default: 0.0)
    """

    def __init__(
        self,
        in_channels: int,
        llm_dim: int = 4096,
        num_prefix_tokens: int = 8,
        projector_type: Literal["mlp", "linear"] = "mlp",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.llm_dim = llm_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.projector_type = projector_type
        self.dropout_rate = dropout

        out_total = num_prefix_tokens * llm_dim
        if hidden_dim is None:
            hidden_dim = max(in_channels * 2, 512)

        if projector_type == "linear":
            self.net = layers.Dense(out_total)
        else:  # "mlp"
            self.net = keras.Sequential(
                [
                    layers.Dense(hidden_dim),
                    layers.LayerNormalization(),
                    layers.Activation("gelu"),
                    layers.Dropout(dropout) if dropout > 0.0 else layers.Identity(),
                    layers.Dense(out_total),
                    layers.LayerNormalization(),
                ]
            )

    def call(self, graph_embedding: any, training: Optional[bool] = None) -> any:
        """Project graph embedding into soft prefix token embeddings.

        Args:
            graph_embedding: Tensor of shape `(batch_size, in_channels)` or `(in_channels,)`.

        Returns:
            Prefix tensor of shape `(batch_size, num_prefix_tokens, llm_dim)`.
        """
        graph_embedding = ops.convert_to_tensor(graph_embedding)
        # Ensure 2D (batch_size, in_channels)
        if len(ops.shape(graph_embedding)) == 1:
            graph_embedding = ops.expand_dims(graph_embedding, axis=0)

        batch_size = ops.shape(graph_embedding)[0]
        flat_proj = self.net(graph_embedding, training=training)
        # Reshape to (batch_size, num_prefix_tokens, llm_dim)
        prefix_tokens = ops.reshape(flat_proj, (batch_size, self.num_prefix_tokens, self.llm_dim))
        return prefix_tokens


class KGLLMConnector(keras.Model):
    r"""High-level Knowledge Graph to Large Language Model (KG-LLM) Connector.

    Connects a Knowledge Graph encoder (e.g. `RGCNSubGraphEncoder` or `TransEPrefixEncoder`)
    with a `GraphPrefixProjector` to produce soft prompt prefix embeddings and inject them
    into Llama 3, Mistral, or other LLMs.

    Example:
        ```python
        from k3_node.rag import RGCNSubGraphEncoder, GraphPrefixProjector, KGLLMConnector

        encoder = RGCNSubGraphEncoder(in_channels=16, hidden_channels=32, out_channels=64, num_relations=5)
        projector = GraphPrefixProjector(in_channels=64, llm_dim=4096, num_prefix_tokens=4)
        connector = KGLLMConnector(encoder=encoder, projector=projector)

        # Generate prefix embeddings for LLM prompt augmentation
        prefix_embeds = connector.encode_subgraph(subgraph)  # [1, 4, 4096]

        # Inject into LLM input embeddings [batch, seq_len, 4096]
        augmented_inputs = connector.inject_prefix(text_token_embeds, prefix_embeds)
        ```

    Args:
        encoder: Graph or KGE encoder (e.g., `RGCNSubGraphEncoder`, `TransEPrefixEncoder`).
        projector: `GraphPrefixProjector` instance.
    """

    def __init__(
        self,
        encoder: keras.layers.Layer,
        projector: GraphPrefixProjector,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.projector = projector

    @property
    def num_prefix_tokens(self) -> int:
        return self.projector.num_prefix_tokens

    @property
    def llm_dim(self) -> int:
        return self.projector.llm_dim

    def call(self, *args, **kwargs) -> any:
        """Encode graph inputs and project directly into LLM prefix tokens."""
        graph_emb = self.encoder(*args, **kwargs)
        return self.projector(graph_emb)

    def encode_subgraph(self, subgraph: SubgraphResult) -> any:
        """Encode an extracted SubgraphResult into LLM prefix tokens.

        Returns:
            Tensor of shape `(1, num_prefix_tokens, llm_dim)`.
        """
        if hasattr(self.encoder, "encode_subgraph"):
            graph_emb = self.encoder.encode_subgraph(subgraph)
        else:
            graph_emb = self.encoder(subgraph.x, subgraph.edge_index)
        return self.projector(graph_emb)

    @staticmethod
    def inject_prefix(text_embeddings: any, prefix_embeddings: any) -> any:
        """Prepend graph prefix embeddings to text token embeddings along sequence dimension.

        Args:
            text_embeddings: Tensor of shape `(batch, seq_len, llm_dim)`.
            prefix_embeddings: Tensor of shape `(batch, num_prefix_tokens, llm_dim)`.

        Returns:
            Concatenated tensor of shape `(batch, num_prefix_tokens + seq_len, llm_dim)`.
        """
        text_embeddings = ops.convert_to_tensor(text_embeddings)
        prefix_embeddings = ops.convert_to_tensor(prefix_embeddings)

        # Match batch size if prefix was computed for single batch
        b_text = ops.shape(text_embeddings)[0]
        b_prefix = ops.shape(prefix_embeddings)[0]
        if b_prefix == 1 and b_text > 1:
            prefix_embeddings = ops.repeat(prefix_embeddings, b_text, axis=0)

        return ops.concatenate([prefix_embeddings, text_embeddings], axis=1)

    @staticmethod
    def extend_attention_mask(attention_mask: any, num_prefix_tokens: int) -> any:
        """Extend LLM binary attention mask with 1s for the prepended prefix tokens.

        Args:
            attention_mask: Tensor of shape `(batch, seq_len)`.
            num_prefix_tokens: Number of prefix tokens prepended.

        Returns:
            Extended mask of shape `(batch, num_prefix_tokens + seq_len)`.
        """
        attention_mask = ops.convert_to_tensor(attention_mask)
        batch_size = ops.shape(attention_mask)[0]
        prefix_mask = ops.ones((batch_size, num_prefix_tokens), dtype=attention_mask.dtype)
        return ops.concatenate([prefix_mask, attention_mask], axis=1)
