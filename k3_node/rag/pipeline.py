"""High-level GraphRAG Pipeline integrating retrieval, verbalization, and LLM prefix injection."""

from typing import Dict, List, Literal, Optional, Sequence, Union
import numpy as np
from keras import ops

from k3_node.rag.encoders import RGCNSubGraphEncoder, TransEPrefixEncoder
from k3_node.rag.projector import GraphPrefixProjector, KGLLMConnector
from k3_node.rag.subgraph import KGEntityRetriever, SubgraphResult, extract_subgraph
from k3_node.rag.verbalizer import format_llm_prompt, verbalize_subgraph


class GraphRAG:
    r"""End-to-End Graph-Augmented Generation (GraphRAG) Pipeline for Knowledge Graphs.

    Provides a unified API for:
    1. Extracting subgraphs around query entities.
    2. Verbalizing structured facts into markdown or natural language prompts for Llama 3 / Mistral.
    3. Projecting GNN (RGCN) or KGE (TransE) embeddings into dense prefix vectors for soft prompt augmentation.

    Example:
        ```python
        import k3_node as k3
        from k3_node.rag import GraphRAG

        # Setup GraphRAG with your knowledge graph
        rag = GraphRAG(
            edge_index=edge_index,
            edge_type=edge_type,
            entity_to_id={"Aspirin": 0, "Headache": 1, "COX-1": 2},
            relation_to_id={"treats": 0, "inhibits": 1},
            llm_dim=4096,  # Llama 3 / Mistral embedding dimension
            num_prefix_tokens=4,
        )

        # 1. Text-based GraphRAG prompt generation
        prompt = rag.build_prompt(
            query="How does Aspirin alleviate headache?",
            entities=["Aspirin", "Headache"],
            model_family="llama3",
        )

        # 2. Dense Prefix Vector encoding
        subgraph = rag.retrieve(["Aspirin"], num_hops=2)
        prefix_embeds = rag.encode_prefix(subgraph)  # [1, 4, 4096]
        ```

    Args:
        edge_index: Graph connectivity tensor of shape `(2, num_edges)`.
        edge_type: Optional relation IDs tensor of shape `(num_edges,)`.
        entity_to_id: Dictionary mapping entity strings to node IDs.
        relation_to_id: Optional dictionary mapping relation strings to relation IDs.
        x: Optional node feature tensor.
        num_relations: Optional total number of relation types.
        encoder_type: Subgraph encoder architecture (`"rgcn"`, `"transe"`, or a custom encoder).
        hidden_dim: Hidden dimension for GNN/KGE encoder. (default: 64)
        encoder_out_dim: Output dimension of graph encoder before LLM projection. (default: 128)
        llm_dim: Target LLM embedding dimension (e.g. 4096 for Llama 3 / Mistral). (default: 4096)
        num_prefix_tokens: Number of virtual prefix tokens to generate. (default: 8)
    """

    def __init__(
        self,
        edge_index: any,
        edge_type: Optional[any] = None,
        entity_to_id: Optional[Dict[str, int]] = None,
        relation_to_id: Optional[Dict[str, int]] = None,
        x: Optional[any] = None,
        num_relations: Optional[int] = None,
        encoder_type: Literal["rgcn", "transe", "none"] = "rgcn",
        hidden_dim: int = 64,
        encoder_out_dim: int = 128,
        llm_dim: int = 4096,
        num_prefix_tokens: int = 8,
        connector: Optional[KGLLMConnector] = None,
    ):
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.x = x
        self.entity_to_id = entity_to_id or {}
        self.relation_to_id = relation_to_id or {}
        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}

        self.retriever = KGEntityRetriever(
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            x=self.x,
        )

        if connector is not None:
            self.connector = connector
        elif encoder_type == "none":
            self.connector = None
        else:
            n_rel = num_relations
            if n_rel is None and self.relation_to_id:
                n_rel = len(self.relation_to_id)
            elif n_rel is None and edge_type is not None:
                n_rel = int(np.max(ops.convert_to_numpy(edge_type))) + 1
            else:
                n_rel = n_rel or 1

            in_ch = ops.shape(x)[-1] if x is not None else hidden_dim
            if encoder_type == "rgcn":
                encoder = RGCNSubGraphEncoder(
                    in_channels=in_ch,
                    hidden_channels=hidden_dim,
                    out_channels=encoder_out_dim,
                    num_relations=n_rel,
                )
            elif encoder_type == "transe":
                num_nodes = len(self.entity_to_id) if self.entity_to_id else int(ops.max(edge_index)) + 1
                encoder = TransEPrefixEncoder(
                    num_nodes=num_nodes,
                    num_relations=n_rel,
                    embedding_dim=hidden_dim,
                    out_channels=encoder_out_dim,
                )
            else:
                raise ValueError(f"Unknown encoder_type: {encoder_type}")

            projector = GraphPrefixProjector(
                in_channels=encoder_out_dim,
                llm_dim=llm_dim,
                num_prefix_tokens=num_prefix_tokens,
            )
            self.connector = KGLLMConnector(encoder=encoder, projector=projector)

    def retrieve(
        self,
        entities: Union[Sequence[Union[str, int]], str, int],
        num_hops: int = 2,
        max_nodes_per_hop: Optional[int] = None,
        directed: bool = False,
    ) -> SubgraphResult:
        """Retrieve multi-hop subgraph around specified entities."""
        return self.retriever.retrieve_subgraph(
            entities=entities,
            num_hops=num_hops,
            max_nodes_per_hop=max_nodes_per_hop,
            directed=directed,
        )

    def verbalize(
        self,
        subgraph: SubgraphResult,
        format_style: Literal["triples", "markdown", "natural"] = "markdown",
        max_triples: Optional[int] = 50,
    ) -> str:
        """Verbalize extracted subgraph into text knowledge for LLM prompt context."""
        return verbalize_subgraph(
            subgraph=subgraph,
            id_to_entity=self.id_to_entity,
            id_to_relation=self.id_to_relation,
            format_style=format_style,
            max_triples=max_triples,
        )

    def build_prompt(
        self,
        query: str,
        entities: Optional[Sequence[Union[str, int]]] = None,
        num_hops: int = 2,
        format_style: Literal["triples", "markdown", "natural"] = "markdown",
        model_family: Literal["llama3", "mistral", "chatml", "standard"] = "llama3",
        system_prompt: Optional[str] = None,
    ) -> str:
        """End-to-end prompt builder: extracts subgraph and formats full LLM prompt."""
        if entities is None or len(entities) == 0:
            entities = self.retriever.find_entities_in_text(query)

        subgraph = self.retrieve(entities=entities, num_hops=num_hops)
        context = self.verbalize(subgraph=subgraph, format_style=format_style)
        return format_llm_prompt(
            query=query,
            context=context,
            system_prompt=system_prompt,
            model_family=model_family,
        )

    def encode_prefix(self, subgraph: SubgraphResult) -> any:
        """Encode extracted subgraph into LLM prefix embeddings.

        Returns:
            Tensor of shape `(1, num_prefix_tokens, llm_dim)`.
        """
        if self.connector is None:
            raise ValueError("No KGLLMConnector configured on this GraphRAG instance.")
        return self.connector.encode_subgraph(subgraph)
