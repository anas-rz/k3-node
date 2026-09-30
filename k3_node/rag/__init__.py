"""GraphRAG & KG-LLM Connectors for K3-Node.

Enables extracting subgraphs around retrieved entities, verbalizing structured knowledge
into LLM prompt contexts (Llama 3, Mistral, ChatML), and projecting GNN (RGCN) and KGE (TransE)
subgraph embeddings into dense prefix vectors for soft-prompt LLM augmentation.
"""

from k3_node.rag.subgraph import (
    SubgraphResult,
    extract_subgraph,
    KGEntityRetriever,
)
from k3_node.rag.verbalizer import (
    subgraph_to_triples,
    verbalize_subgraph,
    format_llm_prompt,
)
from k3_node.rag.encoders import (
    RGCNSubGraphEncoder,
    TransEPrefixEncoder,
    GNNSubGraphEncoder,
)
from k3_node.rag.projector import (
    GraphPrefixProjector,
    KGLLMConnector,
)
from k3_node.rag.pipeline import (
    GraphRAG,
)

__all__ = [
    # Subgraph Extraction
    "SubgraphResult",
    "extract_subgraph",
    "KGEntityRetriever",
    # Verbalization & Prompts
    "subgraph_to_triples",
    "verbalize_subgraph",
    "format_llm_prompt",
    # GNN & KGE Encoders
    "RGCNSubGraphEncoder",
    "TransEPrefixEncoder",
    "GNNSubGraphEncoder",
    # Connectors & Projectors
    "GraphPrefixProjector",
    "KGLLMConnector",
    # Unified Pipeline
    "GraphRAG",
]
