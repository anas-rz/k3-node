# GraphRAG & KG-LLM Connectors

The `k3_node.rag` module provides a comprehensive suite of tools for connecting Knowledge Graphs to Large Language Models (LLMs) such as Meta Llama 3 and Mistral.

---

## 1. Subgraph Extraction

### extract_subgraph
::: k3_node.rag.extract_subgraph

**Usage Example:**
```python
import keras
from keras import ops
from k3_node.rag import extract_subgraph

# A small knowledge graph with 5 nodes and 6 directed edges
edge_index = ops.convert_to_tensor([[0, 0, 2, 3, 4, 4], [1, 2, 3, 1, 1, 2]], dtype="int32")
edge_type = ops.convert_to_tensor([0, 1, 2, 3, 0, 1], dtype="int32")
x = keras.random.normal((5, 16))

# Extract 1-hop enclosing subgraph around node 0
subgraph = extract_subgraph(entities=[0], edge_index=edge_index, edge_type=edge_type, x=x, num_hops=1)
print("Nodes in subgraph:", subgraph.nodes)
print("Subgraph edges:", subgraph.edge_index.shape)
```

---

### KGEntityRetriever
::: k3_node.rag.KGEntityRetriever

**Usage Example:**
```python
from k3_node.rag import KGEntityRetriever

entity_to_id = {"Aspirin": 0, "Headache": 1, "COX-1": 2}
relation_to_id = {"treats": 0, "inhibits": 1}

retriever = KGEntityRetriever(
    entity_to_id=entity_to_id,
    relation_to_id=relation_to_id,
    edge_index=edge_index,
    edge_type=edge_type,
)

# Extract entities mentioned in free-form text query
matched_entities = retriever.find_entities_in_text("What treats Headache?")
print(matched_entities)  # ['Headache']

# Retrieve multi-hop neighborhood
subgraph = retriever.retrieve_subgraph(matched_entities, num_hops=2)
```

---

## 2. Graph Verbalization & Prompt Formatting

### verbalize_subgraph
::: k3_node.rag.verbalize_subgraph

**Usage Example:**
```python
from k3_node.rag import verbalize_subgraph

# Markdown format
markdown_context = verbalize_subgraph(
    subgraph,
    id_to_entity={0: "Aspirin", 1: "Headache", 2: "COX-1"},
    id_to_relation={0: "treats", 1: "inhibits"},
    format_style="markdown",
)
print(markdown_context)
```

---

### format_llm_prompt
::: k3_node.rag.format_llm_prompt

**Usage Example:**
```python
from k3_node.rag import format_llm_prompt

# Llama 3 prompt formatting
prompt = format_llm_prompt(
    query="How does Aspirin treat Headache?",
    context=markdown_context,
    model_family="llama3",
)
print(prompt)
```

---

## 3. Subgraph GNN & KGE Encoders

### RGCNSubGraphEncoder
::: k3_node.rag.RGCNSubGraphEncoder

**Usage Example:**
```python
from k3_node.rag import RGCNSubGraphEncoder

encoder = RGCNSubGraphEncoder(
    in_channels=16,
    hidden_channels=32,
    out_channels=64,
    num_relations=4,
    num_layers=2,
    pooling="center",
)

# Encode subgraph
emb = encoder.encode_subgraph(subgraph)
print("Graph representation shape:", emb.shape)  # (1, 64)
```

---

### TransEPrefixEncoder
::: k3_node.rag.TransEPrefixEncoder

**Usage Example:**
```python
from k3_node.rag import TransEPrefixEncoder
from k3_node.layers.kge import TransE

# Option A: reuse a pretrained TransE model
transe = TransE(num_nodes=100, num_relations=10, hidden_channels=32)
encoder = TransEPrefixEncoder(kge_model=transe, out_channels=64)

# Option B: standalone encoder
encoder = TransEPrefixEncoder(num_nodes=100, num_relations=10, embedding_dim=32, out_channels=64)
emb = encoder.encode_subgraph(subgraph)
```

---

## 4. Connectors & Virtual Prefix Projectors

### GraphPrefixProjector
::: k3_node.rag.GraphPrefixProjector

**Usage Example:**
```python
import keras
from k3_node.rag import GraphPrefixProjector

projector = GraphPrefixProjector(
    in_channels=64,
    llm_dim=4096,           # Llama 3 / Mistral embedding dimension
    num_prefix_tokens=8,    # Number of soft prefix tokens
    projector_type="mlp",
)

graph_embedding = keras.random.normal((1, 64))
prefix_tokens = projector(graph_embedding)
print(prefix_tokens.shape)  # (1, 8, 4096)
```

---

### KGLLMConnector
::: k3_node.rag.KGLLMConnector

**Usage Example:**
```python
import keras
from k3_node.rag import RGCNSubGraphEncoder, GraphPrefixProjector, KGLLMConnector

encoder = RGCNSubGraphEncoder(in_channels=16, hidden_channels=32, out_channels=64, num_relations=5)
projector = GraphPrefixProjector(in_channels=64, llm_dim=4096, num_prefix_tokens=4)
connector = KGLLMConnector(encoder=encoder, projector=projector)

# Encode extracted subgraph directly into LLM prefix tokens
prefix_embeds = connector.encode_subgraph(subgraph)  # [1, 4, 4096]

# Inject prefix into LLM text token embeddings
text_token_embeds = keras.random.normal((1, 15, 4096))
augmented_embeds = connector.inject_prefix(text_token_embeds, prefix_embeds)
print(augmented_embeds.shape)  # (1, 19, 4096)
```

---

## 5. Unified End-to-End Pipeline

### GraphRAG
::: k3_node.rag.GraphRAG

**Usage Example:**
```python
from k3_node.rag import GraphRAG

rag = GraphRAG(
    edge_index=edge_index,
    edge_type=edge_type,
    entity_to_id={"Aspirin": 0, "Headache": 1, "COX-1": 2},
    relation_to_id={"treats": 0, "inhibits": 1},
    llm_dim=4096,
    num_prefix_tokens=4,
)

# Text-based GraphRAG prompt
prompt = rag.build_prompt("What does Aspirin treat?", model_family="llama3")

# Soft prompt prefix encoding
subgraph = rag.retrieve(["Aspirin"], num_hops=1)
prefix_embeds = rag.encode_prefix(subgraph)  # (1, 4, 4096)
```
