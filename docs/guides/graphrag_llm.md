# Supercharging Llama 3 / Mistral with K3-Node Knowledge Graph Embeddings

Modern Large Language Models (LLMs) like **Meta Llama 3** and **Mistral 7B** excel at generative reasoning, but frequently struggle with factual precision, hallucinations, and multi-hop relational reasoning over enterprise and domain-specific knowledge bases.

**K3-Node** provides native **GraphRAG & KG-LLM connectors** under `k3_node.rag`, enabling you to augment LLMs using two complementary paradigms:

1. **Text-based GraphRAG**: Automatically retrieve multi-hop subgraphs around query entities, verbalize them into structured markdown facts, and inject them into prompt templates (Llama 3, Mistral, ChatML).
2. **Soft Prompt / Prefix Injection (KG-LLM)**: Encode relational subgraphs with Relational GNNs (RGCN) or Knowledge Graph Embeddings (TransE) and project them through a `GraphPrefixProjector` into continuous prefix vectors (shape `[1, num_prefix_tokens, 4096]`) prepended directly to the LLM's embedding space.

```
       [ User Query ]
             │
             ▼
   [ Entity Recognition ]
             │
             ▼
[ Subgraph Extraction (k3_node.rag) ]
      ┌──────┴─────────────────────────────────┐
      ▼                                        ▼
[ Text-based Verbalization ]          [ GNN / KGE Encoder ]
(Markdown / Natural Triples)           (RGCN / TransE)
      │                                        │
      ▼                                        ▼
[ Llama 3 / Mistral Prompt ]          [ GraphPrefixProjector ]
(Structured System Context)           (Virtual Tokens: [1, K, 4096])
      │                                        │
      └──────────────────┬─────────────────────┘
                         ▼
        [ Augmented LLM Forward Pass / Generation ]
```

---

## 1. Defining a Knowledge Graph

Let's build a sample biomedical knowledge graph connecting drugs, targets, biological pathways, and diseases:

```python
import numpy as np
import keras
from keras import ops
import k3_node as k3
from k3_node.rag import GraphRAG, extract_subgraph, verbalize_subgraph

# Entities and Relations
entity_to_id = {
    "Aspirin": 0,
    "Headache": 1,
    "COX-1": 2,
    "Prostaglandin": 3,
    "Ibuprofen": 4,
    "Inflammation": 5,
}

relation_to_id = {
    "treats": 0,
    "inhibits": 1,
    "synthesizes": 2,
    "causes": 3,
}

# Graph Edges (head, tail) and Relation Types
edges = np.array([
    [0, 0, 2, 3, 3, 4, 4],  # Head entities
    [1, 2, 3, 1, 5, 1, 2],  # Tail entities
], dtype=np.int64)

edge_type = np.array([0, 1, 2, 3, 3, 0, 1], dtype=np.int64)
edge_index = ops.convert_to_tensor(edges, dtype="int32")
edge_type = ops.convert_to_tensor(edge_type, dtype="int32")

# Optional node features (e.g. text embeddings of entity descriptions)
x = keras.random.normal((6, 64))
```

---

## 2. Subgraph Extraction Around Retrieved Entities

When a user asks a question, we first identify the mentioned entities and extract the multi-hop relational subgraph around them:

```python
from k3_node.rag import KGEntityRetriever

retriever = KGEntityRetriever(
    entity_to_id=entity_to_id,
    relation_to_id=relation_to_id,
    edge_index=edge_index,
    edge_type=edge_type,
    x=x,
)

# Step 1: Detect entities in user query
query = "How does Aspirin alleviate headache and inflammation?"
found_entities = retriever.find_entities_in_text(query)
print("Detected Entities:", found_entities)
# Output: ['Aspirin', 'Inflammation', 'Headache']

# Step 2: Extract 2-hop enclosing subgraph
subgraph = retriever.retrieve_subgraph(found_entities, num_hops=2)
print(f"Extracted Subgraph: {subgraph.num_nodes} nodes, {subgraph.num_edges} edges")
```

The returned `SubgraphResult` contains:
- `edge_index`: Relabeled subgraph connectivity.
- `edge_type`: Preserved relation IDs.
- `nodes`: Original global node IDs.
- `center_nodes`: Indices of the query entities in the subgraph.
- `subgraph.to_data()`: Exports directly as a `k3_node.data.Data` object!

---

## 3. Text-Based GraphRAG with Llama 3 & Mistral

Text-based GraphRAG verbalizes the extracted subgraph into clean markdown facts or natural sentences and wraps them into standard LLM instruction formats.

### Verbalizing Subgraph Facts

```python
from k3_node.rag import verbalize_subgraph, format_llm_prompt

# Verbalize into markdown facts
context_md = verbalize_subgraph(
    subgraph=subgraph,
    id_to_entity={v: k for k, v in entity_to_id.items()},
    id_to_relation={v: k for k, v in relation_to_id.items()},
    format_style="markdown",
)
print(context_md)
```
**Output:**
```markdown
### Retrieved Knowledge Graph Facts:
- **Aspirin** — *treats* -> **Headache**
- **Aspirin** — *inhibits* -> **COX-1**
- **COX-1** — *synthesizes* -> **Prostaglandin**
- **Prostaglandin** — *causes* -> **Headache**
- **Prostaglandin** — *causes* -> **Inflammation**
```

### Formatting LLM Prompts

#### For Meta Llama 3 / 3.1:
```python
llama3_prompt = format_llm_prompt(
    query=query,
    context=context_md,
    model_family="llama3",
)
print(llama3_prompt)
```
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert assistant augmented with a Knowledge Graph. Use the provided Knowledge Graph Context to accurately answer the user's question.<|eot_id|><|start_header_id|>user<|end_header_id|>

Knowledge Graph Context:
### Retrieved Knowledge Graph Facts:
- **Aspirin** — *treats* -> **Headache**
- **Aspirin** — *inhibits* -> **COX-1**
- **COX-1** — *synthesizes* -> **Prostaglandin**
- **Prostaglandin** — *causes* -> **Headache**
- **Prostaglandin** — *causes* -> **Inflammation**

Question: How does Aspirin alleviate headache and inflammation?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

#### For Mistral 7B / Mixtral:
```python
mistral_prompt = format_llm_prompt(
    query=query,
    context=context_md,
    model_family="mistral",
)
```
```
<s>[INST] You are an expert assistant augmented with a Knowledge Graph...

Knowledge Graph Context:
### Retrieved Knowledge Graph Facts:
...

Question: How does Aspirin alleviate headache and inflammation? [/INST]
```

---

## 4. Soft Prompting / Prefix Injection (KG-LLM Connectors)

Rather than converting graph topology to text tokens (which consumes context window and loses continuous structural information), **KG-LLM soft prompting** encodes the relational subgraph into dense **virtual prefix tokens** directly compatible with the LLM's embedding space.

For example, **Llama-3-8B** and **Mistral-7B** both use an embedding dimension of $D_{\text{LLM}} = 4096$.

### Method A: Relational GNN Encoder (RGCN)

`RGCNSubGraphEncoder` applies multi-relational graph convolutions:

```python
from k3_node.rag import RGCNSubGraphEncoder, GraphPrefixProjector, KGLLMConnector

# Step 1: RGCN multi-relational backbone
encoder = RGCNSubGraphEncoder(
    in_channels=64,
    hidden_channels=128,
    out_channels=256,
    num_relations=len(relation_to_id),
    num_layers=2,
    pooling="center",  # Focuses on the retrieved seed entities
)

# Step 2: Projector mapping 256-d graph embedding to 8 Llama 3 prefix tokens
projector = GraphPrefixProjector(
    in_channels=256,
    llm_dim=4096,           # Llama 3 / Mistral embedding dimension
    num_prefix_tokens=8,    # 8 virtual tokens prepended to prompt
    projector_type="mlp",
)

# Step 3: Combined connector
connector = KGLLMConnector(encoder=encoder, projector=projector)

# Encode subgraph into prefix embeddings: shape [1, 8, 4096]
prefix_embeddings = connector.encode_subgraph(subgraph)
print("Virtual prefix shape:", prefix_embeddings.shape)
# Output: (1, 8, 4096)
```

### Method B: Knowledge Graph Embedding Encoder (TransE)

If you already trained a TransE or RotatE model on your KG, you can reuse its geometric embeddings:

```python
from k3_node.layers.kge import TransE
from k3_node.rag import TransEPrefixEncoder

# Pretrained TransE model
transe = TransE(num_nodes=len(entity_to_id), num_relations=len(relation_to_id), hidden_channels=64)

# Wrap into KGE prefix encoder
kge_encoder = TransEPrefixEncoder(
    kge_model=transe,
    out_channels=256,
    pooling="center",
)

connector_kge = KGLLMConnector(
    encoder=kge_encoder,
    projector=GraphPrefixProjector(in_channels=256, llm_dim=4096, num_prefix_tokens=8),
)

prefix_embeddings = connector_kge.encode_subgraph(subgraph)  # [1, 8, 4096]
```

---

## 5. Injecting Prefix Vectors into Hugging Face Llama 3 / Mistral

Here is how you inject the K3-Node prefix vectors into a Hugging Face `transformers` pipeline:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # or "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# 1. Tokenize query
text_inputs = tokenizer(query, return_tensors="pt").to(model.device)

# 2. Get input word embeddings from LLM embedding layer
with torch.no_grad():
    token_embeds = model.get_input_embeddings()(text_inputs["input_ids"])  # [1, seq_len, 4096]

# 3. Convert K3-Node prefix embeddings to PyTorch tensor
prefix_tensor = torch.from_numpy(np.array(prefix_embeddings)).to(model.device, dtype=token_embeds.dtype)

# 4. Concatenate: [Prefix Tokens (8)] + [Text Tokens (seq_len)]
augmented_embeds = KGLLMConnector.inject_prefix(token_embeds, prefix_tensor)
extended_mask = KGLLMConnector.extend_attention_mask(text_inputs["attention_mask"], num_prefix_tokens=8)

# 5. Generate with graph-augmented embeddings!
output_ids = model.generate(
    inputs_embeds=augmented_embeds,
    attention_mask=extended_mask,
    max_new_tokens=150,
    temperature=0.2,
)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("LLM Response:\n", response)
```

---

## 6. End-to-End One-Stop Pipeline with `GraphRAG`

For convenience, `k3_node.rag.GraphRAG` bundles retrieval, verbalization, and prefix encoding into a single unified pipeline:

```python
from k3_node.rag import GraphRAG

# Initialize pipeline
rag = GraphRAG(
    edge_index=edge_index,
    edge_type=edge_type,
    entity_to_id=entity_to_id,
    relation_to_id=relation_to_id,
    x=x,
    encoder_type="rgcn",
    hidden_dim=64,
    encoder_out_dim=256,
    llm_dim=4096,
    num_prefix_tokens=8,
)

# 1. Retrieve & Verbalize for text prompt
prompt = rag.build_prompt(
    query="Explain the biological pathway of Aspirin and COX-1",
    model_family="llama3",
)

# 2. Extract & Encode prefix vectors for soft prompt
subgraph = rag.retrieve(["Aspirin", "COX-1"], num_hops=2)
prefix_embeds = rag.encode_prefix(subgraph)  # [1, 8, 4096]
```
