"""Tests for k3_node.rag GraphRAG and KG-LLM connectors."""

import numpy as np
import pytest
from keras import ops

import k3_node as k3
from k3_node.layers.kge import TransE
from k3_node.rag import (
    GraphPrefixProjector,
    GraphRAG,
    KGEntityRetriever,
    KGLLMConnector,
    RGCNSubGraphEncoder,
    SubgraphResult,
    TransEPrefixEncoder,
    extract_subgraph,
    format_llm_prompt,
    subgraph_to_triples,
    verbalize_subgraph,
)


@pytest.fixture
def sample_kg():
    """Create a sample knowledge graph for testing.

    Graph:
        0 (Aspirin) --[0: treats]--> 1 (Headache)
        0 (Aspirin) --[1: inhibits]--> 2 (COX-1)
        2 (COX-1) --[2: produces]--> 3 (Prostaglandin)
        3 (Prostaglandin) --[3: causes]--> 1 (Headache)
        4 (Ibuprofen) --[0: treats]--> 1 (Headache)
        4 (Ibuprofen) --[1: inhibits]--> 2 (COX-1)
    """
    edges = np.array(
        [
            [0, 0, 2, 3, 4, 4],
            [1, 2, 3, 1, 1, 2],
        ],
        dtype=np.int64,
    )
    edge_type = np.array([0, 1, 2, 3, 0, 1], dtype=np.int64)
    x = np.random.randn(5, 16).astype(np.float32)

    entity_to_id = {
        "Aspirin": 0,
        "Headache": 1,
        "COX-1": 2,
        "Prostaglandin": 3,
        "Ibuprofen": 4,
    }
    relation_to_id = {
        "treats": 0,
        "inhibits": 1,
        "produces": 2,
        "causes": 3,
    }

    return {
        "edge_index": ops.convert_to_tensor(edges, dtype="int32"),
        "edge_type": ops.convert_to_tensor(edge_type, dtype="int32"),
        "x": ops.convert_to_tensor(x, dtype="float32"),
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
        "num_nodes": 5,
        "num_relations": 4,
    }


def test_extract_subgraph_1hop(sample_kg):
    sub = extract_subgraph(
        entities=[0],  # Aspirin
        edge_index=sample_kg["edge_index"],
        edge_type=sample_kg["edge_type"],
        x=sample_kg["x"],
        num_hops=1,
        directed=False,
    )

    assert isinstance(sub, SubgraphResult)
    # Neighbors of 0 within 1 hop: 0, 1, 2
    assert set(sub.nodes.tolist()) == {0, 1, 2}
    assert sub.center_nodes.tolist() == [sub.mapping[0]]
    assert sub.num_nodes == 3
    assert sub.num_edges > 0
    assert sub.x is not None
    assert ops.shape(sub.x)[0] == 3

    # Check to_data()
    data = sub.to_data()
    assert isinstance(data, k3.data.Data)
    assert ops.shape(data.edge_index)[0] == 2


def test_extract_subgraph_2hop_multiseed(sample_kg):
    sub = extract_subgraph(
        entities=[0, 3],  # Aspirin & Prostaglandin
        edge_index=sample_kg["edge_index"],
        edge_type=sample_kg["edge_type"],
        num_hops=2,
    )
    assert sub.num_nodes == 5
    assert len(sub.center_nodes) == 2


def test_kg_entity_retriever(sample_kg):
    retriever = KGEntityRetriever(
        entity_to_id=sample_kg["entity_to_id"],
        relation_to_id=sample_kg["relation_to_id"],
        edge_index=sample_kg["edge_index"],
        edge_type=sample_kg["edge_type"],
        x=sample_kg["x"],
    )

    # Lookup
    assert retriever.get_entity_id("Aspirin") == 0
    assert retriever.get_entity_id("aspirin") == 0  # case-insensitive
    assert retriever.get_entity_id("Unknown") is None

    # Text entity extraction
    query = "Does Aspirin or Ibuprofen treat headache?"
    found = retriever.find_entities_in_text(query)
    assert "Aspirin" in found
    assert "Ibuprofen" in found
    assert "Headache" in found

    # Subgraph retrieval
    sub = retriever.retrieve_subgraph(["Aspirin"], num_hops=1)
    assert sub.num_nodes >= 2


def test_verbalization(sample_kg):
    sub = extract_subgraph(
        entities=[0],
        edge_index=sample_kg["edge_index"],
        edge_type=sample_kg["edge_type"],
        num_hops=1,
    )

    id_to_e = {v: k for k, v in sample_kg["entity_to_id"].items()}
    id_to_r = {v: k for k, v in sample_kg["relation_to_id"].items()}

    # Triples list
    triples = subgraph_to_triples(sub, id_to_e, id_to_r)
    assert len(triples) > 0
    assert ("Aspirin", "treats", "Headache") in triples

    # Markdown format
    md_text = verbalize_subgraph(sub, id_to_e, id_to_r, format_style="markdown")
    assert "**Aspirin**" in md_text
    assert "*treats*" in md_text

    # Natural language format
    natural_text = verbalize_subgraph(sub, id_to_e, id_to_r, format_style="natural")
    assert "Aspirin treats Headache." in natural_text


def test_prompt_formatting():
    context = "- **Aspirin** — *treats* -> **Headache**"
    query = "What treats headache?"

    # Llama 3
    llama3_prompt = format_llm_prompt(query, context, model_family="llama3")
    assert "<|start_header_id|>system<|end_header_id|>" in llama3_prompt
    assert "<|start_header_id|>user<|end_header_id|>" in llama3_prompt
    assert "Knowledge Graph Context:" in llama3_prompt
    assert query in llama3_prompt

    # Mistral
    mistral_prompt = format_llm_prompt(query, context, model_family="mistral")
    assert "<s>[INST]" in mistral_prompt
    assert "[/INST]" in mistral_prompt

    # ChatML
    chatml_prompt = format_llm_prompt(query, context, model_family="chatml")
    assert "<|im_start|>system" in chatml_prompt


def test_rgcn_subgraph_encoder(sample_kg):
    encoder = RGCNSubGraphEncoder(
        in_channels=16,
        hidden_channels=32,
        out_channels=64,
        num_relations=sample_kg["num_relations"],
        num_layers=2,
        pooling="center",
    )

    sub = extract_subgraph(
        entities=[0],
        edge_index=sample_kg["edge_index"],
        edge_type=sample_kg["edge_type"],
        x=sample_kg["x"],
        num_hops=1,
    )

    emb = encoder.encode_subgraph(sub)
    assert ops.shape(emb) == (1, 64)

    # Test other pooling options
    encoder_mean = RGCNSubGraphEncoder(
        in_channels=16,
        hidden_channels=32,
        out_channels=64,
        num_relations=sample_kg["num_relations"],
        pooling="mean",
    )
    emb_mean = encoder_mean.encode_subgraph(sub)
    assert ops.shape(emb_mean) == (1, 64)


def test_transe_prefix_encoder(sample_kg):
    transe = TransE(
        num_nodes=sample_kg["num_nodes"],
        num_relations=sample_kg["num_relations"],
        hidden_channels=32,
    )

    kge_encoder = TransEPrefixEncoder(
        kge_model=transe,
        out_channels=64,
        pooling="center",
    )

    sub = extract_subgraph(
        entities=[0, 1],
        edge_index=sample_kg["edge_index"],
        edge_type=sample_kg["edge_type"],
        num_hops=1,
    )

    emb = kge_encoder.encode_subgraph(sub)
    assert ops.shape(emb) == (1, 64)

    # Test standalone initialization without pretrained model
    standalone_encoder = TransEPrefixEncoder(
        num_nodes=10,
        num_relations=4,
        embedding_dim=32,
        out_channels=64,
    )
    emb_standalone = standalone_encoder.encode_subgraph(sub)
    assert ops.shape(emb_standalone) == (1, 64)


def test_graph_prefix_projector():
    projector = GraphPrefixProjector(
        in_channels=64,
        llm_dim=256,  # test dimension
        num_prefix_tokens=4,
        projector_type="mlp",
    )

    graph_emb = ops.convert_to_tensor(np.random.randn(1, 64).astype(np.float32))
    prefix = projector(graph_emb)
    assert ops.shape(prefix) == (1, 4, 256)

    # Linear projector
    lin_projector = GraphPrefixProjector(
        in_channels=64,
        llm_dim=256,
        num_prefix_tokens=4,
        projector_type="linear",
    )
    prefix_lin = lin_projector(graph_emb)
    assert ops.shape(prefix_lin) == (1, 4, 256)


def test_kg_llm_connector(sample_kg):
    encoder = RGCNSubGraphEncoder(
        in_channels=16,
        hidden_channels=32,
        out_channels=64,
        num_relations=sample_kg["num_relations"],
    )
    projector = GraphPrefixProjector(
        in_channels=64,
        llm_dim=512,
        num_prefix_tokens=4,
    )
    connector = KGLLMConnector(encoder=encoder, projector=projector)

    sub = extract_subgraph(
        entities=[0],
        edge_index=sample_kg["edge_index"],
        edge_type=sample_kg["edge_type"],
        x=sample_kg["x"],
        num_hops=1,
    )

    prefix_tokens = connector.encode_subgraph(sub)
    assert ops.shape(prefix_tokens) == (1, 4, 512)

    # Test prefix injection into LLM text tokens
    text_tokens = ops.convert_to_tensor(np.random.randn(1, 10, 512).astype(np.float32))
    augmented = connector.inject_prefix(text_tokens, prefix_tokens)
    assert ops.shape(augmented) == (1, 14, 512)

    # Test attention mask extension
    attn_mask = ops.ones((1, 10), dtype="int32")
    extended_mask = connector.extend_attention_mask(attn_mask, num_prefix_tokens=4)
    assert ops.shape(extended_mask) == (1, 14)


def test_graph_rag_pipeline(sample_kg):
    # Pipeline with RGCN
    rag_rgcn = GraphRAG(
        edge_index=sample_kg["edge_index"],
        edge_type=sample_kg["edge_type"],
        entity_to_id=sample_kg["entity_to_id"],
        relation_to_id=sample_kg["relation_to_id"],
        x=sample_kg["x"],
        encoder_type="rgcn",
        hidden_dim=32,
        encoder_out_dim=64,
        llm_dim=256,
        num_prefix_tokens=4,
    )

    # Retrieval
    sub = rag_rgcn.retrieve(["Aspirin"], num_hops=1)
    assert sub.num_nodes >= 2

    # Prompt builder
    prompt = rag_rgcn.build_prompt("What does Aspirin treat?", model_family="llama3")
    assert "Aspirin" in prompt
    assert "<|begin_of_text|>" in prompt

    # Prefix encoding
    prefix = rag_rgcn.encode_prefix(sub)
    assert ops.shape(prefix) == (1, 4, 256)

    # Pipeline with TransE
    rag_transe = GraphRAG(
        edge_index=sample_kg["edge_index"],
        edge_type=sample_kg["edge_type"],
        entity_to_id=sample_kg["entity_to_id"],
        relation_to_id=sample_kg["relation_to_id"],
        encoder_type="transe",
        hidden_dim=32,
        encoder_out_dim=64,
        llm_dim=256,
        num_prefix_tokens=4,
    )
    prefix_transe = rag_transe.encode_prefix(sub)
    assert ops.shape(prefix_transe) == (1, 4, 256)
