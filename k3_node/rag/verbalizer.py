"""Verbalization and LLM prompt formatting for GraphRAG."""

from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
import numpy as np
from keras import ops

from k3_node.rag.subgraph import SubgraphResult


def subgraph_to_triples(
    subgraph: SubgraphResult,
    id_to_entity: Optional[Dict[int, str]] = None,
    id_to_relation: Optional[Dict[int, str]] = None,
) -> List[Tuple[str, str, str]]:
    """Convert a SubgraphResult into a list of (head, relation, tail) string triples.

    Args:
        subgraph: The extracted SubgraphResult.
        id_to_entity: Optional map from original entity integer ID to entity string.
        id_to_relation: Optional map from relation integer ID to relation string.

    Returns:
        List of `(head, relation, tail)` string triples.
    """
    if subgraph.num_edges == 0:
        return []

    edge_index_np = np.asarray(ops.convert_to_numpy(subgraph.edge_index)).astype(np.int64)
    row, col = edge_index_np[0], edge_index_np[1]

    # Map subgraph indices back to original node IDs if mapping/nodes available
    if subgraph.nodes is not None and len(subgraph.nodes) > 0:
        orig_row = subgraph.nodes[row]
        orig_col = subgraph.nodes[col]
    else:
        orig_row, orig_col = row, col

    edge_type_np = None
    if subgraph.edge_type is not None:
        edge_type_np = np.asarray(ops.convert_to_numpy(subgraph.edge_type)).astype(np.int64)

    triples = []
    for i in range(len(row)):
        h_id = int(orig_row[i])
        t_id = int(orig_col[i])
        h_name = id_to_entity.get(h_id, f"Node_{h_id}") if id_to_entity else f"Node_{h_id}"
        t_name = id_to_entity.get(t_id, f"Node_{t_id}") if id_to_entity else f"Node_{t_id}"

        if edge_type_np is not None:
            r_id = int(edge_type_np[i])
            r_name = id_to_relation.get(r_id, f"rel_{r_id}") if id_to_relation else f"rel_{r_id}"
        else:
            r_name = "connected_to"

        triples.append((h_name, r_name, t_name))

    return triples


def verbalize_subgraph(
    subgraph: SubgraphResult,
    id_to_entity: Optional[Dict[int, str]] = None,
    id_to_relation: Optional[Dict[int, str]] = None,
    format_style: Literal["triples", "markdown", "natural"] = "markdown",
    max_triples: Optional[int] = 50,
) -> str:
    """Verbalize an extracted subgraph into textual knowledge for LLM prompt augmentation.

    Args:
        subgraph: Extracted SubgraphResult around retrieved entities.
        id_to_entity: Optional dictionary mapping node ID to entity name.
        id_to_relation: Optional dictionary mapping relation ID to relation name.
        format_style:
            - `"triples"`: List of `(Head, Relation, Tail)` text triples.
            - `"markdown"`: Markdown bullet list with facts.
            - `"natural"`: Natural language sentences (`"Head relation Tail."`).
        max_triples: Maximum number of facts to include in the context.

    Returns:
        Formatted textual context string ready to be injected into an LLM prompt.
    """
    triples = subgraph_to_triples(subgraph, id_to_entity, id_to_relation)
    if not triples:
        return "No relevant knowledge graph facts retrieved."

    if max_triples is not None and len(triples) > max_triples:
        triples = triples[:max_triples]

    if format_style == "triples":
        lines = [f"({h}, {r}, {t})" for h, r, t in triples]
        return "\n".join(lines)

    elif format_style == "natural":
        lines = []
        for h, r, t in triples:
            # Clean relation string (replace underscores with spaces)
            rel_str = r.replace("_", " ")
            lines.append(f"{h} {rel_str} {t}.")
        return " ".join(lines)

    else:  # "markdown"
        lines = ["### Retrieved Knowledge Graph Facts:"]
        for h, r, t in triples:
            lines.append(f"- **{h}** — *{r}* -> **{t}**")
        return "\n".join(lines)


def format_llm_prompt(
    query: str,
    context: str,
    system_prompt: Optional[str] = None,
    model_family: Literal["llama3", "mistral", "chatml", "standard"] = "llama3",
) -> str:
    """Format query and retrieved KG context into prompt templates for LLMs.

    Supported model families include:
    - `"llama3"`: Meta Llama 3 / 3.1 instruct template.
    - `"mistral"`: Mistral / Mixtral instruct template.
    - `"chatml"`: OpenAI / Qwen ChatML template.
    - `"standard"`: General markdown system/user format.

    Args:
        query: The user's input question or instruction.
        context: The verbalized knowledge graph context.
        system_prompt: Optional system prompt to instruct the LLM.
        model_family: Target model prompt format. (default: "llama3")

    Returns:
        Formatted prompt string.
    """
    if system_prompt is None:
        system_prompt = (
            "You are an expert assistant augmented with a Knowledge Graph. "
            "Use the provided Knowledge Graph Context to accurately answer the user's question."
        )

    if model_family == "llama3":
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Knowledge Graph Context:\n{context}\n\n"
            f"Question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif model_family == "mistral":
        return (
            f"<s>[INST] {system_prompt}\n\n"
            f"Knowledge Graph Context:\n{context}\n\n"
            f"Question: {query} [/INST]"
        )
    elif model_family == "chatml":
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\nKnowledge Graph Context:\n{context}\n\nQuestion: {query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:  # "standard"
        return (
            f"System: {system_prompt}\n\n"
            f"Knowledge Graph Context:\n{context}\n\n"
            f"User Question: {query}\n\n"
            f"Assistant:"
        )
