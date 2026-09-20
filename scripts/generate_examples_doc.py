"""
Script to scan examples/**/*.ipynb and generate:
1. docs/examples/index.md: List / gallery of examples with links to Colab and GitHub source.
2. docs/examples/<slug>.md: Dedicated page for each example in keras.io/examples style with
   'View in Colab' and 'GitHub source' action buttons, parsed markdown and code cells.
"""
import glob
import json
import os
import re

REPO_URL = "https://github.com/anas-rz/k3-node/blob/main"
COLAB_URL = "https://colab.research.google.com/github/anas-rz/k3-node/blob/main"


try:
    from scripts.convert_all_pyg_examples import METADATA
except ImportError:
    try:
        from convert_all_pyg_examples import METADATA
    except ImportError:
        METADATA = {}


def get_backend(path: str) -> str:
    path_lower = path.lower()
    if "tensorflow" in path_lower:
        return "TensorFlow"
    elif "torch" in path_lower or "pytorch" in path_lower:
        return "PyTorch"
    elif "jax" in path_lower:
        return "JAX"
    return "Multi-Backend"


def get_example_meta(path: str) -> dict:
    filename = os.path.basename(path)
    stem = os.path.splitext(filename)[0]
    backend = get_backend(path)

    if stem in METADATA:
        m = METADATA[stem]
        title = m["title"]
        description = m["desc"]
        dataset = m["dataset"]
        layer = m["layer"]
        icon = m["icon"]
    elif "arxiv" in filename.lower():
        title = "Node Classification on OGBN-Arxiv with ARMAConv"
        description = (
            "Large-scale node classification on the `ogbn-arxiv` citation benchmark "
            "using K3-Node's `ARMAConv` layer, Spektral graph preprocessing, and a custom TensorFlow training loop."
        )
        dataset = "ogbn-arxiv"
        layer = "ARMAConv"
        icon = ":material-google:"
    elif "planetoid" in filename.lower() or "cora" in filename.lower():
        title = "Node Classification on Cora with GatedGraphConv"
        description = (
            "Node classification on the standard `Planetoid Cora` citation graph "
            "using K3-Node's `GatedGraphConv` layer, PyTorch Geometric dataset loading, and PyTorch backend optimization."
        )
        dataset = "Cora (Planetoid)"
        layer = "GatedGraphConv"
        icon = ":material-fire:"
    else:
        clean_name = stem.replace("_", " ").title()
        title = f"{clean_name} ({backend})"
        description = f"Graph Neural Network example demonstrating {clean_name} with K3-Node."
        dataset = "Graph Benchmark"
        layer = "GNN"
        icon = ":material-cube-outline:"

    return {
        "title": title,
        "description": description,
        "backend": backend,
        "dataset": dataset,
        "layer": layer,
        "icon": icon,
        "filename": filename,
        "stem": stem,
        "rel_path": path,
        "doc_file": f"{stem}.md",
        "github_url": f"{REPO_URL}/{path}",
        "colab_url": f"{COLAB_URL}/{path}",
    }


def parse_cell_output(cell: dict) -> str:
    outputs = cell.get("outputs", [])
    text_chunks = []
    for out in outputs:
        otype = out.get("output_type")
        if otype == "stream":
            text = "".join(out.get("text", []))
            text_chunks.append(text)
        elif otype == "execute_result" or otype == "display_data":
            data = out.get("data", {})
            if "text/plain" in data:
                text_chunks.append("".join(data["text/plain"]))
    return "\n".join(text_chunks).strip()


def generate_single_example_page(nb_path: str, meta: dict) -> str:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    lines = []
    lines.append(f"# {meta['title']}")
    lines.append("")
    lines.append(f"**Author:** K3-Node Team<br>")
    lines.append(f"**Backend:** {meta['backend']}<br>")
    lines.append(f"**Dataset:** `{meta['dataset']}`<br>")
    lines.append(f"**Description:** {meta['description']}")
    lines.append("")
    lines.append(
        f'[:simple-googlecolab: **View in Colab**]({meta["colab_url"]}){{ .md-button .md-button--primary }} &nbsp; '
        f'[:octicons-mark-github-16: **GitHub source**]({meta["github_url"]}){{ .md-button }}'
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    cells = nb.get("cells", [])
    for idx, cell in enumerate(cells):
        ctype = cell.get("cell_type")
        src = "".join(cell.get("source", [])).strip()
        if not src:
            continue

        if ctype == "markdown":
            lines.append(src)
            lines.append("")
        elif ctype == "code":
            # Handle pip installation commands
            if src.startswith("!"):
                lines.append("## Setup & Installation")
                lines.append("")
                lines.append("```bash")
                for line in src.splitlines():
                    if line.startswith("!"):
                        lines.append(line.lstrip("!"))
                    else:
                        lines.append(line)
                lines.append("```")
                lines.append("")
                continue

            # Check if cell begins with a section comment like # Load data
            first_line = src.splitlines()[0]
            code_body = src
            if first_line.startswith("# ") and not first_line.startswith("#!"):
                heading = first_line.lstrip("# ").strip()
                lines.append(f"## {heading}")
                lines.append("")
                # Keep remaining code or full code
                code_lines = src.splitlines()[1:]
                code_body = "\n".join(code_lines).strip()
                if not code_body:
                    continue

            lines.append("```python")
            lines.append(code_body)
            lines.append("```")
            lines.append("")

            # Render output if present
            output_text = parse_cell_output(cell)
            if output_text:
                lines.append('??? example "View Output"')
                lines.append("    ```text")
                for oline in output_text.splitlines():
                    lines.append(f"    {oline}")
                lines.append("    ```")
                lines.append("")

    return "\n".join(lines)


def generate_all():
    os.makedirs("docs/examples", exist_ok=True)
    notebook_paths = sorted(glob.glob("examples/**/*.ipynb", recursive=True))

    examples_list = []
    for nb_path in notebook_paths:
        meta = get_example_meta(nb_path)
        examples_list.append(meta)

        # Generate individual example page
        page_content = generate_single_example_page(nb_path, meta)
        target_path = os.path.join("docs/examples", meta["doc_file"])
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(page_content)
        print(f"Generated example page: {target_path}")

    # Generate main docs/examples/index.md (ONLY list of examples)
    index_lines = [
        "# Code Examples",
        "",
        "Welcome to the **K3-Node Code Examples**. This page indexes end-to-end runnable tutorials demonstrating how to build, train, and evaluate Graph Neural Networks with K3-Node across multiple frameworks and backends.",
        "",
        "---",
        "",
        "## Available Examples",
        "",
        '<div class="grid cards" markdown>',
        "",
    ]

    for ex in examples_list:
        index_lines.append(f"-   {ex['icon']} __{ex['title']}__")
        index_lines.append("")
        index_lines.append("    ---")
        index_lines.append("")
        index_lines.append(f"    {ex['description']}")
        index_lines.append("")
        index_lines.append(f"    - **Backend**: {ex['backend']}")
        index_lines.append(f"    - **Dataset**: `{ex['dataset']}`")
        index_lines.append(f"    - **Key Layer**: `{ex['layer']}`")
        index_lines.append("")
        index_lines.append(
            f"    [:octicons-arrow-right-24: Read Tutorial]({ex['doc_file']}){{ .md-button .md-button--primary }} &nbsp; "
            f"[:simple-googlecolab: View in Colab]({ex['colab_url']}){{ .md-button }} &nbsp; "
            f"[:octicons-mark-github-16: GitHub source]({ex['github_url']}){{ .md-button }}"
        )
        index_lines.append("")

    index_lines.extend([
        "</div>",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "| Backend | Example | Dataset | Key Layer | Colab | Source |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for ex in examples_list:
        index_lines.append(
            f"| **{ex['backend']}** | [{ex['title']}]({ex['doc_file']}) | `{ex['dataset']}` | `{ex['layer']}` | "
            f"[Open in Colab]({ex['colab_url']}) | [GitHub source]({ex['github_url']}) |"
        )

    index_lines.append("")

    with open("docs/examples/index.md", "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines))
    print("Generated docs/examples/index.md (gallery of examples).")


if __name__ == "__main__":
    generate_all()
