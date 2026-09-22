"""
Script to scan examples/<category>/*.ipynb and generate (all output is build-only;
none of it is committed, see docs/examples/ in .gitignore):
1. docs/examples/index.md: Overview page listing the example categories.
2. docs/examples/<category>.md: Gallery of examples within one category, with links
   to Colab and GitHub source.
3. docs/examples/<slug>.md: Dedicated page for each example in keras.io/examples style
   with 'View in Colab' and 'GitHub source' action buttons, parsed markdown and code cells.

The category for each example is taken directly from its parent directory under
examples/, so reorganizing the examples/ folder is all that's needed to update the
generated docs.
"""
import glob
import json
import os
import re

REPO_URL = "https://github.com/anas-rz/k3-node/blob/main"
COLAB_URL = "https://colab.research.google.com/github/anas-rz/k3-node/blob/main"

# Display metadata for each examples/<slug>/ subdirectory.
CATEGORY_META = {
    "node_classification": {
        "title": "Node Classification",
        "icon": ":material-graph:",
        "description": "Core GNN layer demos that classify nodes in a single graph (mostly Cora).",
    },
    "large_scale_training": {
        "title": "Large-Scale & Scalable Training",
        "icon": ":material-server-network:",
        "description": "Sampling, partitioning, and OGB-scale techniques for training GNNs on big graphs.",
    },
    "inductive_learning": {
        "title": "Inductive Learning",
        "icon": ":material-transit-connection-variant:",
        "description": "Learning on the PPI multi-graph benchmark where test graphs are unseen at train time.",
    },
    "link_prediction": {
        "title": "Link Prediction",
        "icon": ":material-vector-link:",
        "description": "Predicting edges: static, dynamic, and signed link prediction.",
    },
    "knowledge_graphs": {
        "title": "Knowledge Graphs",
        "icon": ":material-graph-outline:",
        "description": "Knowledge graph embeddings, relational/entity classification, and relational databases.",
    },
    "graph_classification": {
        "title": "Graph Classification",
        "icon": ":material-shape-outline:",
        "description": "Whole-graph classification with pooling methods on TU datasets and MNIST superpixels.",
    },
    "molecular_property_prediction": {
        "title": "Molecular Property Prediction",
        "icon": ":material-molecule:",
        "description": "Property prediction on molecular graphs (QM9, ZINC, MoleculeNet).",
    },
    "point_cloud_3d": {
        "title": "Point Cloud & 3D",
        "icon": ":material-axis-arrow:",
        "description": "Geometric deep learning on point clouds and meshes.",
    },
    "representation_learning": {
        "title": "Representation Learning",
        "icon": ":material-vector-combine:",
        "description": "Unsupervised and self-supervised graph representation learning.",
    },
    "clustering": {
        "title": "Clustering",
        "icon": ":material-scatter-plot:",
        "description": "Unsupervised node and graph clustering.",
    },
    "utilities_and_misc": {
        "title": "Utilities & Misc",
        "icon": ":material-toolbox-outline:",
        "description": "Training infrastructure and other examples that don't fit a single GNN task.",
    },
}


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


def get_category(path: str) -> str:
    """Derive the category slug from the notebook's parent directory under examples/."""
    rel = os.path.relpath(path, "examples")
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else ""


def get_example_meta(path: str) -> dict:
    filename = os.path.basename(path)
    stem = os.path.splitext(filename)[0]
    backend = get_backend(path)
    category = get_category(path)

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
        "category": category,
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


def generate_category_page(category: str, examples_list: list) -> str:
    meta = CATEGORY_META.get(category, {
        "title": category.replace("_", " ").title(),
        "icon": ":material-folder-outline:",
        "description": "",
    })

    lines = [
        f"# {meta['title']}",
        "",
        meta["description"],
        "",
        "[:octicons-arrow-left-24: All categories](index.md)",
        "",
        "---",
        "",
        '<div class="grid cards" markdown>',
        "",
    ]

    for ex in sorted(examples_list, key=lambda e: e["title"]):
        lines.append(f"-   {ex['icon']} __{ex['title']}__")
        lines.append("")
        lines.append(f"    {ex['description']}")
        lines.append("")
        lines.append(f"    `{ex['dataset']}` · `{ex['layer']}`")
        lines.append("")
        lines.append(
            f"    [:octicons-arrow-right-24: Read Tutorial]({ex['doc_file']}){{ .md-button .md-button--primary }} &nbsp; "
            f"[:simple-googlecolab:]({ex['colab_url']}){{ .md-button title=\"View in Colab\" }} &nbsp; "
            f"[:octicons-mark-github-16:]({ex['github_url']}){{ .md-button title=\"GitHub source\" }}"
        )
        lines.append("")

    lines.extend(["</div>", ""])
    return "\n".join(lines)


def generate_all():
    os.makedirs("docs/examples", exist_ok=True)
    notebook_paths = sorted(glob.glob("examples/**/*.ipynb", recursive=True))

    examples_by_category = {}
    for nb_path in notebook_paths:
        meta = get_example_meta(nb_path)
        examples_by_category.setdefault(meta["category"], []).append(meta)

        # Generate individual example page
        page_content = generate_single_example_page(nb_path, meta)
        target_path = os.path.join("docs/examples", meta["doc_file"])
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(page_content)
        print(f"Generated example page: {target_path}")

    # Generate one gallery page per category
    for category, examples_list in examples_by_category.items():
        page_content = generate_category_page(category, examples_list)
        target_path = os.path.join("docs/examples", f"{category}.md")
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(page_content)
        print(f"Generated category page: {target_path}")

    # Generate docs/examples/index.md: categories, then a flat list of all notebooks
    index_lines = [
        "# Code Examples",
        "",
        "Welcome to the **K3-Node Code Examples**. Browse by category, or jump straight "
        "to a notebook below.",
        "",
        "## Categories",
        "",
    ]

    sorted_categories = sorted(
        examples_by_category, key=lambda c: CATEGORY_META.get(c, {}).get("title", c)
    )
    for category in sorted_categories:
        examples_list = examples_by_category[category]
        meta = CATEGORY_META.get(category, {
            "title": category.replace("_", " ").title(),
            "description": "",
        })
        count = len(examples_list)
        index_lines.append(
            f"- [{meta['title']}]({category}.md) — {count} example{'s' if count != 1 else ''}"
        )

    index_lines.extend(["", "## All Examples", ""])

    all_examples = sorted(
        (ex for examples_list in examples_by_category.values() for ex in examples_list),
        key=lambda e: e["title"],
    )
    for ex in all_examples:
        category_title = CATEGORY_META.get(ex["category"], {}).get(
            "title", ex["category"].replace("_", " ").title()
        )
        index_lines.append(f"- [{ex['title']}]({ex['doc_file']}) — *{category_title}*")

    index_lines.append("")

    with open("docs/examples/index.md", "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines))
    print("Generated docs/examples/index.md (categories + notebook list).")


if __name__ == "__main__":
    generate_all()
