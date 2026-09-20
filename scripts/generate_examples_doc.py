"""
Script to scan examples/**/*.ipynb and generate docs/examples.md
linking all interactive notebooks and embedding their complete runnable code.
"""
import glob
import json
import os

REPO_URL = "https://github.com/anas-rz/k3-node/blob/main"
COLAB_URL = "https://colab.research.google.com/github/anas-rz/k3-node/blob/main"


def get_backend(path: str) -> str:
    path_lower = path.lower()
    if "tensorflow" in path_lower:
        return "TensorFlow"
    elif "torch" in path_lower or "pytorch" in path_lower:
        return "PyTorch"
    elif "jax" in path_lower:
        return "JAX"
    return "Multi-Backend"


def get_task_description(path: str) -> str:
    filename = os.path.basename(path)
    if "arxiv" in filename.lower():
        return "Node Classification on OGBN-Arxiv using ARMAConv"
    elif "planetoid" in filename.lower() or "cora" in filename.lower():
        return "Node Classification on Cora using GatedGraphConv"
    else:
        name = os.path.splitext(filename)[0].replace("_", " ").title()
        return f"Graph Model Training: {name}"


def generate_examples_page():
    notebook_paths = sorted(glob.glob("examples/**/*.ipynb", recursive=True))

    lines = [
        "# Interactive Examples & Complete Code",
        "",
        "This page automatically indexes all interactive notebooks from the [`examples/`](https://github.com/anas-rz/k3-node/tree/main/examples) directory.",
        "You can run any notebook directly in **Google Colab**, inspect the notebook on GitHub, or copy the complete, self-contained Python code below.",
        "",
        "---",
        "",
        "## Summary of Examples",
        "",
        "| Backend | Task / Dataset | Layer / Model | Notebook | Colab |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    examples_meta = []
    for nb_path in notebook_paths:
        backend = get_backend(nb_path)
        task = get_task_description(nb_path)
        filename = os.path.basename(nb_path)
        github_link = f"{REPO_URL}/{nb_path}"
        colab_link = f"{COLAB_URL}/{nb_path}"
        colab_badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_link})"

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_blocks = []
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                code_lines = [
                    l for l in cell.get("source", []) if not l.strip().startswith("!")
                ]
                if any(l.strip() for l in code_lines):
                    code_blocks.append("".join(code_lines).strip())

        full_code = "\n\n".join(code_blocks)

        layer_info = "ARMAConv" if "arxiv" in nb_path else "GatedGraphConv"

        lines.append(
            f"| **{backend}** | {task} | `{layer_info}` | [`{filename}`]({github_link}) | {colab_badge} |"
        )

        examples_meta.append({
            "backend": backend,
            "task": task,
            "filename": filename,
            "rel_path": nb_path,
            "github_link": github_link,
            "colab_link": colab_link,
            "colab_badge": colab_badge,
            "layer_info": layer_info,
            "full_code": full_code,
        })

    lines.extend(["", "---", ""])

    for ex in examples_meta:
        b = ex["backend"]
        t = ex["task"]
        f = ex["filename"]
        rp = ex["rel_path"]
        gh = ex["github_link"]
        badge = ex["colab_badge"]

        lines.append(f"## {t} ({b})")
        lines.append("")
        lines.append(
            f"{badge} &nbsp; [View on GitHub]({gh})"
        )
        lines.append("")
        lines.append(f"**File Location**: [`{rp}`]({gh})")
        lines.append("")
        lines.append("### Complete Runnable Code")
        lines.append("")
        lines.append("```python")
        lines.append(ex["full_code"])
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    content = "\n".join(lines)
    os.makedirs("docs", exist_ok=True)
    with open("docs/examples.md", "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Generated docs/examples.md from {len(notebook_paths)} notebooks.")


if __name__ == "__main__":
    generate_examples_page()
