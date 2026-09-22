#!/usr/bin/env python
"""Convert examples/*.ipynb from the 5-cell (markdown/code/markdown/code/markdown)
layout to a 2-cell Colab-friendly layout:

  Cell 1 [markdown]: title + one paragraph explaining the problem, approach,
                      and what the code cell does.
  Cell 2 [code]:      pip installs followed immediately by the full K3-Node
                       implementation (dataset load through evaluation).

Notebooks already in the 2-cell form are skipped.
"""
import json
import re
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def extract_meta(md_source: str) -> dict:
    title_match = re.search(r"^#\s+(.+)$", md_source, re.MULTILINE)
    task_match = re.search(r"\*\*Task:\*\*\s*(.+)", md_source)
    dataset_match = re.search(r"\*\*Dataset:\*\*\s*(.+)", md_source)
    layer_match = re.search(r"\*\*Key Layer/Model:\*\*\s*(.+)", md_source)
    desc_match = re.search(r"\*\*Description:\*\*\s*(.+)", md_source)

    def clean(s):
        if s is None:
            return None
        return s.strip().rstrip("  ").strip()

    return {
        "title": clean(title_match.group(1)) if title_match else None,
        "task": clean(task_match.group(1)) if task_match else None,
        "dataset": clean(dataset_match.group(1)) if dataset_match else None,
        "layer": clean(layer_match.group(1)) if layer_match else None,
        "desc": clean(desc_match.group(1)) if desc_match else None,
    }


def extract_code_facts(code_source: str) -> dict:
    model_classes = re.findall(r"^class\s+(\w+)\(keras\.Model\)", code_source, re.MULTILINE)
    optimizer_match = re.search(r"keras\.optimizers\.(\w+)\(", code_source)
    epochs_matches = re.findall(r"epochs=(\d+)", code_source)
    dataset_calls = re.findall(r"= (\w+)\(\s*root=", code_source)
    return {
        "model_class": model_classes[0] if model_classes else None,
        "optimizer": optimizer_match.group(1) if optimizer_match else None,
        "epochs": epochs_matches[-1] if epochs_matches else None,
        "dataset_class": dataset_calls[0] if dataset_calls else None,
    }


def strip_trailing_period(s: str) -> str:
    return s.rstrip(".").strip()


def build_paragraph(meta: dict, facts: dict) -> str:
    title = meta["title"] or "K3-Node Example"
    task = meta["task"] or "A graph learning task"
    dataset = (meta["dataset"] or "a graph dataset").strip("`")
    layer = (meta["layer"] or "a K3-Node layer").strip("`")
    desc = strip_trailing_period(meta["desc"] or "")

    parts = []
    parts.append(
        f"{task} on {dataset}"
        + (f": {desc}." if desc else ".")
    )

    approach = f"This notebook implements the approach with `{layer}`"
    if facts["model_class"]:
        approach += f" inside a `{facts['model_class']}` model"
    if facts["optimizer"] and facts["epochs"]:
        approach += f", trained with the {facts['optimizer']} optimizer for {facts['epochs']} epochs"
    elif facts["optimizer"]:
        approach += f", trained with the {facts['optimizer']} optimizer"
    elif facts["epochs"]:
        approach += f" for {facts['epochs']} epochs"
    approach += ", evaluating the result on held-out data."
    parts.append(approach)

    parts.append(
        "The single code cell below installs **K3-Node**, loads the dataset, "
        f"defines the model using K3-Node's `{layer}` on **Keras 3**, compiles and trains it, "
        "and reports the resulting metric — the same code runs unchanged on the PyTorch, "
        "TensorFlow, or JAX backend by switching the `KERAS_BACKEND` environment variable."
    )

    return f"# {title}\n\n" + " ".join(parts)


def convert(nb_path: Path) -> bool:
    nb = json.loads(nb_path.read_text())
    cells = nb["cells"]
    types = tuple(c["cell_type"] for c in cells)
    if types == ("markdown", "code"):
        return False  # already converted
    if types != ("markdown", "code", "markdown", "code", "markdown"):
        print(f"SKIP (unexpected shape {types}): {nb_path.name}", file=sys.stderr)
        return False

    md0 = "".join(cells[0]["source"])
    install_src = "".join(cells[1]["source"])
    impl_src = "".join(cells[3]["source"])

    meta = extract_meta(md0)
    facts = extract_code_facts(impl_src)

    install_lines = [l for l in install_src.split("\n") if l.strip().startswith("!pip")]
    merged_code = (
        "# Setup environment and install dependencies\n"
        + "\n".join(install_lines)
        + "\n\n"
        + impl_src
    )

    paragraph = build_paragraph(meta, facts)

    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": cells[0].get("metadata", {}),
            "source": [paragraph],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": cells[3].get("metadata", cells[1].get("metadata", {})),
            "outputs": [],
            "source": [merged_code],
        },
    ]
    nb["cells"] = new_cells
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return True


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    notebooks = sorted(EXAMPLES_DIR.glob("*.ipynb"))
    if only:
        wanted = set(only)
        notebooks = [n for n in notebooks if n.stem in wanted or n.name in wanted]

    converted, skipped = 0, 0
    for nb_path in notebooks:
        if convert(nb_path):
            converted += 1
            print(f"converted: {nb_path.name}")
        else:
            skipped += 1
    print(f"\n{converted} converted, {skipped} skipped")


if __name__ == "__main__":
    main()
