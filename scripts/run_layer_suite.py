import os
import sys
import subprocess
import json
import re

CHECKLIST_PATH = "LAYERS_CHECKLIST.md"

def update_checklist(layer_name, backend_results):
    """
    Updates the row for `layer_name` in LAYERS_CHECKLIST.md.
    backend_results: dict mapping 'torch', 'tensorflow', 'jax' to result dict.
    """
    if not os.path.exists(CHECKLIST_PATH):
        return

    with open(CHECKLIST_PATH, "r") as f:
        content = f.read()

    # Determine status strings
    status_map = {}
    learning_verified = True
    notes = []

    for b in ["torch", "tensorflow", "jax"]:
        res = backend_results.get(b, {})
        status = res.get("status", "UNTESTED")
        if status == "PASS":
            l0 = res.get("initial_loss", 0.0)
            l1 = res.get("final_loss", 0.0)
            red = res.get("loss_reduced", False)
            if not red:
                learning_verified = False
            status_map[b] = f"[x] PASS ({l0:.2f}->{l1:.2f})"
        elif status == "SKIP":
            status_map[b] = "[-] SKIP"
            reason = res.get("reason", "")
            if reason and reason not in notes:
                notes.append(reason)
            learning_verified = False
        elif status == "FAIL":
            err = res.get("error", "Unknown error").replace("\n", " ")
            status_map[b] = "[ ] FAIL"
            notes.append(f"{b} error: {err[:50]}...")
            learning_verified = False
        else:
            status_map[b] = "[ ] Untested"
            learning_verified = False

    learn_str = "[x]" if learning_verified else ("[-]" if all(r.get("status") == "SKIP" for r in backend_results.values()) else "[ ]")
    notes_str = "; ".join(notes)

    with open(CHECKLIST_PATH, "r") as f:
        lines = f.readlines()

    out_lines = []
    updated = False
    for line in lines:
        if line.startswith("| `") and f"`{layer_name}`" in line:
            parts = [p.strip() for p in line.split("|")]
            # parts: ['', '`layer_name`', '`module`', 'torch', 'tf', 'jax', 'learn', 'notes', '']
            if len(parts) >= 8 and parts[1] == f"`{layer_name}`":
                parts[3] = f" {status_map.get('torch', '[ ]')} "
                parts[4] = f" {status_map.get('tensorflow', '[ ]')} "
                parts[5] = f" {status_map.get('jax', '[ ]')} "
                parts[6] = f" {learn_str} "
                parts[7] = f" {notes_str} " if notes_str else " "
                line = "|".join(parts) + "\n"
                updated = True
        out_lines.append(line)

    if updated:
        with open(CHECKLIST_PATH, "w") as f:
            f.writelines(out_lines)


def update_summary_stats():
    """Recalculates and updates the summary statistics table in LAYERS_CHECKLIST.md."""
    if not os.path.exists(CHECKLIST_PATH):
        return

    with open(CHECKLIST_PATH, "r") as f:
        lines = f.readlines()

    # Parse categories and their table rows
    current_cat = None
    cat_stats = {}

    for line in lines:
        if line.startswith("## "):
            cat_name = line.replace("## ", "").strip()
            if cat_name != "Overall Progress Summary":
                current_cat = cat_name
                cat_stats[current_cat] = {"total": 0, "tested": 0, "passed": 0, "failed": 0, "skip": 0}
        elif current_cat and line.strip().startswith("|") and not line.strip().startswith("| :---") and not line.strip().startswith("| Layer Name") and not line.strip().startswith("| Category"):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 7:
                cat_stats[current_cat]["total"] += 1
                torch_st = parts[2]
                tf_st = parts[3]
                jax_st = parts[4]
                if any("[x] PASS" in st for st in [torch_st, tf_st, jax_st]):
                    cat_stats[current_cat]["tested"] += 1
                    if all("[x] PASS" in st for st in [torch_st, tf_st, jax_st]):
                        cat_stats[current_cat]["passed"] += 1
                    else:
                        cat_stats[current_cat]["failed"] += 1
                elif any("FAIL" in st for st in [torch_st, tf_st, jax_st]):
                    cat_stats[current_cat]["tested"] += 1
                    cat_stats[current_cat]["failed"] += 1
                elif any("SKIP" in st for st in [torch_st, tf_st, jax_st]):
                    cat_stats[current_cat]["skip"] += 1

    # Rewrite summary section
    out_lines = []
    in_summary = False
    for line in lines:
        if line.startswith("## Overall Progress Summary"):
            in_summary = True
            out_lines.append(line)
            out_lines.append("\n")
            out_lines.append("| Category | Total | Tested | Passed | Failed | Base/Abstract/Vendor |\n")
            out_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for cat, st in cat_stats.items():
                out_lines.append(f"| {cat} | {st['total']} | {st['tested']} | {st['passed']} | {st['failed']} | {st['skip']} |\n")
            continue
        if in_summary:
            if line.startswith("---"):
                in_summary = False
                out_lines.append(line)
            continue
        out_lines.append(line)

    with open(CHECKLIST_PATH, "w") as f:
        f.writelines(out_lines)


def run_layer(layer_name):
    print(f"\n==========================================")
    print(f"Testing Layer: {layer_name}")
    print(f"==========================================")
    results = {}
    for b in ["torch", "tensorflow", "jax"]:
        env = os.environ.copy()
        env["KERAS_BACKEND"] = b
        if "USE_GPU" not in env:
            env["CUDA_VISIBLE_DEVICES"] = ""
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        cmd = ["env/bin/python", "scripts/test_layer.py", layer_name]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        
        # Parse JSON from last line of stdout
        lines = [l.strip() for l in proc.stdout.strip().split("\n") if l.strip()]
        res = None
        for l in reversed(lines):
            try:
                res = json.loads(l)
                break
            except Exception:
                continue
        
        if res is None:
            res = {"status": "FAIL", "error": proc.stderr.strip() or proc.stdout.strip()}

        results[b] = res
        status = res.get("status")
        if status == "PASS":
            l0 = res.get("initial_loss")
            l1 = res.get("final_loss")
            print(f"  [{b:10s}] PASS: loss {l0:.4f} -> {l1:.4f} (reduced: {res.get('loss_reduced')})")
        elif status == "SKIP":
            print(f"  [{b:10s}] SKIP: {res.get('reason')}")
        else:
            err = res.get("error", "")
            print(f"  [{b:10s}] FAIL: {err[:100]}")

    update_checklist(layer_name, results)
    update_summary_stats()
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        layers_to_run = sys.argv[1:]
    else:
        # Run all layers found in LAYERS_CHECKLIST.md
        layers_to_run = []
        with open(CHECKLIST_PATH) as f:
            for line in f:
                if line.startswith("| `"):
                    l_name = line.split("`")[1]
                    layers_to_run.append(l_name)

    print(f"Loaded {len(layers_to_run)} layers to test.")
    for idx, layer_name in enumerate(layers_to_run, 1):
        print(f"\nProgress: {idx}/{len(layers_to_run)}")
        run_layer(layer_name)

