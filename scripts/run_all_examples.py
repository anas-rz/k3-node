#!/usr/bin/env python
"""Execute every examples/*.ipynb K3-Node implementation cell under a given
Keras backend, logging pass/fail/timeout per notebook.

Usage:
    python scripts/run_all_examples.py <backend> [--timeout SECONDS] [--only name1,name2]

Writes per-notebook logs to logs/examples_run/<notebook>.<backend>.log and
extracted scripts to logs/examples_run/<notebook>.<backend>.py, and appends
results to logs/examples_run/summary.jsonl (one JSON object per line) so
progress can be inspected while the sweep is still running.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
LOG_DIR = ROOT / "logs" / "examples_run"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = LOG_DIR / "summary.jsonl"
_summary_lock = Lock()


def extract_code(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text())
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    src = "".join(code_cells[-1]["source"])
    # Strip Jupyter/Colab shell-magic lines (e.g. "!pip install ..."), which
    # are valid in a notebook cell but not in a plain .py script.
    lines = [l for l in src.split("\n") if not l.strip().startswith("!")]
    return "\n".join(lines)


def run_one(nb_path: Path, backend: str, timeout: int) -> dict:
    name = nb_path.stem
    code = extract_code(nb_path)
    script_path = LOG_DIR / f"{name}.{backend}.py"
    script_path.write_text(code)
    log_path = LOG_DIR / f"{name}.{backend}.log"

    env = os.environ.copy()
    env["KERAS_BACKEND"] = backend
    env["MPLBACKEND"] = "Agg"
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

    start = time.time()
    result = {"notebook": name, "backend": backend}
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(ROOT),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        elapsed = time.time() - start
        result["elapsed"] = round(elapsed, 1)
        result["status"] = "PASS" if proc.returncode == 0 else "FAIL"
    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
        result["elapsed"] = timeout
    except Exception as e:  # pragma: no cover
        result["status"] = "ERROR"
        result["elapsed"] = round(time.time() - start, 1)
        result["exception"] = str(e)

    with _summary_lock:
        with open(SUMMARY_PATH, "a") as sf:
            sf.write(json.dumps(result) + "\n")
    print(f"{result['status']:8s} {backend:10s} {name} ({result.get('elapsed','-')}s)", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=["torch", "tensorflow", "jax"])
    ap.add_argument("--timeout", type=int, default=480)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", type=str, default=None, help="comma-separated notebook stems")
    args = ap.parse_args()

    notebooks = sorted(EXAMPLES_DIR.glob("*.ipynb"))
    if args.only:
        wanted = set(args.only.split(","))
        notebooks = [n for n in notebooks if n.stem in wanted]

    if args.workers <= 1:
        for nb_path in notebooks:
            run_one(nb_path, args.backend, args.timeout)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, nb_path, args.backend, args.timeout) for nb_path in notebooks]
            for _ in as_completed(futs):
                pass

    print(f"DONE backend={args.backend}", flush=True)


if __name__ == "__main__":
    main()
