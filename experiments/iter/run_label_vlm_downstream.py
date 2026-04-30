"""One-shot driver for the label/ y-lookup VLM downstream eval.

Stages (each idempotent / cached):
  1. Generate QA from GT polylines  -> label_qa.json
  2. Extract curves via pipeline    -> curves_cache/<id>.json
  3. Run Gemini Conditions A and B  -> llm_cache/<qid>__{A,B}.json
  4. Score & write summary          -> label_results.csv, label_summary.md

Usage:
    python experiments/iter/run_label_vlm_downstream.py [--stage all|qa|extract|llm|score]
    python experiments/iter/run_label_vlm_downstream.py --pilot 20
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import os

KMEANS_ROOT = Path(os.environ.get("KMEANS_ROOT", Path(__file__).resolve().parents[2]))
PY = sys.executable


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(KMEANS_ROOT))
    return p.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "qa", "extract", "llm", "score"],
                    default="all")
    ap.add_argument("--pilot", type=int, default=0,
                    help="run on first N images only (cap)")
    ap.add_argument("--workers_extract", type=int, default=2)
    ap.add_argument("--workers_llm", type=int, default=8)
    args = ap.parse_args()

    base_qa = [PY, "experiments/iter/qa_generators/gen_label_qa.py"]
    base_ext = [PY, "experiments/iter/extract_label_curves.py",
                "--workers", str(args.workers_extract)]
    base_llm = [PY, "experiments/iter/vlm_qa_label.py",
                "--workers", str(args.workers_llm)]
    base_score = [PY, "experiments/iter/score_label.py"]

    if args.pilot > 0:
        base_ext += ["--limit", str(args.pilot)]
        base_llm += ["--limit", str(args.pilot)]

    if args.stage in ("all", "qa"):
        rc = run(base_qa)
        if rc != 0: sys.exit(rc)
    if args.stage in ("all", "extract"):
        rc = run(base_ext)
        if rc != 0: sys.exit(rc)
    if args.stage in ("all", "llm"):
        rc = run(base_llm)
        if rc != 0: sys.exit(rc)
    if args.stage in ("all", "score"):
        rc = run(base_score)
        if rc != 0: sys.exit(rc)


if __name__ == "__main__":
    main()
