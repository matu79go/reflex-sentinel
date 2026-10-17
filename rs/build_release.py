"""How the Hugging Face dataset was built from the raw experiment files (kept for transparency).
  - runs/*.jsonl : agent runs and resampled retries, task text replaced by markers (redact.py)
  - features/    : hidden states averaged over loop-1 layers 41-71 (1-indexed), [steps, 2 points, 5120], float16
  - scores/      : layer-1 scores (point A, out-of-fold), layer-3 LLM scores, and monitors trained on all runs
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--branch", nargs="+", required=True)
    ap.add_argument("--feat", required=True, help="full hidden states from extract.py ([steps, 2, 160, 5120])")
    ap.add_argument("--llm", required=True)
    ap.add_argument("--band", default="40-70")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    for d in ("runs", "features", "scores"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    py = [sys.executable, f"{here}/redact.py", "redact", "--tasks", a.tasks]
    subprocess.run(py + ["--inp", a.runs, "--out", f"{a.out}/runs/agent_runs.jsonl"], check=True)
    for b in a.branch:
        subprocess.run(py + ["--inp", b, "--out", f"{a.out}/runs/{os.path.basename(b).replace('branch_', 'branch_runs_')}"], check=True)
    l0, l1 = map(int, a.band.split("-"))
    with open(f"{a.out}/features/index.jsonl", "w") as idx:
        for line in open(f"{a.feat}/index.jsonl"):
            j = json.loads(line)
            arr = np.load(f"{a.feat}/{j['id']}.npy", mmap_mode="r")[:, :, l0:l1 + 1].astype(np.float32).mean(2).astype(np.float16)
            np.save(f"{a.out}/features/{j['id']}.npy", arr)
            idx.write(json.dumps({**j, "shape": list(arr.shape), "band": f"loop1 layers {l0}-{l1} (0-indexed), mean"}) + "\n")
    with open(f"{a.out}/scores/llm_scores.jsonl", "w") as f:
        for line in open(a.llm):
            f.write(line)
    print("done")


if __name__ == "__main__":
    main()
