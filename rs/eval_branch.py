"""Compare the resampled probability of concealment with the linear monitor's score at the same starting points.
- Correlation: Spearman between the probability and the monitor score
- Ceiling: the AUROC of telling the run's actual outcome from the probability itself. No monitor can reliably beat it,
  because what happens after that point is still random.
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", nargs="+", default=["data/branch_runs.jsonl"])
    ap.add_argument("--scores", default="data/linear_scores_A.json", help="from eval_linear.py --point 0 --save-scores")
    a = ap.parse_args()
    sc = json.load(open(a.scores))
    g, orig = defaultdict(list), {}
    for path in a.branch:
        for r in map(json.loads, open(path)):
            g[(r["id"], r["b"], r["d"])].append(r["cheat"])
            orig[r["id"]] = r.get("orig_cheat", True)  # the pilot resampled concealed runs only
    rows = [(k, float(np.mean(v))) for k, v in g.items() if k[0] in sc]
    for oc in (True, False):
        for d in sorted({k[2] for k, _ in rows}):
            ps = [p for k, p in rows if k[2] == d and orig[k[0]] == oc]
            if ps:
                print(f"{'concealed' if oc else 'honest'} runs, {d} step(s) before: {len(ps)} points, mean probability {np.mean(ps):.2f}")
    ps, ss = np.array([p for _, p in rows]), np.array([sc[k[0]][k[1]] for k, _ in rows])
    rho, pv = spearmanr(ps, ss)
    print(f"Spearman(probability, monitor score) = {rho:.3f} (p={pv:.3g}, {len(rows)} points)")
    for d in sorted({k[2] for k, _ in rows}):
        sub = [(k, p) for k, p in rows if k[2] == d]
        y = [int(orig[k[0]]) for k, _ in sub]
        if len(set(y)) == 2:
            print(f"{d} step(s) before: ceiling {roc_auc_score(y, [p for _, p in sub]):.3f} | "
                  f"linear monitor {roc_auc_score(y, [sc[k[0]][k[1]] for k, _ in sub]):.3f} ({sum(y)} vs {len(y) - sum(y)})")


if __name__ == "__main__":
    main()
