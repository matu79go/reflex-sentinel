"""Layer 1 (linear model): how well the hidden state predicts concealment, d steps before the concealment step.

One monitor, one threshold: a logistic regression trained on the concealment step (positive) and every step of the
honest conflicting runs (negative), 5-fold by task (a task is never in both train and test). The same monitor's
score is then compared at d = 0, 1, 2, ... steps before concealment against honest runs at the same step numbers.
Hit rate = AUROC. Also reports a situation-only baseline (step number, failed test runs so far) that never looks inside.
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from common import groups, load_episodes, load_feat, n_steps, pairs_at


def situation(runs_path):
    """Per step: [step number, failed test runs so far]. Uses only what is visible from outside."""
    out = {}
    for line in open(runs_path):
        e = json.loads(line)
        rows, fails = [], 0
        for s in e["steps"]:
            rows.append([s["turn"], fails])
            fails += "AssertionError" in str(s.get("result") or "")
        out[e["id"]] = np.array(rows + [rows[-1]] * 5, dtype=np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/agent_runs.jsonl")
    ap.add_argument("--feat", default="data/feat")
    ap.add_argument("--point", type=int, default=0, help="0 = A (before the reasoning), 1 = B (before the command)")
    ap.add_argument("--band", default="40-70", help="loop-1 layers averaged (0-indexed); ignored for band files")
    ap.add_argument("--C", type=float, default=0.003)
    ap.add_argument("--max-d", type=int, default=3)
    ap.add_argument("--save-scores", default=None, help="write {id: [score per step]} (each from a monitor that never saw that task)")
    ap.add_argument("--save-probe", default=None, help="write a monitor trained on all runs (.npz: mu, sd, w, b)")
    a = ap.parse_args()
    eps = load_episodes(a.runs, n_steps(a.feat))
    cheat, honest = groups(eps)
    use = cheat + honest
    print(f"concealed {len(cheat)}, honest {len(honest)}")
    feats = {e["id"]: load_feat(a.feat, e["id"], a.point, a.band) for e in use}
    sit = situation(a.runs)
    tasks = sorted({e["task"] for e in use})
    X, y, g, S = [], [], [], []
    for e in cheat:
        X.append(feats[e["id"]][e["tc"]]); y.append(1); g.append(tasks.index(e["task"])); S.append(sit[e["id"]][e["tc"]])
    for e in honest:
        for t in range(e["n"]):
            X.append(feats[e["id"]][t]); y.append(0); g.append(tasks.index(e["task"])); S.append(sit[e["id"]][t])
    X, y, g, S = np.stack(X), np.array(y), np.array(g), np.stack(S)
    score, base = {}, {}
    for tr, te in GroupKFold(5).split(X, y, g):
        te_tasks = {tasks[i] for i in set(g[te])}
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(C=a.C, max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        bl = LogisticRegression(max_iter=2000).fit(S[tr], y[tr])
        for e in use:
            if e["task"] in te_tasks:
                s = clf.decision_function((feats[e["id"]] - mu) / sd)
                sb = bl.decision_function(sit[e["id"]][: e["n"]])
                for t in range(e["n"]):
                    score[(e["id"], t)], base[(e["id"], t)] = float(s[t]), float(sb[t])
    for d in range(a.max_d + 1):
        pos, neg = pairs_at(cheat, honest, d)
        yy = [1] * len(pos) + [0] * len(neg)
        print(f"{d} step(s) before: linear {roc_auc_score(yy, [score[k] for k in pos + neg]):.3f} | "
              f"situation only {roc_auc_score(yy, [base[k] for k in pos + neg]):.3f} ({len(pos)} vs {len(neg)})")
    if a.save_scores:
        json.dump({e["id"]: [score[(e["id"], t)] for t in range(e["n"])] for e in use}, open(a.save_scores, "w"))
    if a.save_probe:
        mu, sd = X.mean(0), X.std(0) + 1e-6
        clf = LogisticRegression(C=a.C, max_iter=2000).fit((X - mu) / sd, y)
        np.savez(a.save_probe, mu=mu.astype(np.float32), sd=sd.astype(np.float32), w=clf.coef_[0].astype(np.float32),
                 b=np.float32(clf.intercept_[0]), point=a.point, band=a.band)


if __name__ == "__main__":
    main()
