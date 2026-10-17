"""Layer 2 (JEPA): a world model of how the hidden state moves from step to step, trained on honest runs only
(honest conflicting runs + original runs). It never sees a concealment example. The energy of a step = how far the
actual state is from the predicted one. 5-fold by task; scored like eval_linear.py.
"""
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from common import groups, jepa_fit, load_episodes, load_feat, n_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/agent_runs.jsonl")
    ap.add_argument("--feat", default="data/feat")
    ap.add_argument("--point", type=int, default=0)
    ap.add_argument("--band", default="40-70")
    ap.add_argument("--k", type=int, default=2, help="number of previous steps the predictor sees")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--max-d", type=int, default=3)
    a = ap.parse_args()
    eps = load_episodes(a.runs, n_steps(a.feat))
    cheat, honest = groups(eps)
    orig = [e for e in eps if e["split"] == "original"]
    feats = {e["id"]: load_feat(a.feat, e["id"], a.point, a.band) for e in cheat + honest + orig}
    tasks = sorted({e["task"] for e in cheat + honest})
    score = {}
    for tr, te in GroupKFold(5).split(np.zeros(len(tasks)), groups=np.arange(len(tasks))):
        trs, tes = {tasks[i] for i in tr}, {tasks[i] for i in te}
        energy = jepa_fit([feats[e["id"]] for e in honest + orig if e["task"] in trs], a.k, a.dim)
        for e in cheat + honest:
            if e["task"] in tes:
                en = energy(feats[e["id"]])
                for t in range(e["n"]):
                    score[(e["id"], t)] = float(en[t - a.k]) if t >= a.k else float(np.median(en))
    for d in range(a.max_d + 1):
        pos = [(e["id"], e["tc"] - d) for e in cheat if e["tc"] - d >= a.k]  # needs k earlier steps
        turns = {t for _, t in pos}
        neg = [(e["id"], t) for e in honest for t in turns if t < e["n"]]
        yy = [1] * len(pos) + [0] * len(neg)
        print(f"{d} step(s) before: JEPA {roc_auc_score(yy, [score[k] for k in pos + neg]):.3f} ({len(pos)} vs {len(neg)})")


if __name__ == "__main__":
    main()
