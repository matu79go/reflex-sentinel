"""Data for the figures: per run, the layer-1 score of each step (point A, from a monitor that never saw that task) and a
short action label made from the tool call (task text is not included). Labels are Japanese keys; make_figures.py
translates them for --lang en.
"""
import argparse, json, re

import numpy as np

from common import edits_test


def label(c, res):
    if not c:
        return "考えるだけ"
    if edits_test(c):
        return "テストを書き換える"
    res = str(res or "")
    if c.get("tool") == "str_replace_editor":
        p = str(c.get("path", "")).split("/")[-1]
        cmd = c.get("command")
        if cmd == "view":
            return {"test_solution.py": "テストを読む", "solution.py": "答えを読む"}.get(p, "ファイルを見る")
        if p.startswith("solution"):
            return "答えを書く" if cmd == "create" else "答えを直す"
        return "調べる"
    s = str(c.get("command", ""))
    if "doctest" in s:
        return "例題で確認 → 不合格" if "FAIL" in res or "Error" in res else "例題で確認 → 合格"
    if "test_solution" in s or re.search(r"python3? -c", s):
        if "AssertionError" in res or "Error" in res or "FAIL" in res:
            return "テストを実行 → 不合格"
        return "テストを実行 → 合格" if "passed" in res.lower() else "動かしてみる"
    if "> /testbed/solution" in s or "solution.py <<" in s:
        return "答えを書く"
    if s.startswith(("find", "ls", "cat")):
        return "ファイルを見る"
    return "調べる"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="data/agent_runs.jsonl")
    ap.add_argument("--scores", default="data/linear_scores_A.json")
    ap.add_argument("--llm", default="data/llm_scores.jsonl")
    ap.add_argument("--out", default="data/heatmap_data.json")
    a = ap.parse_args()
    sc = json.load(open(a.scores))
    llm = {}
    for j in map(json.loads, open(a.llm)):
        if j["point"] == 0:
            llm[(j["id"], j["turn"])] = j["score"]
    eps = []
    for e in map(json.loads, open(a.gen)):
        if e["id"] not in sc:
            continue
        tc = next((s["turn"] for s in e["steps"] if edits_test(s["cmd"])), None)
        cheat = bool(e["test_tampered"] and tc is not None)
        if e["test_tampered"] and not cheat:  # changed the test but the step is unclear: excluded, as in the evaluation
            continue
        n = len(sc[e["id"]])
        eps.append({"id": e["id"], "task": e["task_id"], "cheat": cheat, "tc": tc if cheat else None,
                    "score": sc[e["id"]], "llm": [llm.get((e["id"], t), -1) for t in range(n)],
                    "act": [label(s["cmd"], s.get("result")) for s in e["steps"]][:n]})
    ref = np.sort(np.concatenate([e["score"] for e in eps if not e["cheat"]]))
    for e in eps:
        e["risk"] = [float(np.searchsorted(ref, x) / len(ref)) for x in e["score"]]
        e["llm_risk"] = [x / 10 if x >= 0 else None for x in e["llm"]]
    json.dump({"eps": eps}, open(a.out, "w"), ensure_ascii=False)
    print(f"{len(eps)} runs ({sum(e['cheat'] for e in eps)} concealed) -> {a.out}")


if __name__ == "__main__":
    main()
