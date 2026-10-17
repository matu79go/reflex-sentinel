"""Task text in the released runs is replaced by markers, because the task data has no stated license.

Every line of a task's prompt / test / original_test that is at least MIN_LEN characters long (after stripping) is
replaced, wherever it appears in a run (messages, commands, tool outputs), by ⟦ilcb:<field>:<line>⟧. Restoring puts the
exact text back from the public task data, so the restored runs are identical to the originals.

  python redact.py restore --tasks data/tasks.jsonl --inp agent_runs.redacted.jsonl --out data/agent_runs.jsonl
"""
import argparse
import json
import re

MIN_LEN = 16
FIELDS = ("prompt", "test", "original_test")
MARK = re.compile(r"⟦ilcb:(prompt|test|original_test):(\d+)⟧")


def table(task):
    t = {}
    for f in FIELDS:
        for i, line in enumerate(task[f].split("\n")):
            s = line.strip()
            if len(s) >= MIN_LEN and s not in t:
                t[s] = f"⟦ilcb:{f}:{i}⟧"
    return t


def walk(x, fn):
    if isinstance(x, str):
        return fn(x)
    if isinstance(x, list):
        return [walk(v, fn) for v in x]
    if isinstance(x, dict):
        return {k: walk(v, fn) for k, v in x.items()}
    return x


def redactor(task):
    t = table(task)
    if not t:
        return lambda s: s
    pat = re.compile("|".join(re.escape(k) for k in sorted(t, key=len, reverse=True)))
    return lambda s: pat.sub(lambda m: t[m.group(0)], s)


def restorer(task):
    return lambda s: MARK.sub(lambda m: task[m.group(1)].split("\n")[int(m.group(2))].strip(), s)


def task_of(rec, tasks):
    tid = rec.get("task_id") or rec["id"].split("__")[1]
    split = rec.get("split") or "conflicting"
    return tasks[(tid, split)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["redact", "restore"])
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--inp", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = {(t["task_id"], t["split"]): t for t in map(json.loads, open(a.tasks))}
    n = 0
    with open(a.out, "w") as f:
        for path in a.inp:
            for line in open(path):
                rec = json.loads(line)
                task = task_of(rec, tasks)
                fn = redactor(task) if a.mode == "redact" else restorer(task)
                keep = {k: rec[k] for k in ("id", "task_id", "split") if k in rec}
                f.write(json.dumps({**walk(rec, fn), **keep}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{a.mode}: {n} records -> {a.out}")


if __name__ == "__main__":
    main()
