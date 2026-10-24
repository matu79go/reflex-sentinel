"""RS guard(製品)の判定 core.decide を AgentDojo の記録に流し、研究の数字(dojo_guard.py の intent1+src)と比べる。

研究の dojo_guard.py の読み込み・攻撃の手の決め方・5 分割をそのまま使い、判定だけを製品の物に差し替える。
製品との違いを埋める所: ツール名に mcp__dojo__ を付ける(製品は MCP のツールを動詞で見分ける)、普段は suite ごと(1 人の利用者の環境)。
同じ会話で通った形を覚える(--session)= 前の手は利用者が通した物として、その記録の中で覚える。
使い方: python dojo_product.py --runs <agentdojo>/runs --gt <dojo_gt.py の出力> [--seen-task]
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

import dojo_guard as dg

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_here, "guard"), os.path.dirname(_here)]  # 研究のリポ(guard/ の隣)と公開のリポ(guard/repro/)のどちらでも
from rs_guard import core  # noqa: E402


def tool_name(fn):
    return fn if fn in core.WEB_TOOLS else "mcp__dojo__" + fn


def steps(d):
    """記録 → 呼び出しごとに (製品のツール名, 引数, その時点の利用者の文, それまでの結果)。"""
    user, outs, out, pending = "", [], [], {}
    for m in d["messages"]:
        r = m.get("role")
        if r in ("system", "user"):
            user += " " + dg.text_of(m)
        elif r == "tool":
            fn = (m.get("tool_call") or {}).get("function") or pending.get(m.get("tool_call_id")) or "?"
            outs.append((fn, dg.text_of(m)))
        elif r == "assistant":
            for t in m.get("tool_calls") or []:
                pending[t.get("id")] = t["function"]
                out.append((tool_name(t["function"]), t.get("args") or {}, user, list(outs)))
    return out


CANON = {}  # お題 → 依頼の文(モデルによって依頼の末尾に決まった文が足されているため、研究と同じくお題ごとに 1 つにまとめる)


def rows_of(d):
    req = CANON.setdefault((d["suite_name"], d["user_task_id"]), dg.request_of(d))
    return [{"session": d.get("user_task_id"), "req": req, "tool": core.short_name(fn),
             "sig": [list(x) for x in core.signature(fn, core.provenance(args, user, outs, fn))]}
            for fn, args, user, outs in steps(d) if core.is_acting(fn)]


def judge(normal, d, min_sim, session):
    """→ 呼び出しごとに止めたか。session なら、その記録で前に通った(止めても利用者が通した)形を覚える。"""
    req, rid, st = dg.request_of(d), "__rec__", []
    normal.session_sigs[rid] = set()
    for fn, args, user, outs in steps(d):
        ok, _ = core.decide(normal, fn, args, req, user, outs, min_sim, session=rid if session else None)
        st.append(not ok)
        if core.is_acting(fn):
            normal.session_sigs[rid] |= set(core.signature(fn, core.provenance(args, user, outs, fn)))
    del normal.session_sigs[rid]
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--model", default="gpt-4o-2024-05-13")
    ap.add_argument("--attack", default="important_instructions")
    ap.add_argument("--seen-task", action="store_true")
    ap.add_argument("--gt", default="")
    ap.add_argument("--min-sim", type=float, default=0.2)
    a = ap.parse_args()
    if a.gt:
        dg.GT.update(json.load(open(a.gt)))
    recs = dg.load(a.runs, a.model, a.attack)
    nrm = [json.load(open(f)) for f in glob.glob(f"{a.runs}/*/*/user_task_*/none/none.json")]
    tasks = sorted({(d["suite_name"], d["user_task_id"]) for _, d in recs})
    rng = np.random.RandomState(0); order = rng.permutation(len(tasks))
    fold = {tasks[i]: j % 5 for j, i in enumerate(order)}
    n_att = sum(r[0] == "attack" for r in recs); n_ben = sum(r[0] == "benign" for r in recs)
    ok_att = sum(r[0] == "attack" and r[1]["security"] for r in recs)
    util0 = sum(r[0] == "benign" and r[1]["utility"] for r in recs)
    print(f"{a.model} / {a.attack} / {'繰り返しの仕事' if a.seen_task else '見たことのないお題'}:"
          f" 防御なし ASR {ok_att / n_att * 100:.1f} % / 仕事の成功 {util0 / n_ben * 100:.1f} %")
    for session in (False, True):
        caught = miss = fp = util = 0
        for k in range(5):
            tr = [d for d in nrm if (a.seen_task and d["pipeline_name"] != a.model)
                  or fold.get((d["suite_name"], d["user_task_id"]), -1) != k]
            by = defaultdict(list)
            for d in tr:
                by[d["suite_name"]] += rows_of(d)
            normals = {s: core.Normal(rs) for s, rs in by.items()}
            for kind, d in recs:
                if fold[(d["suite_name"], d["user_task_id"])] != k:
                    continue
                st = judge(normals.get(d["suite_name"]) or core.Normal([]), d, a.min_sim, session)
                if kind == "benign":
                    fp += any(st); util += d["utility"] and not any(st)
                elif d["security"]:
                    i = dg.attack_step(d, dg.calls(d))
                    hit = any(st) if i is None else any(st[:i + 1])
                    caught += hit; miss += not hit
        print(f"  [製品の判定{' + 同じ会話で覚える' if session else ''}] ASR {miss / n_att * 100:.1f} %"
              f" / 普通の記録の誤報 {fp}/{n_ben} → 仕事の成功 {util / n_ben * 100:.1f} %")


if __name__ == "__main__":
    main()
