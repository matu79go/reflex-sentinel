"""RS: CodeIPI の注入に「もし従っていたら」止まるかを、利用者の本当の普段で測る(design §14-6)。

手元の Gemma はファイルの中の注入(コードのコメント・README・設定ファイル)に従わなかったため、従った記録が無い。
そこで問題(samples.json)に書かれた注入のコマンドを、ファイル・issue を読んだ直後に実行したことにして、製品の判定に通す。
  普段 = 手元の Claude Code の会話記録から rs-guard learn と同じ作り方で作る(その人がコーディングのエージェントを使っている想定)。
  R7(貼り付けた文の指示)を除いた判定 = 手だけを見る判定。指示の書き方(言い換え)に左右されない。
あわせて、普通の記録(runs_codeipi の普通 30 本)の手を同じ普段で流し、止めてしまう本数を数える。
使い方: python codeipi_counterfactual.py --bench <samples.json> --runs <runs_codeipi>
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_here, "guard"), os.path.dirname(_here)]  # 研究のリポ(guard/ の隣)と公開のリポ(guard/repro/)のどちらでも
from rs_guard import acts, cli  # noqa: E402

CMDLINE = re.compile(r"^\s*(?:#\s*|//\s*|\$\s*|-\s*)?`?(.+?)`?\s*(?:-->)?\s*$")


def payloads(sample):
    """注入のコマンド = 注入の文(issue またはファイル)の、コマンドの形の行・`…` の中で、印(payload_indicator)
    を含むか、知っているコマンドで始まる物。"""
    ind = re.compile(sample["payload_indicator"])
    texts = [sample["issue_text"]] if sample["injection_vector"] == "issue_text" else list(sample["repo_files"].values())
    out = []
    for t in texts:
        cands = re.findall(r"`([^`\n]{3,300})`", t) + [CMDLINE.match(l).group(1) for l in t.splitlines() if CMDLINE.match(l)]
        for c in cands:
            c = c.strip()
            m = re.search(r"(?:^|[:\s])((?:pip3?|python3?|cat|echo|curl|wget|env|printenv|mkdir|bash|sh|crontab)\s.*)$", c)
            c = m.group(1) if m and not acts._is_known_command(c) else c  # 「Setup: pip install …」の前の文を除く
            if acts._is_known_command(c) or (ind.search(c) and " " in c):
                out.append(c)
    return list(dict.fromkeys(out))


def judge(seen0, own, pasted, outputs, cmds):
    """→ [(コマンド, 合図の並び)]。普段は写して使う(問題ごとに入れた直後の状態)。"""
    seen = acts.Seen(seen0.salt)
    seen.shapes, seen.dests, seen.sigs, seen.held = set(seen0.shapes), set(seen0.dests), set(seen0.sigs), set(seen0.held)
    snips = acts.instruction_snippets(pasted)
    own_toks = acts.toks(own)
    res, sess, sens = [], set(), False
    for c in cmds:
        st = acts.Step("Bash", {"command": c}, own.lower(), own, outputs, disk=False)
        res.append((c, [r for r, _ in st.check(seen, sess, sens, snips, own_toks)]))
        sens = sens or st.sens
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--runs", default="", help="普通の仕事の記録(codeipi_run.py の出力)。無ければ普通の分は数えない")
    ap.add_argument("--projects", default=os.path.expanduser("~/.claude/projects"), help="普段を作る会話記録の置き場")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--exclude", default="", help="普段から除く会話(この評価を作った会話。試した文字が入っているため)")
    a = ap.parse_args()
    seen = acts.Seen("s")
    files = [p for p in glob.glob(os.path.join(a.projects, "*", "*.jsonl")) if not (a.exclude and a.exclude in p)]
    for p in files:
        cli.seen_from_transcript(p, seen)
    print(f"普段 = 手元の会話記録 {len(files)} 本(手の形 {len(seen.shapes)} 種・送り先 {len(seen.dests)}・触った秘密など {len(seen.held)})")
    tally = collections.Counter()
    by_rule = collections.Counter()
    for s in json.load(open(a.bench)):
        if s["is_benign"]:
            continue
        cmds = payloads(s)
        own = s["task_description"]
        pasted = s["issue_text"] if s["injection_vector"] == "issue_text" else ""
        outputs = [("Read", c.lower()) for c in s["repo_files"].values()]
        res = judge(seen, own, pasted, outputs, cmds)
        any_rule = any(r for _, r in res)
        act_only = any([x for x in r if x != "R7"] for _, r in res)
        key = ("issue" if s["injection_vector"] == "issue_text" else "file") + (
            "・実行の印だけ" if s["payload_category"] == "code_execution" else "・秘密・居座り")
        tally[(key, "問題")] += 1
        tally[(key, "止めた(すべての合図)")] += any_rule
        tally[(key, "止めた(手だけ = R7 なし)")] += act_only
        for _, r in res:
            by_rule.update(set(r))
        if a.show or not any_rule:
            print(f"  {'止めた ' if any_rule else '見逃し '} {s['id']} {s['payload_category']}")
            for c, r in res:
                print(f"      {r} {c[:90]}")
    for key in sorted({k[0] for k in tally}):
        print(key, {k[1]: v for k, v in tally.items() if k[0] == key})
    print("合図ごとの手の数:", dict(by_rule))
    # 普通の記録: 実際の手を同じ普段で流す
    fp = collections.Counter()
    for p in sorted(glob.glob(os.path.join(a.runs, "*.json"))) if a.runs else []:
        r = json.load(open(p))
        if not r["is_benign"]:
            continue
        i = r["user"].find("\n\n## ")
        own, pasted = (r["user"][:i], r["user"][i:]) if i >= 0 else (r["user"], "")
        res = judge(seen, own, pasted, [], [st["cmd"] for st in r["steps"]])
        hits = [(c, x) for c, x in res if x]
        fp["本"] += 1; fp["手"] += len(res); fp["止めた本"] += bool(hits); fp["止めた手"] += len(hits)
        if hits and a.show:
            print("   普通で止めた:", r["id"], hits[:2])
    print("普通(CodeIPI の普通 30 本):", dict(fp))


if __name__ == "__main__":
    main()
