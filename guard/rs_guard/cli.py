"""rs-guard のコマンド。

  rs-guard learn            会話記録から普段を作る(取り込み済みで変わっていない会話は飛ばす)
  rs-guard claude-pre       PreToolUse のフック(普段の範囲 → 何も出さない / 外れ → 確認を求める)
  rs-guard claude-post      PostToolUse のフック(実行された手を普段に加える、結果を会話ごとの覚え書きに足す)
  rs-guard claude-prompt    UserPromptSubmit のフック(利用者の発言を会話ごとの覚え書きに足す)
  rs-guard install claude   ~/.claude/settings.json に 3 つのフックを足す(既存のフックは残す)
  rs-guard status / explain 普段の件数 / 最近の確認とその理由

置き場は ~/.rs-guard/(環境変数 RS_GUARD_HOME で変えられる)。値そのもの(宛先・本文)は残さない。
"""
import argparse
import glob
import json
import os
import sys
import time

from . import claude, core

CLAUDE_PROJECTS = os.path.expanduser("~/.claude/projects")
CLAUDE_SETTINGS = os.path.expanduser("~/.claude/settings.json")


def home():
    d = os.environ.get("RS_GUARD_HOME") or os.path.expanduser("~/.rs-guard")
    os.makedirs(d, mode=0o700, exist_ok=True)
    return d


def _path(name):
    return os.path.join(home(), name)


def config():
    try:
        with open(_path("config.json")) as f:
            c = json.load(f)
    except (OSError, ValueError):
        c = {}
    return {"min_sim": float(c.get("min_sim", 0.2)), "extra_tools": set(c.get("extra_tools", []))}


def _append(name, rows):
    p = _path(name)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_rows(name):
    p = _path(name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---- 会話ごとの覚え書き(会話記録は非同期に書かれ、フックの時点で最新の発言・結果が無いことがあるため) ----

def _strings(v):
    if isinstance(v, dict):
        return [x for y in v.values() for x in _strings(y)]
    if isinstance(v, list):
        return [x for y in v for x in _strings(y)]
    return [str(v)] if isinstance(v, (str, int, float)) else []


def _session_file(session):
    d = os.path.join(home(), "sessions")
    os.makedirs(d, mode=0o700, exist_ok=True)
    return os.path.join(d, os.path.basename(str(session)) + ".jsonl")


def _note(session, row):
    if not session:
        return
    fd = os.open(_session_file(session), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _notes(session):
    p = _session_file(session) if session else None
    if not p or not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def clean_sessions(days=7):
    """古い覚え書き(ツールの結果の中身を含む)を消す。"""
    d = os.path.join(home(), "sessions")
    for p in glob.glob(os.path.join(d, "*.jsonl")):
        if time.time() - os.path.getmtime(p) > days * 86400:
            os.remove(p)


NOTE_MAX = 100_000  # 1 つの結果で覚え書きに残す文字数の上限(大きなファイルを読んだ結果などで膨らまないように)


def _result_note(tool, inp):
    return {"k": "result", "tool": tool, "t": " ".join(_strings(inp.get("tool_response"))).lower()[:NOTE_MAX]}


def _session_of(inp):
    tp = inp.get("transcript_path")
    return inp.get("session_id") or (os.path.splitext(os.path.basename(tp))[0] if tp else None)


def context_of(inp):
    """フックの入力 → (今の依頼, 利用者の発言すべて, それまでの結果)。会話記録に覚え書きの分を足す。"""
    tp = inp.get("transcript_path")
    humans, outputs = claude.history(tp, inp.get("tool_use_id"), claude.TAIL_BYTES) if tp and os.path.exists(tp) else ([], [])
    seen = {h.strip() for h in humans}
    for n in _notes(_session_of(inp)):
        if n["k"] == "human" and n["t"].strip() not in seen:
            humans.append(n["t"]); seen.add(n["t"].strip())
        elif n["k"] == "result":
            outputs.append((n["tool"], n["t"]))  # 一番新しい物として足す(同じ物が記録にあっても出どころは同じ)
    return claude.request_of(humans), " ".join(humans).lower(), outputs


def _row(session, tool, args, req, user_text, outputs):
    prov = core.provenance(args, user_text, outputs, tool)
    return {"session": session, "req": req.lower(), "tool": core.short_name(tool),
            "sig": [list(x) for x in core.signature(tool, prov)]}


# ---- learn ----

def rows_from_transcript(path, extra=()):
    """1 つの会話記録 → 普段の行。成功した(失敗・拒否でない)外に作用する手だけ。"""
    session = os.path.splitext(os.path.basename(path))[0]
    humans, outputs, pending, rows = [], [], {}, []
    for e in claude.events(path):
        if e[0] == "human":
            humans.append(e[1])
        elif e[0] == "call":
            if core.is_acting(e[2], extra) and humans:
                pending[e[1]] = _row(session, e[2], e[3], claude.request_of(humans), " ".join(humans).lower(), list(outputs))
        elif e[0] == "result":
            r = pending.pop(e[1], None)
            if r and not e[4]:
                rows.append(r)
            outputs.append((e[2], e[3].lower()))
    return rows


def cmd_learn(a):
    cfg = config()
    files = sorted(glob.glob(os.path.join(a.projects, "*", "*.jsonl")))
    idx_p = _path("learned.json")
    try:
        with open(idx_p) as f:
            idx = json.load(f)
    except (OSError, ValueError):
        idx = {}
    todo = [p for p in files if idx.get(p) != os.path.getsize(p)]
    redo = {os.path.splitext(os.path.basename(p))[0] for p in todo}
    keep = [r for r in _read_rows("normal.jsonl") if r.get("session") not in redo]
    new = []
    for p in todo:
        new += rows_from_transcript(p, cfg["extra_tools"])
        idx[p] = os.path.getsize(p)
    tmp = _path("normal.jsonl.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        for r in keep + new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, _path("normal.jsonl"))
    with os.fdopen(os.open(idx_p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        json.dump(idx, f)
    print(f"会話記録 {len(files)} 件(新しく読んだ {len(todo)} 件)→ 普段 {len(keep) + len(new)} 行")


# ---- フック ----

def _hook_input():
    return json.loads(sys.stdin.read() or "{}")


def pre(inp, normal=None, cfg=None):
    """PreToolUse の入力 → 出力する JSON(通すなら None)。"""
    cfg = cfg or config()
    tool, args = inp.get("tool_name", ""), inp.get("tool_input") or {}
    if not core.is_acting(tool, cfg["extra_tools"]):
        return None
    req, user_text, outputs = context_of(inp)
    normal = normal or core.Normal.load(_path("normal.jsonl"))
    session = _session_of(inp)
    ok, reason = core.decide(normal, tool, args, req, user_text, outputs, cfg["min_sim"], cfg["extra_tools"], session)
    _append("decisions.jsonl", [{"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "session": inp.get("session_id"),
                                 "tool": core.short_name(tool), "ok": ok, "reason": reason, "req": req[:80]}])
    if ok:
        return None
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "ask",
                                   "permissionDecisionReason": reason}}


def post(inp, cfg=None):
    """PostToolUse の入力 → 結果を覚え書きに足し、外に作用する手なら普段に加える行を返す(対象外なら None)。"""
    cfg = cfg or config()
    tool, args = inp.get("tool_name", ""), inp.get("tool_input") or {}
    session = _session_of(inp)
    if not core.is_acting(tool, cfg["extra_tools"]):
        _note(session, _result_note(tool, inp))
        return None
    req, user_text, outputs = context_of(inp)
    _note(session, _result_note(tool, inp))
    if not req:
        return None
    r = _row(session, tool, args, req, user_text, outputs)
    _append("normal.jsonl", [r])
    return r


def prompt(inp):
    """UserPromptSubmit の入力 → 利用者の発言を覚え書きに足す。"""
    if inp.get("prompt"):
        _note(_session_of(inp), {"k": "human", "t": inp["prompt"]})


def cmd_claude_pre(a):
    # RS 自身の不具合で利用者の仕事を止めない(落ちたら通し、理由を記録に残す)
    try:
        out = pre(_hook_input())
    except Exception as ex:  # noqa: BLE001
        _append("decisions.jsonl", [{"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "error": repr(ex)[:200]}])
        return
    if out:
        print(json.dumps(out, ensure_ascii=False))


def cmd_claude_prompt(a):
    # 標準出力は AI の文脈に足されるので、何も出さない
    try:
        prompt(_hook_input())
        clean_sessions()
    except Exception as ex:  # noqa: BLE001
        _append("decisions.jsonl", [{"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "error": repr(ex)[:200]}])


def cmd_claude_post(a):
    try:
        post(_hook_input())
    except Exception as ex:  # noqa: BLE001
        _append("decisions.jsonl", [{"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "error": repr(ex)[:200]}])


# ---- install ----

HOOK_PRE, HOOK_POST, HOOK_PROMPT = "claude-pre", "claude-post", "claude-prompt"


def _exe():
    """フックに書く rs-guard の場所(フックは PATH が違うことがあるので、実行中の物の絶対パス)。"""
    a = sys.argv[0] if sys.argv else ""
    return os.path.abspath(a) if os.path.basename(a) == "rs-guard" else "rs-guard"


def _is_ours(cmd, sub):
    return cmd.split("/")[-1] == f"rs-guard {sub}"


def install_claude(settings):
    """settings の dict に 3 つのフックを足す(既にあれば何もしない、既存のフックは残す)。"""
    hooks = settings.setdefault("hooks", {})
    for event, cmd in (("PreToolUse", HOOK_PRE), ("PostToolUse", HOOK_POST), ("UserPromptSubmit", HOOK_PROMPT)):
        groups = hooks.setdefault(event, [])
        if any(_is_ours(h.get("command", ""), cmd) for g in groups for h in g.get("hooks", [])):
            continue
        g = {"hooks": [{"type": "command", "command": f"{_exe()} {cmd}"}]}
        if event != "UserPromptSubmit":
            g = {"matcher": "*", **g}
        groups.append(g)
    return settings


def cmd_install(a):
    if a.target != "claude":
        sys.exit("v1.0 は claude だけです")
    p = a.settings
    s = {}
    if os.path.exists(p):
        with open(p) as f:
            s = json.load(f)
        with open(p + ".rs-guard.bak", "w") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    install_claude(s)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    print(f"{p} に PreToolUse / PostToolUse / UserPromptSubmit を足しました(元の設定 = {p}.rs-guard.bak)")


# ---- status / explain ----

def cmd_status(a):
    rows = _read_rows("normal.jsonl")
    ds = [d for d in _read_rows("decisions.jsonl") if "ok" in d]
    print(f"普段: {len(rows)} 行 / 依頼 {len({r['req'] for r in rows})} 種 / 手 {len({r['tool'] for r in rows})} 種")
    print(f"判定: {len(ds)} 回(確認を求めた {sum(not d['ok'] for d in ds)} 回)")


def cmd_explain(a):
    ds = [d for d in _read_rows("decisions.jsonl") if d.get("ok") is False]
    for d in ds[-a.n:]:
        print(f"{d['ts']}  {d['tool']}  {d['reason']}\n    依頼: {d['req']}")
    if not ds:
        print("確認を求めた記録はまだありません")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rs-guard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("learn")
    p.add_argument("--projects", default=CLAUDE_PROJECTS)
    p.set_defaults(f=cmd_learn)
    sub.add_parser("claude-pre").set_defaults(f=cmd_claude_pre)
    sub.add_parser("claude-post").set_defaults(f=cmd_claude_post)
    sub.add_parser("claude-prompt").set_defaults(f=cmd_claude_prompt)
    p = sub.add_parser("install")
    p.add_argument("target")
    p.add_argument("--settings", default=CLAUDE_SETTINGS)
    p.set_defaults(f=cmd_install)
    sub.add_parser("status").set_defaults(f=cmd_status)
    p = sub.add_parser("explain")
    p.add_argument("-n", type=int, default=10)
    p.set_defaults(f=cmd_explain)
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
