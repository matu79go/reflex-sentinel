"""rs-guard のコマンド。

  rs-guard learn            会話記録から普段を作る(取り込み済みで変わっていない会話は飛ばす)
  rs-guard claude-pre       PreToolUse のフック(普段の範囲 → 何も出さない / 外れ → 確認を求める)
  rs-guard claude-post      PostToolUse のフック(実行された手を普段に加える、結果を会話ごとの覚え書きに足す)
  rs-guard claude-prompt    UserPromptSubmit のフック(利用者の発言を会話ごとの覚え書きに足す)
  rs-guard install <道具>   道具の設定にフックを足す(既存のフックは残す)。道具 = claude / codex / gemini / cursor /
                            copilot / cline / windsurf / all(入っている物すべて)
  rs-guard hook <道具> <出来事>  各道具のフックの入口(入力・出力の形は agents.py でそろえる)
  rs-guard status / explain 普段の件数 / 最近の確認とその理由

判定 = core.decide(MCP と Web の手を似た過去の依頼と比べる)+ acts(すべての手。シェル・ファイル・スクリプトを含む)。
置き場は ~/.rs-guard/(環境変数 RS_GUARD_HOME で変えられる)。値そのもの(宛先・本文)は残さない(送り先はハッシュ)。
"""
import argparse
import glob
import secrets
import json
import os
import sys
import time

import hashlib

from . import acts, agents, claude, core

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
    if c.get("lang"):
        os.environ.setdefault("RS_GUARD_LANG", str(c["lang"]))  # 理由の文の言葉(ja / en)。環境変数が先
    return {"min_sim": float(c.get("min_sim", 0.2)), "extra_tools": set(c.get("extra_tools", [])),
            "deny_rules": set(c.get("deny_rules", []))}  # すぐ止める合図(R1x・R2r・R3・R5・R6・R7・D)。他は確認を求める


def salt():
    """送り先のハッシュに混ぜる値(この機械だけの乱数)。"""
    p = _path("salt")
    if not os.path.exists(p):
        with os.fdopen(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
            f.write(secrets.token_hex(16))
    with open(p) as f:
        return f.read().strip()


def load_seen():
    return acts.Seen.load(_path("seen.jsonl"), salt())


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


def context_of(inp, full=False):
    """フックの入力 → (今の依頼, 利用者の発言すべて, それまでの結果)。会話記録に覚え書きの分を足す。
    full なら、すべての手を見る判定の分(本人の言葉・貼り付けた文の指示・この会話で実行した手)の dict も返す。"""
    tp = inp.get("transcript_path")
    pastes = []
    humans, outputs = claude.history(tp, inp.get("tool_use_id"), claude.TAIL_BYTES, pastes) if tp and os.path.exists(tp) else ([], [])
    seen = {h.strip() for h in humans}
    sess = {"sigs": set(), "sens": False}
    for n in _notes(_session_of(inp)):
        if n["k"] == "human" and n["t"].strip() not in seen:
            humans.append(n["t"]); seen.add(n["t"].strip())
        elif n["k"] == "result":
            outputs.append((n["tool"], n["t"]))  # 一番新しい物として足す(同じ物が記録にあっても出どころは同じ)
        elif n["k"] == "act":
            sess["sigs"] |= {tuple(x) for x in n.get("sigs", [])}
            sess["sens"] = sess["sens"] or bool(n.get("sens"))
    base = claude.request_of(humans), " ".join(humans).lower(), outputs
    if not full:
        return base
    owns, snips = [], []
    for h in humans + pastes:
        own, pasted = acts.split_own(h) if h in humans else ("", h)
        owns.append(own)
        snips += acts.instruction_snippets(pasted)
    own_text = " ".join(owns)
    return base + ({"own_text": own_text, "own_toks": acts.toks(own_text), "snips": snips, **sess},)


def _row(session, tool, args, req, user_text, outputs):
    prov = core.provenance(args, user_text, outputs, tool)
    return {"session": session, "req": req.lower(), "tool": core.short_name(tool),
            "sig": [list(x) for x in core.signature(tool, prov)]}


# ---- learn ----

def rows_from_transcript(path, extra=(), events=None):
    """1 つの会話記録 → 普段の行。成功した(失敗・拒否でない)外に作用する手だけ。events = 会話記録の読み方(道具ごと)。"""
    session = os.path.splitext(os.path.basename(path))[0]
    humans, outputs, pending, rows = [], [], {}, []
    for e in (events or claude.events)(path):
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


def seen_from_transcript(path, seen, events=None):
    """1 つの会話記録 → 普段(すべての手)に足す行。実行された(失敗・拒否でない)手だけ。"""
    humans, owns, outputs, pending, rows, files = [], [], [], {}, [], {}
    for e in (events or claude.events)(path):
        if e[0] == "human":
            humans.append(e[1]); owns.append(acts.split_own(e[1])[0])
        elif e[0] == "call" and humans:
            args = e[3] if isinstance(e[3], dict) else {}
            pending[e[1]] = acts.Step(e[2], args, " ".join(humans).lower(), " ".join(owns), list(outputs), files=files, disk=False)
            if e[2] == "Write" and args.get("file_path"):
                files[args["file_path"]] = str(args.get("content", ""))
        elif e[0] == "result":
            st = pending.pop(e[1], None)
            if st and not e[4]:
                rows += st.learned(seen)
            outputs.append((e[2], e[3].lower()))
    return rows


def cmd_learn(a):
    cfg = config()
    files = sorted(glob.glob(os.path.join(a.projects, "*", "*.jsonl")))
    codex_dir = os.path.expanduser("~/.codex/sessions") if a.codex == "auto" and a.projects == CLAUDE_PROJECTS else (
        a.codex if a.codex != "auto" else "")
    codex = agents.codex_files(codex_dir) if codex_dir and os.path.isdir(codex_dir) else []
    files += codex
    reader = {p: agents.codex_events for p in codex}
    idx_p = _path("learned.json")
    try:
        with open(idx_p) as f:
            idx = json.load(f)
    except (OSError, ValueError):
        idx = {}
    todo = [p for p in files if idx.get(p) != os.path.getsize(p)]
    if not os.path.exists(_path("seen.jsonl")):
        todo = files  # すべての手を見る普段が無い(v1.0 に上げた直後)= すべて読み直す
    seen = load_seen()
    seen_rows = []
    redo = {os.path.splitext(os.path.basename(p))[0] for p in todo}
    keep = [r for r in _read_rows("normal.jsonl") if r.get("session") not in redo]
    new = []
    for p in todo:
        new += rows_from_transcript(p, cfg["extra_tools"], reader.get(p))
        seen_rows += seen_from_transcript(p, seen, reader.get(p))
        idx[p] = os.path.getsize(p)
    _append("seen.jsonl", seen_rows)
    tmp = _path("normal.jsonl.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        for r in keep + new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, _path("normal.jsonl"))
    with os.fdopen(os.open(idx_p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        json.dump(idx, f)
    print(f"会話記録 {len(files)} 件(うち Codex {len(codex)} 件、新しく読んだ {len(todo)} 件)→ 普段 {len(keep) + len(new)} 行・"
          f"手の形 {len(seen.shapes)} 種・送り先 {len(seen.dests)} 件")


# ---- フック ----

def _hook_input():
    return json.loads(sys.stdin.read() or "{}")


def pre(inp, normal=None, cfg=None):
    """PreToolUse の入力 → 出力する JSON(通すなら None)。Claude Code の形。"""
    decision, reason, _ = judge(inp, normal, cfg)
    if decision == "allow":
        return None
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision,
                                   "permissionDecisionReason": reason}}


def _fingerprint(tool, args):
    return hashlib.sha256(json.dumps([tool, args], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


def _confirmed(session, fp):
    """確認の画面が無い道具: 前に止めた同じ手が、その後に利用者の発言をはさんで、もう一度来たか。"""
    after = None
    for n in _notes(session):
        if n["k"] == "denied" and n.get("fp") == fp:
            after = False
        elif n["k"] == "human" and after is False:
            after = True
    return after is True


def judge(inp, normal=None, cfg=None, retry=False):
    """共通の形の入力 → (allow / ask / deny, 理由, 合図)。
    retry = 確認の画面が無い道具。確認の代わりに一度止め、利用者の発言の後に同じ手が来たら通す。"""
    cfg = cfg or config()
    tool, args = inp.get("tool_name", ""), inp.get("tool_input") or {}
    req, user_text, outputs, x = context_of(inp, full=True)
    session = _session_of(inp)
    acting = core.is_acting(tool, cfg["extra_tools"])
    rules, reasons = [], []
    if acting:
        normal = normal or core.Normal.load(_path("normal.jsonl"))
        ok, why = core.decide(normal, tool, args, req, user_text, outputs, cfg["min_sim"], cfg["extra_tools"], session)
        if not ok:
            rules.append("D"); reasons.append(why)
    st = acts.Step(tool, args, user_text, x["own_text"], outputs, cwd=inp.get("cwd", ""))
    for r, d in sorted(st.check(load_seen(), x["sigs"], x["sens"], x["snips"], x["own_toks"]), key=lambda f: acts.ORDER.index(f[0])):
        if r not in rules:
            rules.append(r); reasons.append(acts.reason(r, d, core._ja()))
    ok, reason = not rules, " ".join(reasons)
    decision = "allow" if ok else "deny" if set(rules) & cfg["deny_rules"] else "ask"
    if retry and decision == "ask":
        fp = _fingerprint(tool, args)
        if _confirmed(session, fp):
            decision, rules, reason = "allow", ["confirmed"], ""
        else:
            _note(session, {"k": "denied", "fp": fp})
            decision = "deny"
            reason += (" (利用者に確かめてください。良いと言われたら、同じ手をもう一度実行すると通ります)" if core._ja() else
                       " (Ask the user. If they approve, run the same action again and it will go through.)")
    if acting or rules:
        _append("decisions.jsonl", [{"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "session": inp.get("session_id"),
                                     "id": inp.get("tool_use_id"), "tool": core.short_name(tool), "ok": decision == "allow",
                                     "reason": reason, "rules": rules, "req": req[:80], "agent": inp.get("_agent", "claude")}])
    return decision, reason, rules


def post(inp, cfg=None):
    """PostToolUse の入力 → 結果を覚え書きに足し、外に作用する手なら普段に加える行を返す(対象外なら None)。"""
    cfg = cfg or config()
    tool, args = inp.get("tool_name", ""), inp.get("tool_input") or {}
    session = _session_of(inp)
    req, user_text, outputs, x = context_of(inp, full=True)
    seen = load_seen()
    st = acts.Step(tool, args, user_text, x["own_text"], outputs, cwd=inp.get("cwd", ""))
    _append("seen.jsonl", st.learned(seen))  # 実行された手 = 普段(すべての手)に加える
    _note(session, {"k": "act", "sigs": [list(s) for s in st.sig | set(st.web)], "sens": st.sens})
    _note(session, _result_note(tool, inp))
    if inp.get("tool_use_id") and inp["tool_use_id"] in _asked_ids():
        _append("done.jsonl", [{"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "id": inp["tool_use_id"]}])  # 確認のあと通した
    if not core.is_acting(tool, cfg["extra_tools"]):
        return None
    if not req:
        return None
    r = _row(session, tool, args, req, user_text, outputs)
    r["id"] = inp.get("tool_use_id")  # 確認のあと通したかを、判定の記録と突き合わせるため
    _append("normal.jsonl", [r])
    return r


def _asked_ids(tail=200_000):
    """最近の判定の記録で、確認を求めた手の番号(記録の末尾だけ読む)。"""
    p = _path("decisions.jsonl")
    if not os.path.exists(p):
        return set()
    start = max(0, os.path.getsize(p) - tail)
    with open(p, "rb") as f:
        f.seek(start)
        lines = f.read().decode("utf-8", errors="replace").splitlines()[1 if start else 0:]  # 途中から読んだ時だけ最初の行を捨てる
    out = set()
    for l in lines:
        try:
            d = json.loads(l)
        except ValueError:
            continue
        if d.get("ok") is False and d.get("id"):
            out.add(d["id"])
    return out


def approved_ids():
    """確認のあと実行された手の番号(すべての手 = done.jsonl、MCP・Web = 普段の行の id)。"""
    return {r.get("id") for r in _read_rows("normal.jsonl") + _read_rows("done.jsonl") if r.get("id")}


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


def cmd_hook(a):
    """各道具のフックの入口。RS 自身の不具合で仕事を止めない(落ちたら通す形を返し、記録に残す)。"""
    agent, event = a.agent, a.event
    kind = agents.EVENTS.get(agent, {}).get(event)
    out, err, code = (agents.render(agent, event, "allow", "") if kind == "pre" else (agents.render_other(agent, kind), "", 0))
    try:
        raw = _hook_input()
        kind, inps = agents.normalize(agent, event, raw)
        for i in inps:
            i["_agent"] = agent
        if kind == "pre" and inps:
            order = {"allow": 0, "ask": 1, "deny": 2}
            worst, reasons = "allow", []
            for i in inps:
                d, r, _ = judge(i, retry=agent not in agents.ASK_OK)
                if order[d] > order[worst]:
                    worst = d
                if r:
                    reasons.append(r)
            out, err, code = agents.render(agent, event, worst, " ".join(reasons))
        elif kind == "post":
            for i in inps:
                post(i)
        elif kind == "prompt":
            for i in inps:
                prompt(i)
            clean_sessions()
    except Exception as ex:  # noqa: BLE001
        _append("decisions.jsonl", [{"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "agent": agent, "error": repr(ex)[:200]}])
    if out:
        print(out)
    if err:
        print(err, file=sys.stderr)
    sys.exit(code)


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


def _edit_json(p, fn):
    """JSON の設定を読み、fn で足し、書き戻す。初めて触る時だけ元を .rs-guard.bak に残す。"""
    s = {}
    if os.path.exists(p):
        with open(p) as f:
            s = json.load(f)
        if not os.path.exists(p + ".rs-guard.bak"):
            with open(p + ".rs-guard.bak", "w") as f:
                json.dump(s, f, ensure_ascii=False, indent=2)
    fn(s)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
    return p


def _hook_cmd(agent, event):
    return f"{_exe()} hook {agent} {event}"


def _has(entries, agent, event):
    return any(f"hook {agent} {event}" in json.dumps(e) for e in entries)


def _add_grouped(hooks, agent, events, hook, matcher):
    """Claude Code・Codex・Gemini の形: {出来事: [{matcher, hooks: [...]}]}。"""
    for ev, kind in events.items():
        groups = hooks.setdefault(ev, [])
        if _has(groups, agent, ev):
            continue
        g = {"hooks": [{"type": "command", "command": _hook_cmd(agent, ev), **hook}]}
        groups.append({"matcher": matcher, **g} if kind != "prompt" and matcher else g)


def _copilot_home():
    return os.environ.get("COPILOT_HOME") or os.path.expanduser("~/.copilot")


INSTALL_PATHS = {  # 道具 → 入っているかを見る場所
    "codex": "~/.codex", "gemini": "~/.gemini", "cursor": "~/.cursor", "copilot": None,
    "cline": "~/Documents/Cline", "windsurf": "~/.codeium/windsurf",
}


def install_agent(agent, settings_path=None):
    """→ 書いた設定の場所。"""
    ev = agents.EVENTS[agent]
    if agent == "claude":
        return _edit_json(settings_path or CLAUDE_SETTINGS, install_claude)
    if agent == "codex":  # 使う前に Codex の中で /hooks から信頼する必要がある
        return _edit_json(os.path.expanduser("~/.codex/hooks.json"),
                          lambda s: _add_grouped(s.setdefault("hooks", {}), agent, ev, {"timeout": 30}, "*"))
    if agent == "gemini":
        return _edit_json(os.path.expanduser("~/.gemini/settings.json"),
                          lambda s: _add_grouped(s.setdefault("hooks", {}), agent, ev,
                                                 {"name": "rs-guard", "timeout": 30000}, ".*"))
    if agent == "cursor":
        def f(s):
            s.setdefault("version", 1)
            h = s.setdefault("hooks", {})
            for e in ev:
                if not _has(h.get(e, []), agent, e):
                    h.setdefault(e, []).append({"command": _hook_cmd(agent, e), "timeout": 30})
        return _edit_json(os.path.expanduser("~/.cursor/hooks.json"), f)
    if agent == "copilot":  # Copilot CLI と VS Code の Copilot が読む利用者の置き場。rs-guard だけのファイル
        p = os.path.join(_copilot_home(), "hooks", "rs-guard.json")
        return _edit_json(p, lambda s: s.update({"version": 1, "hooks": {
            e: [{"type": "command", "bash": _hook_cmd(agent, e), "timeoutSec": 30}] for e in ev}}))
    if agent == "cline":  # 出来事の名前のファイル(拡張子なし・実行できる)を置く
        d = os.path.expanduser("~/Documents/Cline/Rules/Hooks")
        os.makedirs(d, exist_ok=True)
        for e in ev:
            p = os.path.join(d, e)
            if os.path.exists(p) and "rs-guard" not in open(p, errors="replace").read():
                print(f"  {p} は別のフックがあるので書きません(手で rs-guard の行を足してください)")
                continue
            with open(p, "w") as f:
                f.write(f"#!/bin/sh\nexec {_hook_cmd(agent, e)}\n")
            os.chmod(p, 0o755)
        return d
    if agent == "windsurf":
        def f(s):
            h = s.setdefault("hooks", {})
            for e in ev:
                if not _has(h.get(e, []), agent, e):
                    h.setdefault(e, []).append({"command": _hook_cmd(agent, e), "show_output": False})
        return _edit_json(os.path.expanduser("~/.codeium/windsurf/hooks.json"), f)
    raise ValueError(agent)


def installed_agents():
    out = ["claude"]
    for a, p in INSTALL_PATHS.items():
        if (p and os.path.isdir(os.path.expanduser(p))) or (a == "copilot" and os.path.isdir(_copilot_home())):
            out.append(a)
    return out


NOTES = {
    "codex": "Codex を開いて /hooks で rs-guard のフックを信頼してください(信頼するまで動きません)。確認の画面が無いので、"
             "普段に無い手は一度止まり、あなたが良いと言った後のやり直しで通ります。",
    "gemini": "Gemini CLI で /hooks enable-all が要ることがあります。確認の画面が無いので、一度止めて、良いと言った後のやり直しで通します。",
    "cline": "Cline の設定の Features でフックを有効にしてください。取り消しだけなので、一度止めて、良いと言った後のやり直しで通します。",
    "windsurf": "止めるだけなので、一度止めて、良いと言った後のやり直しで通します。",
    "cursor": "ファイルの書き込みは実行前のフックが無いので、書いた後に普段へ入れるだけです(シェル・MCP・読み込みは実行前に見ます)。",
}


def cmd_install(a):
    targets = installed_agents() if a.target == "all" else [a.target]
    for t in targets:
        if t not in agents.EVENTS:
            sys.exit(f"知らない道具です: {t}(claude / codex / gemini / cursor / copilot / cline / windsurf / all)")
        p = install_agent(t, a.settings if t == "claude" else None)
        print(f"{t}: {p} にフックを足しました(元の設定は .rs-guard.bak)")
        if t in NOTES:
            print("  " + NOTES[t])


# ---- status / explain ----

def cmd_status(a):
    rows = _read_rows("normal.jsonl")
    ds = [d for d in _read_rows("decisions.jsonl") if "ok" in d]
    print(f"普段: {len(rows)} 行 / 依頼 {len({r['req'] for r in rows})} 種 / 手 {len({r['tool'] for r in rows})} 種")
    print(f"判定: {len(ds)} 回(確認を求めた {sum(not d['ok'] for d in ds)} 回)")


def cmd_explain(a):
    """最近の確認と理由、そのあと通したか(実行後のフックが来て普段に入った)・拒否したか。"""
    ds = [d for d in _read_rows("decisions.jsonl") if d.get("ok") is False or "confirmed" in (d.get("rules") or [])]
    done = approved_ids()
    for d in ds[-a.n:]:
        if d.get("ok"):
            what = "→ 確かめて通した(利用者が良いと言った後のやり直し)"
        else:
            what = "→ 通した" if d.get("id") in done else "→ 拒否(または未実行)"
        tag = f"[{d['agent']}]  " if d.get("agent") and d["agent"] != "claude" else ""
        print(f"{d['ts']}  {tag}{d['tool']}  {what}\n    {d['reason']}\n    依頼: {d['req']}")
    if not ds:
        print("確認を求めた記録はまだありません")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rs-guard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("learn")
    p.add_argument("--projects", default=CLAUDE_PROJECTS)
    p.add_argument("--codex", default="auto", help="Codex の会話記録の置き場(auto = ~/.codex/sessions)")
    p.set_defaults(f=cmd_learn)
    sub.add_parser("claude-pre").set_defaults(f=cmd_claude_pre)
    sub.add_parser("claude-post").set_defaults(f=cmd_claude_post)
    sub.add_parser("claude-prompt").set_defaults(f=cmd_claude_prompt)
    p = sub.add_parser("hook")
    p.add_argument("agent")
    p.add_argument("event")
    p.set_defaults(f=cmd_hook)
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
