"""各道具のフックの入力・出力と会話記録を、判定の共通の形(Claude Code の形)にそろえる。

共通の形 = {"session_id", "tool_name", "tool_input", "tool_use_id", "cwd", "transcript_path", "prompt", "tool_response"}。
手の名前は Bash / Write / Edit / Read / WebFetch / WebSearch / mcp__<サーバー>__<ツール> にそろえる。
道具ごとの資料(2026-10 に確認):
  codex    https://learn.chatgpt.com/docs/hooks            PreToolUse / PostToolUse / UserPromptSubmit(確認は出せず拒否だけ)
  gemini   https://geminicli.com/docs/hooks/reference/     BeforeTool / AfterTool / BeforeAgent(拒否だけ)
  cursor   https://cursor.com/docs/hooks                   beforeShellExecution / beforeMCPExecution / beforeReadFile …(確認を出せる)
  copilot  https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-hooks-reference
                                                           preToolUse / postToolUse / userPromptSubmitted(確認を出せる。VS Code も同じ)
  cline    https://docs.cline.bot/features/hooks           PreToolUse / PostToolUse / UserPromptSubmit(取り消しだけ)
  windsurf https://docs.devin.ai/desktop/cascade/hooks     pre_run_command / pre_mcp_tool_use / pre_write_code …(止めるだけ)
  antigravity https://antigravity.google/docs/hooks?tab=ide PreToolUse / PostToolUse(確認を出せる。発言のフックは無い)
"""
import json
import os
import re

# 道具ごとに: フックの出来事の名前 → pre / post / prompt
EVENTS = {
    "claude": {"PreToolUse": "pre", "PostToolUse": "post", "UserPromptSubmit": "prompt"},
    "codex": {"PreToolUse": "pre", "PostToolUse": "post", "UserPromptSubmit": "prompt"},
    "gemini": {"BeforeTool": "pre", "AfterTool": "post", "BeforeAgent": "prompt"},
    "cursor": {"beforeShellExecution": "pre", "beforeMCPExecution": "pre", "beforeReadFile": "pre",
               "afterShellExecution": "post", "afterMCPExecution": "post", "afterFileEdit": "post",
               "beforeSubmitPrompt": "prompt"},
    "copilot": {"preToolUse": "pre", "postToolUse": "post", "userPromptSubmitted": "prompt"},
    "cline": {"PreToolUse": "pre", "PostToolUse": "post", "UserPromptSubmit": "prompt"},
    "windsurf": {"pre_run_command": "pre", "pre_mcp_tool_use": "pre", "pre_write_code": "pre", "pre_read_code": "pre",
                 "post_run_command": "post", "post_mcp_tool_use": "post", "post_write_code": "post",
                 "post_read_code": "post", "pre_user_prompt": "prompt"},
    "antigravity": {"PreToolUse": "pre", "PostToolUse": "post"},
}
ASK_OK = {"claude", "cursor", "copilot", "antigravity"}  # 確認の画面を出せる道具。他は拒否して、利用者が良いと言った後のやり直しを通す


def _json(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return {"_raw": v}
    return v if isinstance(v, dict) else {}


def _text(v):
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False) if v is not None else ""


PATCH_FILE = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+)$", re.M)


def patch_edits(patch):
    """apply_patch の文 → [("Write"|"Edit", 引数)](ファイルごと)。"""
    out = []
    parts = PATCH_FILE.split(patch or "")
    for i in range(1, len(parts) - 2, 3):
        kind, path, body = parts[i], parts[i + 1].strip(), parts[i + 2]
        added = "\n".join(l[1:] for l in body.splitlines() if l.startswith("+"))
        removed = "\n".join(l[1:] for l in body.splitlines() if l.startswith("-"))
        if kind == "Add":
            out.append(("Write", {"file_path": path, "content": added}))
        elif kind == "Update":
            out.append(("Edit", {"file_path": path, "old_string": removed, "new_string": added}))
        else:
            out.append(("Bash", {"command": f"rm {path}"}))
    return out


def _mcp(server, tool):
    return f"mcp__{server}__{tool}"


def _tool(agent, name, args):
    """道具の手の名前・引数 → [(共通の名前, 共通の引数)](apply_patch は複数のファイルになることがある)。"""
    a = _json(args)
    if agent in ("claude",):
        return [(name, a)]
    if agent == "codex":
        if name == "apply_patch":
            return patch_edits(a.get("command") or a.get("input") or a.get("_raw", "")) or [("Edit", {})]
        return [(name, a)]  # Bash(command)・mcp__…・その他はそのまま
    if agent == "gemini":
        if name == "run_shell_command":
            return [("Bash", {"command": a.get("command", "")})]
        if name == "write_file":
            return [("Write", {"file_path": a.get("file_path", ""), "content": a.get("content", "")})]
        if name == "replace":
            return [("Edit", {"file_path": a.get("file_path", ""), "old_string": a.get("old_string", ""),
                              "new_string": a.get("new_string", "")})]
        if name in ("read_file", "read_many_files"):
            return [("Read", {"file_path": a.get("file_path") or " ".join(a.get("include") or [])})]
        if name == "web_fetch":
            urls = re.findall(r"https?://\S+", a.get("prompt", ""))
            return [("WebFetch", {"url": u}) for u in urls] or [("WebFetch", {"url": ""})]
        if name == "google_web_search":
            return [("WebSearch", {"query": a.get("query", "")})]
        return [(name, a)]
    if agent == "copilot":
        if name in ("bash", "powershell", "Bash"):
            return [("Bash", {"command": a.get("command", "")})]
        if name in ("create", "Write"):
            return [("Write", {"file_path": a.get("path") or a.get("file_path", ""),
                               "content": a.get("file_text") or a.get("content", "")})]
        if name in ("edit", "Edit"):
            return [("Edit", {"file_path": a.get("path") or a.get("file_path", ""),
                              "old_string": a.get("old_str") or a.get("old_string", ""),
                              "new_string": a.get("new_str") or a.get("new_string", "")})]
        if name in ("view", "Read"):
            return [("Read", {"file_path": a.get("path") or a.get("file_path", "")})]
        if name == "web_fetch":
            return [("WebFetch", {"url": a.get("url", "")})]
        return [(name, a)]
    if agent == "cline":
        if name == "execute_command":
            return [("Bash", {"command": a.get("command", "")})]
        if name == "write_to_file":
            return [("Write", {"file_path": a.get("path", ""), "content": a.get("content", "")})]
        if name == "replace_in_file":
            d = a.get("diff", "")
            new = "\n".join(re.findall(r"=======\n(.*?)\n\+\+\+\+\+\+\+ REPLACE", d, re.S))
            old = "\n".join(re.findall(r"------- SEARCH\n(.*?)\n=======", d, re.S))
            return [("Edit", {"file_path": a.get("path", ""), "old_string": old, "new_string": new or d})]
        if name == "read_file":
            return [("Read", {"file_path": a.get("path", "")})]
        if name == "use_mcp_tool":
            return [(_mcp(a.get("server_name", "?"), a.get("tool_name", "?")), _json(a.get("arguments", {})))]
        if name in ("web_fetch", "browser_action"):
            return [("WebFetch", {"url": a.get("url", "")})]
        return [(name, a)]
    if agent == "antigravity":  # 引数の名前は版で変わり得るので、いくつかの書き方を受ける
        g = lambda *ks: next((a[k] for k in ks if a.get(k)), "")
        if name == "run_command":
            return [("Bash", {"command": g("CommandLine", "commandLine", "command", "Command")})]
        if name == "write_to_file":
            return [("Write", {"file_path": g("TargetFile", "targetFile", "path", "file_path"),
                               "content": g("CodeContent", "codeContent", "content")})]
        if name in ("replace_file_content", "multi_replace_file_content"):
            chunks = a.get("ReplacementChunks") or a.get("replacementChunks") or []
            new = "\n".join(str(c.get("ReplacementContent") or c.get("replacementContent") or "") for c in chunks if isinstance(c, dict))
            old = "\n".join(str(c.get("TargetContent") or c.get("targetContent") or "") for c in chunks if isinstance(c, dict))
            return [("Edit", {"file_path": g("TargetFile", "targetFile", "path", "file_path"),
                              "old_string": old or g("TargetContent", "targetContent"),
                              "new_string": new or g("ReplacementContent", "replacementContent")})]
        if name == "view_file":
            return [("Read", {"file_path": g("AbsolutePath", "absolutePath", "path", "file_path")})]
        if name == "read_url_content":
            return [("WebFetch", {"url": g("Url", "url")})]
        if name == "search_web":
            return [("WebSearch", {"query": g("query", "Query")})]
        if name == "send_message":
            return [(name, a)]
        return [(name, a)]
    return [(name, a)]


def normalize(agent, event, raw):
    """フックの入力 → (種類 pre/post/prompt, [共通の形の入力])。"""
    kind = EVENTS.get(agent, {}).get(event)
    if kind is None:
        return None, []
    base = {"cwd": raw.get("cwd") or "", "transcript_path": None, "tool_use_id": None}
    if agent == "claude":
        return kind, [raw]
    if agent in ("codex",):
        base.update(session_id=raw.get("session_id"), tool_use_id=raw.get("tool_use_id"))
        if kind == "prompt":
            return kind, [{**base, "prompt": raw.get("prompt", "")}]
        resp = raw.get("tool_response")
        return kind, [{**base, "tool_name": n, "tool_input": a, "tool_response": resp}
                      for n, a in _tool(agent, raw.get("tool_name", ""), raw.get("tool_input"))]
    if agent == "gemini":
        base.update(session_id=raw.get("session_id"))
        if kind == "prompt":
            return kind, [{**base, "prompt": raw.get("prompt", "")}]
        name = raw.get("tool_name", "")
        ctx = raw.get("mcp_context") or {}
        if name.startswith("mcp_") and ctx.get("server_name"):
            tool = ctx.get("tool_name") or name[len("mcp_" + ctx["server_name"] + "_"):]
            calls = [(_mcp(ctx["server_name"], tool), _json(raw.get("tool_input")))]
        else:
            calls = _tool(agent, name, raw.get("tool_input"))
        resp = (raw.get("tool_response") or {}).get("llmContent") if isinstance(raw.get("tool_response"), dict) else raw.get("tool_response")
        cwd = (_json(raw.get("tool_input")).get("dir_path") or raw.get("cwd") or "")
        return kind, [{**base, "cwd": cwd, "tool_name": n, "tool_input": a, "tool_response": resp} for n, a in calls]
    if agent == "cursor":
        base.update(session_id=raw.get("conversation_id"), tool_use_id=raw.get("generation_id"))
        if kind == "prompt":
            return kind, [{**base, "prompt": raw.get("prompt", "")}]
        if event in ("beforeShellExecution", "afterShellExecution"):
            return kind, [{**base, "tool_name": "Bash", "tool_input": {"command": raw.get("command", "")},
                           "tool_response": raw.get("output", "")}]
        if event in ("beforeMCPExecution", "afterMCPExecution"):
            return kind, [{**base, "tool_name": _mcp(raw.get("mcp_server_name", "?"), raw.get("tool_name", "?")),
                           "tool_input": _json(raw.get("tool_input")), "tool_response": raw.get("result_json", "")}]
        if event == "beforeReadFile":
            return kind, [{**base, "tool_name": "Read", "tool_input": {"file_path": raw.get("file_path", "")}}]
        if event == "afterFileEdit":
            edits = raw.get("edits") or []
            return kind, [{**base, "tool_name": "Edit", "tool_input": {
                "file_path": raw.get("file_path", ""), "old_string": "\n".join(e.get("old_string", "") for e in edits),
                "new_string": "\n".join(e.get("new_string", "") for e in edits)}}]
    if agent == "copilot":
        sid = raw.get("sessionId") or raw.get("session_id")
        base.update(session_id=sid)
        if kind == "prompt":
            return kind, [{**base, "prompt": raw.get("prompt", "")}]
        name = raw.get("toolName") or raw.get("tool_name", "")
        args = raw.get("toolArgs") if "toolArgs" in raw else raw.get("tool_input")
        res = raw.get("toolResult") or {}
        resp = res.get("textResultForLlm") if isinstance(res, dict) else res
        return kind, [{**base, "tool_name": n, "tool_input": a, "tool_response": resp} for n, a in _tool(agent, name, args)]
    if agent == "cline":
        base.update(session_id=raw.get("taskId"), cwd=(raw.get("workspaceRoots") or [""])[0])
        if kind == "prompt":
            up = raw.get("userPromptSubmit") or {}
            return kind, [{**base, "prompt": up.get("prompt") or raw.get("prompt", "")}]
        part = raw.get("preToolUse") or raw.get("postToolUse") or {}
        resp = part.get("result") or part.get("output")
        return kind, [{**base, "tool_name": n, "tool_input": a, "tool_response": resp}
                      for n, a in _tool(agent, part.get("toolName", ""), part.get("parameters"))]
    if agent == "windsurf":
        ti = raw.get("tool_info") or {}
        base.update(session_id=raw.get("trajectory_id"), tool_use_id=raw.get("execution_id"), cwd=ti.get("cwd") or "")
        if kind == "prompt":
            return kind, [{**base, "prompt": ti.get("user_prompt", "")}]
        if event.endswith("run_command"):
            return kind, [{**base, "tool_name": "Bash", "tool_input": {"command": ti.get("command_line", "")}}]
        if event.endswith("mcp_tool_use"):
            return kind, [{**base, "tool_name": _mcp(ti.get("mcp_server_name", "?"), ti.get("mcp_tool_name", "?")),
                           "tool_input": _json(ti.get("mcp_tool_arguments")), "tool_response": ti.get("mcp_result")}]
        if event.endswith("write_code"):
            edits = ti.get("edits") or []
            return kind, [{**base, "tool_name": "Edit", "tool_input": {
                "file_path": ti.get("file_path", ""), "old_string": "\n".join(e.get("old_string", "") for e in edits),
                "new_string": "\n".join(e.get("new_string", "") for e in edits)}}]
        if event.endswith("read_code"):
            return kind, [{**base, "tool_name": "Read", "tool_input": {"file_path": ti.get("file_path", "")}}]
    if agent == "antigravity":
        base.update(session_id=raw.get("conversationId"), cwd=(raw.get("workspacePaths") or [""])[0])
        tc = raw.get("toolCall") or {}
        resp = raw.get("toolResult") or raw.get("result") or raw.get("output")
        return kind, [{**base, "tool_name": n, "tool_input": a, "tool_response": resp}
                      for n, a in _tool(agent, tc.get("name", ""), tc.get("args"))]
    return kind, []


def render(agent, event, decision, reason):
    """判定 → (標準出力の文, 標準エラーの文, 終わりの番号)。decision = allow / ask / deny。"""
    if agent == "claude":
        if decision == "allow":
            return "", "", 0
        return json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision,
                                                  "permissionDecisionReason": reason}}, ensure_ascii=False), "", 0
    if agent == "codex":
        if decision == "allow":
            return "", "", 0
        return json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                                  "permissionDecisionReason": reason}}, ensure_ascii=False), "", 0
    if agent == "gemini":
        if decision == "allow":
            return "{}", "", 0
        return json.dumps({"decision": "deny", "reason": reason}, ensure_ascii=False), "", 0
    if agent == "cursor":
        if event == "beforeReadFile":  # ask は無い
            d = "allow" if decision == "allow" else "deny"
            return json.dumps({"permission": d, **({"user_message": reason} if reason else {})}, ensure_ascii=False), "", 0
        out = {"permission": decision}
        if reason:
            out.update(user_message=reason, agent_message=reason)
        return json.dumps(out, ensure_ascii=False), "", 0
    if agent == "copilot":
        if decision == "allow":
            return "{}", "", 0
        return json.dumps({"permissionDecision": decision, "permissionDecisionReason": reason}, ensure_ascii=False), "", 0
    if agent == "cline":
        if decision == "allow":
            return json.dumps({"cancel": False}), "", 0
        return json.dumps({"cancel": True, "errorMessage": reason}, ensure_ascii=False), "", 0
    if agent == "windsurf":
        return ("", "", 0) if decision == "allow" else ("", reason, 2)
    if agent == "antigravity":  # allow = 次のフック・既定の方針に任せる
        return json.dumps({"decision": decision, **({"reason": reason} if reason else {})}, ensure_ascii=False), "", 0
    return "", "", 0


def render_other(agent, kind):
    """pre 以外(post・prompt)の出力: 何もしないことを各道具の形で返す。"""
    if agent == "gemini":
        return "{}"
    if agent == "cursor" and kind == "prompt":
        return json.dumps({"continue": True})
    if agent == "cline":
        return json.dumps({"cancel": False})
    if agent in ("copilot", "antigravity"):
        return "{}"
    return ""


# ---- 会話記録(普段を作る): Codex ----

RE_CMD = re.compile(r"""\bcmd\s*:\s*("(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`)""")
RE_PATCH = re.compile(r"""apply_patch\(\s*("(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`)""")


def _jsstr(s):
    if s.startswith("`"):
        return s[1:-1].replace("\\`", "`")
    try:
        return json.loads(s)
    except ValueError:
        return s[1:-1]


def codex_events(path):
    """Codex の会話記録(~/.codex/sessions/**/rollout-*.jsonl)→ claude.events と同じ形の並び。
    手 = exec の中の tools.exec_command({cmd}) と tools.apply_patch(…)、関数の呼び出し(exec_command・shell・apply_patch・MCP)。
    exec の中の手は、まとめて返る 1 つの結果を、それぞれの手の結果とする。"""
    subs = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            p = d.get("payload") or {}
            if d.get("type") == "session_meta" and p.get("thread_source") == "subagent":
                return  # 下請けのエージェントの会話(依頼は親のエージェントから)は普段に入れない
            if d.get("type") != "response_item":
                continue
            t, cid = p.get("type"), p.get("call_id")
            calls = []
            if t == "message" and p.get("role") == "user":
                txt = " ".join(c.get("text", "") for c in p.get("content") or [] if isinstance(c, dict))
                if txt.strip() and not txt.lstrip().startswith("<"):
                    yield ("human", txt)
            elif t == "custom_tool_call" and p.get("name") == "exec":
                src = p.get("input", "")
                calls = [("Bash", {"command": _jsstr(m.group(1))}) for m in RE_CMD.finditer(src)]
                for m in RE_PATCH.finditer(src):
                    calls += patch_edits(_jsstr(m.group(1)))
            elif t == "custom_tool_call" and p.get("name") == "apply_patch":
                calls = patch_edits(p.get("input", ""))
            elif t == "function_call":
                name, args = p.get("name", ""), _json(p.get("arguments"))
                if name in ("exec_command", "shell", "local_shell"):
                    cmd = args.get("cmd") or args.get("command")
                    cmd = " ".join(cmd[2:] if cmd[:2] == ["bash", "-lc"] else cmd) if isinstance(cmd, list) else cmd
                    calls = [("Bash", {"command": cmd or ""})]
                elif name == "apply_patch":
                    calls = patch_edits(args.get("input", ""))
                elif str(p.get("namespace") or "").startswith("mcp"):
                    calls = [(_mcp(p["namespace"].split("__")[-1], name), args)]
            elif t in ("custom_tool_call_output", "function_call_output") and cid in subs:
                out = p.get("output")
                txt = " ".join(c.get("text", "") for c in out if isinstance(c, dict)) if isinstance(out, list) else _text(out)
                err = bool(re.search(r"Script failed|exit_code\"?:\s*[1-9]", txt[:400]))
                for sid, n in subs.pop(cid):
                    yield ("result", sid, n, txt, err)
            if calls and cid:
                subs[cid] = []
                for i, (n, a) in enumerate(calls):
                    subs[cid].append((f"{cid}#{i}", n))
                    yield ("call", f"{cid}#{i}", n, a)


def codex_files(home=None):
    root = home or os.path.expanduser("~/.codex/sessions")
    out = []
    for dp, _, fs in os.walk(root):
        out += [os.path.join(dp, f) for f in fs if f.endswith(".jsonl")]
    return sorted(out)
