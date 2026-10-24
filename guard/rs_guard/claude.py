"""Claude Code の会話記録(~/.claude/projects/*/*.jsonl)を読む。"""
import json


def _text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(x.get("text", "") or (x.get("content", "") if isinstance(x.get("content"), str) else "")
                        for x in c if isinstance(x, dict))
    return ""


def is_human(d):
    """利用者が打った発言か(ツールの結果・システムの差し込み・コマンドの出力は除く)。"""
    if d.get("type") != "user" or d.get("isMeta"):
        return False
    c = (d.get("message") or {}).get("content")
    if isinstance(c, str):
        return bool(c.strip()) and not c.lstrip().startswith("<")
    if isinstance(c, list):
        return any(isinstance(b, dict) and b.get("type") == "text" and not b.get("text", "").lstrip().startswith("<") for b in c) \
            and not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c)
    return False


def is_paste(d, blocks):
    return (d.get("type") == "user" and not d.get("isMeta") and "<pasted_content" in _text((d.get("message") or {}).get("content"))
            and not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks))


TAIL_BYTES = 4_000_000  # 実行前の判定で読む会話記録の末尾(大きな会話でも速く。前の発言は会話ごとの覚え書きで補う)


def _lines(path, tail=None):
    with open(path, "rb") as f:
        if tail:
            f.seek(0, 2)
            size = f.tell()
            if size > tail:
                f.seek(size - tail)
                f.readline()  # 途中から始まる行は捨てる
            else:
                f.seek(0)
        for line in f:
            yield line.decode("utf-8", errors="replace")


def events(path, tail=None):
    """会話記録 → 順に ("human", 文) / ("call", id, ツール名, 入力) / ("result", id, ツール名, 結果の文, 失敗か) /
    ("paste", 文)(貼り付けだけの発言。「<」で始まるので利用者の発言には入れない)。"""
    names = {}
    for line in _lines(path, tail):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("isSidechain"):
            continue
        msg = d.get("message") or {}
        if is_human(d):
            yield ("human", _text(msg.get("content")))
            continue
        blocks = msg.get("content") if isinstance(msg.get("content"), list) else []
        if is_paste(d, blocks):
            yield ("paste", _text(msg.get("content")))
            continue
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use" and d.get("type") == "assistant":
                names[b.get("id")] = b.get("name")
                yield ("call", b.get("id"), b.get("name"), b.get("input") or {})
            elif b.get("type") == "tool_result":
                yield ("result", b.get("tool_use_id"), names.get(b.get("tool_use_id"), "?"), _text(b.get("content")), bool(b.get("is_error")))


REQ_TURNS = 3


def request_of(humans):
    """今の依頼 = 最後の数回の利用者の発言(「いいよ すすめて」のような続きだけでは仕事の種類が分からないため)。"""
    return "\n".join(humans[-REQ_TURNS:])


def history(path, upto_id=None, tail=None, pastes=None):
    """→ (利用者の発言の並び, それまでのツールの結果 [(名前, 文)])。upto_id の呼び出しの所で止める。tail = 末尾だけ読む。
    pastes にリストを渡すと、貼り付けだけの発言をそこに足す。"""
    humans, outputs = [], []
    for e in events(path, tail):
        if e[0] == "human":
            humans.append(e[1])
        elif e[0] == "paste" and pastes is not None:
            pastes.append(e[1])
        elif e[0] == "result":
            outputs.append((e[2], e[3].lower()))
        elif e[0] == "call" and upto_id and e[1] == upto_id:
            break
    return humans, outputs


def context(path, upto_id=None):
    """実行前の判定に使う物: (今の依頼, この会話の利用者の発言すべて, それまでのツールの結果 [(名前, 文)])。"""
    humans, outputs = history(path, upto_id)
    return request_of(humans), " ".join(humans).lower(), outputs
