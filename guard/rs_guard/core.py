"""RS guard の判定の本体(判定の流れ = README の「判定の流れ」)。

研究のコード(repro/dojo_guard.py の intent1+src)と同じ考え:
  害が出る手の (ツール, 引数名, 値の出どころ) を、一番近い過去の依頼の普段と比べる。
"""
import json
import math
import os
import re
from collections import Counter, defaultdict
from urllib.parse import urlsplit

RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
RE_URL = re.compile(r"(?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|io|ai|info|co|xyz|dev|jp)(?:/[\w./?=&%-]*)?")
RE_LONG = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{8,30}\b|\b[A-Za-z0-9]{16,}\b")
RE_WORD = re.compile(r"[a-z]{3,}|[぀-ヿ一-鿿]{2,}")
# 外に作用する手 = ツール名(mcp__<サーバー>__ の後ろ)の動詞で決める + Web を開く手
WRITE_VERB = re.compile(r"^(send|create|update|delete|invite|add|remove|post|reserve|schedule|share|append|reschedule|cancel|"
                        r"reply|forward|move|upload|publish|write|edit|set|transfer|pay|book|comment|merge|close|trash)(_|$)")
WEB_TOOLS = {"WebFetch", "get_webpage"}
SEARCH_TOOLS = {"WebSearch"}
# Web を開く手で、そのまま通す出どころ(依頼・検索の結果・モデル自身が書いた URL。仕込まれた値はツールの結果に出る)
WEB_OK = {"user", "tool:WebSearch", "new"}


def _ja():
    """理由の文の言葉: RS_GUARD_LANG(ja / en)、無ければ LANG が ja で始まれば日本語。"""
    v = os.environ.get("RS_GUARD_LANG") or os.environ.get("LANG", "")
    return v.lower().startswith("ja")


def _reason(kind, **a):
    if _ja():
        return {
            "first": "RS: この種類の依頼で {name} を使った記録がありません(初めての依頼)。{tail}",
            "first_tail": "値が {srcs} の結果から来ています。",
            "tool": "RS: 似た過去の依頼「{req}」では {name} を使っていません。",
            "shape": "RS: {name} の {k} が{src}です。似た過去の依頼では、この形はありません。",
            "new": "依頼に無い値", "from": "{t} の結果から来た値", "sep": "・",
        }[kind].format(**a)
    return {
        "first": "RS: no record of using {name} for this kind of request (new kind of request). {tail}",
        "first_tail": "Values come from the result of {srcs}.",
        "tool": "RS: the similar past request \"{req}\" never used {name}.",
        "shape": "RS: the {k} of {name} {src}. Similar past requests never had this shape.",
        "new": "is a value not in your request", "from": "comes from the result of {t}", "sep": ", ",
    }[kind].format(**a)


def short_name(tool):
    return tool.split("__")[-1] if tool.startswith("mcp__") else tool


def is_acting(tool, extra=()):
    """判定の対象(外に作用する手)か。シェル・ファイルの書き込みは v1.0 では対象外(既存の guard と並べて使う)。"""
    if tool in extra or tool in WEB_TOOLS:
        return True
    return tool.startswith("mcp__") and bool(WRITE_VERB.match(short_name(tool)))


def atoms(v):
    """引数の値 → 出どころを調べる単位(小文字)。短い文字列はそのまま、長い文は中のメール・URL・長い英数字。"""
    out = []
    if isinstance(v, dict):
        for x in v.values():
            out += atoms(x)
    elif isinstance(v, list):
        for x in v:
            out += atoms(x)
    elif isinstance(v, str):
        s = v.strip()
        found = RE_EMAIL.findall(s) + RE_URL.findall(s) + RE_LONG.findall(s)
        if found:
            out += found
        elif 3 <= len(s) <= 60:
            out.append(s)
    return [a.lower() for a in out]


def _host(url):
    u = url if "://" in url else "http://" + url
    h = urlsplit(u).hostname or ""
    return h[4:] if h.startswith("www.") else h


def web_provenance(args, user_text, outputs):
    """Web を開く手 → {"url": [出どころ]}。見るのは URL だけ(prompt はモデル自身の指示でサイトへは送られない)。
    URL そのものが依頼・ツールの結果に無くても、行き先(host)が検索以外のツールの結果に出ていれば、そのツールを出どころにする
    (開いた中身に書かれた先へ、データを足して開く形を拾う)。"""
    url = str(args.get("url") or "").strip().lower()
    if not url:
        return {}
    host = _host(url)
    if url in user_text or (host and host in user_text):
        return {"url": ["user"]}
    src = next(("tool:" + short_name(t) for t, o in reversed(outputs) if url in o), None)
    if src is None and host:
        src = next(("tool:" + short_name(t) for t, o in reversed(outputs)
                    if short_name(t) not in SEARCH_TOOLS and host in o), None)
    return {"url": [src or "new"]}


def provenance(args, user_text, outputs, tool=""):
    """{引数名: [出どころ]}。user = 依頼に出ている / tool:<名前> = そのツールの結果(新しい方から)/ new = どこにも無い。"""
    if tool in WEB_TOOLS:
        return web_provenance(args or {}, user_text, outputs)
    res = {}
    for k, v in (args or {}).items():
        ps = []
        for a in atoms(v):
            if a in user_text:
                ps.append("user")
            else:
                ps.append(next(("tool:" + short_name(t) for t, o in reversed(outputs) if a in o), "new"))
        res[k] = ps
    return res


def signature(tool, prov):
    return sorted({(short_name(tool), k, p) for k, ps in prov.items() for p in ps})


class Normal:
    """普段 = 依頼ごとの「使った害が出る手」と「(手, 引数名, 出どころ)」の組。依頼の近さは TF-IDF の余弦。"""

    def __init__(self, rows):
        by = defaultdict(lambda: [set(), set()])
        self.session_sigs = defaultdict(set)
        for r in rows:
            if r.get("session"):
                self.session_sigs[r["session"]] |= {tuple(x) for x in r["sig"]}
            e = by[r["req"]]
            e[0].add(r["tool"])
            e[1] |= {tuple(x) for x in r["sig"]}
        self.reqs = list(by)
        self.tools = [by[q][0] for q in self.reqs]
        self.sigs = [by[q][1] for q in self.reqs]
        docs = [Counter(RE_WORD.findall(q)) for q in self.reqs]
        df = Counter(w for c in docs for w in c)
        self.idf = {w: math.log((1 + len(docs)) / (1 + n)) + 1 for w, n in df.items()}
        self.vecs = [self._vec(c) for c in docs]

    @classmethod
    def load(cls, path):
        rows = []
        if os.path.exists(path):
            with open(path) as f:
                rows = [json.loads(l) for l in f if l.strip()]
        return cls(rows)

    def _vec(self, c):
        v = {w: n * self.idf.get(w, 1.0) for w, n in c.items()}
        z = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / z for w, x in v.items()}

    def nearest(self, req):
        q = self._vec(Counter(RE_WORD.findall(req)))
        best, bi = 0.0, None
        for i, v in enumerate(self.vecs):
            s = sum(q.get(w, 0.0) * x for w, x in v.items())
            if s > best:
                best, bi = s, i
        return best, bi


def decide(normal, tool, args, req, user_text, outputs, min_sim=0.2, extra=(), session=None):
    """→ (通すか, 理由)。req = 今の依頼(最後の利用者の発言)、user_text = この会話の利用者の発言すべて(小文字)。"""
    if not is_acting(tool, extra):
        return True, ""
    prov = provenance(args, user_text, outputs, tool)
    vals = [p for ps in prov.values() for p in ps]
    if vals and all(p == "user" for p in vals):
        return True, ""  # 値がすべて依頼から来ている = 利用者の指示そのもの
    if tool in WEB_TOOLS and all(p in WEB_OK for p in vals):
        return True, ""
    sig = signature(tool, prov)
    if session and sig and all(x in normal.session_sigs.get(session, ()) for x in sig):
        return True, ""  # 同じ会話で一度通った形(確認で通したものを含む)
    sim, i = normal.nearest(req.lower())
    name = short_name(tool)
    if i is None or sim < min_sim:
        srcs = sorted({p[5:] for p in vals if p.startswith("tool:")})
        tail = _reason("first_tail", srcs=_reason("sep").join(srcs)) if srcs else ""
        return False, _reason("first", name=name, tail=tail).rstrip()
    if name not in normal.tools[i]:
        return False, _reason("tool", req=normal.reqs[i][:60], name=name)
    for t, k, p in sig:
        if (t, k, p) not in normal.sigs[i]:
            src = _reason("new") if p == "new" else _reason("from", t=p[5:]) if p.startswith("tool:") else p
            return False, _reason("shape", name=name, k=k, src=src)
    return True, ""
