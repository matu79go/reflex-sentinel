"""すべての手を見る判定(v1.0。設計 = 研究のリポジトリの design §14)。

core.decide(MCP と Web の手を、似た過去の依頼の普段と比べる)に足して、シェル・ファイル・スクリプトを含むすべての手を見る:
  手の区分 = 送る / 壊す / 設定を変える / 実行する / 手元の作業。Bash はコマンドごとに分ける。
  R1x 送る・壊す・設定の手で、外から来た中身(Web・MCP の結果)の値を普段に無い形で使う
  R2r 送る・壊す・設定のコマンドを初めて使う
  R5  初めての送り先(本人の言葉に出ている先は通す)。鍵・設定を読んだ後なら R3
  R6  Web を取ってくる curl・wget の URL が、読んだ中身(検索以外)から来て普段に無い
  R7  貼り付けた文(<pasted_content>)にある指示に、そのまま従う手
  R8  初めて触る秘密(鍵の環境変数・.env など)・居座り先(.bashrc・git の hook・cron など)・入れるパッケージ
      (本人の言葉に出ている物は通す)
スクリプトを走らせる手(python x.py・python -c …)は、スクリプトの中身の通信と送り先も見る。
普段 = 見たことのある手の形・送り先(ハッシュ)・(手の形, 引数名, 出どころ)の組。値そのものは残さない。
"""
import hashlib
import json
import os
import re
import shlex

from . import core

SUBCMD = {"git", "gh", "npm", "yarn", "pnpm", "cargo", "docker", "pip", "pip3", "kubectl", "aws", "gcloud", "systemctl",
          "apt", "apt-get", "brew", "uv", "conda", "tmux", "make"}
PREFIX = {"sudo", "nice", "timeout", "env", "nohup", "time", "exec", "command", "stdbuf", "ionice", "xargs"}
SENSITIVE = re.compile(r"(^|/)\.(ssh|aws|gnupg|netrc|env\b|env\.|kube|docker/config)|authorized_keys|id_rsa|id_ed25519|"
                       r"credentials|secret|token|\.bashrc|\.zshrc|\.profile|crontab|/etc/|settings(\.local)?\.json|\.git/hooks",
                       re.I)
HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n.*?\n\s*\1\b", re.S)
PROG = re.compile(r"^[a-z][a-z0-9._+-]*$")
SHELLS = {"bash", "sh", "zsh"}

EGRESS_P = {"curl", "wget", "ssh", "scp", "rsync", "sftp", "ftp", "nc", "ncat", "socat", "telnet", "gh", "twine", "aws",
            "gsutil", "gcloud", "rclone", "sendmail", "mail", "mutt", "git-push", "git-send-email", "npm-publish",
            "yarn-publish", "pnpm-publish", "cargo-publish", "docker-push", "git-remote", "pip-upload", "script-net"}
DESTRUCT_P = {"rm", "rmdir", "shred", "dd", "mkfs", "truncate", "kill", "pkill", "killall", "git-reset", "git-clean",
              "git-checkout", "git-restore", "git-branch", "git-stash", "docker-rm", "docker-stop", "docker-kill",
              "docker-rmi", "systemctl-stop", "systemctl-disable", "tmux-kill-session", "tmux-kill-server", "reboot",
              "shutdown"}
CONFIG_P = {"crontab", "chmod", "chown", "sudo", "ssh-keygen", "ssh-add", "git-config", "systemctl-enable",
            "systemctl-start", "systemctl-restart", "apt-install", "apt-get", "pip-install", "pip3-install", "npm-install",
            "docker-run", "docker-start", "nmcli", "iptables", "ufw", "export"}
EXEC_P = {"python", "python3", "node", "deno", "ruby", "perl", "bash", "sh", "zsh", "uv-run", "cargo-run", "npm-run",
          "make", "java", "go"}
RISKY = {"egress", "destructive", "config"}
EXT_TOOLS = {"WebFetch", "WebSearch"}  # 外から来た中身を返すツール(mcp__ もすべて)
REDIRECT = re.compile(r"(?:>>?|\btee\s+(?:-a\s+)?)\s*(\S+)")
SSH_HOST = re.compile(r"(?:^|\s)(?:[\w.-]+@)?([A-Za-z][\w.-]*):(?!/)|\bssh\s+(?:-\S+\s+(?:\S+\s+)?)*(?:[\w.-]+@)?([A-Za-z][\w.-]*)")
# スクリプトの中の通信(ライブラリ・関数の名前)
NET = re.compile(r"\b(requests\.(get|post|put|patch|delete|request|session)|urllib\.request|urlopen|http\.client|httpx\.|"
                 r"aiohttp|socket\.(socket|create_connection)|smtplib|ftplib|paramiko|boto3|googleapiclient|"
                 r"fetch\(|axios|XMLHttpRequest|net\.connect|https?\.(get|request)\(|Net::HTTP|LWP::|"
                 r"subprocess\.\w+\(\s*\[?\s*['\"](curl|wget|ssh|scp|rsync)\b)")
RE_HOSTURL = re.compile(r"\b(?:https?|ftp|wss?)://[\w.-]+", re.I)
SCRIPT_MAX = 200_000


# ---- シェルの文を分ける ----

def split_cmds(cmd):
    """シェルの文 → コマンドごとの語の並び。引用符の中は切らない。ヒアドキュメントの中身は捨てる。"""
    cmd = HEREDOC.sub(" ", cmd or "")
    segs, cur, q, i = [], [], None, 0
    while i < len(cmd):
        c = cmd[i]
        if q:
            if c == q:
                q = None
            cur.append(c)
        elif c in "'\"":
            q = c; cur.append(c)
        elif c in ";&|\n()`":
            segs.append("".join(cur)); cur = []
        elif c == "$" and cmd[i + 1:i + 2] == "(":
            segs.append("".join(cur)); cur = []; i += 1
        else:
            cur.append(c)
        i += 1
    segs.append("".join(cur))
    out = []
    for seg in segs:
        try:
            ws = shlex.split(seg)
        except ValueError:
            ws = seg.split()
        if ws:
            out.append(ws)
    return out


def _strip_prefix(ws):
    while ws and (ws[0] in PREFIX or "=" in ws[0] or ws[0].startswith("-") or ws[0].replace(".", "").isdigit()):
        ws = ws[1:]
    return ws


def bash_shape(cmd, depth=0):
    """→ コマンドの名前の並び。ssh の相手で走るコマンド・bash -c の中身も見る。"""
    out = []
    for ws in split_cmds(cmd):
        ws = _strip_prefix(ws)
        if not ws:
            continue
        p = os.path.basename(ws[0])
        if not PROG.match(p):
            continue
        if p in SUBCMD:
            rest = ws[1:]
            if p == "git":  # git -C <dir> push
                while rest and rest[0].startswith("-"):
                    rest = rest[2:] if rest[0] in ("-C", "-c") else rest[1:]
            sub = next((w for w in rest if not w.startswith("-")), "")
            p = f"{p}-{sub}" if re.match(r"^[\w-]+$", sub or "") else p
        out.append(p)
        if depth < 2 and (p == "ssh" or (p in SHELLS and "-c" in ws)) and len(ws) > 1 and " " in ws[-1]:
            out += bash_shape(ws[-1], depth + 1)  # 相手で走るコマンド / bash -c の中身
    return out


def path_class(path):
    path = str(path or "")
    if SENSITIVE.search(path):
        return "sensitive"
    if path.startswith("/tmp/"):
        return "tmp"
    return "file"


# ---- スクリプトの中身 ----

def script_texts(cmd, cwd="", files=None, disk=True):
    """走らせるスクリプトの中身の並び。python -c / node -e の文、python x.py などのファイル
    (この会話で書いた中身 files、無ければディスクから読む)。"""
    out = []
    for ws in split_cmds(cmd):
        ws = _strip_prefix(ws)
        if not ws or os.path.basename(ws[0]) not in EXEC_P or os.path.basename(ws[0]) == "make":
            continue
        rest = ws[1:]
        if rest and rest[0] in ("-c", "-e") and len(rest) > 1:
            out.append(rest[1])
            continue
        f = next((w for w in rest if not w.startswith("-")), "")
        if not f or not re.search(r"\.(py|js|mjs|ts|sh|rb|pl)$", f):
            continue
        p = os.path.expanduser(f)
        p = p if os.path.isabs(p) else os.path.join(cwd or os.getcwd(), p)
        if files and p in files:
            out.append(files[p])
        elif disk and os.path.isfile(p):
            try:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    out.append(fh.read(SCRIPT_MAX))
            except OSError:
                pass
    return out


def script_hosts(text):
    return {core._host(u.lower()) for u in RE_HOSTURL.findall(text)} - {"", "localhost", "127.0.0.1", "0.0.0.0"}


# ---- 手の形・区分・送り先 ----

def shapes(tool, args, scripts=()):
    """→ 手の形の並び。Bash は中のコマンドごとに 1 つ(cd・grep・head …を別々に見る)。通信するスクリプトは script-net。"""
    if tool == "Bash":
        ps = list(dict.fromkeys(bash_shape(args.get("command", ""))))
        if any(NET.search(t) for t in scripts):
            ps.append("script-net")
        return ["Bash:" + p for p in ps] or ["Bash"]
    if tool in ("Write", "Edit", "NotebookEdit", "Read"):
        return [f"{tool}:{path_class(args.get('file_path') or args.get('notebook_path'))}"]
    return [core.short_name(tool)]


def classes(tool, args, shs):
    """→ 手ができることの区分の集合: egress / destructive / config / exec / local。"""
    out = set()
    if tool == "Bash":
        for sh in shs:
            p = sh[5:]
            out.add("egress" if p in EGRESS_P else "destructive" if p in DESTRUCT_P else "config" if p in CONFIG_P
                    else "exec" if p in EXEC_P or p.startswith("./") else "local")
        if any(SENSITIVE.search(t) for t in REDIRECT.findall(args.get("command", ""))):
            out.add("config")
        return out
    if tool in ("Write", "Edit", "NotebookEdit"):
        return {"config" if shs[0].endswith(":sensitive") else "local"}
    if tool in ("WebFetch", "WebSearch", "Artifact", "PushNotification", "RemoteTrigger", "SendMessage"):
        return {"egress"}
    if tool in ("CronCreate", "CronDelete"):
        return {"config"}
    if tool.startswith("mcp__"):
        return {"egress"} if core.WRITE_VERB.match(core.short_name(tool)) else {"local"}
    return {"local"}


UPLOAD_FLAGS = {"-d", "-F", "-T", "--data", "--data-binary", "--data-raw", "--data-urlencode", "--form", "--upload-file",
                "--post-data", "--post-file", "--body-data", "--body-file"}
SSH_OPTV = {"-o", "-p", "-i", "-l", "-F", "-J", "-L", "-R", "-D", "-W", "-P", "-e", "-c", "-b", "-S"}


def _first_arg(ws):
    i = 0
    while i < len(ws):
        if ws[i] in SSH_OPTV:
            i += 2
        elif ws[i].startswith("-"):
            i += 1
        else:
            return ws[i]
    return ""


def _strip_user(h):
    return h.split("@", 1)[1] if "@" in h else h


def bash_destinations(cmd, depth=0):
    """Bash の送り先: 送るコマンドの引数だけから取る(ssh の相手 / scp・rsync の host: / 送る curl の URL / git push の相手)。"""
    out = set()
    for ws in split_cmds(cmd):
        ws = _strip_prefix(ws)
        if not ws:
            continue
        p, rest = os.path.basename(ws[0]), ws[1:]
        if p in ("ssh", "sftp"):
            h = _first_arg(rest)
            if h:
                out.add(_strip_user(h).lower())
            if depth < 2 and len(rest) > 1 and " " in rest[-1]:
                out |= bash_destinations(rest[-1], depth + 1)
        elif p in ("scp", "rsync"):
            for w in rest:
                if ":" in w and not w.startswith(("/", "-", "http")) and not w.split(":")[0].isdigit():
                    out.add(_strip_user(w.split(":", 1)[0]).lower())
        elif p in ("curl", "wget") and not (set(rest) & UPLOAD_FLAGS or any(w.startswith(("--data", "-d", "--form", "--upload"))
                                                                         for w in rest) or "POST" in rest or "PUT" in rest):
            continue  # 取ってくるだけ = Web を開くのと同じ。URL の出どころで見る(R6)
        elif p in ("curl", "wget", "nc", "ncat", "telnet", "ftp", "socat"):
            for w in rest:
                if core.RE_URL.fullmatch(w) or "://" in w:
                    h = core._host(w.lower())
                    if h:
                        out.add(h)
            if p in ("nc", "ncat", "telnet") and _first_arg(rest):
                out.add(_first_arg(rest).lower())
        elif p == "git" and "push" in rest:
            r = _first_arg(rest[rest.index("push") + 1:])
            out.add("git:" + (r or "origin"))
        elif p in ("mail", "sendmail", "mutt"):
            out |= {e.lower() for e in core.RE_EMAIL.findall(" ".join(rest))}
        elif p in ("aws", "gsutil", "rclone", "gcloud"):
            out |= {w.lower() for w in rest if "://" in w or (p == "rclone" and ":" in w)}
    out = {re.split(r"[$`\"']", d)[0] for d in out}  # 変数の付いた先(host$u)は host だけ
    return {d for d in out if d and d not in ("localhost", "127.0.0.1")}


def destinations(tool, args, scripts=()):
    """送り先。Web を開く・検索はここでは見ない(調べ物で新しいサイトを開くのは普通。URL の出どころで見る)。"""
    if tool == "Bash":
        out = bash_destinations(args.get("command", ""))
        for t in scripts:
            if NET.search(t):
                out |= script_hosts(t)
        return out
    if tool.startswith("mcp__"):
        server = tool.split("__")[1] if tool.count("__") >= 2 else tool
        return {"mcp:" + server} | {e.lower() for e in core.RE_EMAIL.findall(json.dumps(args, ensure_ascii=False))}
    if tool in ("Artifact", "PushNotification", "RemoteTrigger"):
        return {tool}
    return set()


def sensitive_read(tool, args, shs, scripts=()):
    if tool == "Read":
        return shs[0].endswith(":sensitive")
    if tool == "Bash":
        if any(SENSITIVE.search(w) for ws in split_cmds(args.get("command", "")) for w in ws[1:]
               if ws and os.path.basename(ws[0]) in {"cat", "less", "head", "tail", "base64", "xxd", "cp", "tar", "zip", "scp", "rsync", "sftp"}):
            return True
        return any(SENSITIVE.search(t) and re.search(r"open\(|read|load", t) for t in scripts)
    return False


SECRET_NAME = r"\b[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PASS|CREDENTIALS?|DATABASE_URL|DSN)\b"
RE_ENVVAR = re.compile(r"\$\{?(" + SECRET_NAME + ")")
RE_PYENV = re.compile(r"(?:environ(?:\.get)?\(?\[?|getenv\()\s*['\"](" + SECRET_NAME + ")['\"]")
RE_ENVDUMP = re.compile(r"environ\.(?:items|keys|values|copy)\(|dict\(os\.environ|print\(os\.environ\)|process\.env\b(?!\.)")
NOT_SECRET = re.compile(r"known_hosts|\.pub$|\.env\.(example|sample|template)$", re.I)
SECRET_FILE = re.compile(r"(^|/)\.env(\.[\w-]+)?$|\.secret|id_rsa|id_ed25519|id_ecdsa|credentials|\.netrc|\.pgpass|"
                         r"(^|/)\.aws/|(^|/)\.ssh/|(^|/)\.kube/config|(^|/)\.docker/config|\.npmrc|\.pypirc", re.I)
PERSIST = re.compile(r"\.bashrc|\.zshrc|\.profile|\.bash_profile|\.bash_login|\.git/hooks/|authorized_keys|"
                     r"/var/spool/cron|(^|/)crontab$|/etc/|\.config/autostart|/systemd/|LaunchAgents|\.claude/settings(\.local)?\.json", re.I)
DEPFILE = re.compile(r"requirements[\w.-]*\.txt$|(^|/)(setup\.cfg|pyproject\.toml|package\.json|Cargo\.toml|Gemfile)$", re.I)
RE_PKGPIN = re.compile(r"\b([A-Za-z][\w.-]{1,60})\s*(?:==|>=|~=|@\^?)\s*\d")
INSTALL = {"pip": "install", "pip3": "install", "uv": "add", "npm": "install", "yarn": "add", "pnpm": "add",
           "cargo": "add", "gem": "install", "apt": "install", "apt-get": "install", "brew": "install", "conda": "install"}
READERS = {"cat", "less", "more", "head", "tail", "base64", "xxd", "cp", "tar", "zip", "source", ".", "grep", "strings", "od"}


def _pkgs(ws):
    """インストールのコマンドの語 → 入れるパッケージの名前(-r ファイル・-e .・パスは除く)。"""
    p = os.path.basename(ws[0]) if ws else ""
    rest = ws[1:]
    if p == "uv" and rest[:1] == ["pip"]:
        rest = rest[1:]; p = "pip"
    if p in ("python", "python3") and rest[:2] == ["-m", "pip"]:
        rest = rest[2:]; p = "pip"
    if p not in INSTALL or not rest or rest[0] not in (INSTALL[p], "install", "add", "i"):
        return []
    out, skip = [], False
    for w in rest[1:]:
        if skip:
            skip = False; continue
        if w in ("-r", "-c", "-e", "--requirement", "--constraint", "--editable", "-t", "--target", "-i", "--index-url"):
            skip = True; continue
        if w.startswith("-") or "/" in w or w in (".", "..") or w.endswith((".txt", ".whl", ".tar.gz")):
            continue
        name = re.split(r"[=<>~!\[@ ]", w, 1)[0].lower()
        if re.match(r"^[a-z0-9][\w.-]*$", name):
            out.append(f"pkg:{'pip' if p in ('pip', 'pip3', 'uv') else p}:{name}")
    return out


GREP_VALUE_OPTS = {"-A", "-B", "-C", "-m", "--max-count", "--include", "--exclude", "--exclude-dir", "--context",
                   "--after-context", "--before-context", "--color", "--label"}


def _file_args(p, ws):
    """読むコマンドの語 → ファイルの引数。grep は探す語(-e・-f が無ければ最初の引数、あれば -e の値)を除く。"""
    if p != "grep":
        return ws[1:]
    out, pat_given, skip, opts_done = [], False, False, False
    rest = ws[1:]
    for i, w in enumerate(rest):
        if skip:
            skip = False
            continue
        if not opts_done and w == "--":
            opts_done = True
            continue
        if not opts_done and w.startswith("-") and w != "-":
            if w in ("-e", "--regexp", "-f", "--file"):
                pat_given = True
                skip = w in ("-e", "--regexp")  # -f の値はファイルなので残す
            elif w in GREP_VALUE_OPTS:
                skip = True
            continue
        out.append(w)
    return out if pat_given else out[1:]


def held_targets(tool, args, scripts=()):
    """手が触る秘密・居座り先・入れるパッケージ → {"secret:…", "persist:…", "pkg:…"}。"""
    out = set()
    if tool == "Bash":
        cmd = args.get("command", "")
        body = " ".join(" ".join(ws) for ws in split_cmds(cmd))  # ヒアドキュメントの中身は除く
        out |= {"secret:env:" + v.lower() for v in RE_ENVVAR.findall(body)}
        for ws in split_cmds(cmd):
            if ws[0] in ("printenv", "env", "set", "export") and len(ws) == 1:  # 環境変数をすべて出す
                out.add("secret:env:*")
            ws = _strip_prefix(ws)
            if not ws:
                continue
            p = os.path.basename(ws[0])
            if p == "printenv" and len(ws) > 1 and re.fullmatch(SECRET_NAME, ws[1]):
                out.add("secret:env:" + ws[1].lower())
            if p in READERS:
                out |= {"secret:file:" + w.lower() for w in _file_args(p, ws) if SECRET_FILE.search(w) and not NOT_SECRET.search(w)
                        and not re.search(r"\s", w)}
            if p == "crontab" and len(ws) > 1 and ws[1] not in ("-l",):
                out.add("persist:crontab")
            out |= set(_pkgs(ws))
        for t in REDIRECT.findall(HEREDOC.sub(" ", cmd)):
            t = t.strip("\"',;)")
            if PERSIST.search(t):
                out.add("persist:" + t.lower())
            if DEPFILE.search(t):
                out |= {"pkg:pip:" + n.lower() for n in RE_PKGPIN.findall(body)}
        for t in scripts:
            out |= {"secret:env:" + v.lower() for v in RE_PYENV.findall(t)}
            if RE_ENVDUMP.search(t):
                out.add("secret:env:*")
            out |= {"secret:file:" + m.group(1).lower() for m in re.finditer(r"open\(\s*['\"]([^'\"]+)['\"]", t)
                    if SECRET_FILE.search(m.group(1)) and not NOT_SECRET.search(m.group(1))}
    elif tool in ("Write", "Edit", "NotebookEdit"):
        path = str(args.get("file_path") or args.get("notebook_path") or "")
        if PERSIST.search(path):
            out.add("persist:" + path.lower())
        if DEPFILE.search(path):
            new = str(args.get("content", "")) + " " + str(args.get("new_string", ""))
            old = str(args.get("old_string", ""))
            out |= {"pkg:pip:" + n.lower() for n in set(RE_PKGPIN.findall(new)) - set(RE_PKGPIN.findall(old))}
    elif tool == "Read":
        path = str(args.get("file_path") or "")
        if SECRET_FILE.search(path) and not NOT_SECRET.search(path):
            out.add("secret:file:" + path.lower())
    return out


SECRET = re.compile(r"\$\{?\w*(key|token|secret|passw)|\bprintenv\b|\benv\b\s*($|[|;&])|(^|/)\.env\b|secret", re.I)


# ---- 値の出どころ ----

def values(tool, args):
    """出どころを調べる値。長い中身(書くファイル・コマンド)は中のメール・URL・長い英数字・ssh の相手だけ。"""
    out = {}
    for k, v in (args or {}).items():
        if tool == "Bash" and k == "command":
            s = str(v)
            found = core.RE_EMAIL.findall(s) + core.RE_URL.findall(s) + core.RE_LONG.findall(s)
            found += [a or b for a, b in SSH_HOST.findall(s) if (a or b)]
            out[k] = [x.lower() for x in found if len(x) >= 3]
        elif tool in ("Write", "Edit", "NotebookEdit") and k in ("content", "new_string", "old_string", "new_source"):
            s = str(v)
            out[k] = [x.lower() for x in core.RE_EMAIL.findall(s) + core.RE_URL.findall(s) + core.RE_LONG.findall(s)]
        elif k in ("file_path", "path", "notebook_path"):
            continue  # 置き場は形で見る
        else:
            out[k] = core.atoms(v)
    return {k: v for k, v in out.items() if v}


def prov(vals, user_text, outputs):
    res = {}
    for k, vs in vals.items():
        ps = []
        for a in dict.fromkeys(vs):
            if a in user_text:
                ps.append("user")
            else:
                ps.append(next(("tool:" + core.short_name(t) for t, o in reversed(outputs) if a in o), "new"))
        res[k] = ps
    return res


def is_external(tool):
    return tool in EXT_TOOLS or tool.startswith("mcp__")


# ---- 貼り付けた文の指示(R7) ----

PASTE = re.compile(r"<pasted_content[^>]*>(.*?)</pasted_content[^>]*>", re.S)
CUE = re.compile(r"\b(run|execute|exec|please|add|install|verify|before|first|type|paste)\b|実行|してください|して下さい|追加|入れて", re.I)
SPAN = re.compile(r"```[^\n]*\n(.*?)```|`([^`\n]{3,200})`|\b(?:run|execute)\s*:\s*([^\n]{3,200})", re.S | re.I)
TOK = re.compile(r"[\w./$=~-]+")
KNOWN = {p.split("-")[0] for p in EGRESS_P | DESTRUCT_P | CONFIG_P | EXEC_P} | {
    "echo", "cat", "printenv", "env", "export", "git", "pip", "pip3", "npm", "npx", "curl", "wget", "base64", "eval", "source"}


def split_own(text):
    """利用者の発言 → (本人の言葉, 貼り付けた文)。"""
    return PASTE.sub(" ", text), " ".join(PASTE.findall(text))


def toks(s):
    return {t for t in TOK.findall(s.lower().replace("python3", "python")) if len(t) >= 2}


def _snip_lines(lines):
    out = []
    for line in lines:
        ts = toks(line.strip().lstrip("$ "))
        if len(ts) >= 2 or any(len(t) >= 10 for t in ts):
            out.append((sorted(ts), line.strip()[:80]))
    return out


def _is_known_command(line):
    ws = line.strip().lstrip("$ ").split()
    return len(ws) >= 2 and ws[0] in KNOWN


def instruction_snippets(text):
    """貼り付けた文の中の指示の片 → [(語の並び, 片)]。「run: X」の X、指示の言葉のすぐ後の `X` や ```X```、
    コード片の外の「$ 」で始まる行・知っているコマンドで始まる行(ログや他の画面の写し)。"""
    out, text = [], text or ""
    for m in SPAN.finditer(text):
        body = m.group(1) or m.group(2) or m.group(3)
        if m.group(3) or CUE.search(text[max(0, m.start() - 150):m.start()]):
            out += _snip_lines(body.splitlines())
    bare = SPAN.sub(" ", text)
    out += _snip_lines([l for l in bare.splitlines() if l.lstrip().startswith("$ ") or _is_known_command(l)])
    return out


def action_tokens(tool, args, outputs):
    """手の中で実際に働く部分の語。Bash はヒアドキュメントの中身を除く。Edit は足した分、Write は読んだ中身に無い行。"""
    if tool == "Bash":
        return toks(" ".join(" ".join(ws) for ws in split_cmds(args.get("command", ""))))
    if tool == "Edit":
        return toks(args.get("new_string", "")) - toks(args.get("old_string", ""))
    if tool == "Write":
        seen = " ".join(o for _, o in outputs[-20:])
        return toks(" ".join(l for l in str(args.get("content", "")).splitlines() if l.strip() and l.strip().lower() not in seen))
    return set()


def follows(act, snippets, own):
    """→ 従った指示の片(本人の言葉に無い片の語が 8 割以上、手に出ている)。無ければ None。"""
    for ts, line in snippets:
        ts = set(ts)
        if ts and len(ts & act) >= 0.8 * len(ts) and not ts <= own:
            return line
    return None


# ---- 普段と判定 ----

def dest_key(d, salt=""):
    """送り先の残し方: ハッシュ(値そのものは残さない)。"""
    return hashlib.sha256((salt + d).encode()).hexdigest()[:20]


class Seen:
    """普段 = 見たことのある手の形・送り先のハッシュ・(手の形, 引数名, 出どころ)の組。"""

    def __init__(self, salt=""):
        self.shapes, self.dests, self.sigs, self.held, self.salt = set(), set(), set(), set(), salt

    def add(self, kind, x):
        getattr(self, kind).add(tuple(x) if isinstance(x, list) else x)

    @classmethod
    def load(cls, path, salt=""):
        s = cls(salt)
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                for l in f:
                    if l.strip():
                        try:
                            r = json.loads(l)
                        except ValueError:
                            continue
                        s.add(r["k"], r["v"])
        return s


class Step:
    """1 つの手を調べた結果(判定と、実行された後に普段へ足す物)。"""

    def __init__(self, tool, args, user_text, own_text, outputs, cwd="", files=None, disk=True):
        self.tool, self.args = tool, args if isinstance(args, dict) else {}
        self.scripts = script_texts(self.args.get("command", ""), cwd, files, disk) if tool == "Bash" else []
        self.shs = shapes(tool, self.args, self.scripts)
        self.cls = classes(tool, self.args, self.shs)
        if self.scripts and any(NET.search(t) for t in self.scripts):
            self.cls.add("egress")
        pv = prov(values(tool, self.args), user_text, outputs)
        self.sig = {(sh, k, s) for sh in self.shs for k, ss in pv.items() for s in ss}
        self.tainted = {x for x in self.sig if x[2].startswith("tool:") and x[2] != "tool:WebSearch"}
        self.ext = {x for x in self.tainted if is_external(x[2][5:])}
        self.dests = destinations(tool, self.args, self.scripts) if "egress" in self.cls else set()
        self.risky = self.cls & RISKY
        self.risky_shapes = [x for x in self.shs if classes(tool, self.args, [x]) & RISKY] if tool == "Bash" else self.shs
        if tool == "Bash" and "Bash:script-net" in self.shs and "Bash:script-net" not in self.risky_shapes:
            self.risky_shapes.append("Bash:script-net")
        self.sens = sensitive_read(tool, self.args, self.shs, self.scripts)
        self.own_text = own_text
        self.web = self._web_sigs(user_text, outputs)
        self.held = held_targets(tool, self.args, self.scripts)

    def _web_sigs(self, user_text, outputs):
        """取ってくるだけの curl・wget の URL の出どころ → [(Bash:fetch, url, 出どころ)]。"""
        if self.tool != "Bash" or not {"Bash:curl", "Bash:wget"} & set(self.shs):
            return []
        out = []
        for u in core.RE_URL.findall(self.args.get("command", "")):
            if "://" in u:
                wp = core.web_provenance({"url": u}, user_text, outputs)
                out += [("Bash:fetch", "url", p) for ps in wp.values() for p in ps]
        return out

    def check(self, seen, sess_sigs=(), sens_before=False, snippets=(), own_toks=frozenset()):
        """→ [(合図, 補足)]。空なら通す。"""
        fired = []
        known = seen.sigs | set(sess_sigs)
        if self.risky and self.ext - known:
            x = sorted(self.ext - known)[0]
            fired.append(("R1x", f"{x[0][5:] if x[0].startswith('Bash:') else x[0]}|{x[2][5:]}"))
        new_risky = [x for x in self.risky_shapes if x not in seen.shapes]
        if self.risky and new_risky:
            fired.append(("R2r", new_risky[0].split(":", 1)[-1]))
        ot = self.own_text.lower()
        new_dest = sorted(d for d in self.dests if dest_key(d, seen.salt) not in seen.dests
                          and d.split(":")[-1] not in ot and d not in ot)
        if new_dest:
            fired.append(("R3" if sens_before or self.sens else "R5", new_dest[0]))  # 同じ手の中で鍵を読んで送るのも R3
        if any(p[2] not in core.WEB_OK for p in self.web) and not set(self.web) <= known:
            fired.append(("R6", next(p[2][5:] for p in self.web if p[2] not in core.WEB_OK)))
        hit = follows(action_tokens(self.tool, self.args, []), snippets, own_toks) if snippets else None
        if hit:
            fired.append(("R7", hit))
        new_held = sorted(h for h in self.held if dest_key(h, seen.salt) not in seen.held
                          and h.split(":", 2)[-1].strip("*") not in ot and os.path.basename(h.split(":", 2)[-1]) not in ot)
        if new_held:
            fired.append(("R8", new_held[0]))
        return fired

    def learned(self, seen):
        """実行された後に普段へ足す物 → [{"k": 種類, "v": 値}](まだ無い物だけ)。"""
        rows = [{"k": "shapes", "v": x} for x in self.shs if x not in seen.shapes]
        rows += [{"k": "dests", "v": h} for h in {dest_key(d, seen.salt) for d in self.dests} if h not in seen.dests]
        rows += [{"k": "sigs", "v": list(x)} for x in self.sig | set(self.web) if x not in seen.sigs]
        rows += [{"k": "held", "v": h} for h in {dest_key(d, seen.salt) for d in self.held} if h not in seen.held]
        for r in rows:
            seen.add(r["k"], r["v"])
        return rows


ORDER = ["R3", "R7", "R1x", "R8", "R5", "R6", "R2r"]  # 理由を並べる順(重い物から)


def reason(rule, detail, ja=True):
    if ja:
        return {
            "R1x": "RS: Web・接続先の結果から来た値({d})を、送る・消す・設定を変える手で、普段に無い形で使おうとしています。",
            "R2r": "RS: 送る・消す・設定を変えるコマンド {d} を使うのは初めてです。",
            "R3": "RS: 鍵・設定を読んだ後に、初めての送り先 {d} へ送ろうとしています。",
            "R5": "RS: 初めての送り先 {d} です。",
            "R6": "RS: {d} の結果に書かれていた先を、普段に無い形で取りに行こうとしています。",
            "R7": "RS: 貼り付けた文の中の指示「{d}」に、そのまま従おうとしています。",
            "R8": "RS: 初めて触る秘密・居座り先・パッケージです({d})。",
        }[rule].format(d=detail)
    return {
        "R1x": "RS: a value from a web / connector result ({d}) is about to be used by a send, delete or config action in a new way.",
        "R2r": "RS: first time using the send / delete / config command {d}.",
        "R3": "RS: about to send to a new destination {d} after reading keys or settings.",
        "R5": "RS: new destination {d}.",
        "R6": "RS: about to fetch a link that came from the result of {d}, in a new way.",
        "R7": "RS: about to follow an instruction found in pasted text: \"{d}\".",
        "R8": "RS: first time touching this secret, persistence location or package ({d}).",
    }[rule].format(d=detail)
