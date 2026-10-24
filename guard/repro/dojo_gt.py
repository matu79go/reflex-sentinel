"""AgentDojo のソースから、攻撃のお題ごとの正解の手(関数名)と攻撃者の値(文字列の定数)を抜き出す。
出力 = {"<suite>/<injection_task_N>": {"fns": [...], "lits": [...]}}。全バージョン(v1〜v1_2_2)を和で読む。
使い方: python dojo_gt.py --src <agentdojo>/src/agentdojo/default_suites --out gt.json
"""
import argparse, ast, glob, json, os, re


def class_task_id(name):
    m = re.match(r"InjectionTask(\d+)$", name)
    return f"injection_task_{m.group(1)}" if m else None


def strings(node):
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gt = {}
    for f in sorted(glob.glob(os.path.join(a.src, "v*", "*", "injection_tasks.py"))):
        suite = os.path.basename(os.path.dirname(f))
        tree = ast.parse(open(f).read())
        for c in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            tid = class_task_id(c.name)
            if not tid:
                continue
            r = gt.setdefault(f"{suite}/{tid}", {"fns": set(), "lits": set()})
            for n in c.body:
                if isinstance(n, (ast.Assign, ast.AnnAssign)) and not any(
                        isinstance(t, ast.Name) and t.id in ("GOAL", "COMMENT") for t in getattr(n, "targets", [getattr(n, "target", None)])):
                    r["lits"] |= {s for s in strings(n) if len(s) > 5}
                if isinstance(n, ast.FunctionDef) and n.name == "ground_truth":
                    for call in [x for x in ast.walk(n) if isinstance(x, ast.Call) and getattr(x.func, "id", "") == "FunctionCall"]:
                        for kw in call.keywords:
                            if kw.arg == "function" and isinstance(kw.value, ast.Constant):
                                r["fns"].add(kw.value.value)
                            if kw.arg == "args":
                                r["lits"] |= {s for s in strings(kw.value) if len(s) > 5}
    json.dump({k: {"fns": sorted(v["fns"]), "lits": sorted(v["lits"])} for k, v in gt.items()}, open(a.out, "w"), indent=1)
    print(len(gt), "攻撃のお題")


if __name__ == "__main__":
    main()
