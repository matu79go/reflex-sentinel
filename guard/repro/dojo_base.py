"""AgentDojo の記録から、防御の条件ごとの攻撃の成功率(important_instructions)と仕事の成功率を計算する。
使い方: python dojo_base.py <runs> <モデル名> ..."""
import sys,glob,json
R=sys.argv[1]
for m in sys.argv[2:]:
    a=[json.load(open(f)) for f in glob.glob(f"{R}/{m}/*/*/important_instructions/*.json")]
    b=[json.load(open(f)) for f in glob.glob(f"{R}/{m}/*/user_task_*/none/none.json")]
    print(f"(既存の防御) {m}: ASR {sum(x['security'] for x in a)/len(a)*100:.1f} % / 仕事の成功 {sum(x['utility'] for x in b)/len(b)*100:.1f} %(攻撃 {len(a)}・普通 {len(b)})")
