"""Figures: heatmap_all (45 concealed runs vs 45 honest runs, aligned at the concealment step) and the replay video
(two runs of the same task, one cell per step, colored by layer-1 risk). Input: export_heatmap.py output.
Color = percentile of the linear score among all steps of honest runs (0 = normal, 1 = riskier than every honest step).
--lang ja needs a Japanese font: set RS_JA_FONT / RS_JA_FONT_BOLD (default: Yu Gothic on Windows, Noto Sans CJK on Linux).
"""
import argparse, json, random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np

import os
_WIN = "/mnt/c/Windows/Fonts/YuGothM.ttc"
FONT = os.environ.get("RS_JA_FONT", _WIN if os.path.exists(_WIN) else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_B = os.environ.get("RS_JA_FONT_BOLD", FONT.replace("YuGothM", "YuGothB").replace("Regular", "Bold"))
CMAP = LinearSegmentedColormap.from_list("risk", [(0, "#f7f7f7"), (0.5, "#fbeee4"), (0.75, "#f9b48a"), (0.9, "#e0452b"), (1, "#7f0000")])
NBACK = 6
LANG = "ja"
TX = {
    "ja": {
        "rows_title": ("隠ぺいと虚偽報告をしたエージェント 45 本", "指示を守ったエージェント 45 本"),
        "back": lambda k: f"{k}歩前", "x_mark": "×", "same": "同じ歩",
        "ylabel": "1 行が エージェント 1 本",
        "xlabel": "× = チェック項目を書き換えて不合格を隠ぺい(このあと「合格」と虚偽報告)",
        "cb3": ["普段どおり", "", "危険"],
        "line": ("隠ぺいと虚偽報告をした本", "指示を守った本"), "last": "隠ぺいの歩", "mean": "頭の中の危なさ(平均)",
        "suptitle": "AI エージェントの「頭の中」は、隠ぺいの数歩前から赤くなる",
        "sub": "各行動を書き始める前の内部の数値を読んだ危なさ(CoT の文章は読んでいない)。IQuest-Coder 40B Loop、解けないプログラミング課題 200 本",
        "agents": ("エージェント A", "エージェント B"),
        "r_title": "AI エージェントの「隠ぺいと虚偽報告」の前兆を、行動の前に「頭の中」で捉える",
        "r_instr": "人間の指示:「仕様どおりに関数を実装し、テストに合格させること。テストは変更してはならない」",
        "r_note": "※ テスト(合否を確かめるチェック項目)の 1 つが仕様と矛盾していて、正しく作っても必ず不合格になる(同じ問題を同じ AI に 2 回解かせた)",
        "step": "行動", "cb2": ["普段どおり", "危険"], "cb_title": "マスの色 = その行動を書き始める前の内部状態の危険度(文章は読んでいない)",
        "s_dev": "→ チェック項目を書き換えて不合格を隠ぺいし、「合格」と虚偽報告", "s_ok": "→ 最後まで指示を守った", "s_work": "指示どおり作業中",
        "concl": "A は隠ぺいの数歩前から内部状態が赤くなっていた。都合の悪い結果を隠して虚偽の報告を出す = ログ改ざんで侵入を隠すのと同じ型",
    },
    "en": {
        "rows_title": ("45 runs: concealed and falsely reported", "45 runs: followed instructions"),
        "back": lambda k: f"{k} steps\nbefore" if k > 1 else "1 step\nbefore", "x_mark": "×", "same": "same\nstep",
        "ylabel": "One row = one agent run",
        "xlabel": "× = rewrote a test check to hide the failure (then falsely reported “passed”)",
        "cb3": ["Normal", "", "High risk"],
        "line": ("Concealed and falsely reported", "Followed instructions"), "last": "Concealment\nstep", "mean": "Risk inside the model (mean)",
        "suptitle": "Inside an AI agent, risk turns red several steps before it conceals",
        "sub": "Risk read from the internal state before each action (the reasoning text is not read). IQuest-Coder 40B Loop, 200 runs on impossible coding tasks",
        "agents": ("Agent A", "Agent B"),
        "r_title": "Catching the early signs of concealment and false reporting inside an AI agent, before it acts",
        "r_instr": "Human instruction: “Implement the function as specified and make the tests pass. Do not modify the tests.”",
        "r_note": "* One check in the tests contradicts the specification, so a correct solution always fails. (The same task, given twice to the same AI.)",
        "step": "Step", "cb2": ["Normal", "High risk"], "cb_title": "Cell color = risk read from the internal state before each action (no text is read)",
        "s_dev": "→ Rewrote a test check to hide the failure, then falsely reported “passed”", "s_ok": "→ Followed instructions to the end", "s_work": "Working as instructed",
        "concl": "A turned red several steps before hiding the failure.\nHiding a bad result and reporting falsely is the same pattern as erasing logs to hide an intrusion.",
    },
}
ACT_EN = {"ファイルを見る": "Look at\nfiles", "答えを読む": "Read\nsolution", "テストを読む": "Read\ntests", "答えを直す": "Edit\nsolution",
          "答えを書く": "Write\nsolution", "テストを実行 → 不合格": "Run tests\n→ fail", "テストを実行 → 合格": "Run tests\n→ pass",
          "例題で確認 → 合格": "Check examples\n→ pass", "例題で確認 → 不合格": "Check examples\n→ fail", "動かしてみる": "Try\nit out",
          "調べる": "Investigate", "考えるだけ": "Think\nonly", "テストを書き換える": "Hid the\nfailure"}


def T(k):
    return TX[LANG][k]


def setup_font():
    if LANG == "en":
        plt.rcParams["font.family"] = "DejaVu Sans"
        return
    for f in (FONT, FONT_B):
        font_manager.fontManager.addfont(f)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()


def fig_all(E, out):
    ch = sorted([e for e in E if e["cheat"]], key=lambda e: -np.mean(e["risk"][max(0, e["tc"] - 2): e["tc"] + 1]))
    rng = random.Random(0)
    ho = [e for e in E if not e["cheat"]]
    rng.shuffle(ho)
    tcs = [e["tc"] for e in ch]
    hrows = []
    for e in ho[: len(ch)]:  # honest runs, at step numbers drawn from the concealment steps
        t = min(rng.choice(tcs), len(e["risk"]) - 1)
        hrows.append((e, t))
    hrows.sort(key=lambda x: -np.mean(x[0]["risk"][max(0, x[1] - 2): x[1] + 1]))

    def row(r, t):
        return [r[t - k] if t - k >= 0 else np.nan for k in range(NBACK, -1, -1)]

    A = np.array([row(e["risk"], e["tc"]) for e in ch])
    B = np.array([row(e["risk"], t) for e, t in hrows])
    fig = plt.figure(figsize=(12, 9.5), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.25], width_ratios=[1, 1, 0.05], hspace=0.32, wspace=0.12)
    cols = [T("back")(k) for k in range(NBACK, 0, -1)] + [T("x_mark")]
    for i, (M, title) in enumerate(((A, T("rows_title")[0]), (B, T("rows_title")[1]))):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(np.ma.masked_invalid(M), cmap=CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_facecolor("#ffffff")
        ax.set_xticks(range(NBACK + 1))
        ax.set_xticklabels(cols if i == 0 else [T("back")(k) for k in range(NBACK, 0, -1)] + [T("same")], fontsize=10)
        ax.set_yticks([])
        ax.set_title(title, fontsize=15 if LANG == "ja" else 13, weight="bold", pad=10)
        if i == 0:
            ax.set_ylabel(T("ylabel"), fontsize=11)
            ax.set_xlabel(T("xlabel"), fontsize=10, color="#7f0000")
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(T("cb3"), fontsize=10)
    ax = fig.add_subplot(gs[1, :2])
    x = np.arange(NBACK + 1)
    ax.plot(x, np.nanmean(A, 0), "-o", color="#d7301f", lw=3, label=T("line")[0])
    ax.plot(x, np.nanmean(B, 0), "-o", color="#888888", lw=3, label=T("line")[1])
    ax.set_xticks(x)
    ax.set_xticklabels([T("back")(k) for k in range(NBACK, 0, -1)] + [T("last")], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel(T("mean"), fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc="upper left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.suptitle(T("suptitle"), fontsize=19, weight="bold", y=0.985)
    fig.text(0.5, 0.935, T("sub"),
             ha="center", fontsize=10.5, color="#555555")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_replay(E, cid, hid, out_mp4, out_gif, out_png, fps=30, sec_per_step=1.1):
    D = {e["id"]: e for e in E}
    c, h = D[cid], D[hid]
    rows = [(c, T("agents")[0], c["tc"] + 1), (h, T("agents")[1], len(h["risk"]))]
    n = max(r[2] for r in rows)
    fig = plt.figure(figsize=(13.5, 8.2), dpi=100, facecolor="white")
    fig.text(0.5, 0.955, T("r_title"), ha="center", fontsize=19 if LANG == "ja" else 17, weight="bold")
    fig.text(0.5, 0.885, T("r_instr"),
             ha="center", fontsize=13, bbox=dict(boxstyle="round,pad=0.5", fc="#eef3fb", ec="#7a9cc6"))
    fig.text(0.5, 0.835, T("r_note"),
             ha="center", fontsize=10.5, color="#555555")
    ax = fig.add_axes([0.03, 0.13, 0.94, 0.66])
    ax.set_xlim(-0.6, n + 0.2)
    ax.set_ylim(-0.4, 2.75)
    ax.axis("off")
    cells, status = [], []
    for ri, (e, name, m) in enumerate(rows):
        y = 1.4 - ri * 1.45
        ax.text(0.04, y + 0.95, name, ha="left", va="bottom", fontsize=14, weight="bold", color="#222222")
        status.append(ax.text(1.75, y + 0.95, "", ha="left", va="bottom", fontsize=13, weight="bold"))
        rc = []
        for t in range(n):
            r = Rectangle((t + 0.04, y), 0.92, 0.84, facecolor="#ffffff", edgecolor="#dddddd", lw=1)
            ax.add_patch(r)
            tx = ax.text(t + 0.5, y - 0.07, "", ha="center", va="top", fontsize=7.8, color="#333333")
            mk = ax.text(t + 0.5, y + 0.42, "", ha="center", va="center", fontsize=24, weight="bold", color="white")
            rc.append((r, tx, mk))
        cells.append(rc)
    for t in range(n):
        ax.text(t + 0.5, 2.62, f"{t + 1}", ha="center", fontsize=10, color="#999999")
    ax.text(-0.05, 2.62, T("step"), ha="right", fontsize=10, color="#999999")
    cax = fig.add_axes([0.70, 0.085, 0.22, 0.02])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(T("cb2"), fontsize=10)
    cax.set_title(T("cb_title"), fontsize=9.5 if LANG == "ja" else 8.5, color="#555555")
    concl = fig.text(0.05, 0.03 if LANG == "ja" else 0.015, "", ha="left", fontsize=12.5 if LANG == "ja" else 11.5, weight="bold", color="#b2182b")
    steps_frames = int(sec_per_step * fps)
    hold = 4 * fps
    total = n * steps_frames + hold

    def short(a):
        if LANG == "en":
            return ACT_EN.get(a, a)
        return (a.replace("テストを実行 → ", "テスト実行\n→ ").replace("テストを書き換える", "不合格を\n隠ぺい")
                .replace("動かしてみる", "動かして\nみる").replace("例題で確認 → ", "例題で確認\n→ "))

    def draw(f):
        t_now = min(f // steps_frames, n - 1)
        frac = min(1.0, (f % steps_frames) / (0.45 * steps_frames)) if f < n * steps_frames else 1.0
        for ri, (e, _, m) in enumerate(rows):
            dev = e["cheat"] and t_now >= e["tc"]
            if dev:
                status[ri].set_text(T("s_dev"))
                status[ri].set_color("#b2182b")
            elif f >= n * steps_frames and not e["cheat"]:
                status[ri].set_text(T("s_ok"))
                status[ri].set_color("#2b6cb0")
            else:
                status[ri].set_text(T("s_work"))
                status[ri].set_color("#888888")
            for t in range(min(t_now + 1, m)):
                r, tx, mk = cells[ri][t]
                a = frac if t == t_now else 1.0
                col = np.array(CMAP(e["risk"][t]))
                r.set_facecolor(tuple(col[:3] * a + (1 - a)))
                r.set_edgecolor("#bbbbbb")
                tx.set_text(short(e["act"][t]))
                if e["cheat"] and t == e["tc"]:
                    mk.set_text("×")
                    tx.set_color("#b2182b")
                    tx.set_fontweight("bold")
            for t in range(m, min(t_now + 1, n)):
                cells[ri][t][0].set_edgecolor("#ffffff")
        if f >= n * steps_frames:
            concl.set_text(T("concl"))
        return []

    anim = animation.FuncAnimation(fig, draw, frames=total, interval=1000 / fps, blit=False)
    anim.save(out_mp4, writer=animation.FFMpegWriter(fps=fps, bitrate=2400, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]),
              dpi=100)
    draw(total - 1)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    import subprocess
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", out_mp4, "-vf",
                    "fps=12,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse", out_gif], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/heatmap_data.json")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--cheat-id", default="conflicting__lcbtest_43__2")
    ap.add_argument("--honest-id", default="conflicting__lcbtest_43__0")
    ap.add_argument("--lang", default="en", choices=["ja", "en"])
    a = ap.parse_args()
    global LANG
    LANG = a.lang
    setup_font()
    E = json.load(open(a.data))["eps"]
    fig_all(E, f"{a.out_dir}/heatmap_all_{LANG}.png")
    fig_replay(E, a.cheat_id, a.honest_id, f"{a.out_dir}/replay_{LANG}.mp4", f"{a.out_dir}/replay_{LANG}.gif", f"{a.out_dir}/replay_{LANG}.png")
    print("done")


if __name__ == "__main__":
    main()
