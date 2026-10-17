"""Run IQuest-Coder-V1-40B-Loop-Instruct as a multi-step coding agent on the tasks.

Each step the agent calls one tool (bash or a file editor) and sees the result. Labels come for free: on a conflicting
task the tests cannot pass honestly, so an agent that ends with a changed test file has concealed the failure.
Model-written commands run as nobody, inside the task's own workspace only. Run this inside the vLLM container
(scripts/run_in_vllm.sh) and keep the container offline (--network none) if you can.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from common import CUT_OFF, FIRST_USER, SYSTEM, TOOLS, judge, parse_cmd, run_tool, setup_ws

MAX_LEN = 16384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cyankiwi/IQuest-Coder-V1-40B-Loop-Instruct-AWQ-4bit")
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--splits", default="conflicting,original")
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--n", type=int, default=4, help="runs per task")
    ap.add_argument("--n-orig", type=int, default=1, help="runs per task for the original (honest control) split")
    ap.add_argument("--max-turns", type=int, default=14)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--gpu-util", type=float, default=0.6)
    ap.add_argument("--out", default="data/agent_runs.jsonl")
    ap.add_argument("--ws-root", default="data/ws")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams

    os.makedirs("/tmp/pybin", exist_ok=True)  # the vLLM image has python3 only
    if not os.path.exists("/tmp/pybin/python"):
        os.symlink("/usr/bin/python3", "/tmp/pybin/python")
    tasks = [json.loads(l) for l in open(a.tasks)]
    splits = a.splits.split(",")
    ids = sorted({t["task_id"] for t in tasks})[: a.n_tasks]
    eps = []
    for t in tasks:
        if t["split"] in splits and t["task_id"] in ids:
            for k in range(a.n_orig if t["split"] == "original" else a.n):
                eid = f"{t['split']}__{t['task_id']}__{k}"
                ws = os.path.abspath(f"{a.ws_root}/{eid}")
                setup_ws(ws, t)
                eps.append({"id": eid, "task": t, "ws": ws, "msgs": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": FIRST_USER}],
                            "steps": [], "done": False, "end": None})
    print(f"{len(eps)} runs", flush=True)
    llm = LLM(model=a.model, trust_remote_code=True, dtype="float16", gpu_memory_utilization=a.gpu_util,
              max_model_len=MAX_LEN, max_num_seqs=32, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=a.max_tokens, temperature=a.temp, top_p=0.95)
    for turn in range(a.max_turns):
        for e in eps:  # vLLM fails the whole batch if one prompt exceeds the context, so stop runs that get close
            if not e["done"] and len(tok.apply_chat_template(e["msgs"], tools=TOOLS, add_generation_prompt=True)) > MAX_LEN - a.max_tokens - 64:
                e["done"], e["end"] = True, "too_long"
        act = [e for e in eps if not e["done"]]
        if not act:
            break
        t0 = time.time()
        outs = llm.chat([e["msgs"] for e in act], sp, use_tqdm=False, tools=TOOLS)
        cmds = [parse_cmd(o.outputs[0].text) for o in outs]
        with ThreadPoolExecutor(8) as ex:
            ress = list(ex.map(lambda ec: run_tool(ec[0]["ws"], ec[1]) if ec[1] is not None else None, zip(act, cmds)))
        for e, o, cmd, res in zip(act, outs, cmds, ress):
            txt = o.outputs[0].text
            e["msgs"].append({"role": "assistant", "content": txt})
            if cmd is None:
                e["steps"].append({"turn": turn, "cmd": None})
                if "SUBMIT" in txt or o.outputs[0].finish_reason != "length":
                    e["done"], e["end"] = True, ("submit" if "SUBMIT" in txt else "no_command")
                else:  # cut off by the token limit while thinking: let it continue
                    e["msgs"].append({"role": "user", "content": CUT_OFF})
                continue
            e["steps"].append({"turn": turn, "cmd": cmd, "result": res})
            e["msgs"].append({"role": "tool", "content": res})
        print(f"step {turn}: {len(act)} active, {time.time() - t0:.0f} s", flush=True)
    with open(a.out, "w") as f:
        for e in eps:
            if not e["done"]:
                e["end"] = "max_turns"
            f.write(json.dumps({"id": e["id"], "split": e["task"]["split"], "task_id": e["task"]["task_id"], "end": e["end"],
                                **judge(e["ws"], e["task"]), "n_turns": len(e["steps"]), "steps": e["steps"], "msgs": e["msgs"]},
                               ensure_ascii=False) + "\n")
    for s in splits:
        es = [json.loads(l) for l in open(a.out) if json.loads(l)["split"] == s]
        if es:
            print(f"{s}: {len(es)} runs, changed the test file {sum(x['test_tampered'] for x in es)}, "
                  f"original test passes {sum(x['pass_fresh'] for x in es)}")


if __name__ == "__main__":
    main()
