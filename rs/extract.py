"""Re-read each agent run with the same model and save the hidden states at two points per step:
  A (point 0) = right before the step's first token (the reasoning has not started)
  B (point 1) = right before the tool call (the reasoning is written, the command is not)
Each point is the last token's residual stream after every layer: 80 layers x 2 loops = 160 vectors of 5120.
max_tokens=1 so only the prefill runs; the last 160 hook calls belong to that point.
Output: <out-dir>/<id>.npy (float16, [steps, 2, 160, 5120]) and <out-dir>/index.jsonl. Full size is about 11 GB for 250 runs.
--band L0-L1 instead keeps only the mean over loop-1 layers L0..L1 (0-indexed): [steps, 2, 5120], about 70 MB in total.
"""
import argparse
import json
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import numpy as np

from common import TOOLS, call_pos

GRAB = []


def add_hooks(model):
    for l in model.model.layers:
        l.register_forward_hook(lambda m, i, o: GRAB.append((o[0] if isinstance(o, tuple) else o)[-1].detach().half().cpu()))
    return len(model.model.layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cyankiwi/IQuest-Coder-V1-40B-Loop-Instruct-AWQ-4bit")
    ap.add_argument("--runs", default="data/agent_runs.jsonl")
    ap.add_argument("--out-dir", default="data/feat")
    ap.add_argument("--band", default=None, help="e.g. 40-70: save only the loop-1 mean over these layers")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    import torch

    os.makedirs(a.out_dir, exist_ok=True)
    llm = LLM(model=a.model, trust_remote_code=True, dtype="float16", gpu_memory_utilization=0.45,
              max_model_len=20480, max_num_seqs=1, enforce_eager=True, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    n_layers = llm.apply_model(add_hooks)[0]
    per = 2 * n_layers  # the loop model calls each layer twice per token
    sp = SamplingParams(max_tokens=1, temperature=0)
    ipath = f"{a.out_dir}/index.jsonl"
    done = {json.loads(l)["id"] for l in open(ipath)} if os.path.exists(ipath) else set()
    idx = open(ipath, "a")
    for line in open(a.runs):
        e = json.loads(line)
        if e["id"] in done:
            continue
        msgs, feats, turns = e["msgs"], [], []
        for k, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            head = tok.apply_chat_template(msgs[:k], tools=TOOLS, add_generation_prompt=True, tokenize=False)
            cut = call_pos(m["content"])
            pts = []
            for p in (head, head + (m["content"][:cut] if cut >= 0 else m["content"])):
                GRAB.clear()
                llm.generate(p, sp, use_tqdm=False)
                pts.append(torch.stack(GRAB[-per:]))
            feats.append(torch.stack(pts))
            turns.append({"msg_index": k, "has_cmd": cut >= 0})
        arr = torch.stack(feats).float().numpy()
        if a.band:
            l0, l1 = map(int, a.band.split("-"))
            arr = arr[:, :, l0:l1 + 1].mean(2)
        arr = arr.astype(np.float16)
        np.save(f"{a.out_dir}/{e['id']}.npy", arr)
        idx.write(json.dumps({"id": e["id"], "shape": list(arr.shape), "turns": turns}) + "\n")
        idx.flush()
        print(f"{e['id']} {arr.shape}", flush=True)


if __name__ == "__main__":
    main()
