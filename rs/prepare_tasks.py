"""Download the tasks (Impossible-LiveCodeBench, easy version, from ImpossibleBench) into one jsonl file.
Splits: conflicting (tests contradict the specification), oneoff, original (no contradiction; the honest control). 50 tasks each."""
import argparse
import json

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="arianaazarbal/impossible_livecodebench_easy")
    ap.add_argument("--out", default="data/tasks.jsonl")
    a = ap.parse_args()
    n = 0
    with open(a.out, "w") as f:
        for split in ("conflicting", "oneoff", "original"):
            for r in load_dataset(a.dataset, split=split):
                f.write(json.dumps({**r, "split": split}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} tasks -> {a.out}")


if __name__ == "__main__":
    main()
