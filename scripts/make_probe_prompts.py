"""Write English config-probe prompts (pilot split only + hand-written math/code) to data/bench/.

Used to check that each drafter is configured sensibly on English before any comparison
(WS1.6). Never uses the tuning or report splits.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HANDWRITTEN = [
    ("probe-math-1", "math", "A train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. "
     "What is its average speed for the whole journey? Show your reasoning step by step."),
    ("probe-math-2", "math", "Solve for x: 3(x - 4) + 2x = 5x - 12 + x. Explain each step."),
    ("probe-code-1", "code", "Write a Python function that returns the n-th Fibonacci number "
     "iteratively, with a docstring and three doctest examples."),
]


def main() -> None:
    out = []
    for task, n in (("chat", 3), ("qa", 3)):
        rows = [json.loads(x) for x in (REPO / "data/bench" / f"{task}.jsonl").read_text(
            encoding="utf-8").splitlines() if x.strip()]
        pilot = [r for r in rows if r["split"] == "pilot" and r["lang"] == "en"][:n]
        out += [{"prompt_id": f"probe-{task}-{r['item_id']}", "language": "en", "task": task,
                 "messages": r["messages"]} for r in pilot]
    out += [{"prompt_id": pid, "language": "en", "task": t,
             "messages": [{"role": "user", "content": text}]} for pid, t, text in HANDWRITTEN]
    dst = REPO / "data/bench/probes_en.jsonl"
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"wrote {len(out)} prompts to {dst}")


if __name__ == "__main__":
    main()
