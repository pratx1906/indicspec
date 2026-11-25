import json
from pathlib import Path

from indicspec.harness.sweep import Cell, load_prompts


def _write(bench: Path, task: str, rows: list[dict]) -> None:
    bench.mkdir(parents=True, exist_ok=True)
    (bench / f"{task}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def _row(task, item, lang, split="pilot", n=5):
    return {
        "task": task,
        "item_id": item,
        "lang": lang,
        "split": split,
        "prompt_token_ids": list(range(n)),
    }


def test_load_prompts_keeps_parallel_items_and_excludes_long(tmp_path: Path):
    rows = [_row("qa", i, lang) for i in range(10) for lang in ("en", "ta")]
    rows.append(_row("qa", 99, "en", n=100))  # too long, and only in en
    rows.append(_row("qa", 50, "en", split="report"))  # other split
    _write(tmp_path, "qa", rows)
    _write(tmp_path, "chat", [_row("chat", 1, "en")])  # task not requested
    p, excl = load_prompts(tmp_path, "pilot", ["en", "ta"], ["qa"], 50, 4, seed=1)
    assert excl == {"en": 1}
    assert [x.prompt_id for x in p["en"]] == [x.prompt_id for x in p["ta"]]
    assert len(p["en"]) == 4 and all(x.task == "qa" for x in p["en"])
    p2, _ = load_prompts(tmp_path, "pilot", ["en", "ta"], ["qa"], 50, 4, seed=1)
    assert [x.prompt_id for x in p2["en"]] == [x.prompt_id for x in p["en"]]  # deterministic


def test_cell_key():
    assert Cell("ta", 8, 2).key == "ta|c8|r2"
