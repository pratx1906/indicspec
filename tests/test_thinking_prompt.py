import pytest

from indicspec.harness.loadgen import Prompt
from indicspec.harness.sweep import QWEN3_EMPTY_THINK, enable_thinking


def test_enable_thinking_strips_empty_think_block() -> None:
    head = [151644, 77091, 198]  # <|im_start|>assistant\n
    p = Prompt("qa:1", "ta", "qa", head + QWEN3_EMPTY_THINK)
    q = enable_thinking(p)
    assert q.token_ids == head
    assert (q.prompt_id, q.language, q.task) == ("qa:1", "ta", "qa")
    assert p.token_ids == head + QWEN3_EMPTY_THINK  # original untouched


def test_enable_thinking_rejects_other_templates() -> None:
    with pytest.raises(ValueError):
        enable_thinking(Prompt("qa:1", "en", "qa", [1, 2, 3, 4, 5]))


def test_select_pool_slice_keeps_items_parallel() -> None:
    from indicspec.harness.sweep import select_pool_slice

    def mk(lang: str, sizes: list[int]) -> list[Prompt]:
        return [Prompt(f"t:{i}", lang, "qa", [0] * s) for i, s in enumerate(sizes)]

    prompts = {"en": mk("en", [5, 5, 5, 5, 5]), "ta": mk("ta", [5, 50, 5, 5, 5])}
    out = select_pool_slice(prompts, limit=10, offset=1, n=2)  # item 1 too long in ta
    assert [p.prompt_id for p in out["en"]] == ["t:2", "t:3"]
    assert [p.prompt_id for p in out["ta"]] == ["t:2", "t:3"]
