"""Regenerate the arm x engine x model table in docs/ENGINE_STATUS.md from smoke records.

Reads every ``results/raw/smoke/*/status.json`` (+ manifest), keeps the latest attempt per
(engine, arm, model) and all earlier failures, and rewrites the block between the markers
``<!-- BEGIN GENERATED -->`` / ``<!-- END GENERATED -->``. Never hand-edit inside the markers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
BEGIN, END = "<!-- BEGIN GENERATED -->", "<!-- END GENERATED -->"


ROOT_CAUSE = re.compile(
    r"(ValueError|NotImplementedError|AssertionError|FileNotFoundError|ImportError|"
    r"ModuleNotFoundError|TypeError|KeyError|torch\.OutOfMemoryError|CUDA error)"
)


def _first_error(run_dir: Path, status: dict[str, Any]) -> str:
    """Prefer the specific root-cause exception line over generic wrapper errors."""
    exc = status.get("exception") or ""
    candidates: list[str] = []
    log = run_dir / "server.log"
    if log.exists():
        candidates = log.read_text(errors="replace").splitlines()
    candidates += exc.splitlines()
    roots = [ln for ln in candidates if ROOT_CAUSE.search(ln) and "raise" not in ln]
    if roots:
        return roots[-1].split("] ")[-1].strip()[:300]
    return exc.splitlines()[-1][:300] if exc else ""


def collect(smoke_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for st in sorted(smoke_dir.glob("*/status.json")):
        s = json.loads(st.read_text(encoding="utf-8"))
        tau = None
        req = st.parent / "requests.jsonl"
        per_lang: dict[str, float] = {}
        if req.exists():
            for line in req.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r.get("acceptance_length"):
                    per_lang[r["language"]] = r["acceptance_length"]
        d = s.get("spec_counter_delta") or {}
        drafts = d.get("vllm:spec_decode_num_drafts_total")
        acc = d.get("vllm:spec_decode_num_accepted_tokens_total")
        if drafts:
            tau = (acc + drafts) / drafts
        rows.append(
            {
                "run_id": s.get("run_id", st.parent.name),
                "engine": s.get("engine"),
                "arm": s.get("arm"),
                "model": s.get("model"),
                "ok": bool(s.get("ok")),
                "tau_pooled": tau,
                "tau_by_lang": per_lang,
                "peak_vram_mib": s.get("peak_vram_mib"),
                "error": "" if s.get("ok") else _first_error(st.parent, s),
            }
        )
    return rows


def render(rows: list[dict[str, Any]]) -> str:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in rows:
        latest[(r["engine"], r["arm"], r["model"])] = r  # run ids sort by timestamp
    out = [
        "| Engine | Arm | Model | Status | τ pooled (smoke) | τ by language (smoke) "
        "| Peak VRAM MiB | Latest run / error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (e, a, m), r in sorted(latest.items()):
        tau_s = "" if r["tau_pooled"] is None else f"{r['tau_pooled']:.2f}"
        tl = " ".join(f"{k}:{v:.2f}" for k, v in sorted(r["tau_by_lang"].items()))
        out.append(
            f"| {e} | {a} | {m} | {'works' if r['ok'] else '**fails**'} | "
            f"{tau_s} | {tl} | "
            f"{r['peak_vram_mib'] or ''} | `{r['run_id']}`"
            f"{'' if r['ok'] else ' — ' + r['error'].replace('|', '/')} |"
        )
    fails = [
        r for r in rows if not r["ok"] and latest[(r["engine"], r["arm"], r["model"])] is not r
    ]
    if fails:
        out += ["", "Earlier failed attempts (kept for the record):", ""]
        out += [f"- `{r['run_id']}`: {r['error'] or 'see status.json'}" for r in fails]
    out += [
        "",
        "Smoke τ values come from 6 short prompts (one per language, 64 tokens) and are **not** "
        "results; they only show the speculative path is active.",
    ]
    return "\n".join(out)


def main() -> int:
    doc = REPO / "docs/ENGINE_STATUS.md"
    text = doc.read_text(encoding="utf-8")
    table = render(collect(REPO / "results/raw/smoke"))
    block = f"{BEGIN}\n{table}\n{END}"
    if BEGIN in text:
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n## Smoke results (generated)\n\n" + block + "\n"
    doc.write_text(text, encoding="utf-8")
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
