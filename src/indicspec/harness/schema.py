"""Load and validate the versioned JSON Schemas in ``schemas/``."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA_DIR = Path(__file__).resolve().parents[3] / "schemas"


@cache
def load_schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))


def validate(obj: dict[str, Any], name: str) -> None:
    """Raise jsonschema.ValidationError if ``obj`` does not match schema ``name``."""
    jsonschema.validate(obj, load_schema(name))


def validate_jsonl(path: Path, name: str = "request_log.v1") -> int:
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                validate(json.loads(line), name)
                n += 1
    return n
