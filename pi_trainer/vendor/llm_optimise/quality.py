"""Deterministic task scoring. Never executes generated code."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    expected: Any
    evaluator: str = "exact"
    system: str = "Follow the instruction. Return only the requested answer, without explanation."
    tolerance: float = 0.0
    json_schema: dict | None = None

    def __post_init__(self):
        if (
            not isinstance(self.id, str)
            or not self.id
            or not isinstance(self.prompt, str)
            or not self.prompt
        ):
            raise ValueError("task id and prompt must be nonempty strings")
        if not isinstance(self.system, str):
            raise ValueError("system must be a string")
        if self.evaluator not in ("exact", "json", "json_subset", "numeric"):
            raise ValueError("evaluator must be exact, json, json_subset or numeric")
        if (
            not isinstance(self.tolerance, (float, int))
            or not math.isfinite(self.tolerance)
            or self.tolerance < 0
        ):
            raise ValueError("tolerance must be finite and nonnegative")
        if self.evaluator == "numeric":
            if isinstance(self.expected, bool) or not math.isfinite(float(self.expected)):
                raise ValueError("numeric expected value must be finite")
        if self.evaluator == "json_subset" and (
            not isinstance(self.expected, dict) or not self.expected
        ):
            raise ValueError("json_subset requires a nonempty object")

        if self.json_schema is not None and not isinstance(self.json_schema, dict):
            raise ValueError("json_schema must be an object")

    def score(self, output: str) -> float:
        output = output.strip()
        if self.evaluator == "exact":
            return float(output.casefold() == str(self.expected).strip().casefold())
        try:
            if self.evaluator == "numeric":
                n = float(output)
                return float(math.isfinite(n) and abs(n - float(self.expected)) <= self.tolerance)
            value = json.loads(output)
            if self.evaluator == "json":
                return float(_same_json(value, self.expected))
            return float(_subset(value, self.expected))
        except (ValueError, TypeError):
            return 0.0


def _same_json(actual, expected):
    # JSON booleans must never pass as numeric 0/1.
    if isinstance(actual, bool) != isinstance(expected, bool):
        return False
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_same_json(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same_json(a, b) for a, b in zip(actual, expected, strict=False))
        )
    return actual == expected


def _subset(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and _subset(actual[k], v) for k, v in expected.items()
        )
    return _same_json(actual, expected)


def load_tasks(path: str | Path) -> list[Task]:
    tasks = []
    keys = {"id", "prompt", "expected", "evaluator", "system", "tolerance", "json_schema"}
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict) or set(data) - keys:
                    raise ValueError("unknown task keys or non-object row")
                tasks.append(Task(**data))
            except (ValueError, TypeError) as e:
                raise ValueError(f"invalid dataset row {line_no}: {e}") from e
    if not tasks or len({t.id for t in tasks}) != len(tasks):
        raise ValueError("dataset must be nonempty with unique task IDs")
    return tasks
