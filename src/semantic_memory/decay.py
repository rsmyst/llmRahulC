"""Adaptive temporal decay: learn lambda per task / subgoal from feedback.

Endsem extension of the fixed-lambda decay in R(q, v_i). Instead of one global
lambda, each named task keeps its own estimate:

    lambda_task <- (1 - alpha) * lambda_task + alpha * ln(2) / median_useful_age

i.e. the tuner picks the half-life that makes the median age of states the
agent reported useful decay by 50%. Tasks whose useful state is always fresh
get a large lambda (strong recency); tasks that still need old state get a
small lambda. `time_decay="auto"` on retrieve_context uses the learned value.

State is persisted to <db>.decay.json so it survives restarts.
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_MIN_LAMBDA = 0.005
_MAX_LAMBDA = 0.5
_DEFAULT_ALPHA = 0.35
_DEFAULT_LAMBDA = 0.05


@dataclass
class TaskDecay:
    lambda_value: float
    samples: int = 0


class DecayTuner:
    """Per-task exponential-decay rate learned from retrieval feedback."""

    def __init__(
        self,
        path: str | Path,
        default_lambda: float = _DEFAULT_LAMBDA,
        alpha: float = _DEFAULT_ALPHA,
    ):
        self._path = Path(path)
        self._default = default_lambda
        self._alpha = alpha
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskDecay] = {}
        self._load()

    # ------------------------------------------------------------ persistence

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for task, rec in raw.get("tasks", {}).items():
                self._tasks[str(task)] = TaskDecay(
                    lambda_value=float(rec["lambda_value"]),
                    samples=int(rec.get("samples", 0)),
                )
            if "default_lambda" in raw:
                self._default = float(raw["default_lambda"])
        except Exception:
            self._tasks = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(self._path) + ".tmp")
        payload = {
            "default_lambda": self._default,
            "tasks": {k: asdict(v) for k, v in sorted(self._tasks.items())},
        }
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    # ------------------------------------------------------------------ API

    @staticmethod
    def _clamp(value: float) -> float:
        return max(_MIN_LAMBDA, min(_MAX_LAMBDA, value))

    def lambda_for(self, task: str | None = None) -> float:
        """Learned lambda for a task, or the default when never seen."""
        with self._lock:
            if task and task in self._tasks:
                return self._tasks[task].lambda_value
            return self._default

    def observe(self, task: str, useful_ages: list[int]) -> float:
        """Update the task's lambda from the ages of states reported useful.

        `useful_ages` are (t_curr - t_i) for each feedback item id. Empty
        feedback is ignored.
        """
        task = (task or "").strip() or "default"
        ages = [a for a in (int(a) for a in useful_ages) if a >= 0]
        if not ages:
            return self.lambda_for(task)
        ages.sort()
        median_age = ages[len(ages) // 2]
        # Half-life target: median useful age should sit near 50% decay.
        target = math.log(2.0) / max(median_age, 1)
        with self._lock:
            cur = self._tasks.get(task)
            if cur is None:
                new_lambda = target
                samples = 1
            else:
                new_lambda = (1 - self._alpha) * cur.lambda_value + self._alpha * target
                samples = cur.samples + 1
            self._tasks[task] = TaskDecay(
                lambda_value=self._clamp(new_lambda), samples=samples
            )
            self._save()
            return self._tasks[task].lambda_value

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "default_lambda": self._default,
                "alpha": self._alpha,
                "tasks": {
                    task: {"lambda_value": round(t.lambda_value, 4), "samples": t.samples}
                    for task, t in sorted(self._tasks.items())
                },
            }
