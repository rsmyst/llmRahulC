"""High-level memory service managing temporal and state-aware versioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .config import Config
from .decay import DecayTuner
from .embeddings import EmbeddingProvider
from .vectorstore import SemanticBuffer


class MemoryService:
    def __init__(self, config: Config):
        self._config = config
        self._buffer: SemanticBuffer | None = None
        self._embeddings: EmbeddingProvider | None = None
        self._decay: DecayTuner | None = None

    # ------------------------------------------------------------ lazy init

    def _db_path(self) -> Path:
        p = self._config.db_path
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def embeddings(self) -> EmbeddingProvider:
        if self._embeddings is None:
            self._embeddings = EmbeddingProvider(self._config)
        return self._embeddings

    def buffer(self) -> SemanticBuffer:
        if self._buffer is None:
            self._buffer = SemanticBuffer(
                self._db_path(), embed_dim=self.embeddings().dim
            )
            self._buffer.initialize()
        return self._buffer

    def decay_tuner(self) -> DecayTuner:
        if self._decay is None:
            path = self._db_path().with_suffix(".decay.json")
            self._decay = DecayTuner(
                path, default_lambda=self._config.default_decay_lambda
            )
        return self._decay

    # ----------------------------------------------------------------- tools

    def ingest_state(
        self,
        resource_id: str,
        content: str,
        step_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        dependencies: Sequence[str] | str | None = None,
    ) -> dict[str, Any]:
        """Ingest a state or tool execution result with state-aware versioning."""
        if not resource_id or not content:
            raise ValueError("resource_id and content are required")

        buffer = self.buffer()
        vector = self.embeddings().embed_one(content, task_type="RETRIEVAL_DOCUMENT")
        item = buffer.add_item(
            resource_id=resource_id,
            content=content,
            embedding=vector,
            step_id=step_id,
            metadata=metadata,
            dependencies=dependencies,
        )

        return item.to_dict()

    def ingest_tool_output(
        self,
        resource_id: str,
        content: str,
        step_id: int | None = None,
        metadata: dict[str, Any] | None = None,
        dependencies: Sequence[str] | str | None = None,
    ) -> dict[str, Any]:
        """Alias for ingest_state for agentic tool output compatibility."""
        return self.ingest_state(
            resource_id=resource_id,
            content=content,
            step_id=step_id,
            metadata=metadata,
            dependencies=dependencies,
        )

    def retrieve_context(
        self,
        query: str,
        k: int = 5,
        exclude_stale: bool = True,
        time_decay: float | str | None = None,
        min_score: float = 0.0,
        task: str | None = None,
        propagate_dependencies: bool = True,
    ) -> list[dict[str, Any]]:
        """Semantically retrieve relevant state items filtered by temporal decay & staleness.

        time_decay:
            float  -> fixed lambda
            "auto" / "adaptive" -> learned per-task lambda (requires `task`)
            None   -> config default
        """
        if not query or not query.strip():
            raise ValueError("query is required")

        if isinstance(time_decay, str):
            if time_decay.lower() not in ("auto", "adaptive"):
                raise ValueError('time_decay string must be "auto" or "adaptive"')
            decay_lambda = self.decay_tuner().lambda_for(task)
        elif time_decay is None:
            decay_lambda = self._config.default_decay_lambda
        else:
            decay_lambda = float(time_decay)

        vector = self.embeddings().embed_one(query, task_type="RETRIEVAL_QUERY")
        results = self.buffer().search(
            query_vector=vector,
            k=k,
            exclude_stale=exclude_stale,
            time_decay_lambda=decay_lambda,
            min_score=min_score,
            propagate_dependencies=propagate_dependencies,
        )
        out = [item.to_dict() for item in results]
        for row in out:
            row["decay_lambda"] = round(decay_lambda, 4)
        return out

    def record_feedback(
        self, task: str, useful_item_ids: Sequence[int]
    ) -> dict[str, Any]:
        """Teach the adaptive decay model which retrieved states were useful.

        Ages of the reported items update the task's lambda so future
        `time_decay="auto"` retrievals favour the right horizon.
        """
        if not task or not str(task).strip():
            raise ValueError("task is required")
        ids = [int(i) for i in (useful_item_ids or [])]
        items = self.buffer().items_by_ids(ids)
        ages = [it.age_steps or 0 for it in items]
        new_lambda = self.decay_tuner().observe(task, ages)
        return {
            "task": task.strip(),
            "feedback_items": len(items),
            "unknown_item_ids": sorted(set(ids) - {it.item_id for it in items}),
            "lambda_value": round(new_lambda, 4),
        }

    def get_decay_stats(self) -> dict[str, Any]:
        """Per-task learned decay rates."""
        return self.decay_tuner().stats()

    def get_resource_history(self, resource_id: str) -> list[dict[str, Any]]:
        """Retrieve complete version evolution timeline for a resource."""
        if not resource_id or not resource_id.strip():
            raise ValueError("resource_id is required")
        items = self.buffer().get_resource_history(resource_id.strip())
        return [item.to_dict() for item in items]

    def dependency_graph(self) -> dict[str, Any]:
        """Current dependency graph between resources + invalidation status."""
        return self.buffer().dependency_graph()

    def list_resources(self) -> dict[str, dict[str, Any]]:
        """List all tracked resources and their version states."""
        return self.buffer().list_resources()

    def advance_logical_clock(self, steps: int = 1) -> dict[str, Any]:
        """Advance the system logical clock step."""
        new_step = self.buffer().advance_clock(steps=steps)
        return {"logical_clock_step": new_step}

    def stats(self) -> dict[str, Any]:
        s = {"buffer": self.buffer().stats()}
        s["embedding_backend"] = self.embeddings().backend_name
        s["decay"] = self.decay_tuner().stats()
        return s

    def clear_memory(self) -> dict[str, Any]:
        self.buffer().clear()
        decay_path = self._db_path().with_suffix(".decay.json")
        if decay_path.exists():
            try:
                decay_path.unlink()
            except OSError:
                pass
        self._decay = None
        return self.stats()
