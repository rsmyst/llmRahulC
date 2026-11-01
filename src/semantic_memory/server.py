"""MCP server exposing the Temporal & State-Aware Semantic Memory Buffer to agentic LLMs.

Tools exposed:
- ingest_state         : store a resource state / tool output (tagged with resource_id,
                         logical clock, and optional dependency resource ids)
- ingest_tool_output   : alias for ingest_state for agent compatibility
- retrieve_context     : top-K semantic retrieval with state-aware versioning, time decay
                         (fixed lambda or "auto" learned per task) and dependency invalidation
- get_resource_history : inspect version evolution timeline for a resource
- list_resources       : list all tracked resources and active version states
- dependency_graph     : cross-resource dependency edges + invalidation status
- record_feedback      : report useful retrieved items so adaptive decay can learn
- get_decay_stats      : learned per-task decay lambdas
- advance_clock        : step the logical clock forward
- get_memory_stats     : buffer statistics (items, active vs stale count, clock step)
- clear_memory         : reset the buffer
Resources: memory://stats, memory://resources, memory://graph, memory://decay
"""

from __future__ import annotations

import logging
import os
from typing import Any
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
for _name in ["mcp", "httpx", "httpcore", "sentence_transformers", "transformers"]:
    logging.getLogger(_name).setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from mcp.server.fastmcp import FastMCP

from .config import Config
from .service import MemoryService

mcp = FastMCP("semantic-memory-buffer")


def _service() -> MemoryService:
    if not hasattr(mcp, "_memory_service"):
        mcp._memory_service = MemoryService(Config.from_env())
    return mcp._memory_service


@mcp.tool()
def ingest_state(
    resource_id: str,
    content: str,
    step_id: int | None = None,
    metadata: dict[str, Any] | None = None,
    dependencies: list[str] | None = None,
) -> dict[str, Any]:
    """Ingest a state or tool output into the semantic memory buffer.

    Automatically tags the entry with the current logical clock (step_id) and
    resource_id, incrementing version numbers and marking previous versions as
    superseded. Declare `dependencies` (resource ids this state was derived
    from) so mutation of a dependency also invalidates this state transitively.
    """
    return _service().ingest_state(
        resource_id=resource_id,
        content=content,
        step_id=step_id,
        metadata=metadata,
        dependencies=dependencies,
    )


@mcp.tool()
def ingest_tool_output(
    resource_id: str,
    content: str,
    step_id: int | None = None,
    metadata: dict[str, Any] | None = None,
    dependencies: list[str] | None = None,
) -> dict[str, Any]:
    """Ingest a raw tool output into semantic memory (alias for ingest_state)."""
    return _service().ingest_tool_output(
        resource_id=resource_id,
        content=content,
        step_id=step_id,
        metadata=metadata,
        dependencies=dependencies,
    )


@mcp.tool()
def retrieve_context(
    query: str,
    k: int = 5,
    exclude_stale: bool = True,
    time_decay: float | str | None = None,
    min_score: float = 0.0,
    task: str | None = None,
    propagate_dependencies: bool = True,
) -> dict[str, Any]:
    """Semantically retrieve relevant state items.

    Retrieval score formula:
        R(q, v_i) = 0    if superseded, or dependency-invalid when
                          propagate_dependencies=True (both under exclude_stale)
                  = cos(q, v_i) * exp(-lambda * (t_curr - t_i))   otherwise

    time_decay: number for a fixed lambda, or "auto"/"adaptive" to use the
    lambda learned for `task` via record_feedback. Omit for the config default.
    """
    return {
        "results": _service().retrieve_context(
            query=query,
            k=k,
            exclude_stale=exclude_stale,
            time_decay=time_decay,
            min_score=min_score,
            task=task,
            propagate_dependencies=propagate_dependencies,
        )
    }


@mcp.tool()
def record_feedback(
    task: str,
    useful_item_ids: list[int],
) -> dict[str, Any]:
    """Report which retrieved item_ids were useful for a task/subgoal.

    Updates the adaptive decay model: future retrieve_context calls with
    time_decay="auto" and the same `task` will use the learned lambda.
    """
    return _service().record_feedback(task=task, useful_item_ids=useful_item_ids)


@mcp.tool()
def get_decay_stats() -> dict[str, Any]:
    """Learned per-task decay lambdas and sample counts."""
    return _service().get_decay_stats()


@mcp.tool()
def dependency_graph() -> dict[str, Any]:
    """Cross-resource dependency graph: nodes, edges, dependents, invalidations."""
    return _service().dependency_graph()


@mcp.tool()
def get_resource_history(resource_id: str) -> dict[str, Any]:
    """Retrieve the complete version history timeline for a given resource_id."""
    return {
        "resource_id": resource_id,
        "history": _service().get_resource_history(resource_id),
    }


@mcp.tool()
def list_resources() -> dict[str, Any]:
    """List all tracked resources, their version counts, and active status."""
    return {"resources": _service().list_resources()}


@mcp.tool()
def advance_clock(steps: int = 1) -> dict[str, Any]:
    """Advance the logical clock by a given number of steps."""
    return _service().advance_logical_clock(steps=steps)


@mcp.tool()
def get_memory_stats() -> dict[str, Any]:
    """Return buffer statistics (items, active, stale count, clock step)."""
    return _service().stats()


@mcp.tool()
def clear_memory() -> dict[str, Any]:
    """Destructively clear all memory items."""
    return _service().clear_memory()


@mcp.resource("memory://stats")
def stats_resource() -> dict[str, Any]:
    """JSON statistics for the semantic memory buffer."""
    return _service().stats()


@mcp.resource("memory://resources")
def resources_resource() -> dict[str, Any]:
    """JSON list of all tracked resources in the memory buffer."""
    return _service().list_resources()


@mcp.resource("memory://graph")
def graph_resource() -> dict[str, Any]:
    """JSON dependency graph between resources with invalidation status."""
    return _service().dependency_graph()


@mcp.resource("memory://decay")
def decay_resource() -> dict[str, Any]:
    """JSON per-task learned decay lambdas."""
    return _service().get_decay_stats()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
