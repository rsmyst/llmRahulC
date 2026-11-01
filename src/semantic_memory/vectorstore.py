"""State-aware semantic memory buffer backed by FAISS vector index & metadata store.

Implements Contribution 1 (Rahul Saravanan): Temporal & State-Aware Versioning.
Mathematical formulation:
Let q be the current query embedding, and v_i be the vector embedding of
historical state i. Let t_curr be the current logical clock step, and t_i be
the creation step of v_i. The retrieval score R(q, v_i) is calculated as:

    R(q, v_i) = 0                                       if exists j s.t. (id_j = id_i) and (t_j > t_i)
              = cos(q, v_i) * exp(-lambda * (t_curr - t_i))   otherwise

Endsem extension: when propagate_dependencies is on, a state whose dependency
resource was mutated after its own step (or whose dependency chain is itself
invalid) is also zeroed:

    R(q, v_i) = 0 if i is lineage-stale or transitively dependency-invalid
              = cos(q, v_i) * exp(-lambda * (t_curr - t_i))   otherwise

Dependency edges are declared at ingest time via `dependencies=[resource_id, ...]`
and persist across restarts inside the JSONL metadata file.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import faiss
import numpy as np


@dataclass
class MemoryItem:
    item_id: int
    resource_id: str
    version: int
    step_id: int
    content: str
    metadata: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    is_latest: bool = True
    superseded_by_step: int | None = None
    invalidated_by: list[str] = field(default_factory=list)
    stale_reason: str | None = None
    similarity: float | None = None
    age_steps: int | None = None
    decay_factor: float | None = None
    adjusted_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "item_id": self.item_id,
            "resource_id": self.resource_id,
            "version": self.version,
            "step_id": self.step_id,
            "is_latest": self.is_latest,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
        if self.dependencies:
            d["dependencies"] = list(self.dependencies)
        if self.superseded_by_step is not None:
            d["superseded_by_step"] = self.superseded_by_step
        if self.invalidated_by:
            d["invalidated_by"] = list(self.invalidated_by)
        if self.stale_reason:
            d["stale_reason"] = self.stale_reason
        if self.similarity is not None:
            d["similarity"] = round(self.similarity, 6)
        if self.age_steps is not None:
            d["age_steps"] = self.age_steps
        if self.decay_factor is not None:
            d["decay_factor"] = round(self.decay_factor, 6)
        if self.adjusted_score is not None:
            d["retrieval_score"] = round(self.adjusted_score, 6)
            d["adjusted_score"] = round(self.adjusted_score, 6)
        d["content_preview"] = (
            self.content[:400] + "..." if len(self.content) > 400 else self.content
        )
        return d


def _normalize_deps(dependencies: Sequence[str] | str | None, self_id: str) -> list[str]:
    if dependencies is None:
        return []
    if isinstance(dependencies, str):
        dependencies = [dependencies]
    out: list[str] = []
    seen: set[str] = set()
    for dep in dependencies:
        dep = str(dep).strip()
        if not dep or dep == self_id or dep in seen:
            continue
        seen.add(dep)
        out.append(dep)
    return out


class SemanticBuffer:
    def __init__(self, db_path: str | Path, embed_dim: int = 768):
        db = Path(db_path)
        self._meta_path = db.with_suffix(".jsonl")
        self._index_path = db.with_suffix(".index")
        self._clock_path = db.with_suffix(".clock")
        self._embed_dim = embed_dim
        self._lock = threading.RLock()

        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(embed_dim))
        self._items: dict[int, dict[str, Any]] = {}
        self._by_resource: dict[str, list[dict[str, Any]]] = {}
        self._step = 0

    # ------------------------------------------------------------- lifecycle

    def initialize(self) -> None:
        with self._lock:
            if self._index_path.exists() and self._index_path.stat().st_size > 0:
                try:
                    index = faiss.read_index(str(self._index_path))
                    if index.d != self._embed_dim:
                        raise ValueError(
                            f"index dim {index.d} != configured embed dim {self._embed_dim}"
                        )
                    self._index = index
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not load FAISS index at {self._index_path}: {exc}. "
                        "Delete the .index/.jsonl pair or set SM_EMBED_DIM to match."
                    ) from exc
            if self._meta_path.exists():
                with open(self._meta_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        self._index_item(rec)
            item_steps = [int(r["step_id"]) for r in self._items.values()]
            clock = 0
            if self._clock_path.exists():
                try:
                    clock = int(self._clock_path.read_text(encoding="utf-8").strip() or 0)
                except (OSError, ValueError):
                    clock = 0
            self._step = max(max(item_steps, default=0), clock)

    def _persist_clock(self) -> None:
        """Persist the logical clock (advances may happen without ingests)."""
        tmp = Path(str(self._clock_path) + ".tmp")
        tmp.write_text(str(self._step), encoding="utf-8")
        os.replace(tmp, self._clock_path)

    def _index_item(self, rec: dict[str, Any]) -> None:
        """Register a record in the id and per-resource lookup indexes."""
        self._items[int(rec["item_id"])] = rec
        self._by_resource.setdefault(str(rec.get("resource_id", "")), []).append(rec)

    def _save(self) -> None:
        """Atomic write of index + metadata (temp file + rename)."""
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)
        idx_tmp = Path(str(self._index_path) + ".tmp")
        meta_tmp = Path(str(self._meta_path) + ".tmp")
        faiss.write_index(self._index, str(idx_tmp))
        with open(meta_tmp, "w", encoding="utf-8") as fh:
            for item_id in sorted(self._items):
                fh.write(json.dumps(self._items[item_id]) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(idx_tmp, self._index_path)
        os.replace(meta_tmp, self._meta_path)

    # ------------------------------------------------------------------ clock

    def latest_step(self) -> int:
        with self._lock:
            return self._step

    def next_step(self) -> int:
        with self._lock:
            self._step += 1
            self._persist_clock()
            return self._step

    def advance_clock(self, steps: int = 1) -> int:
        """Advance the logical clock by `steps`."""
        with self._lock:
            self._step += max(1, steps)
            self._persist_clock()
            return self._step

    # ---------------------------------------------------------------- ingest

    def add_item(
        self,
        resource_id: str,
        content: str,
        embedding: list[float],
        step_id: int | None = None,
        metadata: dict | None = None,
        created_at: float | None = None,
        dependencies: Sequence[str] | str | None = None,
    ) -> MemoryItem:
        """Ingest a state or tool execution result with state-aware versioning."""
        if created_at is None:
            created_at = time.time()
        metadata = metadata or {}
        deps = _normalize_deps(dependencies, resource_id)

        with self._lock:
            if step_id is None:
                self._step += 1
                step_id = self._step
            else:
                self._step = max(self._step, step_id)

            prior = self._by_resource.get(resource_id, [])
            new_version_num = (max(int(r.get("version", 1)) for r in prior) + 1) if prior else 1

            for rec in prior:
                if rec.get("is_latest"):
                    rec["is_latest"] = False
                    rec["superseded_by_step"] = step_id

            item_id = max(self._items, default=0) + 1
            item_record: dict[str, Any] = {
                "item_id": item_id,
                "resource_id": resource_id,
                "version": new_version_num,
                "step_id": step_id,
                "content": content,
                "metadata": metadata,
                "dependencies": deps,
                "created_at": created_at,
                "is_latest": True,
                "superseded_by_step": None,
            }
            self._index_item(item_record)

            vec = np.asarray([embedding], dtype=np.float32)
            if vec.shape[1] != self._embed_dim:
                raise ValueError(
                    f"embedding dim {vec.shape[1]} != buffer embed dim {self._embed_dim}"
                )
            ids = np.asarray([item_id], dtype=np.int64)
            self._index.add_with_ids(vec, ids)
            self._save()

        return MemoryItem(
            item_id=item_id,
            resource_id=resource_id,
            version=new_version_num,
            step_id=step_id,
            content=content,
            metadata=metadata,
            dependencies=deps,
            created_at=created_at,
            is_latest=True,
            superseded_by_step=None,
        )

    # ----------------------------------------------------- dependency graph

    def _latest_by_resource(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for res, recs in self._by_resource.items():
            best = None
            for rec in recs:
                if best is None or int(rec.get("step_id", 0)) > int(best.get("step_id", 0)):
                    best = rec
            if best is not None:
                latest[res] = best
        return latest

    def _dependency_invalidations(
        self,
        rec: dict[str, Any],
        latest: dict[str, dict[str, Any]],
        memo: dict[int, set[str]],
        visiting: set[int],
    ) -> set[str]:
        """Transitively invalidated dependency resource ids for one record."""
        rid = int(rec["item_id"])
        if rid in memo:
            return memo[rid]
        if rid in visiting:
            return set()
        visiting.add(rid)
        invalid: set[str] = set()
        for dep in rec.get("dependencies", []):
            dep_rec = latest.get(dep)
            if dep_rec is None:
                continue
            if int(dep_rec["step_id"]) > int(rec["step_id"]):
                invalid.add(dep)
            else:
                invalid |= self._dependency_invalidations(dep_rec, latest, memo, visiting)
        visiting.discard(rid)
        memo[rid] = invalid
        return invalid

    def dependency_graph(self) -> dict[str, Any]:
        """Current-state dependency graph + invalidation status per resource."""
        with self._lock:
            latest = self._latest_by_resource()
            memo: dict[int, set[str]] = {}
            nodes = sorted(latest)
            edges = []
            reverse: dict[str, list[str]] = {n: [] for n in nodes}
            invalid: dict[str, list[str]] = {}
            for res, rec in latest.items():
                deps = list(rec.get("dependencies", []))
                for dep in deps:
                    edges.append({"from": res, "to": dep, "since_step": int(rec["step_id"])})
                    reverse.setdefault(dep, []).append(res)
                inv = sorted(self._dependency_invalidations(rec, latest, memo, set()))
                if inv:
                    invalid[res] = inv
            return {
                "nodes": nodes,
                "edges": edges,
                "dependents": {k: sorted(set(v)) for k, v in reverse.items() if v},
                "invalid": invalid,
                "logical_clock_step": self._step,
            }

    # --------------------------------------------------------------- retrieve

    def search(
        self,
        query_vector: list[float],
        k: int = 5,
        exclude_stale: bool = True,
        time_decay_lambda: float = 0.05,
        min_score: float = 0.0,
        propagate_dependencies: bool = True,
    ) -> list[MemoryItem]:
        """Perform State-Aware & Temporal retrieval.

        Mathematical scoring:
            R(q, v_i) = 0 if superseded and exclude_stale is True
                      = 0 if dependency-invalid and propagate_dependencies is True
                      = cos(q, v_i) * exp(-lambda * (t_curr - t_i)) otherwise
        """
        with self._lock:
            n = self._index.ntotal
            if n == 0:
                return []
            k = max(1, int(k))
            # Exact flat index: fetch every candidate so stale filtering can
            # never starve the top-k (a fixed fetch_k could).
            scores, ids = self._index.search(
                np.asarray([query_vector], dtype=np.float32), n
            )
            ids = ids[0].tolist()
            scores = scores[0].tolist()
            current_clock = self._step
            items = self._items
            latest = self._latest_by_resource() if propagate_dependencies else {}

            results: list[MemoryItem] = []
            memo: dict[int, set[str]] = {}
            for item_id, sim in zip(ids, scores):
                if item_id < 0:
                    continue
                rec = items.get(int(item_id))
                if rec is None:
                    continue

                is_latest = bool(rec.get("is_latest", True))
                step_i = int(rec["step_id"])
                age_steps = max(0, current_clock - step_i)

                invalidated_by: list[str] = []
                stale_reason: str | None = None

                if exclude_stale and not is_latest:
                    stale_reason = "superseded"
                elif exclude_stale and propagate_dependencies and rec.get("dependencies"):
                    inv = self._dependency_invalidations(rec, latest, memo, set())
                    if inv:
                        invalidated_by = sorted(inv)
                        stale_reason = "dependency_invalidated"

                if stale_reason is not None:
                    # Piecewise R(q, v_i) = 0: skip entirely so zero-scored
                    # rows cannot occupy top-k slots.
                    continue

                decay_factor = math.exp(-time_decay_lambda * age_steps)
                adjusted_score = float(sim) * decay_factor

                if adjusted_score < min_score:
                    continue

                results.append(
                    MemoryItem(
                        item_id=int(item_id),
                        resource_id=str(rec["resource_id"]),
                        version=int(rec.get("version", 1)),
                        step_id=step_i,
                        content=str(rec["content"]),
                        metadata=dict(rec.get("metadata", {})),
                        dependencies=list(rec.get("dependencies", [])),
                        created_at=float(rec.get("created_at", 0.0)),
                        is_latest=is_latest,
                        superseded_by_step=rec.get("superseded_by_step"),
                        invalidated_by=invalidated_by,
                        stale_reason=stale_reason,
                        similarity=float(sim),
                        age_steps=age_steps,
                        decay_factor=float(decay_factor),
                        adjusted_score=float(adjusted_score),
                    )
                )

        results.sort(key=lambda it: it.adjusted_score or 0.0, reverse=True)
        return results[:k]

    # ------------------------------------------------------------- versioning

    def get_resource_history(self, resource_id: str) -> list[MemoryItem]:
        """Retrieve the complete chronological version history for a given resource."""
        with self._lock:
            matching = self._by_resource.get(resource_id, [])
            current_clock = self._step
            latest = self._latest_by_resource()
            memo: dict[int, set[str]] = {}

        items: list[MemoryItem] = []
        for rec in sorted(matching, key=lambda r: (r.get("version", 1), r["step_id"])):
            step_i = int(rec["step_id"])
            is_latest = bool(rec.get("is_latest", True))
            if is_latest:
                inv = sorted(self._dependency_invalidations(rec, latest, memo, set()))
                reason = "dependency_invalidated" if inv else None
            else:
                inv = []
                reason = "superseded"
            items.append(
                MemoryItem(
                    item_id=int(rec["item_id"]),
                    resource_id=str(rec["resource_id"]),
                    version=int(rec.get("version", 1)),
                    step_id=step_i,
                    content=str(rec["content"]),
                    metadata=dict(rec.get("metadata", {})),
                    dependencies=list(rec.get("dependencies", [])),
                    created_at=float(rec.get("created_at", 0.0)),
                    is_latest=is_latest,
                    superseded_by_step=rec.get("superseded_by_step"),
                    invalidated_by=inv,
                    stale_reason=reason,
                    age_steps=max(0, current_clock - step_i),
                )
            )
        return items

    def list_resources(self) -> dict[str, dict[str, Any]]:
        """List all resources with version counts and latest step info."""
        with self._lock:
            resources: dict[str, dict[str, Any]] = {}
            for res_id, recs in self._by_resource.items():
                entry = {
                    "resource_id": res_id,
                    "total_versions": len(recs),
                    "latest_version": 0,
                    "latest_step": 0,
                    "has_active_state": False,
                    "dependencies": [],
                }
                for rec in recs:
                    ver = rec.get("version", 1)
                    if ver > entry["latest_version"]:
                        entry["latest_version"] = ver
                    step = rec.get("step_id", 0)
                    if step > entry["latest_step"]:
                        entry["latest_step"] = step
                    if rec.get("is_latest"):
                        entry["has_active_state"] = True
                        entry["dependencies"] = list(rec.get("dependencies", []))
                resources[res_id] = entry
            return resources

    def items_by_ids(self, item_ids: Iterable[int]) -> list[MemoryItem]:
        """Fetch records by item id (for feedback / decay bookkeeping)."""
        wanted = {int(i) for i in item_ids}
        out: list[MemoryItem] = []
        with self._lock:
            current_clock = self._step
            for iid in wanted:
                rec = self._items.get(iid)
                if rec is None:
                    continue
                out.append(
                    MemoryItem(
                        item_id=int(rec["item_id"]),
                        resource_id=str(rec["resource_id"]),
                        version=int(rec.get("version", 1)),
                        step_id=int(rec["step_id"]),
                        content=str(rec["content"]),
                        metadata=dict(rec.get("metadata", {})),
                        dependencies=list(rec.get("dependencies", [])),
                        created_at=float(rec.get("created_at", 0.0)),
                        is_latest=bool(rec.get("is_latest", True)),
                        superseded_by_step=rec.get("superseded_by_step"),
                        age_steps=max(0, current_clock - int(rec["step_id"])),
                    )
                )
        out.sort(key=lambda it: it.item_id)
        return out

    # ------------------------------------------------------------------ stats

    def stats(self) -> dict[str, Any]:
        with self._lock:
            n_items = len(self._items)
            n_latest = sum(1 for r in self._items.values() if r.get("is_latest"))
            n_resources = len(self._by_resource)
            n_edges = len(
                {
                    (res, dep)
                    for res, recs in self._by_resource.items()
                    for rec in recs
                    if rec.get("is_latest")
                    for dep in rec.get("dependencies", [])
                }
            )
            return {
                "items": n_items,
                "latest_items": n_latest,
                "stale_items": n_items - n_latest,
                "resources": n_resources,
                "dependency_edges": n_edges,
                "logical_clock_step": self._step,
                "embed_dim": self._embed_dim,
            }

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._by_resource.clear()
            self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self._embed_dim))
            self._step = 0
            for p in (self._index_path, self._meta_path, self._clock_path):
                if p.exists():
                    try:
                        os.remove(p)
                    except OSError:
                        pass
