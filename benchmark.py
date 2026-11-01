"""Long-horizon benchmark for MATP (endsem evaluation).

Scenario: 40 logical steps of an agent evolving a service (3 schema versions,
2 auth versions, dependent ORM/session/routes/tests, docs). Overlapping
mutations mean naive cosine keeps surfacing superseded text while FIFO forgets
early-but-current state.

Compares:
  FIFO (window of last 10 steps)   retention-limited baseline
  Naive RAG (cosine only)          no version awareness
  MATP (lineage only)              R=0 on superseded same-resource versions
  MATP + dependency graph          also R=0 on transitively invalid dependents

Run:  python benchmark.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from semantic_memory.config import Config
from semantic_memory.service import MemoryService

FIFO_WINDOW = 10  # steps retained by the FIFO/context-window baseline


def build_service() -> MemoryService:
    db = os.path.join(tempfile.gettempdir(), "matp_benchmark.db")
    cfg = Config.from_env()
    object.__setattr__(cfg, "buffer_db", db)
    object.__setattr__(cfg, "gemini_api_key", "")  # local Nomic, reproducible
    svc = MemoryService(cfg)
    svc.clear_memory()
    return svc


def ingest_scenario(svc: MemoryService) -> list[tuple[int, str, str, list[str]]]:
    """Returns (step, resource, description, dependencies) timeline."""
    S = [
        (1, "db/schema.sql", "-- v1 legacy\nCREATE TABLE users (user_id INT, username VARCHAR(50), password_hash VARCHAR(255), legacy_pin VARCHAR(6));", []),
        (2, "config/auth.env", "AUTH_MODE=local_credentials\nTOKEN_SECRET=legacy_secret_9988", []),
        (3, "config/cache.yaml", "cache_host: 10.0.0.15\nport: 6379\nmode: single_node", []),
        (4, "src/models/user.py", "class User(Base):\n  __tablename__='users'\n  password_hash=Column(String(255))  # maps to schema v1", ["db/schema.sql"]),
        (5, "api/routes.json", '{"version":1,"endpoint":"/api/v1/auth","protocol":"http"}', ["config/auth.env"]),
        (6, "docs/api.md", "# API v1\nPOST /api/v1/auth with username/password fields.", ["api/routes.json"]),
        (7, "migrations/2026_01.sql", "ALTER TABLE users ADD legacy_pin VARCHAR(6);  -- schema v1 based", ["db/schema.sql"]),
        (8, "src/auth/session.py", "def login(user, password):  # checks password_hash (schema v1)", ["config/auth.env"]),
        (9, "db/schema.sql", "-- v2 oauth\nCREATE TABLE users (user_id INT, username VARCHAR(50), email VARCHAR(100), auth_provider VARCHAR(30), oauth_access_token VARCHAR(512), is_active BOOLEAN);", []),
        (10, "config/auth.env", "AUTH_MODE=oauth2_pkce\nOAUTH_ISSUER=https://auth.company.internal", []),
        (11, "src/models/user.py", "class User(Base):\n  auth_provider=Column(String(30))\n  oauth_access_token=Column(String(512))  # maps to schema v2", ["db/schema.sql"]),
        (12, "tests/test_auth.py", "def test_login(): assert user.password_hash  # written against session v1", ["src/auth/session.py", "config/auth.env"]),
        (13, "README.md", "Quickstart: run with local_credentials auth mode.", ["config/auth.env"]),
        (14, "api/routes.json", '{"version":2,"endpoint":"/api/v2/oauth","protocol":"https","tls":true}', ["config/auth.env"]),
        (15, "docs/api.md", "# API v2\nPOST /api/v2/oauth with PKCE redirect.", ["api/routes.json"]),
        (16, "src/models/user.py", "class User(Base):\n  refresh_token=Column(String(512))  # transitional, still references oauth_access_token", ["db/schema.sql"]),
        (17, "config/cache.yaml", "cache_host: 10.0.0.42\nport: 6379\nmode: cluster\n ttl: 300", []),
        (18, "src/auth/session.py", "def login_pkce(code, verifier):  # exchanges code for oauth token", ["config/auth.env"]),
        (19, "migrations/2026_01.sql", "ALTER TABLE users ADD oauth_access_token VARCHAR(512);  -- schema v2 based", ["db/schema.sql"]),
        (20, "docs/runbook.md", "Rollback: AUTH_MODE=local_credentials (still valid for v1 only).", ["config/auth.env"]),
        (21, "db/schema.sql", "-- v3: split credentials\nCREATE TABLE users (user_id INT, username VARCHAR(50), email VARCHAR(100), is_active BOOLEAN);\nCREATE TABLE user_credentials (user_id INT PRIMARY KEY, password_hash VARCHAR(255), oauth_access_token VARCHAR(512));", []),
        (22, "config/auth.env", "AUTH_MODE=multi_provider\nOAUTH_ISSUER=https://auth.company.internal\nPASSWORD_HASH_REQUIRED=true", []),
        (23, "src/models/user.py", "class User(Base):\n  email=Column(String(100))\n  credentials=relationship('UserCredentials')  # maps to schema v3", ["db/schema.sql"]),
        (24, "src/auth/session.py", "def login(user, secret):  # routes to password or oauth per AUTH_MODE", ["config/auth.env", "db/schema.sql"]),
        (25, "tests/test_auth.py", "def test_multi_provider(): assert session.login(...)  # written against session v3", ["src/auth/session.py"]),
        (26, "api/routes.json", '{"version":3,"endpoints":["/api/v3/login","/api/v3/oauth"],"protocol":"https","tls":true}', ["config/auth.env"]),
        (27, "docs/api.md", "# API v3\nPOST /api/v3/login (password or oauth), POST /api/v3/oauth PKCE.", ["api/routes.json"]),
        (28, "README.md", "Quickstart: multi_provider auth, credentials table required.", ["config/auth.env", "db/schema.sql"]),
        (29, "src/auth/session.py", "def login(user, secret):\n  mode = settings.AUTH_MODE  # multi_provider default", ["config/auth.env"]),
        (30, "src/models/user.py", "class User(Base):\n  email=Column(String(100))\n  is_active=Column(Boolean, default=True)  # schema v3 complete", ["db/schema.sql"]),
        (31, "config/cache.yaml", "cache_host: 10.0.0.42\nport: 6379\nmode: cluster\n ttl: 600", []),
        (32, "docs/runbook.md", "Rollback: AUTH_MODE=multi_provider stays; schema rollback needs migrations table.", ["config/auth.env", "db/schema.sql"]),
        (33, "migrations/2026_01.sql", "CREATE TABLE user_credentials (user_id INT PRIMARY KEY, password_hash VARCHAR(255));  -- schema v3", ["db/schema.sql"]),
        (34, "api/routes.json", '{"version":3.1,"endpoints":["/api/v3/login","/api/v3/oauth","/api/v3/refresh"],"protocol":"https","tls":true}', ["config/auth.env"]),
        (35, "docs/api.md", "# API v3.1\nAdded POST /api/v3/refresh token rotation.", ["api/routes.json"]),
        (36, "tests/test_auth.py", "def test_refresh(): assert session.refresh(...)  # written against routes v3.1", ["src/auth/session.py", "api/routes.json"]),
        (37, "src/auth/session.py", "def refresh(old_token):  # rotates oauth refresh token", ["config/auth.env", "api/routes.json"]),
        (38, "src/models/user.py", "class User(Base):\n  email=Column(String(100))\n  is_active=Column(Boolean, default=True)\n  credentials=relationship('UserCredentials')  # schema v3 final", ["db/schema.sql"]),
        (39, "README.md", "Quickstart: multi_provider auth, credentials table, /api/v3/refresh.", ["config/auth.env", "db/schema.sql", "api/routes.json"]),
        (40, "docs/runbook.md", "Rollback: keep multi_provider; refresh tokens rotate on demand.", ["config/auth.env", "api/routes.json"]),
    ]
    for step, res, content, deps in S:
        svc.ingest_state(res, content, step_id=step, dependencies=deps)
    return S


# Queries with known current-state answers (resource -> must be active hit).
# Legacy-heavy terms are included deliberately so lexical overlap favours stale
# text — that is the failure mode being measured.
QUERIES = [
    (
        "user table columns password_hash legacy_pin oauth access token credentials",
        {"db/schema.sql", "src/models/user.py", "config/auth.env"},
    ),
    (
        "api gateway auth endpoint routes http protocol login v1",
        {"api/routes.json", "config/auth.env"},
    ),
    (
        "test auth login password_hash session credentials",
        {"tests/test_auth.py", "src/auth/session.py"},
    ),
    (
        "migration alter table users password_hash oauth_access_token",
        {"migrations/2026_01.sql"},
    ),
    (
        "cache configuration parameters for services",
        {"config/cache.yaml"},
    ),
]


def fifo_baseline(timeline, t_curr):
    """FIFO keeps only the last FIFO_WINDOW steps: current states born early
    but never re-ingested are gone."""
    retained = {res for step, res, _, _ in timeline if t_curr - step <= FIFO_WINDOW}
    needed = set()
    for _, resources in QUERIES:
        needed |= resources
    forgotten = sorted(r for r in needed if r not in retained)
    return forgotten


def main() -> None:
    print("MATP long-horizon benchmark — 40 logical steps, 10 resources")
    print("Rahul Saravanan, NITK Surathkal (endsem)\n")

    svc = build_service()
    timeline = ingest_scenario(svc)
    t_curr = svc.stats()["buffer"]["logical_clock_step"]
    print(f"ingested {len(timeline)} states, clock t_curr={t_curr}\n")

    # FIFO baseline
    forgotten = fifo_baseline(timeline, t_curr)
    print(f"FIFO (window={FIFO_WINDOW} steps):")
    print(f"  forgotten current resources: {forgotten or 'none'}")
    print(f"  retention: {4 - len(forgotten)}/4 query-relevant resources retained\n")

    # Naive (no stale filter) vs MATP variants across queries
    print(f"{'query':<58} {'method':<16} {'ret':>3} {'lin':>3} {'dep':>3} {'act':>3}")
    print("-" * 94)
    agg = {"Naive": [0, 0, 0, 0], "MATP": [0, 0, 0, 0], "MATP+graph": [0, 0, 0, 0]}
    for query, _ in QUERIES:
        naive_res = svc.retrieve_context(
            query, k=5, time_decay=0.0, exclude_stale=False,
            propagate_dependencies=False,
        )
        matp_res = svc.retrieve_context(
            query, k=5, time_decay=0.05, exclude_stale=True,
            propagate_dependencies=False,
        )
        graph_res = svc.retrieve_context(
            query, k=5, time_decay=0.05, exclude_stale=True,
            propagate_dependencies=True,
        )
        # Items MATP keeps (active by lineage) that the dependency graph drops.
        graph_ids = {r["item_id"] for r in graph_res}
        dep_dropped = [r for r in matp_res if r["item_id"] not in graph_ids]

        rows = []
        lin_n = sum(1 for r in naive_res if not r["is_latest"])
        act_n = sum(1 for r in naive_res if r["is_latest"])
        rows.append(("Naive", naive_res, lin_n, 0, act_n))

        lin_m = sum(1 for r in matp_res if not r["is_latest"])
        act_m = sum(1 for r in matp_res if r["is_latest"])
        rows.append(("MATP", matp_res, lin_m, 0, act_m))

        act_g = sum(1 for r in graph_res if r["is_latest"])
        rows.append(("MATP+graph", graph_res, 0, len(dep_dropped), act_g))

        short = query if len(query) <= 56 else query[:55] + "…"
        for name, results, lin, dep, act in rows:
            agg[name][0] += len(results)
            agg[name][1] += lin
            agg[name][2] += dep
            agg[name][3] += act
            print(
                f"{short:<58} {name:<16} {len(results):>3} {lin:>3} {dep:>3} {act:>3}"
            )
        print("-" * 94)

    print("\nAggregate over all queries (top-5 each; dep = suppressed by graph):")
    print(f"{'method':<16} {'returned':>8} {'lineage-stale':>13} {'dep-drop':>8} {'active':>6} {'active%':>7}")
    for name, (ret, lin, dep, act) in agg.items():
        pct = 100.0 * act / ret if ret else 0.0
        print(
            f"{name:<16} {ret:>8} {lin:>13} {dep:>11} {act:>6} {pct:>6.0f}%"
        )

    # Adaptive decay round trip
    print("\nAdaptive decay (learned per task):")
    hits = svc.retrieve_context(
        "api gateway auth endpoint routes refresh token",
        k=3,
        time_decay=0.05,
        task="oauth-migration",
    )
    # Five more steps pass before the agent reports which hits it actually used:
    # the two newest states still matter at this horizon.
    svc.advance_logical_clock(5)
    useful = [r["item_id"] for r in hits[:2]]
    fb = svc.record_feedback("oauth-migration", useful)
    print(f"  feedback on {fb['feedback_items']} items -> lambda={fb['lambda_value']}")
    auto = svc.retrieve_context(
        "api gateway auth endpoint routes refresh token",
        k=3,
        time_decay="auto",
        task="oauth-migration",
    )
    print(
        f"  auto retrieval used lambda={auto[0]['decay_lambda']}, "
        f"top={auto[0]['resource_id']} v{auto[0]['version']}"
    )

    # Dependency graph audit
    graph = svc.dependency_graph()
    print(f"\nDependency graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
    print(f"  currently invalid: {graph['invalid'] or 'none'}")
    s = svc.stats()
    print(
        f"\nStats: {s['buffer']['items']} items, {s['buffer']['latest_items']} active, "
        f"{s['buffer']['stale_items']} stale, {s['buffer']['dependency_edges']} dep-edges, "
        f"clock={s['buffer']['logical_clock_step']}"
    )


if __name__ == "__main__":
    main()
