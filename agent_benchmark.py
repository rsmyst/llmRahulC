"""Agent-in-the-loop comparison: Gemini 3.8 Flash with vs without the memory MCP server.

Experiment:
- An agent receives 24 sequential notes (facts + filler status updates) across
  24 turns. Conversation history is artificially capped at ~1500 tokens: oldest
  messages are evicted FIFO, so early facts fall out of the context window.
- Two conditions, identical notes and probes:
    no-MCP : plain chat, memory only inside the (truncated) context window
    MCP    : agent may call ingest_state / retrieve_context / get_memory_stats
             on the real semantic-memory-buffer server (stdio subprocess)
- After the notes, 7 probe questions ask about early / mid / late facts.
  Early facts are guaranteed to be outside the 1500-token window by then.

Metrics: probe accuracy (overall + early facts only), prompt/completion/thought
tokens, API call count, tool calls, wall time.

Run:  python agent_benchmark.py            (uses SM_GEMINI_API_KEY from .env)
      python agent_benchmark.py --notes 12  (quicker smoke run)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import httpx

from semantic_memory.config import Config

MODEL = "gemini-3.8-flash"
CONTEXT_TOKEN_LIMIT = 1500  # artificial sliding window
MAX_TOOL_STEPS = 4          # tool round trips allowed per turn

# --------------------------------------------------------------------- notes
# (fact_notes carry a probe; filler is status noise a real session would have)
FACT_NOTES = {
    1: ("The staging database password is hunter2-staging.", "hunter2-staging"),
    2: ("The production API gateway listens on port 8443.", "8443"),
    3: ("The billing service lead engineer is Priya.", "priya"),
    4: ("The legacy monolith codename is Bluebird.", "bluebird"),
    5: ("The data retention policy freezes logs after December 20.", "december"),
    6: ("The primary metrics region is eu-central.", "eu-central"),
    7: ("The on-call rotation for payments is team-wolverine.", "wolverine"),
    8: ("The vendor contract ID for email delivery is VEND-7734.", "vend-7734"),
    13: ("The cache layer was migrated from Redis to KeyDB last week.", "keydb"),
    16: ("The analytics warehouse now runs on DuckDB.", "duckdb"),
    20: ("The mobile client min supported OS version is iOS 16.", "ios 16"),
    22: ("The current sprint goal is to ship OAuth2 PKCE login.", "pkce"),
}

FILLER_NOTES = [
    "Build 4821 passed all unit tests.",
    "The staging pod restarted once overnight, no action needed.",
    "Ticket OPS-119 was closed as duplicate.",
    "Latency dashboard shows p99 at 210 ms, within budget.",
    "Two new hires start next Monday in the platform team.",
    "The nightly backup completed in 42 minutes.",
    "Code freeze for the release branch begins Friday.",
    "Security scan found no new advisories this week.",
    "The design review for the new billing UI is scheduled.",
    "Staging TLS certificates were rotated successfully.",
    "The search index rebuild finished without errors.",
    "A follow-up meeting was booked to discuss queue backlogs.",
    "The observability agent was upgraded to version 3.4.",
    "Three flaky integration tests were quarantined.",
    "The CDN purge for the marketing pages completed.",
]

PROBES = [
    ("What is the staging database password?", FACT_NOTES[1][1], "early"),
    ("Which port does the production API gateway listen on?", FACT_NOTES[2][1], "early"),
    ("Who is the lead engineer for the billing service?", FACT_NOTES[3][1], "early"),
    ("What is the codename of the legacy monolith?", FACT_NOTES[4][1], "early"),
    ("After which month are logs frozen under the retention policy?", FACT_NOTES[5][1], "early"),
    ("Which cache system did the team migrate to?", FACT_NOTES[13][1], "mid"),
    ("What is the sprint goal for OAuth login?", FACT_NOTES[22][1], "late"),
]


def build_notes(n_notes: int) -> list[str]:
    """Interleave fact notes with filler up to n_notes turns."""
    fact_ids = sorted(FACT_NOTES)
    notes: list[str] = []
    fact_idx = 0
    fill_idx = 0
    for turn in range(1, n_notes + 1):
        if fact_idx < len(fact_ids) and turn == fact_ids[fact_idx]:
            notes.append(FACT_NOTES[fact_ids[fact_idx]][0])
            fact_idx += 1
        else:
            notes.append(FILLER_NOTES[fill_idx % len(FILLER_NOTES)])
            fill_idx += 1
    return notes


# ------------------------------------------------------------------ Gemini

def est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token + role overhead)."""
    return len(text) // 4 + 4


def history_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += est_tokens(json.dumps(m.get("content", "")))
        total += est_tokens(m.get("role", ""))
    return total


def trim_history(messages: list[dict], limit: int, system: str) -> list[dict]:
    """FIFO-evict oldest messages until history fits the artificial window."""
    budget = limit - est_tokens(system)
    while messages and history_tokens(messages) > budget:
        messages.pop(0)
        # never leave an orphaned tool response at the front
        while messages and messages[0]["role"] == "tool":
            messages.pop(0)
    return messages


class GeminiAgent:
    def __init__(self, api_key: str, base_url: str, system: str):
        self._url = f"{base_url.rstrip('/')}/models/{MODEL}:generateContent"
        self._headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        self._system = system
        self._client = httpx.Client(timeout=180)
        self.messages: list[dict] = []
        self.stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thought_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
        }

    def _call(self, tools: list[dict] | None) -> dict:
        body: dict = {
            "systemInstruction": {"parts": [{"text": self._system}]},
            "contents": self._messages_payload(),
        }
        if tools:
            body["tools"] = [{"functionDeclarations": tools}]
        # keep latency and token cost predictable; drop thinking if unsupported
        body["generationConfig"] = {
            "temperature": 0.2,
            "thinkingConfig": {"thinkingBudget": 0},
        }

        for attempt in range(6):
            resp = self._client.post(self._url, json=body, headers=self._headers)
            if resp.status_code in (429, 503):
                time.sleep(min(2**attempt, 20))
                continue
            if resp.status_code == 400 and "thinking" in resp.text.lower():
                body["generationConfig"].pop("thinkingConfig", None)
                continue
            resp.raise_for_status()
            data = resp.json()
            self.stats["api_calls"] += 1
            usage = data.get("usageMetadata", {})
            self.stats["prompt_tokens"] += usage.get("promptTokenCount", 0)
            self.stats["completion_tokens"] += usage.get("candidatesTokenCount", 0)
            self.stats["thought_tokens"] += usage.get("thoughtsTokenCount", 0)
            self.stats["total_tokens"] += usage.get("totalTokenCount", 0)
            return data
        raise RuntimeError(f"give up calling {MODEL}: last={resp.status_code} {resp.text[:200]}")

    def _messages_payload(self) -> list[dict]:
        out = []
        for m in self.messages:
            if m["role"] == "user":
                out.append({"role": "user", "parts": [{"text": m["content"]}]})
            elif m["role"] == "model":
                if m.get("functionCall"):
                    out.append({"role": "model", "parts": [{"functionCall": m["functionCall"]}]})
                else:
                    out.append({"role": "model", "parts": [{"text": m["content"]}]})
            elif m["role"] == "tool":
                out.append({
                    "role": "user",
                    "parts": [{"functionResponse": {"name": m["name"], "response": m["content"]}}],
                })
        return out

    def _extract(self, data: dict) -> tuple[str, dict | None]:
        text_parts, fc = [], None
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "functionCall" in part:
                    fc = part["functionCall"]
                elif "text" in part:
                    text_parts.append(part["text"])
        return "".join(text_parts).strip(), fc

    def send(self, user_text: str, tools: list[dict] | None, exec_tool) -> str:
        """One user turn; executes tool calls via exec_tool(name, args) -> dict."""
        self.messages.append({"role": "user", "content": user_text})
        self.messages = trim_history(self.messages, CONTEXT_TOKEN_LIMIT, self._system)

        final_text = ""
        for _ in range(MAX_TOOL_STEPS):
            data = self._call(tools)
            text, fc = self._extract(data)
            if fc is None:
                self.messages.append({"role": "model", "content": text})
                final_text = text
                break
            self.messages.append({"role": "model", "content": "", "functionCall": fc})
            result = exec_tool(fc.get("name", ""), fc.get("args", {}) or fc.get("arguments", {}))
            self.messages.append({"role": "tool", "name": fc.get("name", ""), "content": result})
            self.messages = trim_history(self.messages, CONTEXT_TOKEN_LIMIT, self._system)
            final_text = text  # may be empty when the turn was pure tool call
        else:
            self.messages.append({"role": "model", "content": final_text})
        return final_text


# ---------------------------------------------------------------- conditions

SYSTEM_BASE = (
    "You are a software project assistant. You receive sequential session notes "
    "from a teammate and later answer questions about them. Answer briefly and "
    "directly. If you do not know, say you do not know."
)

SYSTEM_MCP = SYSTEM_BASE + (
    " You also have a persistent memory service: when you learn a fact that may"
    " be asked about later, store it with the ingest_state tool (resource_id"
    " 'session/facts'). Before answering a question, call retrieve_context to"
    " check the memory service."
)


def mcp_tools() -> list[dict]:
    return [
        {
            "name": "ingest_state",
            "description": "Store a fact or state into persistent memory, tagged by resource id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_id": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["resource_id", "content"],
            },
        },
        {
            "name": "retrieve_context",
            "description": "Retrieve relevant stored states from persistent memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_memory_stats",
            "description": "Return memory buffer statistics.",
            "parameters": {"type": "object", "properties": {}},
        },
    ]


def is_correct(answer: str, expected: str) -> bool:
    a = re.sub(r"[^a-z0-9\- ]", "", answer.lower())
    return expected.lower() in a


async def run_mcp_session() -> "MemorySession":
    """Spawn the real MCP server subprocess and open a session."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-W", "ignore", "-m", "semantic_memory"],
        cwd=str(Path(__file__).resolve().parent),
        env={
            **os.environ,
            "SM_BUFFER_DB": os.path.join(tempfile.gettempdir(), "matp_agent_bench.db"),
            "SM_GEMINI_API_KEY": "",  # embeddings stay local; only the agent uses Gemini
            "TRANSFORMERS_VERBOSITY": "error",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONWARNINGS": "ignore",
        },
    )
    ctx, inner, session = await _open(stdio_client, params)
    mem = MemorySession(ctx, inner, session)
    await mem.call("clear_memory", {})
    return mem


async def _open(stdio_client, params):
    # keep the context manager alive for the whole run
    ctx = stdio_client(params)
    read, write = await ctx.__aenter__()
    inner = ClientSession(read, write)
    session = await inner.__aenter__()
    await session.initialize()
    return ctx, inner, session


class MemorySession:
    def __init__(self, outer, inner, session):
        self._outer, self._inner, self._session = outer, inner, session
        self.tool_calls = 0

    async def call(self, name: str, args: dict) -> dict:
        self.tool_calls += 1
        res = await self._session.call_tool(name, args)
        return json.loads(res.content[0].text)

    async def close(self):
        await self._inner.__aexit__(None, None, None)
        await self._outer.__aexit__(None, None, None)


async def run_condition(
    label: str, api_key: str, base_url: str, notes: list[str], use_mcp: bool
) -> dict:
    session = None
    if use_mcp:
        session = await run_mcp_session()
        system = SYSTEM_MCP
        tools = mcp_tools()
    else:
        system = SYSTEM_BASE
        tools = None

    agent = GeminiAgent(api_key, base_url, system)
    tool_counter = {"n": 0}

    async def async_exec(name: str, args: dict) -> dict:
        if not use_mcp:
            return {"ok": True}
        tool_counter["n"] += 1
        return await session.call(name, args)

    t0 = time.time()
    for note in notes:
        await _send_async(agent, note, tools, async_exec)
    probe_results = []
    for question, expected, band in PROBES:
        answer = await _send_async(agent, question, tools, async_exec)
        probe_results.append(
            {
                "question": question,
                "band": band,
                "expected": expected,
                "answer": answer[:160],
                "correct": is_correct(answer, expected),
            }
        )
    wall = time.time() - t0

    if session:
        stats = await session.call("get_memory_stats", {})
        await session.close()
    else:
        stats = None

    early = [p for p in probe_results if p["band"] == "early"]
    return {
        "label": label,
        "probe_results": probe_results,
        "accuracy": sum(p["correct"] for p in probe_results) / len(probe_results),
        "early_accuracy": sum(p["correct"] for p in early) / len(early),
        "tool_calls": tool_counter["n"],
        "wall_seconds": round(wall, 1),
        **agent.stats,
        "memory_stats": stats,
    }


async def _send_async(agent: GeminiAgent, text: str, tools, async_exec) -> str:
    """Run agent.send (sync) with async tool execution via a worker thread."""
    import concurrent.futures

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_sync_send, agent, text, tools, async_exec, loop)
        return await asyncio.wrap_future(fut)


def _sync_send(agent, text, tools, async_exec, loop) -> str:
    # Tool calls happen on the event loop from this worker thread.
    def exec_tool(name, args):
        return asyncio.run_coroutine_threadsafe(async_exec(name, args), loop).result()

    return agent.send(text, tools, exec_tool)


# ---------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", type=int, default=24, help="number of session notes")
    parser.add_argument("--out", default="agent_benchmark_results.json")
    args = parser.parse_args()

    cfg = Config.from_env()
    if not cfg.has_gemini_key:
        sys.exit("SM_GEMINI_API_KEY is required for the agent benchmark (see .env.example)")

    notes = build_notes(args.notes)
    print(f"Agent benchmark — model={MODEL}, notes={len(notes)}, "
          f"context cap={CONTEXT_TOKEN_LIMIT} tokens, probes={len(PROBES)}")
    print(f"early-fact probes: {sum(1 for p in PROBES if p[2]=='early')}, "
          f"total notes token mass ~{sum(est_tokens(n) for n in notes)}\n")

    async def run() -> dict:
        no_mcp = await run_condition(
            "without MCP", cfg.gemini_api_key, cfg.gemini_base_url, notes, use_mcp=False
        )
        print(f"  [done] without MCP: accuracy={no_mcp['accuracy']:.0%} "
              f"early={no_mcp['early_accuracy']:.0%} tokens={no_mcp['total_tokens']}")
        with_mcp = await run_condition(
            "with MCP", cfg.gemini_api_key, cfg.gemini_base_url, notes, use_mcp=True
        )
        print(f"  [done] with MCP:    accuracy={with_mcp['accuracy']:.0%} "
              f"early={with_mcp['early_accuracy']:.0%} tokens={with_mcp['total_tokens']} "
              f"tool_calls={with_mcp['tool_calls']}")
        return {"without_mcp": no_mcp, "with_mcp": with_mcp,
                "config": {"model": MODEL, "context_limit": CONTEXT_TOKEN_LIMIT,
                           "notes": len(notes), "probes": len(PROBES)}}

    results = asyncio.run(run())

    print("\n" + "=" * 78)
    print(f"{'probe':<58} {'band':<6} {'no-MCP':>7} {'MCP':>5}")
    print("-" * 78)
    for p_n, p_m in zip(results["without_mcp"]["probe_results"],
                        results["with_mcp"]["probe_results"]):
        print(f"{p_n['question'][:56]:<58} {p_n['band']:<6} "
              f"{'OK' if p_n['correct'] else 'MISS':>7} "
              f"{'OK' if p_m['correct'] else 'MISS':>5}")
    print("-" * 78)
    a, b = results["without_mcp"], results["with_mcp"]
    print(f"{'accuracy (all probes)':<58} {'':<6} {a['accuracy']:>7.0%} {b['accuracy']:>5.0%}")
    print(f"{'early-fact probes (outside context window)':<58} {'':<6} "
          f"{a['early_accuracy']:>7.0%} {b['early_accuracy']:>5.0%}")
    print(f"{'prompt tokens':<58} {'':<6} {a['prompt_tokens']:>7} {b['prompt_tokens']:>5}")
    print(f"{'completion tokens':<58} {'':<6} {a['completion_tokens']:>7} {b['completion_tokens']:>5}")
    print(f"{'total tokens':<58} {'':<6} {a['total_tokens']:>7} {b['total_tokens']:>5}")
    print(f"{'API calls':<58} {'':<6} {a['api_calls']:>7} {b['api_calls']:>5}")
    print(f"{'memory tool calls':<58} {'':<6} {a['tool_calls']:>7} {b['tool_calls']:>5}")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
