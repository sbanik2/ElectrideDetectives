"""One extra literature tool: semantic paper search through the Valency MCP server.

valency_search(query) -> papers from the Valency corpus whose MEANING matches the query, not its keywords.

Unlike literature_search (arXiv keyword search), Valency's `semantic_search_papers` embeds the query and
ranks a multi-source corpus (arXiv, PubMed, ...) by conceptual similarity, so a description such as
"electrons trapped in lattice cavities acting as anions" finds relevant work without naming it.

The server is remote and reached through `npx mcp-remote`, which handles HTTP transport and OAuth; the first
call of a session may open a browser to sign in. Like every other tool, this one takes ONE string and always
returns a string, so an agent sees either evidence or an explicit failure message.
"""
import asyncio
import json
import os
import threading
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

VALENCY_URL = "https://mcp.valency.io/"
VALENCY_TOOL = "semantic_search_papers"
MAX_RESULTS = 3            # papers returned to the agent
MIN_INTERVAL = 1.0         # seconds between calls (agents run in parallel)
ABSTRACT_CHARS = 1200      # per paper

VALENCY_SERVER = StdioServerParameters(
    command="npx", args=["-y", "mcp-remote", VALENCY_URL], env=os.environ.copy())

_LOCK = threading.Lock()
_last_call = [0.0]


async def _call(tool, args):
    async with stdio_client(VALENCY_SERVER) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool(name=tool, arguments=args)
    text = "\n".join(getattr(c, "text", "") for c in res.content if getattr(c, "text", ""))
    if not text and getattr(res, "structuredContent", None):
        text = json.dumps(res.structuredContent)
    if getattr(res, "isError", False):
        raise RuntimeError(text[:300] or f"Valency tool {tool} returned an error")
    return text


def _run(coro):
    """asyncio.run that also works when called from a notebook cell (which has a running loop)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)          # normal case: worker thread from asyncio.to_thread
    box = {}
    def target():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:
            box["error"] = exc
    t = threading.Thread(target=target)
    t.start(); t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def valency_search(query: str) -> str:
    """Semantic literature search via Valency MCP: papers whose meaning matches the query."""
    try:
        with _LOCK:
            wait = MIN_INTERVAL - (time.monotonic() - _last_call[0])
            if wait > 0:
                time.sleep(wait)
            try:
                raw = _run(_call(VALENCY_TOOL, {"query": query, "limit": MAX_RESULTS}))
            finally:
                _last_call[0] = time.monotonic()
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw[:4000] if raw.strip() else f"Valency search returned nothing for {query!r}."
        papers = data.get("papers", []) if isinstance(data, dict) else []
        if not papers:
            return f"Valency search returned no papers for {query!r}; try other wording."
        return json.dumps([{"id": p.get("id"), "title": p.get("title", ""),
                            "authors": (p.get("authors") or [])[:3],
                            "year": str(p.get("datestamp") or "")[:4],
                            "source": p.get("source"), "doi": p.get("doi"),
                            "citations": p.get("citation_count"),
                            "abstract": (p.get("abstract") or "")[:ABSTRACT_CHARS],
                            "url": p.get("url", "")}
                           for p in papers[:MAX_RESULTS]])
    except Exception as exc:
        return f"Valency search unavailable: {type(exc).__name__}: {exc}"
