"""The Fetch Tax: how many tokens does each retrieval path put into an agent's context?

For every question in questions.json we run up to five retrieval paths and record
how much text enters the model's context, what it cost, and whether the model
answered correctly.

  raw_html        GET the canonical URL, count the tokens an agent would swallow if it
                  pasted the raw page (measurement only, never sent to the model)
  url_context     Gemini's built-in URL context tool reads the canonical URL; Google's
                  own usage object reports the tool-use tokens
  exa_full_text   Exa MCP ``web_fetch_exa`` returns the canonical page as clean markdown
  exa_highlights  Exa MCP ``web_search_exa`` finds pages itself and returns highlights
  google_search   Gemini's built-in Google Search grounding, no URL supplied

Usage:
    uv run python fetch_tax.py run --project PROJECT [--paths ...] [--limit N]
    uv run python fetch_tax.py probe --project PROJECT      # one question, all paths
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from google import genai
from google.genai import errors, types
from mcp.client.client import Client as McpClient

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
QUESTIONS_FILE = HERE / "questions.json"

EXA_MCP_URL = "https://mcp.exa.ai/mcp?tools=web_search_exa,web_fetch_exa"
EXA_API_URL = "https://api.exa.ai/search"
EXA_DYNAMIC_BETA = "dynamic-highlights-2026-08-28"
EXA_NUM_RESULTS = 5
EXA_FULL_TEXT_MAX_CHARS = 200_000

ANSWER_MODEL = "gemini-3.8-flash"
TOKEN_MODEL = "gemini-3.8-flash"

# Published list prices on 2026-10-13, used only to turn tokens into dollars.
PRICE_PER_M_INPUT = 0.75  # Gemini 3.8 Flash input, through 2026-12-31
PRICE_PER_M_OUTPUT = 3.75  # Gemini 3.8 Flash output incl. thinking
PRICE_GOOGLE_SEARCH_QUERY = 14 / 1000  # after 5,000 free requests per month
PRICE_EXA_SEARCH = 7 / 1000  # auto search, up to 10 results, contents included
PRICE_EXA_CONTENTS_PAGE = 1 / 1000  # contents endpoint, per page per content type

PATHS = (
    "raw_html",
    "url_context",
    "exa_full_text",
    "exa_highlights",
    "exa_dynamic",
    "google_search",
)

ANSWER_INSTRUCTION = (
    "Answer in one short sentence with the specific value asked for. "
    "If you cannot find it, reply exactly: not found."
)


@dataclass
class Record:
    question_id: str
    path: str
    model: str
    context_tokens: int = 0  # tokens of retrieved material entering the context
    prompt_tokens: int = 0  # total prompt tokens billed for the answering call
    output_tokens: int = 0
    thought_tokens: int = 0
    context_chars: int = 0
    search_queries: int = 0  # Google Search queries the model chose to run
    pages: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0  # list-price estimate
    measured_cost_usd: float | None = None  # what the provider itself reported
    provider_latency_s: float | None = None
    answer: str = ""
    correct: bool | None = None
    error: str | None = None
    attempts: int = 1
    context_preview: str = ""
    sources: list[str] = field(default_factory=list)


def load_questions(limit: int | None = None) -> list[dict]:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    return questions[:limit] if limit else questions


def grade(answer: str, pattern: str) -> bool:
    """Mechanical grading: the expected value must appear in a one-line answer."""
    if re.search(r"\bnot found\b", answer, re.IGNORECASE):
        return False
    return re.search(pattern, answer, re.IGNORECASE) is not None


def token_cost(prompt_tokens: int, output_tokens: int) -> float:
    return (
        prompt_tokens * PRICE_PER_M_INPUT + output_tokens * PRICE_PER_M_OUTPUT
    ) / 1_000_000


def make_client(project: str, location: str) -> genai.Client:
    return genai.Client(vertexai=True, project=project, location=location)


async def generate_with_retry(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig | None,
    attempts: int = 5,
) -> types.GenerateContentResponse:
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            return await client.aio.models.generate_content(
                model=model, contents=contents, config=config
            )
        except errors.APIError as exc:
            if exc.code not in (429, 500, 503) or attempt == attempts:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
    raise RuntimeError("unreachable")


async def count_tokens(client: genai.Client, text: str) -> int:
    if not text:
        return 0
    result = await client.aio.models.count_tokens(model=TOKEN_MODEL, contents=text)
    return int(result.total_tokens or 0)


def usage_fields(response: types.GenerateContentResponse) -> dict[str, int]:
    usage = response.usage_metadata
    if usage is None:
        return {"prompt": 0, "output": 0, "thought": 0, "tool_use": 0}
    return {
        "prompt": int(usage.prompt_token_count or 0),
        "output": int(usage.candidates_token_count or 0),
        "thought": int(usage.thoughts_token_count or 0),
        "tool_use": int(usage.tool_use_prompt_token_count or 0),
    }


# --------------------------------------------------------------------------- paths


async def path_raw_html(client: genai.Client, q: dict) -> Record:
    rec = Record(q["id"], "raw_html", "-")
    start = time.perf_counter()
    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as http:
        resp = await http.get(
            q["url"], headers={"User-Agent": "fetch-tax-experiment/1.0"}
        )
    body = resp.text
    rec.latency_s = time.perf_counter() - start
    rec.context_chars = len(body)
    rec.context_preview = body[:400]
    rec.context_tokens = await count_tokens(client, body)
    rec.pages = 1
    rec.sources = [q["url"]]
    rec.cost_usd = rec.context_tokens * PRICE_PER_M_INPUT / 1_000_000
    return rec


async def path_url_context(client: genai.Client, q: dict) -> Record:
    rec = Record(q["id"], "url_context", ANSWER_MODEL)
    prompt = (
        f"{q['question']}\nRead this page to answer: {q['url']}\n{ANSWER_INSTRUCTION}"
    )
    config = types.GenerateContentConfig(
        tools=[types.Tool(url_context=types.UrlContext())], temperature=0
    )
    start = time.perf_counter()
    response = await generate_with_retry(client, ANSWER_MODEL, prompt, config)
    rec.latency_s = time.perf_counter() - start
    usage = usage_fields(response)
    rec.context_tokens = usage["tool_use"]
    rec.prompt_tokens = usage["prompt"] + usage["tool_use"]
    rec.output_tokens = usage["output"] + usage["thought"]
    rec.thought_tokens = usage["thought"]
    rec.pages = 1
    rec.sources = [q["url"]]
    rec.answer = (response.text or "").strip()
    rec.correct = grade(rec.answer, q["answer_regex"])
    rec.cost_usd = token_cost(rec.prompt_tokens, rec.output_tokens)
    return rec


async def answer_from_context(
    client: genai.Client, q: dict, context: str, rec: Record
) -> Record:
    prompt = (
        f"Question: {q['question']}\n{ANSWER_INSTRUCTION}\n\n"
        f"Context (search results):\n{context}"
    )
    config = types.GenerateContentConfig(temperature=0)
    start = time.perf_counter()
    response = await generate_with_retry(client, ANSWER_MODEL, prompt, config)
    rec.latency_s += time.perf_counter() - start
    usage = usage_fields(response)
    rec.prompt_tokens = usage["prompt"]
    rec.output_tokens = usage["output"] + usage["thought"]
    rec.thought_tokens = usage["thought"]
    rec.answer = (response.text or "").strip()
    rec.correct = grade(rec.answer, q["answer_regex"])
    rec.cost_usd += token_cost(rec.prompt_tokens, rec.output_tokens)
    return rec


def mcp_text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")


async def path_exa_full_text(client: genai.Client, mcp: McpClient, q: dict) -> Record:
    rec = Record(q["id"], "exa_full_text", ANSWER_MODEL)
    start = time.perf_counter()
    result = await mcp.call_tool(
        "web_fetch_exa", {"urls": [q["url"]], "maxCharacters": EXA_FULL_TEXT_MAX_CHARS}
    )
    rec.latency_s = time.perf_counter() - start
    text = mcp_text(result)
    if getattr(result, "is_error", False):
        rec.error = text[:300]
        return rec
    rec.context_chars = len(text)
    rec.context_preview = text[:400]
    rec.context_tokens = await count_tokens(client, text)
    rec.pages = 1
    rec.sources = [q["url"]]
    rec.cost_usd = PRICE_EXA_CONTENTS_PAGE
    return await answer_from_context(client, q, text, rec)


async def path_exa_highlights(client: genai.Client, mcp: McpClient, q: dict) -> Record:
    rec = Record(q["id"], "exa_highlights", ANSWER_MODEL)
    args = {
        "query": q["question"],
        "numResults": EXA_NUM_RESULTS,
        "objective": (
            "Find the page that states this value directly; prefer official "
            "documentation or the primary source and pull the exact figure."
        ),
    }
    text = ""
    start = time.perf_counter()
    for attempt in range(1, 4):  # the keyless tier sometimes returns no results
        rec.attempts = attempt
        result = await mcp.call_tool("web_search_exa", args)
        text = mcp_text(result)
        if getattr(result, "is_error", False):
            rec.error = text[:300]
            return rec
        if re.search(r"^URL:\s*\S+", text, re.MULTILINE):
            break
        await asyncio.sleep(3 * attempt)
    rec.latency_s = time.perf_counter() - start
    rec.context_chars = len(text)
    rec.context_preview = text[:400]
    rec.context_tokens = await count_tokens(client, text)
    rec.sources = re.findall(r"^URL:\s*(\S+)", text, re.MULTILINE)
    rec.pages = len(rec.sources)
    rec.cost_usd = PRICE_EXA_SEARCH
    return await answer_from_context(client, q, text, rec)


async def path_google_search(client: genai.Client, q: dict) -> Record:
    rec = Record(q["id"], "google_search", ANSWER_MODEL)
    prompt = f"{q['question']}\n{ANSWER_INSTRUCTION}"
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())], temperature=0
    )
    start = time.perf_counter()
    response = await generate_with_retry(client, ANSWER_MODEL, prompt, config)
    rec.latency_s = time.perf_counter() - start
    usage = usage_fields(response)
    rec.context_tokens = usage["tool_use"]
    rec.prompt_tokens = usage["prompt"] + usage["tool_use"]
    rec.output_tokens = usage["output"] + usage["thought"]
    rec.thought_tokens = usage["thought"]
    rec.answer = (response.text or "").strip()
    rec.correct = grade(rec.answer, q["answer_regex"])
    candidate = response.candidates[0] if response.candidates else None
    meta = getattr(candidate, "grounding_metadata", None)
    if meta is not None:
        rec.search_queries = len(meta.web_search_queries or [])
        rec.sources = [
            c.web.uri for c in (meta.grounding_chunks or []) if c.web and c.web.uri
        ]
        rec.pages = len(rec.sources)
    rec.cost_usd = token_cost(rec.prompt_tokens, rec.output_tokens) + (
        rec.search_queries * PRICE_GOOGLE_SEARCH_QUERY
    )
    return rec


async def path_exa_dynamic(client: genai.Client, q: dict) -> Record:
    """Exa /search with Dynamic Highlights (research preview). Needs EXA_API_KEY."""
    rec = Record(q["id"], "exa_dynamic", ANSWER_MODEL)
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        rec.error = "EXA_API_KEY not set"
        return rec
    payload = {
        "query": q["question"],
        "type": "auto",
        "numResults": EXA_NUM_RESULTS,
        "contents": {"highlights": {"dynamic": True}},
    }
    headers = {"x-api-key": api_key, "Exa-Beta": EXA_DYNAMIC_BETA}
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=60) as http:
        resp = await http.post(EXA_API_URL, json=payload, headers=headers)
    rec.latency_s = time.perf_counter() - start
    if resp.status_code != 200:
        rec.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        return rec
    data = resp.json()
    blocks = []
    for r in data.get("results", []):
        highlights = "\n".join(r.get("highlights") or [])
        blocks.append(
            f"Title: {r.get('title')}\nURL: {r.get('url')}\n"
            f"Published: {r.get('publishedDate') or 'N/A'}\nHighlights:\n{highlights}"
        )
        rec.sources.append(r.get("url") or "")
    text = "\n\n---\n\n".join(blocks)
    rec.pages = len(blocks)
    rec.context_chars = len(text)
    rec.context_preview = text[:400]
    rec.context_tokens = await count_tokens(client, text)
    rec.measured_cost_usd = float((data.get("costDollars") or {}).get("total") or 0)
    search_ms = data.get("searchTime")
    rec.provider_latency_s = float(search_ms) / 1000 if search_ms else None
    rec.cost_usd = rec.measured_cost_usd
    return await answer_from_context(client, q, text, rec)


# ----------------------------------------------------------------------------- run


async def run_question(
    client: genai.Client, mcp: McpClient | None, q: dict, paths: tuple[str, ...]
) -> list[Record]:
    out: list[Record] = []
    for path in paths:
        try:
            if path == "raw_html":
                rec = await path_raw_html(client, q)
            elif path == "url_context":
                rec = await path_url_context(client, q)
            elif path == "exa_full_text":
                assert mcp is not None
                rec = await path_exa_full_text(client, mcp, q)
            elif path == "exa_highlights":
                assert mcp is not None
                rec = await path_exa_highlights(client, mcp, q)
            elif path == "exa_dynamic":
                rec = await path_exa_dynamic(client, q)
            elif path == "google_search":
                rec = await path_google_search(client, q)
            else:
                raise ValueError(path)
        except Exception as exc:  # noqa: BLE001 - record and keep going
            rec = Record(
                q["id"], path, ANSWER_MODEL, error=f"{type(exc).__name__}: {exc}"[:300]
            )
        out.append(rec)
        mark = "✓" if rec.correct else ("✗" if rec.correct is False else "-")
        print(
            f"{q['id']} {path:15s} {mark} ctx={rec.context_tokens:>7,} tok "
            f"prompt={rec.prompt_tokens:>7,} ${rec.cost_usd:.4f} {rec.latency_s:5.1f}s"
            + (f"  ERROR {rec.error}" if rec.error else ""),
            flush=True,
        )
        if path.startswith("exa"):
            await asyncio.sleep(0.5)  # keyless MCP tier is rate limited
    return out


async def cmd_run(args: argparse.Namespace) -> None:
    client = make_client(args.project, args.location)
    questions = load_questions(args.limit)
    paths = tuple(args.paths)
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = f"{datetime.now(UTC):%Y%m%d_%H%M%S}"
    out_path = RESULTS_DIR / f"runs_{stamp}.jsonl"
    needs_mcp = any(p.startswith("exa") for p in paths)

    async def run_all(mcp: McpClient | None) -> None:
        with out_path.open("w", encoding="utf-8") as fh:
            for q in questions:
                for rec in await run_question(client, mcp, q, paths):
                    row = asdict(rec) | {"domain": q["domain"], "url": q["url"]}
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()

    if needs_mcp:
        async with McpClient(EXA_MCP_URL) as mcp:
            await run_all(mcp)
    else:
        await run_all(None)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "probe"):
        p = sub.add_parser(name)
        p.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
        p.add_argument("--location", default="global")
        p.add_argument("--paths", nargs="+", default=list(PATHS), choices=PATHS)
        p.add_argument("--limit", type=int, default=1 if name == "probe" else None)
    args = parser.parse_args()
    if not args.project:
        sys.exit("Set --project or GOOGLE_CLOUD_PROJECT.")
    asyncio.run(cmd_run(args))


if __name__ == "__main__":
    main()
