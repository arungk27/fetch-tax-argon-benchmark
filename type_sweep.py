"""Sweep Exa search types on the same questions: does paying for depth buy anything?

For each ``type`` we run every question through /search with ``highlights: true``,
hand the highlights to the answering model, and record tokens, accuracy,
Exa's own ``searchTime`` and ``costDollars``. Deep types may also return a
synthesized answer in ``output.content``; we grade that separately as a bonus.

Usage:
    EXA_API_KEY=... uv run python type_sweep.py --project PROJECT [--model M]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime

import httpx

import fetch_tax
from fetch_tax import (
    DEFAULT_MODEL,
    EXA_API_URL,
    EXA_NUM_RESULTS,
    MODEL_PRICES,
    RESULTS_DIR,
    Record,
    answer_from_context,
    count_tokens,
    grade,
    load_questions,
    make_client,
)

TYPES = ("instant", "fast", "auto", "deep-lite", "deep", "deep-reasoning")


def render(results: list[dict]) -> str:
    blocks = []
    for r in results:
        highlights = "\n".join(r.get("highlights") or [])
        blocks.append(
            f"Title: {r.get('title')}\nURL: {r.get('url')}\n"
            f"Published: {r.get('publishedDate') or 'N/A'}\nHighlights:\n{highlights}"
        )
    return "\n\n---\n\n".join(blocks)


async def one(client, http: httpx.AsyncClient, q: dict, search_type: str) -> dict:
    rec = Record(q["id"], f"exa_type_{search_type}", fetch_tax.ANSWER_MODEL)
    payload = {
        "query": q["question"],
        "type": search_type,
        "numResults": EXA_NUM_RESULTS,
        "contents": {"highlights": True},
    }
    start = time.perf_counter()
    try:
        resp = await http.post(EXA_API_URL, json=payload)
    except httpx.HTTPError as exc:
        rec.error = f"{type(exc).__name__}: {exc}"[:200]
        return asdict(rec) | {"search_type": search_type, "synth_correct": None}
    rec.latency_s = time.perf_counter() - start
    if resp.status_code != 200:
        rec.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        return asdict(rec) | {"search_type": search_type, "synth_correct": None}
    data = resp.json()
    results = data.get("results", [])
    text = render(results)
    rec.pages = len(results)
    rec.sources = [r.get("url") or "" for r in results]
    rec.context_chars = len(text)
    rec.context_preview = text[:400]
    rec.context_tokens = await count_tokens(client, text)
    rec.measured_cost_usd = float((data.get("costDollars") or {}).get("total") or 0)
    search_ms = data.get("searchTime")
    rec.provider_latency_s = float(search_ms) / 1000 if search_ms else None
    rec.cost_usd = rec.measured_cost_usd

    # Deep types can synthesize an answer themselves; grade it as a bonus signal.
    synth = (
        (data.get("output") or {}).get("content")
        if isinstance(data.get("output"), dict)
        else None
    )
    synth_correct = grade(str(synth), q["answer_regex"]) if synth else None

    if results:
        await answer_from_context(client, q, text, rec)
    else:
        rec.answer = "not found"
        rec.correct = False
    return asdict(rec) | {
        "search_type": search_type,
        "synth_answer": (str(synth)[:300] if synth else None),
        "synth_correct": synth_correct,
        "domain": q["domain"],
    }


async def main_async(args: argparse.Namespace) -> None:
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        sys.exit("Set EXA_API_KEY in the environment (never in a file).")
    fetch_tax.ANSWER_MODEL = fetch_tax.TOKEN_MODEL = args.model
    client = make_client(args.project, args.location)
    questions = load_questions(args.limit)
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = f"{datetime.now(UTC):%Y%m%d_%H%M%S}"
    out = RESULTS_DIR / f"type_sweep_{args.model}_{stamp}.jsonl"
    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(http, q, t):
        async with sem:
            return await one(client, http, q, t)

    headers = {"x-api-key": api_key}
    async with httpx.AsyncClient(timeout=120, headers=headers) as http:
        with out.open("w", encoding="utf-8") as fh:
            for t in args.types:
                rows = await asyncio.gather(*(guarded(http, q, t) for q in questions))
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                ok = [r for r in rows if not r["error"]]
                correct = sum(1 for r in ok if r["correct"])
                toks = sorted(r["context_tokens"] for r in ok) or [0]
                lat = sorted(r["provider_latency_s"] or 0 for r in ok) or [0]
                cost = sum(r["measured_cost_usd"] or 0 for r in ok)
                synth = [r for r in ok if r["synth_correct"] is not None]
                print(
                    f"{t:15s} correct {correct:2d}/{len(ok):2d}  median ctx "
                    f"{toks[len(toks) // 2]:>6,} tok  median exa time "
                    f"{lat[len(lat) // 2]:5.2f}s  total ${cost:.3f}  "
                    f"errors {len(rows) - len(ok)}"
                    + (
                        f"  synth {sum(1 for r in synth if r['synth_correct'])}/{len(synth)}"
                        if synth
                        else ""
                    ),
                    flush=True,
                )
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--location", default="global")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_PRICES))
    parser.add_argument("--types", nargs="+", default=list(TYPES), choices=TYPES)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    if not args.project:
        sys.exit("Set --project or GOOGLE_CLOUD_PROJECT.")
    asyncio.run(main_async(args))


def _unused() -> None:  # keep ``re`` for future answer parsing without a lint warning
    re.compile("")


if __name__ == "__main__":
    main()
