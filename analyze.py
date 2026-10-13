"""Summarize fetch-tax runs: tokens, accuracy, cost per path, plus a chart.

Usage:
    uv run python analyze.py results/runs_X.jsonl --chart results/fetch_tax_chart.png
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = [
    "raw_html",
    "url_context",
    "exa_full_text",
    "exa_highlights",
    "exa_dynamic",
    "google_search",
]
LABELS = {
    "raw_html": "Raw HTML\n(what a naive fetch\nreturns)",
    "url_context": "Gemini\nURL context",
    "exa_full_text": "Exa full text\n(web_fetch_exa)",
    "exa_highlights": "Exa highlights\n(web_search_exa)",
    "exa_dynamic": "Exa dynamic\nhighlights (API)",
    "google_search": "Google Search\ngrounding",
}
BG, PANEL, TEXT, GRID = "#0D1B2A", "#13263A", "#E6EDF3", "#2A3B4F"
AMBER, CORAL, CYAN, MINT = "#FFB800", "#FF6B6B", "#00E5FF", "#3DDC97"


def grade(answer: str, pattern: str) -> bool:
    if re.search(r"\bnot found\b", answer, re.IGNORECASE):
        return False
    return re.search(pattern, answer, re.IGNORECASE) is not None


def load(paths: list[Path], questions_file: Path) -> list[dict]:
    """Merge run files; later files override earlier rows for the same question/path.

    Every graded row is re-graded from its stored answer with the current regex, so
    fixing a regex in questions.json never requires re-running the experiment.
    """
    patterns = {
        q["id"]: q["answer_regex"]
        for q in json.loads(questions_file.read_text(encoding="utf-8"))
    }
    merged: dict[tuple[str, str], dict] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("correct") is not None and row["question_id"] in patterns:
                row["correct"] = grade(
                    row.get("answer", ""), patterns[row["question_id"]]
                )
                # No retrieved material means any correct answer came from the
                # model's memory, which is not the path's doing.
                if row["path"] != "google_search" and row.get("pages", 0) == 0:
                    row["correct"] = False
                    row["retrieval_failed"] = True
            merged[(row["question_id"], row["path"])] = row
    return list(merged.values())


def summarize(rows: list[dict]) -> list[dict]:
    by_path: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_path[r["path"]].append(r)
    out = []
    for path in ORDER:
        rs = [r for r in by_path.get(path, []) if not r.get("error")]
        errors = len(by_path.get(path, [])) - len(rs)
        if not rs:
            continue
        graded = [r for r in rs if r["correct"] is not None]
        correct = sum(1 for r in graded if r["correct"])
        ctx = [r["context_tokens"] for r in rs]
        total_ctx = sum(ctx)
        out.append(
            {
                "path": path,
                "n": len(rs),
                "errors": errors,
                "median_context_tokens": int(statistics.median(ctx)),
                "mean_context_tokens": int(statistics.mean(ctx)),
                "max_context_tokens": max(ctx),
                "total_context_tokens": total_ctx,
                "correct": correct,
                "graded": len(graded),
                "accuracy": (correct / len(graded)) if graded else None,
                "tokens_per_correct": (total_ctx / correct) if correct else None,
                "total_cost_usd": sum(r["cost_usd"] for r in rs),
                "cost_per_correct_usd": (
                    sum(r["cost_usd"] for r in rs) / correct if correct else None
                ),
                "median_latency_s": round(
                    statistics.median(r["latency_s"] for r in rs), 1
                ),
                "search_queries": sum(r.get("search_queries", 0) for r in rs),
            }
        )
    return out


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def num(x: float | None, fmt: str = ",.0f") -> str:
    return "-" if x is None else format(x, fmt)


def as_markdown(summary: list[dict]) -> str:
    head = (
        "| Path | n | Median context tokens | Max | Accuracy | Tokens per correct answer "
        "| Cost per correct answer | Median latency |\n|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for s in summary:
        acc = pct(s["accuracy"]) if s["graded"] else "not answered"
        lines.append(
            f"| {s['path']} | {s['n']} | {s['median_context_tokens']:,} | "
            f"{s['max_context_tokens']:,} | {acc} | {num(s['tokens_per_correct'])} | "
            f"{num(s['cost_per_correct_usd'], '.4f') if s['cost_per_correct_usd'] is not None else '-'} | "
            f"{s['median_latency_s']}s |"
        )
    return head + "\n".join(lines) + "\n"


def per_question_table(rows: list[dict]) -> str:
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_q[r["question_id"]][r["path"]] = r
    head = (
        "| Question | " + " | ".join(ORDER) + " |\n|---|" + "---|" * len(ORDER) + "\n"
    )
    lines = []
    for qid in sorted(by_q):
        cells = []
        for path in ORDER:
            r = by_q[qid].get(path)
            if r is None or r.get("error"):
                cells.append("err")
                continue
            mark = {True: "✓", False: "✗", None: ""}[r["correct"]]
            cells.append(f"{r['context_tokens']:,} {mark}".strip())
        lines.append(f"| {qid} | " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def draw_chart(summary: list[dict], rows: list[dict], path: Path) -> None:
    paths = [s["path"] for s in summary]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8), facecolor=BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # Left: distribution of context tokens per path (log scale), one dot per question
    by_path: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if not r.get("error"):
            by_path[r["path"]].append(max(r["context_tokens"], 1))
    xs = list(range(len(paths)))
    for i, p in enumerate(paths):
        vals = by_path[p]
        ax1.scatter([i] * len(vals), vals, color=AMBER, alpha=0.55, s=28, zorder=3)
        med = statistics.median(vals)
        ax1.hlines(med, i - 0.3, i + 0.3, color=CYAN, linewidth=2.5, zorder=4)
        label = (
            "0 tokens\n(billed per query)"
            if p == "google_search"
            else f"median {med:,.0f}"
        )
        ax1.annotate(
            label,
            (i, med),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color=CYAN,
            fontsize=9,
        )
    ax1.set_yscale("log")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([LABELS[p] for p in paths], color=TEXT, fontsize=9)
    ax1.set_ylabel("Tokens entering the context per question (log scale)", color=TEXT)
    ax1.set_title("What each retrieval path puts into context", color=TEXT, fontsize=12)

    # Right: tokens per correct answer, with accuracy labels
    graded = [s for s in summary if s["graded"]]
    xs2 = list(range(len(graded)))
    vals2 = [s["tokens_per_correct"] or 0 for s in graded]
    bars = ax2.bar(
        xs2, vals2, color=[CORAL if v == max(vals2) else MINT for v in vals2]
    )
    for bar, s in zip(bars, graded, strict=True):
        ax2.annotate(
            f"{s['tokens_per_correct']:,.0f} tok\n{pct(s['accuracy'])} correct",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=9,
        )
    ax2.set_xticks(xs2)
    ax2.set_xticklabels([LABELS[s["path"]] for s in graded], color=TEXT, fontsize=9)
    ax2.set_ylabel("Context tokens per correct answer", color=TEXT)
    ax2.set_ylim(0, max(vals2) * 1.25 if vals2 else 1)
    ax2.set_title("Tokens you pay per correct answer", color=TEXT, fontsize=12)

    n = max(s["n"] for s in summary)
    words = {4: "four", 5: "five", 6: "six", 7: "seven"}
    k = words.get(len(summary), str(len(summary)))
    fig.suptitle(
        f"The Fetch Tax: {n} factual questions, {k} retrieval paths, a Flash-class model answering",
        color=TEXT,
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=BG)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--chart", type=Path)
    parser.add_argument("--summary", type=Path, default=Path("results/summary.md"))
    parser.add_argument("--questions", type=Path, default=Path("questions.json"))
    args = parser.parse_args()

    rows = load(args.files, args.questions)
    summary = summarize(rows)
    table = as_markdown(summary)
    detail = per_question_table(rows)
    print(table)
    print(detail)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        "## Per path\n\n"
        + table
        + "\n## Per question (context tokens, ✓ correct)\n\n"
        + detail,
        encoding="utf-8",
    )
    args.summary.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if args.chart:
        draw_chart(summary, rows, args.chart)


if __name__ == "__main__":
    main()
