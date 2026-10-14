"""Same benchmark, two answering models: output tokens and searches, side by side.

Reads two sets of run files (for example the Flash-class run and the Argon run),
re-grades them the same way analyze.py does, and draws one chart comparing what
each model spent on its own side of the bill.

Usage:
    uv run python compare.py --base FLASH_RUNS.jsonl --new ARGON_RUNS.jsonl
        [--chart results/model_compare.png]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import AMBER, BG, CYAN, GRID, LABELS, ORDER, PANEL, TEXT, load

SLATE = "#5B7083"
GROUNDING = ("google_search", "gemini_exa_grounding")


def totals(rows: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(
        lambda: {"output": 0, "queries": 0, "correct": 0, "n": 0}
    )
    for r in rows:
        if r.get("error") or r["path"] == "raw_html":
            continue
        t = out[r["path"]]
        t["output"] += r.get("output_tokens", 0) or 0  # thinking included
        t["queries"] += r.get("search_queries", 0) or 0
        t["correct"] += int(bool(r.get("correct")))
        t["n"] += 1
    return out


def style(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def draw(base: dict, new: dict, labels: tuple[str, str], path: Path) -> None:
    shared = [p for p in ORDER if p in base and p in new]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(17, 6.5), facecolor=BG, gridspec_kw={"width_ratios": [3, 1]}
    )
    for ax in (ax1, ax2):
        style(ax)
    width = 0.38

    # Left: output tokens (thinking included) across all answers, per path
    xs = range(len(shared))
    b_vals = [base[p]["output"] for p in shared]
    n_vals = [new[p]["output"] for p in shared]
    ax1.bar([x - width / 2 for x in xs], b_vals, width, color=SLATE, label=labels[0])
    bars = ax1.bar(
        [x + width / 2 for x in xs], n_vals, width, color=AMBER, label=labels[1]
    )
    for bar, b, n, p in zip(bars, b_vals, n_vals, shared, strict=True):
        ratio = f"{b / n:.1f}x fewer" if n else "none"
        ax1.annotate(
            f"{ratio}\n{new[p]['correct']}/{new[p]['n']} correct",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=CYAN,
            fontsize=9,
        )
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([LABELS[p] for p in shared], color=TEXT, fontsize=9)
    ax1.set_ylabel("Output tokens across 20 answers (thinking included)", color=TEXT)
    ax1.set_title("What each model wrote to answer", color=TEXT, fontsize=12)
    ax1.set_ylim(0, max(b_vals) * 1.2)
    legend = ax1.legend(facecolor=PANEL, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(TEXT)

    # Right: searches the model chose to run on the grounding paths
    g = [p for p in GROUNDING if p in shared]
    gx = range(len(g))
    b_q = [base[p]["queries"] for p in g]
    n_q = [new[p]["queries"] for p in g]
    for offset, vals, color in ((-width / 2, b_q, SLATE), (width / 2, n_q, AMBER)):
        bars = ax2.bar([x + offset for x in gx], vals, width, color=color)
        for bar, v in zip(bars, vals, strict=True):
            ax2.annotate(
                str(v),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=TEXT,
                fontsize=10,
            )
    ax2.set_xticks(list(gx))
    ax2.set_xticklabels([LABELS[p] for p in g], color=TEXT, fontsize=9)
    ax2.set_ylabel("Searches across 20 questions", color=TEXT)
    ax2.set_title("Searches the model chose to run", color=TEXT, fontsize=12)
    ax2.set_ylim(0, max(b_q + n_q) * 1.25 if g else 1)

    fig.suptitle(
        f"Same 20 questions, same retrieval paths: {labels[0]} vs {labels[1]}",
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
    parser.add_argument("--base", nargs="+", type=Path, required=True)
    parser.add_argument("--new", nargs="+", type=Path, required=True)
    parser.add_argument("--base-label", default="a Flash-class model")
    parser.add_argument("--new-label", default="Gemini 4 Argon")
    parser.add_argument("--questions", type=Path, default=Path("questions.json"))
    parser.add_argument("--chart", type=Path, default=Path("results/model_compare.png"))
    args = parser.parse_args()

    base = totals(load(args.base, args.questions))
    new = totals(load(args.new, args.questions))
    print(f"{'path':22s} {'output base/new':>18s} {'queries base/new':>18s}")
    for p in ORDER:
        if p in base and p in new:
            print(
                f"{p:22s} {base[p]['output']:>8,}/{new[p]['output']:<8,} "
                f"{base[p]['queries']:>8}/{new[p]['queries']:<8}"
            )
    draw(base, new, (args.base_label, args.new_label), args.chart)


if __name__ == "__main__":
    main()
