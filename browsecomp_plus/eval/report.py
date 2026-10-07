#!/usr/bin/env python3
"""Summarize a BrowseComp-Plus agent eval run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from metrics import mean
from paths import EVAL_DATA_DIR


def load_summary(run_dir: Path) -> dict:
    path = run_dir / "summary.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Missing {path}")


def percentile(values, q: float) -> float | None:
    """Linearly interpolated percentile (the standard p50/p90 convention)."""
    nums = sorted(float(value) for value in values if value is not None)
    if not nums:
        return None
    position = (len(nums) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(nums) - 1)
    weight = position - lower
    return nums[lower] * (1.0 - weight) + nums[upper] * weight


def print_summary(summary: dict) -> None:
    agg = summary.get("aggregate") or {}
    print(
        f"run={summary.get('run_id')}  "
        f"n={agg.get('n_queries')}  completed={agg.get('n_completed')}"
    )
    print(
        "Evidence ranking  "
        f"evidence nDCG@10={agg.get('evidence_ndcg@10')}  "
        f"gold nDCG@10={agg.get('gold_ndcg@10')}  "
        f"evidence R@10={agg.get('evidence_recall@10')}"
    )
    print(
        "Ranking diagnosis  "
        f"trace-oracle evidence nDCG@10={agg.get('evidence_trace_oracle_ndcg@10')}  "
        f"gold={agg.get('gold_trace_oracle_ndcg@10')}  "
        f"retained seen evidence@10={agg.get('evidence_ranking_retention@10')}  "
        f"gold={agg.get('gold_ranking_retention@10')}"
    )
    ranking_shape = (
        f"Ranking shape  ranked={agg.get('n_ranked')}  "
        f"avg chunks={agg.get('avg_ranked_chunks')}  "
        f"avg docs={agg.get('avg_ranked_docs')}"
    )
    print(ranking_shape)
    print(
        f"trace evidence recall={agg.get('evidence_trace_recall')}  "
        f"avg searches={agg.get('avg_searches')}  "
        f"(text={agg.get('avg_text_searches')}, "
        f"keyword={agg.get('avg_keyword_searches')})  "
        f"avg fetches={agg.get('avg_fetches')}"
    )
    print(
        "Bedrock cost (USD)  ranking="
        f"{agg.get('total_ranking_cost_usd')}  "
        f"total={agg.get('total_cost_usd')}  "
        f"avg_ranking={agg.get('avg_ranking_cost_usd')}  "
        f"avg_cost={agg.get('avg_cost_usd')}"
    )
    print(
        "latency excl. Bedrock retry wait  "
        f"avg={agg.get('avg_latency_s')}s  p50={agg.get('p50_latency_s')}s  "
        f"p90={agg.get('p90_latency_s')}s  "
        f"bedrock wait avg={agg.get('avg_bedrock_wait_s')}s  "
        f"wall avg={agg.get('avg_wall_s')}s"
    )
    print(
        f"search paging: raw chunks={agg.get('search_chunks_raw')}  "
        f"deduped={agg.get('search_chunks_deduped')}  "
        f"shown={agg.get('search_chunks_visible')}"
    )
    print(
        f"fetch payoff: new chunks={agg.get('fetch_chunks_new')}  "
        f"re-read={agg.get('fetch_chunks_reread')}  "
        f"redundant fetches={agg.get('redundant_fetches')}"
    )
    for row in summary.get("queries") or []:
        detail = (
            f"  {row.get('query_id')}: status={row.get('status')} "
            f"searches={row.get('n_searches')} "
            f"(text={row.get('n_text_searches')}, "
            f"keyword={row.get('n_keyword_searches')}) "
            f"fetches={row.get('n_fetches')} "
            f"ev_ndcg={row.get('evidence_ndcg@10')} "
            f"ev_oracle={row.get('evidence_trace_oracle_ndcg@10')} "
            f"ev_retained={row.get('evidence_ranking_retention@10')} "
            f"lat={row.get('latency_s')}s "
            f"bedrock={row.get('bedrock_wait_s')}s "
            f"cost={row.get('cost_usd')} "
            f"ranking={row.get('ranking_cost_usd')} "
            f"error={row.get('error') or ''}"
        )
        print(detail)


def aggregate_query_rows(rows: list[dict]) -> dict:
    completed = [row for row in rows if row.get("status") == "completed"]
    ranked = [row for row in rows if row.get("n_ranked")]
    return {
        "n_queries": len(rows),
        "n_completed": len(completed),
        "n_ranked": len(ranked),
        "n_errors": sum(1 for row in rows if row.get("status") == "error"),
        "n_no_submit": sum(1 for row in rows if row.get("status") == "no_submit"),
        "gold_ndcg@10": mean(row.get("gold_ndcg@10") for row in ranked),
        "evidence_ndcg@10": mean(row.get("evidence_ndcg@10") for row in ranked),
        "gold_ndcg@10_all_queries": mean(row.get("gold_ndcg@10") for row in rows),
        "evidence_ndcg@10_all_queries": mean(
            row.get("evidence_ndcg@10") for row in rows
        ),
        "gold_recall@10": mean(row.get("gold_recall@10") for row in ranked),
        "evidence_recall@10": mean(row.get("evidence_recall@10") for row in ranked),
        "avg_ranked_chunks": mean(row.get("n_ranked_chunks") for row in ranked),
        "avg_ranked_docs": mean(row.get("n_ranked") for row in ranked),
        "gold_trace_recall": mean(row.get("gold_trace_recall") for row in rows),
        "evidence_trace_recall": mean(row.get("evidence_trace_recall") for row in rows),
        "gold_trace_oracle_ndcg@10": mean(
            row.get("gold_trace_oracle_ndcg@10") for row in rows
        ),
        "evidence_trace_oracle_ndcg@10": mean(
            row.get("evidence_trace_oracle_ndcg@10") for row in rows
        ),
        "gold_ranking_retention@10": mean(
            row.get("gold_ranking_retention@10") for row in ranked
        ),
        "evidence_ranking_retention@10": mean(
            row.get("evidence_ranking_retention@10") for row in ranked
        ),
        "avg_searches": mean(row.get("n_searches") for row in rows),
        "avg_text_searches": mean(row.get("n_text_searches") for row in rows),
        "avg_keyword_searches": mean(
            row.get("n_keyword_searches") for row in rows
        ),
        "avg_fetches": mean(row.get("n_fetches") for row in rows),
        "search_chunks_raw": sum(int(row.get("search_chunks_raw") or 0) for row in rows),
        "search_chunks_deduped": sum(
            int(row.get("search_chunks_deduped") or 0) for row in rows
        ),
        "search_chunks_visible": sum(
            int(row.get("search_chunks_visible") or 0) for row in rows
        ),
        "fetch_chunks_new": sum(int(row.get("fetch_chunks_new") or 0) for row in rows),
        "fetch_chunks_reread": sum(int(row.get("fetch_chunks_reread") or 0) for row in rows),
        "redundant_fetches": sum(int(row.get("redundant_fetches") or 0) for row in rows),
        "total_cost_usd": sum(float(row.get("cost_usd") or 0.0) for row in rows),
        "avg_cost_usd": mean(row.get("cost_usd") for row in rows),
        "p50_cost_usd": percentile((row.get("cost_usd") for row in rows), 0.50),
        "p90_cost_usd": percentile((row.get("cost_usd") for row in rows), 0.90),
        "p50_completed_cost_usd": percentile(
            (row.get("cost_usd") for row in completed), 0.50
        ),
        "total_ranking_cost_usd": sum(
            float(row.get("ranking_cost_usd") or 0.0) for row in rows
        ),
        "avg_ranking_cost_usd": mean(row.get("ranking_cost_usd") for row in rows),
        "total_wall_s": sum(float(row.get("wall_s") or 0.0) for row in rows),
        "avg_wall_s": mean(row.get("wall_s") for row in rows),
        "avg_latency_s": mean(row.get("latency_s") for row in rows),
        "p50_latency_s": percentile((row.get("latency_s") for row in rows), 0.50),
        "p90_latency_s": percentile((row.get("latency_s") for row in rows), 0.90),
        "p50_completed_latency_s": percentile(
            (row.get("latency_s") for row in completed), 0.50
        ),
        "avg_bedrock_wait_s": mean(row.get("bedrock_wait_s") for row in rows),
        "total_latency_s": sum(float(row.get("latency_s") or 0.0) for row in rows),
        "total_bedrock_wait_s": sum(
            float(row.get("bedrock_wait_s") or 0.0) for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default="")
    args = parser.parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        runs = sorted((EVAL_DATA_DIR / "runs").glob("*"), reverse=True)
        run_dir = next((path for path in runs if path.is_dir()), None)
        if run_dir is None:
            print("No eval runs found.", file=sys.stderr)
            return 1
    print_summary(load_summary(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
