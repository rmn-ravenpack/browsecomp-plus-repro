#!/usr/bin/env python3
"""Run BrowseComp-Plus queries through a Bedrock Converse tool loop.

Each query is a Python-owned conversation: the model may call search, grep,
and get_document, then submit_ranking to stop retrieval. A second
submit_ranking over the collected evidence is the scored ranking. The loop
stops there; it does not generate a benchmark answer.
Gold answers never go to the model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from agent_loop import check_bedrock, make_bedrock_generate, run_agent_loop, BedrockRetryWait
from corpus_check import coverage_report
from metrics import score_query
from paths import (
    CONFIG_PATH,
    DECRYPTED_JSONL,
    EVAL_DATA_DIR,
    EXPECTED_QUERY_COUNT,
    FINAL_RANKING_PROMPT_PATH,
    MAPPING_JSON,
    PKG_DIR,
    PROMPT_PATH,
    REPO_DIR,
    RUNS_DIR,
    STATE_PATH,
    atomic_write_json,
    docids_from_field,
    load_config,
    load_jsonl,
    sha256_file,
    sha256_text,
)
from report import aggregate_query_rows, print_summary
from session import EvalSession

if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from searcher import BrowsecompSearcher  # noqa: E402

from dotenv import load_dotenv

load_dotenv(REPO_DIR / ".env")


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def format_query_progress(packed: dict[str, Any]) -> str:
    """One-line per-query status after a query finishes."""

    def ndcg(key: str) -> str:
        value = packed.get(key)
        if value is None:
            return "None"
        return f"{float(value):.3f}"

    def seconds(key: str) -> str:
        value = packed.get(key)
        if value is None:
            return "None"
        return f"{float(value):.1f}s"

    def cost(key: str = "cost_usd") -> str:
        value = packed.get(key)
        if value is None:
            return "None"
        return f"{float(value):.4f}"

    ranking_cost = packed.get("ranking_cost_usd")
    cost_part = f"cost={cost()}"
    if ranking_cost is not None:
        cost_part += f" ranking={cost('ranking_cost_usd')}"

    return (
        f"  status={packed.get('status')} searches={packed.get('n_searches')} "
        f"ev_ndcg={ndcg('evidence_ndcg@10')} gold_ndcg={ndcg('gold_ndcg@10')} "
        f"lat={seconds('latency_s')} {cost_part} "
        f"error={packed.get('error') or ''}"
    )


def user_prompt(query: str) -> str:
    return f"USER_QUERY:\n{query.strip()}"


def rendered_agent_prompt(cfg: dict[str, Any]) -> str:
    """Load the retrieval system prompt."""
    del cfg
    rendered = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not rendered:
        raise ValueError("Agent prompt is empty")
    return rendered


def rendered_final_ranking_prompt(cfg: dict[str, Any] | None = None) -> str:
    """Load the system prompt for the final-ranking pass."""
    del cfg
    rendered = FINAL_RANKING_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not rendered:
        raise ValueError("Final ranking prompt is empty")
    return rendered


def resolve_queries(
    *,
    jsonl: Path,
    limit: int | None,
    query_ids: list[str],
) -> list[dict]:
    rows = load_jsonl(jsonl)
    row_ids = [str(row.get("query_id")) for row in rows]
    if len(set(row_ids)) != len(row_ids):
        duplicates = sorted(
            {qid for qid in row_ids if row_ids.count(qid) > 1},
            key=lambda item: (len(item), item),
        )
        raise SystemExit(f"Duplicate query IDs in {jsonl}: {', '.join(duplicates)}")
    by_id = {str(row.get("query_id")): row for row in rows}
    if (
        len(rows) != EXPECTED_QUERY_COUNT
        or len(by_id) != EXPECTED_QUERY_COUNT
    ):
        raise SystemExit(
            f"Decrypted JSONL must contain exactly {EXPECTED_QUERY_COUNT} unique "
            f"official queries; got rows={len(rows)} unique_ids={len(by_id)}"
        )
    if query_ids:
        selected = [by_id[qid] for qid in query_ids if qid in by_id]
        missing = [qid for qid in query_ids if qid not in by_id]
        if missing:
            raise SystemExit(f"Unknown query ids: {', '.join(missing)}")
        return selected
    selected = rows
    if limit is not None:
        selected = selected[:limit]
    return selected


def official_record(
    *,
    query_id: str,
    status: str,
    session: EvalSession,
) -> dict[str, Any]:
    search_counts = session.search_call_counts()
    return {
        "query_id": str(query_id),
        "tool_call_counts": {
            "search": search_counts["text_searches"],
            "grep": search_counts["keyword_searches"],
            "get_document": session.n_fetches(),
            "submit_ranking": (
                2 if session.ranking() else (1 if session.ranking_draft() else 0)
            ),
        },
        "status": "completed" if session.ranking() and status == "completed" else status,
        "retrieved_docids": session.seen_docids(),
        "result": [],
    }


def score_session(
    *,
    row: dict,
    session: EvalSession,
    uploaded: set[str],
    status: str,
    error: str,
    wall_s: float,
    cost_usd: float | None,
    ranking_cost_usd: float | None = None,
    model_usage: dict[str, Any] | None = None,
    bedrock_wait_s: float = 0.0,
) -> dict[str, Any]:
    ranking = session.ranking() or {}
    gold = docids_from_field(row.get("gold_docs"))
    evidence = docids_from_field(row.get("evidence_docs"))
    ranked_documents = ranking.get("documents") or []
    search_counts = session.search_call_counts()
    retrieval_s = float(session.retrieval_wait_s or 0.0)
    if retrieval_s <= 0.0:
        retrieval_s = session.search_elapsed_s() + session.fetch_elapsed_s()
    retrieval_s = round(retrieval_s, 3)
    wait_s = round(max(0.0, float(bedrock_wait_s or 0.0)), 3)
    latency_s = round(max(0.0, float(wall_s) - wait_s), 3)
    metrics = score_query(
        gold_docids=gold,
        evidence_docids=evidence,
        ranked_docids=ranking.get("ranked_docids") or [],
        seen_docids=session.seen_docids(),
        uploaded_docids=uploaded,
        n_ranked_chunks=len(ranked_documents),
    )
    if not status:
        status = "completed" if ranking else "no_submit"
    packed = {
        "query_id": str(row.get("query_id")),
        "status": status,
        "error": error,
        "wall_s": round(wall_s, 3),
        "latency_s": latency_s,
        "bedrock_wait_s": wait_s,
        "retrieval_wait_s": retrieval_s,
        "search_elapsed_s": round(session.search_elapsed_s(), 3),
        "fetch_elapsed_s": round(session.fetch_elapsed_s(), 3),
        "cost_usd": cost_usd,
        "ranking_cost_usd": ranking_cost_usd,
        "model_usage": model_usage or {},
        "n_searches": session.n_searches(),
        "n_text_searches": search_counts["text_searches"],
        "n_keyword_searches": search_counts["keyword_searches"],
        "n_fetches": session.n_fetches(),
        **session.search_chunk_stats(),
        **session.fetch_chunk_stats(),
        "ranking_strategy": ranking.get("ranking_strategy"),
        **metrics,
    }
    atomic_write_json(session.root / "metrics.json", packed)
    atomic_write_json(
        session.root / "official.json",
        official_record(
            query_id=str(row.get("query_id")),
            status=status,
            session=session,
        ),
    )
    return packed


def run_one_query(
    *,
    row: dict,
    run_dir: Path,
    cfg: dict,
    dry_run: bool,
    uploaded: set[str],
    force: bool,
    generate=None,
    searcher: BrowsecompSearcher | None = None,
) -> dict[str, Any]:
    query_id = str(row.get("query_id"))
    qdir = run_dir / "queries" / query_id
    metrics_path = qdir / "metrics.json"
    if metrics_path.is_file() and not force:
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        if cached.get("status") == "completed":
            return cached
    if not force and (qdir / "ranking.json").is_file():
        # The model may have completed before a post-loop artifact write failed.
        # Recover a validated final ranking without paying to run the query again.
        recovered = EvalSession(
            qdir,
            query_id=query_id,
            max_searches=int(cfg.get("max_searches") or 20),
            max_fetches=int(cfg.get("max_fetches") or 10),
        )
        return score_session(
            row=row,
            session=recovered,
            uploaded=uploaded,
            status="completed",
            error="",
            wall_s=0.0,
            cost_usd=None,
            model_usage={"recovered_existing_submission": True},
        )
    if qdir.exists():
        shutil.rmtree(qdir)
    qdir.mkdir(parents=True, exist_ok=True)

    session = EvalSession(
        qdir,
        query_id=query_id,
        max_searches=int(cfg.get("max_searches") or 20),
        max_fetches=int(cfg.get("max_fetches") or 10),
    )
    (qdir / "query.txt").write_text(str(row.get("query") or ""), encoding="utf-8")
    if dry_run:
        return score_session(
            row=row,
            session=session,
            uploaded=uploaded,
            status="dry_run",
            error="",
            wall_s=0.0,
            cost_usd=None,
        )

    if searcher is None:
        searcher = BrowsecompSearcher(
            tag=str(cfg.get("tag") or ""),
            ranking_params=cfg.get("ranking_params"),
        )
    if generate is None:
        retry_wait = BedrockRetryWait()
        generate = make_bedrock_generate(cfg, retry_wait=retry_wait)
    else:
        retry_wait = getattr(generate, "retry_wait", None)
        if not isinstance(retry_wait, BedrockRetryWait):
            retry_wait = BedrockRetryWait()
    started = datetime.now(tz=timezone.utc)
    result = run_agent_loop(
        session=session,
        searcher=searcher,
        query=str(row.get("query") or ""),
        generate=generate,
        cfg=cfg,
        system_prompt=rendered_agent_prompt(cfg),
        final_ranking_system_prompt=rendered_final_ranking_prompt(cfg),
        transcript_path=qdir / "transcript.json",
    )
    wall_s = (datetime.now(tz=timezone.utc) - started).total_seconds()
    return score_session(
        row=row,
        session=session,
        uploaded=uploaded,
        status=result.status,
        error=result.error,
        wall_s=wall_s,
        cost_usd=result.cost_usd,
        ranking_cost_usd=result.ranking_cost_usd,
        bedrock_wait_s=retry_wait.seconds,
        model_usage={
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cache_read_tokens": result.cache_read_tokens,
            "cache_write_tokens": result.cache_write_tokens,
            "ranking_input_tokens": result.ranking_input_tokens,
            "ranking_output_tokens": result.ranking_output_tokens,
            "ranking_cache_read_tokens": result.ranking_cache_read_tokens,
            "ranking_cache_write_tokens": result.ranking_cache_write_tokens,
            "n_turns": result.n_turns,
            "forced_ranking": result.forced_ranking,
        },
    )


def run_query_batch(
    *,
    rows: list[dict],
    run_dir: Path,
    cfg: dict,
    dry_run: bool,
    uploaded: set[str],
    force: bool,
    searcher: BrowsecompSearcher | None,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Run queries in original order; overlap up to `concurrency` in flight."""
    workers = max(1, min(int(concurrency), len(rows) or 1))
    packed: list[dict[str, Any] | None] = [None] * len(rows)

    def run_at(index: int) -> tuple[int, dict[str, Any]]:
        return index, run_one_query(
            row=rows[index],
            run_dir=run_dir,
            cfg=cfg,
            dry_run=dry_run,
            uploaded=uploaded,
            force=force,
            searcher=None if workers > 1 else searcher,
        )

    if workers == 1:
        for index in range(len(rows)):
            print(f"[{index + 1}/{len(rows)}] query_id={rows[index].get('query_id')}", flush=True)
            _, result = run_at(index)
            packed[index] = result
            print(format_query_progress(result), flush=True)
        return [item for item in packed if item is not None]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_at, index) for index in range(len(rows))]
        for future in as_completed(futures):
            index, result = future.result()
            packed[index] = result
            print(
                f"[{index + 1}/{len(rows)}] query_id={rows[index].get('query_id')}",
                flush=True,
            )
            print(format_query_progress(result), flush=True)
    return [item for item in packed if item is not None]


def choose_jsonl(explicit: str) -> Path:
    if explicit:
        return Path(explicit)
    if DECRYPTED_JSONL.is_file():
        return DECRYPTED_JSONL
    raise SystemExit("No decrypted eval JSONL found. Run eval/decrypt_eval.py")


def _run_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Config fields that change retrieval or scoring. Concurrency does not."""
    data = dict(cfg or {})
    data.pop("query_concurrency", None)
    return data


def write_manifest(run_dir: Path, *, cfg: dict, jsonl: Path, query_ids: list[str]) -> dict:
    prompt = rendered_agent_prompt(cfg)
    final_ranking_prompt = rendered_final_ranking_prompt(cfg)
    coverage = None
    if STATE_PATH.is_file():
        coverage = coverage_report(jsonl=jsonl, state_path=STATE_PATH)
    manifest = {
        "created_at": _utc_now(),
        "config": cfg,
        "runtime": "bedrock_converse",
        "model": cfg.get("model") or "",
        "prompt_sha256": sha256_text(prompt),
        "final_ranking_prompt_sha256": sha256_text(final_ranking_prompt),
        "config_sha256": sha256_file(CONFIG_PATH),
        "mapping_sha256": sha256_file(MAPPING_JSON),
        "jsonl_sha256": sha256_file(jsonl),
        "jsonl": str(jsonl),
        "query_ids": query_ids,
        "corpus_coverage": coverage,
    }
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = (
            "config",
            "runtime",
            "model",
            "prompt_sha256",
            "final_ranking_prompt_sha256",
            "config_sha256",
            "mapping_sha256",
            "jsonl_sha256",
            "jsonl",
            "query_ids",
        )
        changed = []
        for key in comparable:
            if key == "config":
                if _run_config(existing.get("config")) != _run_config(
                    manifest.get("config")
                ):
                    changed.append("config")
            elif existing.get(key) != manifest.get(key):
                changed.append(key)
        if changed:
            raise SystemExit(
                "Refusing to mix settings in an existing run. Changed manifest "
                f"fields: {', '.join(changed)}. Start a new run id."
            )
        return existing
    atomic_write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", default="", help="Decrypted JSONL with answers (scorer only)")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N queries")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N queries")
    parser.add_argument("--query-id", action="append", default=[], dest="query_ids")
    parser.add_argument("--run-id", default="", help="Reuse this run directory")
    parser.add_argument("--force", action="store_true", help="Rerun queries that already have metrics")
    parser.add_argument("--dry-run", action="store_true", help="Write session dirs without calling Bedrock")
    parser.add_argument("--check", action="store_true", help="Verify AWS credentials via STS")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Queries to run at once (default: query_concurrency in eval_config.json).",
    )
    args = parser.parse_args()

    if args.check:
        cfg = load_config()
        try:
            ident = check_bedrock(cfg)
        except Exception as exc:
            print(f"bedrock check failed: {exc}", file=sys.stderr)
            return 1
        print(f"runtime=bedrock_converse")
        print(f"model={ident['model']}")
        print(f"region={ident['region']}")
        print("credentials=ok")
        return 0

    cfg = load_config()
    if args.concurrency is not None:
        if args.concurrency < 1:
            raise SystemExit("--concurrency must be at least 1")
        cfg["query_concurrency"] = args.concurrency
    jsonl = choose_jsonl(args.jsonl)
    rows = resolve_queries(
        jsonl=jsonl,
        limit=args.limit,
        offset=args.offset,
        query_ids=args.query_ids,
    )
    if not rows:
        print("No queries selected.", file=sys.stderr)
        return 1

    run_id = args.run_id or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    query_ids = [str(row.get("query_id")) for row in rows]
    write_manifest(run_dir, cfg=cfg, jsonl=jsonl, query_ids=query_ids)

    uploaded: set[str] = set()
    if STATE_PATH.is_file():
        conn = sqlite3.connect(f"file:{STATE_PATH}?mode=ro", uri=True)
        uploaded = {
            str(docid)
            for (docid,) in conn.execute(
                "SELECT docid FROM documents WHERE status='uploaded'"
            )
        }
        conn.close()

    searcher = None
    if not args.dry_run:
        searcher = BrowsecompSearcher(
            tag=str(cfg.get("tag") or ""),
            ranking_params=cfg.get("ranking_params"),
        )

    concurrency = int(cfg.get("query_concurrency") or 1)
    print(
        f"run={run_dir} queries={len(rows)} runtime=bedrock_converse "
        f"model={cfg.get('model') or ''} concurrency={concurrency}"
    )
    packed_rows = run_query_batch(
        rows=rows,
        run_dir=run_dir,
        cfg=cfg,
        dry_run=args.dry_run,
        uploaded=uploaded,
        force=args.force,
        searcher=searcher,
        concurrency=concurrency,
    )

    summary = {
        "run_id": run_id,
        "jsonl": str(jsonl),
        "query_concurrency": int(cfg.get("query_concurrency") or 1),
        "aggregate": aggregate_query_rows(packed_rows),
        "queries": packed_rows,
        "updated_at": _utc_now(),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    print_summary(summary)
    print(f"Wrote {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
