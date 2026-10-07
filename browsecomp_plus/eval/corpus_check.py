#!/usr/bin/env python3
"""Check local upload coverage and optional remote Bigdata indexing readiness."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from paths import (
    DECRYPTED_JSONL,
    EVAL_DATA_DIR,
    EXPECTED_CORPUS_COUNT,
    MAPPING_JSON,
    STATE_PATH,
    atomic_write_json,
    docids_from_field,
    load_jsonl,
    sha256_file,
)

_PKG_DIR = _EVAL_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from bigdata_client import (  # noqa: E402
    API_BASE_URL_DEFAULT,
    BigdataContentClient,
    RateLimiter,
)

load_dotenv(_PKG_DIR.parent / ".env")


def _status_sets(state_path: Path) -> dict[str, set[str]]:
    conn = sqlite3.connect(f"file:{state_path}?mode=ro", uri=True)
    rows = conn.execute("SELECT docid, status FROM documents").fetchall()
    conn.close()
    out: dict[str, set[str]] = {}
    for docid, status in rows:
        out.setdefault(str(status), set()).add(str(docid))
    return out


def _uploaded_rows(state_path: Path) -> list[tuple[str, str]]:
    conn = sqlite3.connect(f"file:{state_path}?mode=ro", uri=True)
    rows = conn.execute(
        """
        SELECT docid, content_id
        FROM documents
        WHERE status = 'uploaded' AND TRIM(content_id) != ''
        ORDER BY length(docid), docid
        """
    ).fetchall()
    conn.close()
    return [(str(docid), str(content_id).upper()) for docid, content_id in rows]


def _latest_remote_rows(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not path.is_file():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[str(row["docid"])] = row
    return latest


def check_remote(
    *,
    state_path: Path,
    target_docids: set[str],
    cache_path: Path,
    api_key: str,
    base_url: str,
    concurrency: int,
    requests_per_minute: int,
    expected_failed: int,
) -> dict:
    uploaded = _uploaded_rows(state_path)
    selected = [
        (docid, content_id)
        for docid, content_id in uploaded
        if not target_docids or docid in target_docids
    ]
    latest = _latest_remote_rows(cache_path)
    pending = [
        pair
        for pair in selected
        if not (
            pair[0] in latest
            and latest[pair[0]].get("content_id") == pair[1]
            and latest[pair[0]].get("status") == "completed"
        )
    ]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    client = BigdataContentClient(
        api_key=api_key,
        rate_limiter=RateLimiter(requests_per_minute),
        base_url=base_url,
    )

    def fetch(pair: tuple[str, str]) -> dict:
        docid, content_id = pair
        data, http_status = client.get_document(content_id)
        return {
            "docid": docid,
            "content_id": content_id,
            "http_status": http_status,
            "status": str((data or {}).get("status") or ""),
            "error_code": str((data or {}).get("error_code") or ""),
        }

    if pending:
        with cache_path.open("a", encoding="utf-8") as output:
            with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
                futures = [executor.submit(fetch, pair) for pair in pending]
                for index, future in enumerate(as_completed(futures), start=1):
                    row = future.result()
                    latest[row["docid"]] = row
                    with write_lock:
                        output.write(json.dumps(row, sort_keys=True) + "\n")
                        output.flush()
                    if index == 1 or index % 500 == 0:
                        print(f"remote checked {index}/{len(pending)}", flush=True)

    selected_rows = [
        latest.get(
            docid,
            {
                "docid": docid,
                "content_id": content_id,
                "http_status": 0,
                "status": "not_checked",
                "error_code": "",
            },
        )
        for docid, content_id in selected
    ]
    counts: dict[str, int] = {}
    for row in selected_rows:
        status = str(row.get("status") or f"http_{row.get('http_status', 0)}")
        counts[status] = counts.get(status, 0) + 1
    nonterminal = [
        str(row["docid"])
        for row in selected_rows
        if row.get("status") not in {"completed", "failed"}
    ]
    failed = [
        str(row["docid"]) for row in selected_rows if row.get("status") == "failed"
    ]
    return {
        "selected": len(selected),
        "counts": counts,
        "expected_failed": expected_failed,
        "ready": len(nonterminal) == 0 and len(failed) == expected_failed,
        "failed_docids": sorted(failed, key=lambda item: (len(item), item)),
        "nonterminal_docids": sorted(
            nonterminal, key=lambda item: (len(item), item)
        ),
        "cache_path": str(cache_path),
    }


def coverage_report(*, jsonl: Path, state_path: Path) -> dict:
    rows = load_jsonl(jsonl)
    statuses = _status_sets(state_path)
    uploaded = statuses.get("uploaded", set())
    skipped = statuses.get("skipped", set())
    failed = statuses.get("failed", set())
    gold: set[str] = set()
    evidence: set[str] = set()
    queries_missing_gold = 0
    queries_missing_evidence = 0
    gold_in_skipped: set[str] = set()
    evidence_in_skipped: set[str] = set()
    for row in rows:
        g = set(docids_from_field(row.get("gold_docs")))
        e = set(docids_from_field(row.get("evidence_docs")))
        gold |= g
        evidence |= e
        if g - uploaded:
            queries_missing_gold += 1
        if e - uploaded:
            queries_missing_evidence += 1
        gold_in_skipped |= g & skipped
        evidence_in_skipped |= e & skipped
    return {
        "n_queries": len(rows),
        "uploaded": len(uploaded),
        "skipped": len(skipped),
        "failed": len(failed),
        "expected_corpus_rows": EXPECTED_CORPUS_COUNT,
        "mapping_sha256": sha256_file(MAPPING_JSON),
        "gold_docs": len(gold),
        "evidence_docs": len(evidence),
        "gold_uploaded": len(gold & uploaded),
        "evidence_uploaded": len(evidence & uploaded),
        "gold_missing": sorted(gold - uploaded),
        "evidence_missing": sorted(evidence - uploaded),
        "gold_in_skipped": sorted(gold_in_skipped),
        "evidence_in_skipped": sorted(evidence_in_skipped),
        "queries_with_missing_gold": queries_missing_gold,
        "queries_with_missing_evidence": queries_missing_evidence,
        "skipped_docids": sorted(skipped, key=lambda item: (len(item), item)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", default="", help="Decrypted eval JSONL")
    parser.add_argument("--out", default="", help="Write JSON report here")
    parser.add_argument(
        "--remote",
        choices=("none", "qrels", "all"),
        default="none",
        help="Query Bigdata readiness for labeled documents or the complete upload.",
    )
    parser.add_argument(
        "--remote-cache",
        default=str(EVAL_DATA_DIR / "corpus_remote_status.jsonl"),
        help="Private resumable JSONL status cache.",
    )
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--requests-per-minute", type=int, default=400)
    parser.add_argument(
        "--expected-failed",
        type=int,
        default=0,
        help="Require exactly this many terminal remote failures.",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("BIGDATA_API_BASE_URL", API_BASE_URL_DEFAULT),
    )
    args = parser.parse_args()
    jsonl = Path(args.jsonl) if args.jsonl else DECRYPTED_JSONL
    if not jsonl.is_file():
        print(f"Missing {jsonl}. Run eval/decrypt_eval.py first.", file=sys.stderr)
        return 1
    if not STATE_PATH.is_file():
        print(f"Missing {STATE_PATH}", file=sys.stderr)
        return 1
    report = coverage_report(jsonl=jsonl, state_path=STATE_PATH)
    if args.remote != "none":
        expected_failed = args.expected_failed
        api_key = os.getenv("BIGDATA_API_KEY", "").strip()
        if not api_key:
            print("BIGDATA_API_KEY is required for --remote", file=sys.stderr)
            return 1
        if args.remote == "qrels":
            target_docids = set(report["gold_missing"]) | set(report["evidence_missing"])
            rows = load_jsonl(jsonl)
            for row in rows:
                target_docids.update(docids_from_field(row.get("gold_docs")))
                target_docids.update(docids_from_field(row.get("evidence_docs")))
        else:
            target_docids = set()
        report["remote"] = check_remote(
            state_path=STATE_PATH,
            target_docids=target_docids,
            cache_path=Path(args.remote_cache),
            api_key=api_key,
            base_url=args.api_base_url,
            concurrency=args.concurrency,
            requests_per_minute=args.requests_per_minute,
            expected_failed=expected_failed,
        )
    out = Path(args.out) if args.out else EVAL_DATA_DIR / "corpus_coverage.json"
    atomic_write_json(out, report)
    print(
        f"queries={report['n_queries']} uploaded={report['uploaded']} "
        f"skipped={report['skipped']} failed={report['failed']}"
    )
    print(
        f"gold {report['gold_uploaded']}/{report['gold_docs']} uploaded  "
        f"evidence {report['evidence_uploaded']}/{report['evidence_docs']} uploaded"
    )
    print(
        f"queries missing gold docs={report['queries_with_missing_gold']}  "
        f"missing evidence docs={report['queries_with_missing_evidence']}"
    )
    if report["gold_in_skipped"] or report["evidence_in_skipped"]:
        print(
            "WARNING: skipped documents appear in qrels: "
            f"gold={report['gold_in_skipped']} evidence={report['evidence_in_skipped']}"
        )
    if "remote" in report:
        remote = report["remote"]
        print(
            f"remote selected={remote['selected']} ready={remote['ready']} "
            f"statuses={json.dumps(remote['counts'], sort_keys=True)}"
        )
    print(f"Wrote {out}")
    if report["gold_missing"] or report["evidence_missing"]:
        return 2
    if "remote" in report and not report["remote"]["ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
