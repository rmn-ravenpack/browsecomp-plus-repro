#!/usr/bin/env python3
"""Download Tevatron/browsecomp-plus-corpus and upload it to Bigdata.

Documents are tagged `browsecomp-plus`. YAML titles become Bigdata filenames.
Translation enrichment is requested only when language detection says the body
is not English. Some very long documents may fail ingestion.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

load_dotenv(_REPO_ROOT / ".env")

from bigdata_client import (  # noqa: E402
    API_BASE_URL_DEFAULT,
    BigdataContentClient,
    RateLimiter,
)
from detect_language import detect_language  # noqa: E402
from state import UploadState  # noqa: E402

HF_DATASET = "Tevatron/browsecomp-plus-corpus"
HF_DATASET_REVISION = "b27b02bc3e45511b8b82a13e6f90ce761df726f6"
DEFAULT_TAG = "browsecomp-plus"
DATA_DIR = _HERE / "data"
HF_CACHE_DIR = DATA_DIR / "hf_cache"
STATE_PATH = DATA_DIR / "upload_state.sqlite"
LOG_DIR = DATA_DIR / "logs"

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)
FRONTMATTER_LINE_RE = re.compile(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$")
DATE_VALUE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}(?::\d{2})?))?"
)
UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
HTML_DOC_RE = re.compile(r"(?is)\A\s*(?:<!--.*?-->\s*)*(?:<!doctype\s+html|<html\b)")
SPAM_LINE_RE = re.compile(
    r"click now to see|click here now|weak erection|small penis|"
    r"casino ohne oasis|casino non aams|online casino australia",
    re.I,
)
TITLE_KEYS = {"title"}
DATE_KEYS = {"date", "published", "published_ts", "pub_date"}
AUTHOR_KEYS = {"author", "authors"}
TRANSIENT_HTTP = {429, 500, 502, 503, 504}
MAPPING_TSV = DATA_DIR / "docid_to_content_id.tsv"
MAPPING_JSONL = DATA_DIR / "docid_to_content_id.jsonl"
MAPPING_JSON = DATA_DIR / "docid_content_id_map.json"


class FilenameRegistry:
    """Keep display names unique within a run (same titles are common)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._used: set[str] = set()

    def seed(self, name: str) -> None:
        cleaned = (name or "").strip()
        if not cleaned:
            return
        with self._lock:
            self._used.add(cleaned)

    def allocate(self, desired: str, docid: str) -> str:
        base = (desired or "").strip() or f"browsecomp-plus-{docid}"
        with self._lock:
            if base not in self._used:
                self._used.add(base)
                return base
            alt = f"{base} ({docid})"
            n = 2
            while alt in self._used:
                alt = f"{base} ({docid}-{n})"
                n += 1
            self._used.add(alt)
            return alt


def configure_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path


def load_corpus(cache_dir: Path):
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    logging.info(
        "Loading %s revision=%s (cache=%s)",
        HF_DATASET,
        HF_DATASET_REVISION,
        cache_dir,
    )
    return load_dataset(
        HF_DATASET,
        revision=HF_DATASET_REVISION,
        split="train",
        cache_dir=str(cache_dir),
    )


def _clean_meta_value(value: str) -> str:
    return value.strip().strip("'\"")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text:
        return {}, ""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        km = FRONTMATTER_LINE_RE.match(line)
        if not km:
            continue
        key = km.group(1).strip().lower()
        val = _clean_meta_value(km.group(2))
        if key and val and key not in meta:
            meta[key] = val
    body = text[match.end() :].lstrip("\n")
    return meta, body


def normalize_published_ts(value: str | None) -> str | None:
    if not value:
        return None
    match = DATE_VALUE_RE.match(value.strip())
    if not match:
        return None
    day = match.group(1)
    clock = match.group(2) or "12:00:00"
    if len(clock) == 5:
        clock = f"{clock}:00"
    return f"{day}T{clock}Z"


def extract_published_ts(text: str) -> str | None:
    meta, _ = parse_frontmatter(text or "")
    for key in DATE_KEYS:
        ts = normalize_published_ts(meta.get(key))
        if ts:
            return ts
    return None


def extract_title(text: str) -> str | None:
    meta, _ = parse_frontmatter(text or "")
    return meta.get("title") or None


def file_name_for(docid: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(docid).strip()) or "unknown"
    return f"browsecomp-plus-{safe}.txt"


def file_name_from_title(title: str | None, docid: str) -> str:
    """Use the corpus title as Bigdata display name / search title (no extension)."""
    cleaned = UNSAFE_FILENAME_RE.sub(" ", title or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if cleaned.lower().endswith(".txt"):
        cleaned = cleaned[:-4].rstrip(" .")
    if not cleaned:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(docid).strip()) or "unknown"
        return f"browsecomp-plus-{safe}"
    if len(cleaned) > 180:
        cleaned = cleaned[:180].rstrip()
    return cleaned


def _strip_leading_title(body: str, title: str | None) -> str:
    if not body or not title:
        return body
    lines = body.lstrip().splitlines()
    if lines and _clean_meta_value(lines[0]) == title.strip():
        return "\n".join(lines[1:]).lstrip("\n")
    return body


class _HTMLTextParser(HTMLParser):
    """Visible text only: drop script/style/head chrome, keep article copy."""

    _SKIP = frozenset({"script", "style", "noscript", "svg", "iframe", "template"})
    _BLOCK = frozenset(
        {
            "br",
            "p",
            "div",
            "h1",
            "h2",
            "h3",
            "h4",
            "li",
            "tr",
            "section",
            "article",
            "header",
            "footer",
            "blockquote",
            "pre",
            "hr",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip = 0
        self._in_head = False
        self._in_title = False

    def handle_starttag(self, tag, attrs):  # noqa: ARG002
        if tag == "head":
            self._in_head = True
        if tag == "title" and not self.title_parts:
            self._in_title = True
        if tag in self._SKIP:
            self._skip += 1
        if tag in self._BLOCK and not self._skip and not self._in_head:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "head":
            self._in_head = False
        if tag == "title":
            self._in_title = False
        if tag in self._SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)
        if self._skip or self._in_head:
            return
        self.parts.append(data)


def looks_like_html_document(text: str) -> bool:
    return bool(HTML_DOC_RE.search((text or "")[:2000]))


def html_document_to_text(text: str) -> tuple[str, str | None]:
    parser = _HTMLTextParser()
    parser.feed(text)
    parser.close()
    html_title = re.sub(r"\s+", " ", "".join(parser.title_parts)).strip() or None
    lines: list[str] = []
    for raw in "".join(parser.parts).splitlines():
        cleaned = re.sub(r"[ \t]+", " ", raw).strip()
        if not cleaned or SPAM_LINE_RE.search(cleaned):
            continue
        lines.append(cleaned)
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return body, html_title


def maybe_strip_html(body: str) -> tuple[str, str | None]:
    """If the corpus row is a full HTML page, keep visible text only.

    Does not touch PDF OCR dumps or books; those are already plaintext.
    """
    if not looks_like_html_document(body):
        return body, None
    extracted, html_title = html_document_to_text(body)
    if len(extracted) < 200:
        return body, html_title
    return extracted, html_title


def prepare_upload_document(text: str) -> tuple[str, str | None, str | None]:
    """Move title/date out of the body; keep author and other leftover metadata on top.

    Returns (prepared_body, title, published_ts).
    """
    meta, body = parse_frontmatter(text or "")
    title = meta.get("title") or None
    published_ts = None
    for key in DATE_KEYS:
        published_ts = normalize_published_ts(meta.get(key))
        if published_ts:
            break
    body = _strip_leading_title(body, title)
    body, html_title = maybe_strip_html(body)
    if not title:
        title = html_title

    leftover_keys = [k for k in meta if k not in TITLE_KEYS and k not in DATE_KEYS]
    header_lines: list[str] = []
    for key in leftover_keys:
        if key in AUTHOR_KEYS:
            header_lines.append(f"{key.capitalize()}: {meta[key]}")
    for key in leftover_keys:
        if key not in AUTHOR_KEYS:
            header_lines.append(f"{key}: {meta[key]}")

    if header_lines:
        prepared = "\n".join(header_lines) + ("\n\n" + body if body.strip() else "\n")
    else:
        prepared = body
    prepared = prepared.strip()
    if prepared:
        prepared += "\n"
    return prepared, title, published_ts


def encode_document(text: str) -> bytes:
    return (text or "").encode("utf-8")


def upload_one(
    client: BigdataContentClient,
    *,
    docid: str,
    url: str,
    text: str,
    tags: list[str],
    share_with_org: bool,
    max_retries: int,
    name_registry: FilenameRegistry | None = None,
) -> dict:
    upload_text, title, published_ts = prepare_upload_document(text)
    file_name = file_name_from_title(title, docid)
    if name_registry is not None:
        file_name = name_registry.allocate(file_name, docid)
    decision = detect_language(upload_text)
    enrichments = ["translation"] if decision.request_translation else None
    payload = encode_document(upload_text)

    result = {
        "docid": docid,
        "url": url,
        "file_name": file_name,
        "language": decision.language,
        "is_english": decision.is_english,
        "translation_requested": bool(enrichments),
        "published_ts": published_ts or "",
        "content_id": "",
        "status": "failed",
        "error": "",
        "detect_reason": decision.reason,
    }

    if not payload.strip():
        result["status"] = "skipped"
        result["error"] = "empty_text"
        return result

    last_error = ""
    data = None
    for attempt in range(max_retries):
        data, status_code = client.create_document(
            file_name,
            published_ts=published_ts,
            tags=tags,
            share_with_org=share_with_org,
            enrichments=enrichments,
        )
        if status_code in TRANSIENT_HTTP or status_code == 0:
            last_error = f"post_{status_code}"
            time.sleep(min(2 ** (attempt + 1), 20))
            continue
        if status_code != 200 or not data or "url" not in data or "id" not in data:
            result["error"] = f"post_{status_code}"
            return result
        break

    if not data or "url" not in data or "id" not in data:
        result["error"] = last_error or "post_max_retries"
        return result

    content_id = str(data["id"])
    result["content_id"] = content_id
    for attempt in range(max_retries):
        ok, put_status = client.put_bytes(data["url"], payload)
        if not ok:
            last_error = f"put_{put_status}"
            if put_status in TRANSIENT_HTTP or put_status == 0:
                time.sleep(min(2 ** (attempt + 1), 20))
                continue
            result["error"] = last_error
            return result

        result["status"] = "uploaded"
        result["error"] = ""
        return result

    result["error"] = last_error or "put_max_retries"
    return result


def cmd_download(args: argparse.Namespace) -> int:
    configure_logging()
    ds = load_corpus(Path(args.cache_dir))
    logging.info("Corpus ready: %s rows", len(ds))
    return 0


def cmd_progress(args: argparse.Namespace) -> int:
    configure_logging()
    state = UploadState(Path(args.state_path))
    logging.info("Upload state: %s", json.dumps(state.counts(), sort_keys=True))
    state.close()
    return 0


def cmd_export_mapping(args: argparse.Namespace) -> int:
    configure_logging()
    state = UploadState(Path(args.state_path))
    tsv_path = Path(args.tsv_path) if args.tsv_path else MAPPING_TSV
    jsonl_path = Path(args.jsonl_path) if args.jsonl_path else MAPPING_JSONL
    json_path = Path(args.json_path) if args.json_path else MAPPING_JSON
    n_map = state.export_mapping(tsv_path, jsonl_path, json_path)
    logging.info(
        "Wrote mapping n=%s tsv=%s jsonl=%s json=%s",
        n_map,
        tsv_path,
        jsonl_path,
        json_path,
    )
    state.close()
    return 0


def _read_id_file(path: Path) -> list[str]:
    ids: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(line)
    return ids


def cmd_upload(args: argparse.Namespace) -> int:
    log_path = configure_logging()
    api_key = os.getenv("BIGDATA_API_KEY", "").strip()
    if not args.dry_run and not api_key:
        logging.error("BIGDATA_API_KEY is not set (repo-root .env or environment)")
        return 1

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] or [DEFAULT_TAG]
    wanted_docids = {x.strip() for x in (args.docid or []) if str(x).strip()}
    if args.docid_file:
        wanted_docids |= {x.strip() for x in _read_id_file(Path(args.docid_file)) if x.strip()}
        logging.info("Loaded %s wanted docids from %s", len(wanted_docids), args.docid_file)
    ds = load_corpus(Path(args.cache_dir))
    state = UploadState(Path(args.state_path))
    skip_ids = state.uploaded_or_skipped_ids()
    if args.retry_failed:
        skip_ids -= state.failed_ids()
    if wanted_docids:
        skip_ids -= wanted_docids

    name_registry = FilenameRegistry()
    for existing_name in state.uploaded_file_names():
        name_registry.seed(existing_name)

    client = None
    if not args.dry_run:
        max_per_minute = max(1, args.rate_limit_per_minute - args.rate_limit_safety_margin)
        client = BigdataContentClient(
            api_key=api_key,
            rate_limiter=RateLimiter(max_per_minute),
            base_url=args.api_base_url,
        )

    logging.info(
        "Starting upload dry_run=%s concurrency=%s limit=%s skip_already=%s tags=%s share_with_org=%s log=%s",
        args.dry_run,
        args.concurrency,
        args.limit,
        len(skip_ids),
        tags,
        args.share_with_org,
        log_path,
    )

    stats = {"uploaded": 0, "failed": 0, "skipped": 0, "dry_run": 0, "translation": 0}
    started = time.monotonic()

    def handle_result(result: dict) -> None:
        state.record(result)
        status = result["status"]
        stats[status] = stats.get(status, 0) + 1
        if result.get("translation_requested"):
            stats["translation"] += 1
        if status == "failed":
            logging.warning(
                "upload failed docid=%s content_id=%s error=%s",
                result["docid"],
                result.get("content_id"),
                result.get("error"),
            )
        done = stats["uploaded"] + stats["failed"] + stats["skipped"] + stats["dry_run"]
        if done == 1 or done % 50 == 0:
            elapsed = time.monotonic() - started
            logging.info(
                "progress n=%s uploaded=%s failed=%s skipped=%s translation=%s elapsed=%.0fs last=%s %s",
                done,
                stats["uploaded"],
                stats["failed"],
                stats["skipped"],
                stats["translation"],
                elapsed,
                result["docid"],
                status,
            )

    def submit_dry_run(row: dict) -> dict:
        docid = str(row.get("docid") or "").strip()
        upload_text, title, published_ts = prepare_upload_document(
            str(row.get("text") or "")
        )
        file_name = file_name_from_title(title, docid)
        file_name = name_registry.allocate(file_name, docid)
        decision = detect_language(upload_text)
        return {
            "docid": docid,
            "url": str(row.get("url") or ""),
            "file_name": file_name,
            "language": decision.language,
            "is_english": decision.is_english,
            "translation_requested": decision.request_translation,
            "published_ts": published_ts or "",
            "content_id": "",
            "status": "dry_run",
            "error": decision.reason,
        }

    processed = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            in_flight: set[Future] = set()
            max_in_flight = max(args.concurrency * 2, args.concurrency)

            for row in ds:
                docid = str(row.get("docid") or "").strip()
                if not docid:
                    continue
                if wanted_docids and docid not in wanted_docids:
                    continue
                if docid in skip_ids:
                    continue
                if args.limit is not None and processed >= args.limit:
                    break
                processed += 1

                while len(in_flight) >= max_in_flight:
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        handle_result(fut.result())

                if args.dry_run:
                    fut = executor.submit(submit_dry_run, dict(row))
                else:
                    assert client is not None
                    fut = executor.submit(
                        upload_one,
                        client,
                        docid=docid,
                        url=str(row.get("url") or ""),
                        text=str(row.get("text") or ""),
                        tags=tags,
                        share_with_org=args.share_with_org,
                        max_retries=args.max_retries,
                        name_registry=name_registry,
                    )
                in_flight.add(fut)

            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    handle_result(fut.result())
    finally:
        if not args.dry_run:
            n_map = state.export_mapping(MAPPING_TSV, MAPPING_JSONL, MAPPING_JSON)
            logging.info(
                "Wrote mapping n=%s tsv=%s jsonl=%s json=%s",
                n_map,
                MAPPING_TSV,
                MAPPING_JSONL,
                MAPPING_JSON,
            )
        logging.info("Finished: %s state=%s", json.dumps(stats), json.dumps(state.counts()))
        state.close()
    return 0 if stats["failed"] == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--cache-dir", default=str(HF_CACHE_DIR))
        p.add_argument("--state-path", default=str(STATE_PATH))

    p_download = sub.add_parser("download", help="Download/cache the Hugging Face corpus")
    add_shared(p_download)
    p_download.set_defaults(func=cmd_download)

    p_progress = sub.add_parser("progress", help="Print upload checkpoint counts")
    add_shared(p_progress)
    p_progress.set_defaults(func=cmd_progress)

    p_upload = sub.add_parser("upload", help="Upload corpus documents to Bigdata")
    add_shared(p_upload)
    p_upload.add_argument("--limit", type=int, default=None, help="Max new documents this run")
    p_upload.add_argument("--docid", action="append", default=[], help="Upload only this corpus docid (repeatable)")
    p_upload.add_argument("--docid-file", default=None, help="File with one corpus docid per line")
    p_upload.add_argument("--concurrency", type=int, default=12)
    p_upload.add_argument("--dry-run", action="store_true")
    p_upload.add_argument("--retry-failed", action="store_true")
    p_upload.add_argument("--tags", default=DEFAULT_TAG)
    p_upload.add_argument("--share-with-org", action="store_true", default=False)
    p_upload.add_argument("--api-base-url", default=os.getenv("BIGDATA_API_BASE_URL", API_BASE_URL_DEFAULT))
    p_upload.add_argument("--rate-limit-per-minute", type=int, default=int(os.getenv("BIGDATA_RATE_LIMIT_PER_MINUTE", "500")))
    p_upload.add_argument("--rate-limit-safety-margin", type=int, default=int(os.getenv("BIGDATA_RATE_LIMIT_SAFETY_MARGIN", "20")))
    p_upload.add_argument("--max-retries", type=int, default=int(os.getenv("BIGDATA_UPLOAD_MAX_RETRIES", "5")))
    p_upload.set_defaults(func=cmd_upload)

    p_export = sub.add_parser(
        "export-mapping",
        help="Write corpus docid <-> Bigdata content_id mapping files",
    )
    add_shared(p_export)
    p_export.add_argument("--tsv-path", default=str(MAPPING_TSV))
    p_export.add_argument("--jsonl-path", default=str(MAPPING_JSONL))
    p_export.add_argument("--json-path", default=str(MAPPING_JSON))
    p_export.set_defaults(func=cmd_export_mapping)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
