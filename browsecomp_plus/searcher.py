"""Isolated BrowseComp-Plus search over Bigdata's fast Search API."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
load_dotenv(_REPO / ".env")

API_BASE = os.getenv("BIGDATA_API_BASE_URL", "https://api.bigdata.com").rstrip("/")
SEARCH_URL = f"{API_BASE}/v1/search"
TAG = "browsecomp-plus"
STATE_PATH = _PKG / "data" / "upload_state.sqlite"


RANKING_PARAMS = {
    "source_boost": 0,
    "freshness_boost": 0,
    "content_diversification": {"enabled": True},
    "reranker": {"enabled": True, "threshold": 0.2},
}

DOCUMENT_RANKING_PARAMS = {
    "source_boost": 0,
    "freshness_boost": 0,
    "content_diversification": {"enabled": False},
    "reranker": {"enabled": False},
}


def format_timestamp(value: Any) -> str:
    """Date the model can use. Search API `timestamp` is the document date."""
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        except (OSError, OverflowError, ValueError):
            return str(value)
    text = str(value).strip()
    if "T" in text:
        return text.split("T", 1)[0]
    return text


@dataclass
class Chunk:
    cnum: int | None
    text: str
    relevance: float


@dataclass
class Hit:
    content_id: str
    docid: str | None
    headline: str
    score: float
    snippet: str
    n_chunks: int
    source_id: str | None = None
    timestamp: str = ""
    chunks: list[Chunk] = field(default_factory=list)

    def to_agent_dict(self) -> dict[str, Any]:
        """Fields the model is allowed to see. No search scores or API extras."""
        doc: dict[str, Any] = {
            "docid": self.docid,
            "headline": self.headline,
            "chunks": [
                {"cnum": chunk.cnum, "text": chunk.text} for chunk in self.chunks
            ],
        }
        return doc


@dataclass
class SearchResult:
    mode: str
    query_sent: str
    status: int
    hits: list[Hit]
    unmapped_ids: list[str]
    audit: dict[str, Any] | None = None
    isolation_ok: bool = True
    error: str = ""
    raw_n_docs: int = 0

    @property
    def ranked_docids(self) -> list[str]:
        return [h.docid for h in self.hits if h.docid]


@dataclass
class DocumentResult:
    docid: str
    content_id: str | None
    status: int
    headline: str = ""
    text: str = ""
    n_chunks: int = 0
    truncated: bool = False
    error: str = ""
    timestamp: str = ""
    cnums: list[int] = field(default_factory=list)


class BrowsecompSearcher:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        state_path: Path = STATE_PATH,
        timeout: int = 90,
        tag: str = TAG,
        ranking_params: dict[str, Any] | None = None,
    ):
        self.api_key = (api_key or os.getenv("BIGDATA_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError("BIGDATA_API_KEY is not set")
        self.timeout = timeout
        self.tag = str(tag or TAG)
        self.ranking_params = dict(ranking_params or RANKING_PARAMS)
        self.content_id_to_docid = self._load_mapping(state_path)
        self.docid_to_content_id = {
            docid: cid for cid, docid in self.content_id_to_docid.items()
        }

    @staticmethod
    def _load_mapping(path: Path) -> dict[str, str]:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = conn.execute(
            """
            SELECT content_id, docid FROM documents
            WHERE status='uploaded' AND TRIM(content_id) != ''
            """
        ).fetchall()
        conn.close()
        return {str(cid).upper(): str(docid) for cid, docid in rows}

    def _post(self, payload: dict) -> tuple[int, dict]:
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        resp = requests.post(SEARCH_URL, json=payload, headers=headers, timeout=self.timeout)
        try:
            data = resp.json() if resp.text else {}
        except ValueError:
            data = {"raw": (resp.text or "")[:500]}
        if not isinstance(data, dict):
            data = {"raw": data}
        return resp.status_code, data

    def _collapse(self, results: list[dict]) -> tuple[list[Hit], list[str]]:
        best: dict[str, Hit] = {}
        order: list[str] = []
        unmapped: list[str] = []
        for doc in results:
            cid = str(doc.get("id") or "").upper()
            if not cid:
                continue
            raw_chunks = doc.get("chunks") or []
            parsed: list[Chunk] = []
            score = 0.0
            snippet = ""
            for chunk in raw_chunks:
                rel = float(chunk.get("relevance") or 0.0)
                text = str(chunk.get("text") or "")
                parsed.append(
                    Chunk(cnum=chunk.get("cnum"), text=text, relevance=rel)
                )
                if rel >= score:
                    score = rel
                    snippet = text
            parsed.sort(key=lambda c: c.relevance, reverse=True)
            src = doc.get("source") or {}
            source_id = src.get("id") if isinstance(src, dict) else src
            hit = Hit(
                content_id=cid,
                docid=self.content_id_to_docid.get(cid),
                headline=str(doc.get("headline") or ""),
                score=score,
                snippet=snippet,
                n_chunks=len(parsed),
                source_id=str(source_id) if source_id else None,
                timestamp=format_timestamp(doc.get("timestamp")),
                chunks=parsed,
            )
            if cid not in best:
                order.append(cid)
                best[cid] = hit
            elif score > best[cid].score:
                best[cid] = hit
            if hit.docid is None:
                unmapped.append(cid)
        hits = [best[cid] for cid in order if best[cid].docid]
        return hits, unmapped

    @staticmethod
    def fast_payload(
        query: str = "",
        *,
        max_chunks: int = 50,
        keyword_all_of: list[str] | None = None,
        keyword_any_of: list[str] | None = None,
        keyword_none_of: list[str] | None = None,
        keyword_search_in: str = "ALL",
        timestamp_start: str | None = None,
        timestamp_end: str | None = None,
        auto_enrich_filters: bool | None = None,
        tag: str = TAG,
        ranking_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        filters: dict[str, Any] = {
            "category": {"mode": "INCLUDE", "values": ["my_files"]},
            "tag": {"any_of": [str(tag or TAG)]},
        }
        timestamp: dict[str, str] = {}
        if timestamp_start:
            timestamp["start"] = timestamp_start
        if timestamp_end:
            timestamp["end"] = timestamp_end
        if timestamp:
            filters["timestamp"] = timestamp
        has_keywords = bool(keyword_all_of or keyword_any_of or keyword_none_of)
        if has_keywords:
            keyword: dict[str, Any] = {"search_in": keyword_search_in}
            # Omit empty lists. Sending all_of: [] with any_of can make the API
            # match nothing even when the terms are in the text.
            if keyword_all_of:
                keyword["all_of"] = list(keyword_all_of)
            if keyword_any_of:
                keyword["any_of"] = list(keyword_any_of)
            if keyword_none_of:
                keyword["none_of"] = list(keyword_none_of)
            filters["keyword"] = keyword
        if auto_enrich_filters is None:
            auto_enrich_filters = False
        query_obj: dict[str, Any] = {
            "filters": filters,
            "ranking_params": dict(ranking_params or RANKING_PARAMS),
            "max_chunks": max_chunks,
            "auto_enrich_filters": bool(auto_enrich_filters),
        }
        if query:
            query_obj["text"] = query
        return {"search_mode": "fast", "query": query_obj}

    def search_fast(
        self,
        query: str = "",
        *,
        max_chunks: int = 50,
        keyword_all_of: list[str] | None = None,
        keyword_any_of: list[str] | None = None,
        keyword_none_of: list[str] | None = None,
        keyword_search_in: str = "ALL",
        timestamp_start: str | None = None,
        timestamp_end: str | None = None,
        auto_enrich_filters: bool | None = None,
    ) -> SearchResult:
        payload = self.fast_payload(
            query,
            max_chunks=max_chunks,
            keyword_all_of=keyword_all_of,
            keyword_any_of=keyword_any_of,
            keyword_none_of=keyword_none_of,
            keyword_search_in=keyword_search_in,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            auto_enrich_filters=auto_enrich_filters,
            tag=self.tag,
            ranking_params=self.ranking_params,
        )
        sent = query or json.dumps((payload["query"]["filters"] or {}).get("keyword"))
        return self._run("fast", sent, payload)

    def document_payload(self, content_id: str, *, max_chunks: int = 200) -> dict[str, Any]:
        return {
            "search_mode": "fast",
            "query": {
                "filters": {
                    "category": {"mode": "INCLUDE", "values": ["my_files"]},
                    "tag": {"any_of": [self.tag]},
                    "document": {"mode": "INCLUDE", "values": [content_id]},
                    "chunk": {"from": 1, "to": max_chunks},
                },
                # This is deterministic chunk retrieval within one known
                # document, not a relevance search.
                "ranking_params": dict(DOCUMENT_RANKING_PARAMS),
                "max_chunks": max_chunks,
                "auto_enrich_filters": False,
            },
        }

    def fetch_document(
        self,
        docid: str,
        *,
        max_chunks: int = 200,
        max_chars: int = 0,
    ) -> DocumentResult:
        """Reassemble one document from its chunks via the document-id filter."""
        content_id = self.docid_to_content_id.get(str(docid))
        if not content_id:
            return DocumentResult(
                docid=str(docid),
                content_id=None,
                status=0,
                error="unknown docid",
            )
        status, data = self._post(self.document_payload(content_id, max_chunks=max_chunks))
        if status >= 400:
            return DocumentResult(
                docid=str(docid),
                content_id=content_id,
                status=status,
                error=json.dumps(data)[:400],
            )
        results = data.get("results") or []
        if not results:
            return DocumentResult(
                docid=str(docid),
                content_id=content_id,
                status=status,
                error="document not found in index",
            )
        doc = results[0]
        chunks = sorted(
            (c for c in (doc.get("chunks") or []) if c.get("text")),
            key=lambda c: c.get("cnum") or 0,
        )
        parts: list[str] = []
        shown_cnums: list[int] = []
        chars_used = 0
        truncated = False
        for chunk in chunks:
            chunk_text = str(chunk.get("text") or "").strip()
            separator = "\n\n" if parts else ""
            # Label each chunk so the caller can address it as docid:cnum, the
            # same chunk identity that search results expose.
            addition = f"{separator}[cnum {int(chunk.get('cnum') or 0)}]\n{chunk_text}"
            if max_chars and chars_used + len(addition) > max_chars:
                remaining = max_chars - chars_used
                if remaining > 0:
                    parts.append(addition[:remaining])
                    shown_cnums.append(int(chunk.get("cnum") or 0))
                truncated = True
                break
            parts.append(addition)
            shown_cnums.append(int(chunk.get("cnum") or 0))
            chars_used += len(addition)
        if len(chunks) >= max_chunks:
            truncated = True
        text = "".join(parts)
        return DocumentResult(
            docid=str(docid),
            content_id=content_id,
            status=status,
            headline=str(doc.get("headline") or ""),
            text=text,
            n_chunks=len(shown_cnums),
            truncated=truncated,
            timestamp=format_timestamp(doc.get("timestamp")),
            cnums=shown_cnums,
        )

    def _run(self, mode: str, query_sent: str, payload: dict) -> SearchResult:
        status, data = self._post(payload)
        if status >= 400:
            return SearchResult(
                mode=mode,
                query_sent=query_sent,
                status=status,
                hits=[],
                unmapped_ids=[],
                isolation_ok=False,
                error=json.dumps(data)[:400],
            )
        docs = data.get("results") or []
        hits, unmapped = self._collapse(docs)
        audit = (data.get("metadata") or {}).get("audit")
        return SearchResult(
            mode=mode,
            query_sent=query_sent,
            status=status,
            hits=hits,
            unmapped_ids=unmapped,
            audit=audit,
            isolation_ok=len(unmapped) == 0,
            raw_n_docs=len(docs),
        )
