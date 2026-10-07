"""Eval tool handlers used by the Bedrock agent loop."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any, Protocol

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from session import EvalSession


class SearcherLike(Protocol):
    def search_fast(
        self,
        query: str = "",
        *,
        max_chunks: int = 50,
        keyword_all_of: list[str] | None = None,
        keyword_any_of: list[str] | None = None,
        keyword_none_of: list[str] | None = None,
        keyword_search_in: str = "ALL",
        auto_enrich_filters: bool | None = None,
    ) -> Any: ...

    def fetch_document(
        self,
        docid: str,
        *,
        max_chunks: int = 200,
        max_chars: int = 0,
    ) -> Any: ...




def budget_line(session: EvalSession) -> str:
    """Expose the eval call ceilings without encouraging budget exhaustion."""
    return (
        f"search/grep {session.n_searches()} of max {session.max_searches}, "
        f"get_document {session.n_fetches()} of max {session.max_fetches}. "
        "These are ceilings, not quotas: submit the draft ranking once further "
        "retrieval would not change which chunks are relevant."
    )


def fetch_note(n_new: int, n_already: int) -> str:
    """Tell the agent whether the fetch paid for itself, so it learns when to skip."""
    if n_new <= 0:
        return (
            "Search had already shown every chunk of this document, so this fetch "
            "added nothing. Prefer fetching documents whose search or grep hits looked "
            "truncated relative to the detail you need."
        )
    return (
        f"{n_new} of {n_new + n_already} chunks are new; search or grep had already shown "
        f"{n_already}."
    )


KEYWORD_FIELDS = {"ALL", "HEADLINE", "BODY"}
MAX_KEYWORD_TERMS = 20
MIN_KEYWORD_CHARS = 3
_GREP_FLAGS = {"-i", "-n", "-w", "-E", "-F", "-e", "-o"}
_GREP_REGEX_CHARS = re.compile(r"[\\\[\](){}^$*?]")


def _clean_terms(values: list[str] | str | None) -> list[str]:
    """One API keyword per list item; punctuation remains part of the keyword."""
    if values is None:
        return []
    if isinstance(values, str):
        items: list[Any] = [values]
    elif isinstance(values, list):
        items = values
    else:
        items = [values]
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if len(text) < MIN_KEYWORD_CHARS:
            continue
        if text not in out:
            out.append(text)
        if len(out) >= MAX_KEYWORD_TERMS:
            return out
    return out


def _tokenize_grep_pattern(pattern: str) -> list[tuple[str, bool]]:
    """Return (text, quoted) atoms so syntax inside quotes stays literal."""
    atoms: list[tuple[str, bool]] = []
    i = 0
    n = len(pattern)
    while i < n:
        while i < n and pattern[i].isspace():
            i += 1
        if i >= n:
            break
        if pattern[i] in "\"'":
            quote = pattern[i]
            i += 1
            start = i
            while i < n and pattern[i] != quote:
                i += 1
            if i >= n:
                raise ValueError("Unclosed quote in grep pattern.")
            atoms.append((pattern[start:i], True))
            i += 1
            continue
        start = i
        while i < n and not pattern[i].isspace():
            i += 1
        atoms.append((pattern[start:i], False))
    return atoms


def parse_grep_pattern(raw: str) -> tuple[list[str], list[str]]:
    """Map a grep-like pattern onto keyword all_of / any_of. Not a regex engine."""
    text = str(raw or "").strip()
    if not text:
        raise ValueError("Provide a grep pattern of one or more tokens.")
    atoms = _tokenize_grep_pattern(text)
    while atoms and not atoms[0][1] and atoms[0][0] in _GREP_FLAGS:
        atoms.pop(0)
    if atoms and not atoms[0][1] and atoms[0][0] == "-v":
        raise ValueError("grep does not support invert match (-v).")
    if not atoms:
        raise ValueError("Provide a grep pattern of one or more tokens.")
    for atom, _quoted in atoms:
        if atom in {".", ".*", ".+", ".?"}:
            raise ValueError(
                "grep is token matching, not regex. Send names or titles, not wildcards."
            )
        if _GREP_REGEX_CHARS.search(atom):
            raise ValueError(
                "grep is token matching, not regex. Send names or titles, not wildcards."
            )
    or_atoms = [atom for atom, quoted in atoms if not quoted and "|" in atom]
    and_atoms = [atom for atom, quoted in atoms if quoted or "|" not in atom]
    if len(or_atoms) > 1:
        raise ValueError(
            "Only one | group per grep. Independent clues are separate calls."
        )
    all_of = _clean_terms(and_atoms)
    any_of = _clean_terms(or_atoms[0].split("|") if or_atoms else [])
    if not all_of and not any_of:
        raise ValueError("Each grep token needs at least 3 characters.")
    return all_of, any_of


def _drop_seen_chunks(
    session: EvalSession,
    raw_hits: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Omit already-shown (docid, cnum) pairs."""
    selected: list[dict[str, Any]] = []
    raw_chunk_count = 0
    deduped_seen = 0
    for hit in raw_hits:
        docid = str(hit["docid"])
        seen = session.seen_chunks(docid)
        kept: list[dict[str, Any]] = []
        for chunk in hit.get("chunks") or []:
            raw_chunk_count += 1
            cnum = chunk.get("cnum")
            try:
                cnum_key = int(cnum) if cnum is not None else None
            except (TypeError, ValueError):
                cnum_key = None
            if cnum_key is not None and cnum_key in seen:
                deduped_seen += 1
                continue
            kept.append(chunk)
        if not kept:
            continue
        selected.append(
            {
                "docid": docid,
                "headline": hit.get("headline") or "",
                "chunks": kept,
            }
        )
    return selected, raw_chunk_count, deduped_seen


def _run_index_search(
    session: EvalSession,
    searcher: SearcherLike,
    *,
    query: str,
    all_of: list[str],
    any_of: list[str],
    none_of: list[str],
    field: str,
    max_chunks: int,
    auto_enrich_filters: bool,
    request_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if session.retrieval_closed():
        return {
            "ok": False,
            "error": "Retrieval closed when the draft ranking was submitted.",
        }
    remaining = session.remaining_searches()
    if remaining <= 0:
        return {
            "ok": False,
            "error": (
                f"Search/grep limit reached ({session.max_searches}). Call submit_ranking."
            ),
        }
    if field not in KEYWORD_FIELDS:
        return {"ok": False, "error": "search_in must be ALL, HEADLINE, or BODY."}
    n_chunks = max(1, min(int(max_chunks), 100))
    request: dict[str, Any] = {
        "search_mode": "fast",
        "text": query,
        "max_chunks": n_chunks,
        "auto_enrich_filters": bool(auto_enrich_filters),
    }
    if request_extra:
        request.update(request_extra)
    search_kwargs: dict[str, Any] = {
        "max_chunks": n_chunks,
        "auto_enrich_filters": bool(auto_enrich_filters),
    }
    if all_of:
        request["keyword_all_of"] = all_of
        search_kwargs["keyword_all_of"] = all_of
    if any_of:
        request["keyword_any_of"] = any_of
        search_kwargs["keyword_any_of"] = any_of
    if none_of:
        request["keyword_none_of"] = none_of
        search_kwargs["keyword_none_of"] = none_of
    if all_of or any_of or none_of:
        request["keyword_search_in"] = field
        search_kwargs["keyword_search_in"] = field
    started = time.perf_counter()
    result = searcher.search_fast(query, **search_kwargs)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    raw_hits = [hit.to_agent_dict() for hit in result.hits if hit.docid]
    hits, raw_chunk_count, deduped_count = _drop_seen_chunks(
        session, raw_hits
    )
    docids = [hit["docid"] for hit in hits]
    visible_chunk_count = sum(len(hit.get("chunks") or []) for hit in hits)
    log_result = {
        "status": result.status,
        "error": result.error,
        "n_hits": len(hits),
        "n_chunks": visible_chunk_count,
        "raw_n_hits": len(raw_hits),
        "raw_n_chunks": raw_chunk_count,
        "deduped_seen_chunks": deduped_count,
        "docids": docids,
        "unmapped_ids": result.unmapped_ids,
        "elapsed_ms": elapsed_ms,
        "cnums_by_docid": {
            hit["docid"]: [
                chunk["cnum"]
                for chunk in hit.get("chunks") or []
                if chunk.get("cnum") is not None
            ]
            for hit in hits
        },
        "evidence_documents": hits,
    }
    try:
        new_docids = session.record_search(request, log_result)
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    if result.error or int(result.status or 0) >= 400:
        return {
            "ok": False,
            "error": result.error or f"Search API returned HTTP {result.status}.",
            "status": result.status,
            "remaining_searches": session.remaining_searches(),
            "budget": budget_line(session),
        }
    payload = {
        "ok": True,
        "budget": budget_line(session),
        "documents": hits,
    }
    notes: list[str] = []
    if '"' in query:
        notes.append(
            "Quotation marks in a search query were ignored: this tool matches "
            "meaning, not strings. An exact phrase belongs in grep."
        )
    n_and_tokens = len(all_of) + (1 if any_of else 0)
    if hits and not query and n_and_tokens >= 3:
        notes.append(
            f"This grep required all {n_and_tokens} tokens in one chunk. A page "
            "that has only the rarest of them will not appear here."
        )
    if not hits and deduped_count:
        notes.append(
            "All returned chunks were already shown in this session. Repeating "
            "the same query will not expose new evidence; search a different "
            "constraint or call get_document on a promising docid."
        )
    elif not hits and (all_of or any_of) and not query:
        if n_and_tokens > 1:
            notes.append(
                f"No chunk contains all {n_and_tokens} tokens together. grep ANDs "
                "tokens within one chunk, so each token added shrinks the result "
                "set. The rarest literal on its own matches more."
            )
        else:
            notes.append(
                "This literal appears in no chunk. Nothing is spelled that way in "
                "the corpus, so a different literal or a search on the meaning is "
                "the next move."
            )
    elif not hits and query:
        notes.append(
            "No chunks matched this meaning. Ask it as a natural question, try a "
            "different sense of an ambiguous word, or grep a single literal. "
            "Adding another constraint to this call would match less."
        )
    elif not new_docids:
        notes.append(
            "No new document: every chunk here comes from a document already in "
            "your context."
        )
    streak = session.calls_without_new_documents()
    if streak >= 3:
        notes.append(
            f"The last {streak} retrieval calls added no new document. Whatever "
            "you are varying is not what is limiting you: change the fact, the "
            "tool, or open a document you already hold."
        )
    if notes:
        payload["note"] = " ".join(notes)
    return payload


def handle_search(
    session: EvalSession,
    searcher: SearcherLike,
    *,
    text: str = "",
    max_chunks: int = 20,
    auto_enrich_filters: bool = False,
) -> dict[str, Any]:
    query = str(text or "").strip()
    if not query:
        return {
            "ok": False,
            "error": "Provide a text query. Use grep for exact tokens.",
        }
    return _run_index_search(
        session,
        searcher,
        query=query,
        all_of=[],
        any_of=[],
        none_of=[],
        field="ALL",
        max_chunks=max_chunks,
        auto_enrich_filters=auto_enrich_filters,
        request_extra={"tool": "search"},
    )


def handle_grep(
    session: EvalSession,
    searcher: SearcherLike,
    *,
    pattern: str = "",
    max_chunks: int = 20,
    auto_enrich_filters: bool = False,
) -> dict[str, Any]:
    try:
        all_of, any_of = parse_grep_pattern(pattern)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return _run_index_search(
        session,
        searcher,
        query="",
        all_of=all_of,
        any_of=any_of,
        none_of=[],
        field="ALL",
        max_chunks=max_chunks,
        auto_enrich_filters=auto_enrich_filters,
        request_extra={"tool": "grep", "pattern": str(pattern or "").strip()},
    )


def handle_get_document(
    session: EvalSession,
    searcher: SearcherLike,
    *,
    docid: str,
    max_chunks: int = 200,
    max_chars: int = 0,
) -> dict[str, Any]:
    if session.retrieval_closed():
        return {
            "ok": False,
            "error": "Retrieval closed when the draft ranking was submitted.",
        }
    if session.remaining_fetches() <= 0:
        return {
            "ok": False,
            "error": (
                f"get_document limit reached ({session.max_fetches}). "
                "Use search or grep, or call submit_ranking."
            ),
        }
    wanted = str(docid or "").strip()
    if not wanted:
        return {"ok": False, "error": "docid is required."}
    if wanted not in set(session.seen_docids()):
        return {
            "ok": False,
            "error": (
                f"docid {wanted} has not appeared in a search or grep result. "
                "Only fetch documents returned by search or grep."
            ),
        }
    already_seen = session.seen_chunks(wanted)
    started = time.perf_counter()
    doc = searcher.fetch_document(wanted, max_chunks=max_chunks, max_chars=max_chars)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    cnums = list(getattr(doc, "cnums", []) or [])
    n_already = len([cnum for cnum in cnums if cnum in already_seen])
    n_new = len(cnums) - n_already
    matches = list(re.finditer(r"(?:^|\n)\[cnum (\d+)\]\n", str(doc.text or "")))
    fetched_chunks: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(doc.text)
        fetched_chunks.append(
            {
                "cnum": int(match.group(1)),
                "text": str(doc.text[match.end() : end]).strip(),
            }
        )
    if not fetched_chunks and len(cnums) == 1:
        fetched_chunks = [{"cnum": cnums[0], "text": str(doc.text or "")}]
    log_result = {
        "n_chunks": doc.n_chunks,
        "n_chars": len(doc.text),
        "chunks_new": n_new,
        "chunks_already_seen": n_already,
        "cnums": cnums,
        "status": doc.status,
        "error": doc.error,
        "elapsed_ms": elapsed_ms,
        "evidence_documents": [
            {
                "docid": wanted,
                "headline": str(doc.headline or ""),
                "chunks": fetched_chunks,
            }
        ],
    }
    try:
        session.record_fetch({"docid": wanted, "max_chunks": max_chunks}, log_result)
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    if doc.error:
        return {
            "ok": False,
            "error": doc.error,
            "budget": budget_line(session),
        }
    body = str(doc.text or "")
    return {
        "ok": True,
        "docid": doc.docid,
        "headline": doc.headline,
        "note": fetch_note(n_new, n_already),
        "truncated": doc.truncated,
        "text": body,
        "documents": [
            {
                "docid": wanted,
                "headline": str(doc.headline or ""),
                "chunks": fetched_chunks,
            }
        ],
        "budget": budget_line(session),
    }


def handle_submit_ranking(
    session: EvalSession,
    *,
    documents: list[Any] | None = None,
    ranking_strategy: str = "",
) -> dict[str, Any]:
    result = session.submit_ranking(
        documents=documents, ranking_strategy=ranking_strategy
    )
    if not result.get("ok"):
        return result
    if result.get("phase") == "draft":
        result["next"] = "Draft ranking recorded and retrieval is now closed."
    else:
        result["next"] = "Final ranking recorded."
    return result


def dispatch_tool(
    session: EvalSession,
    searcher: SearcherLike,
    name: str,
    arguments: dict[str, Any] | None,
    *,
    max_chunks: int = 20,
    document_max_chunks: int = 200,
    document_max_chars: int = 0,
    auto_enrich_filters: bool = False,
) -> dict[str, Any]:
    """Route one tool call to the matching handler."""
    args = arguments or {}
    if name == "search":
        return handle_search(
            session,
            searcher,
            text=str(args.get("text") or ""),
            max_chunks=max_chunks,
            auto_enrich_filters=auto_enrich_filters,
        )
    if name == "grep":
        return handle_grep(
            session,
            searcher,
            pattern=str(args.get("pattern") or ""),
            max_chunks=max_chunks,
            auto_enrich_filters=auto_enrich_filters,
        )
    if name == "get_document":
        return handle_get_document(
            session,
            searcher,
            docid=str(args.get("docid") or ""),
            max_chunks=document_max_chunks,
            max_chars=document_max_chars,
        )
    if name == "submit_ranking":
        return handle_submit_ranking(
            session,
            documents=list(args.get("documents") or []),
            ranking_strategy=str(args.get("ranking_strategy") or ""),
        )
    return {"ok": False, "error": f"Unknown tool: {name}"}
