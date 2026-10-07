"""Owned Bedrock Converse tool loop.

The model sees only the eval tools. The first submit_ranking call is the
retrieval stop signal: it closes search. A fresh evidence-complete pass then
records the scored ranking and the query ends.
"""

from __future__ import annotations

import base64
import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from paths import atomic_write_json
from schemas import (
    bedrock_tool_config,
    phase_tool_names,
)
from session import MAX_RANKED_DOCUMENTS, EvalSession
from tools import (
    SearcherLike,
    dispatch_tool,
)


CACHE_POINT = {"cachePoint": {"type": "default"}}
RETRIEVAL_TOOLS = frozenset({"search", "grep", "get_document"})
RANKING_REVIEW_FULL_TEXT_LIMIT = 20
ROUND_CAP_KEYS = {
    "search": "max_search_calls_per_round",
    "grep": "max_grep_calls_per_round",
    "get_document": "max_fetch_calls_per_round",
}
ROUND_CAP_DEFAULTS = {"search": 5, "grep": 10, "get_document": 5}


def round_caps(cfg: dict[str, Any]) -> dict[str, int]:
    return {
        name: int(cfg.get(ROUND_CAP_KEYS[name]) or ROUND_CAP_DEFAULTS[name])
        for name in ROUND_CAP_DEFAULTS
    }


def over_round_cap(uses: list["ToolUse"], caps: dict[str, int]) -> dict[int, str]:
    """Index the calls past this turn's per-tool ceiling, keeping the earlier ones."""
    used: dict[str, int] = {}
    rejected: dict[int, str] = {}
    for index, use in enumerate(uses):
        cap = caps.get(use.name)
        if cap is None:
            continue
        used[use.name] = used.get(use.name, 0) + 1
        if used[use.name] > cap:
            rejected[index] = (
                f"One turn accepts {cap} {use.name} calls; this was number "
                f"{used[use.name]}. The earlier ones ran. Send this one in the "
                "next turn if the results still leave the gap open."
            )
    return rejected


def transcript_json_safe(value: Any) -> Any:
    """Convert Bedrock binary response fields into inspectable JSON values."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            "__bytes_base64__": base64.b64encode(bytes(value)).decode("ascii")
        }
    if isinstance(value, dict):
        return {str(key): transcript_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [transcript_json_safe(item) for item in value]
    return value


def converse_system_blocks(system: str) -> list[dict[str, Any]]:
    return [{"text": system}, dict(CACHE_POINT)]


def model_uses_manual_cache_points(model_id: str) -> bool:
    """GPT-5.6 Converse uses implicit caching; Claude accepts cachePoint blocks."""
    return ".openai." not in str(model_id) and not str(model_id).startswith("openai.")


def messages_with_cache_points(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy messages and mark up to 3 user turns for Bedrock prompt cache.

    Checkpoints stay off the stored transcript. Bedrock allows 4 including
    the system prompt, so we keep the first user turn plus the last two.
    """
    copied = copy.deepcopy(messages)
    user_idxs = [
        i
        for i, msg in enumerate(copied)
        if msg.get("role") == "user" and isinstance(msg.get("content"), list)
    ]
    keep: set[int] = set()
    if user_idxs:
        keep.add(user_idxs[0])
        keep.add(user_idxs[-1])
        if len(user_idxs) > 1:
            keep.add(user_idxs[-2])
    for i in keep:
        content = [
            block
            for block in copied[i]["content"]
            if not (isinstance(block, dict) and "cachePoint" in block)
        ]
        content.append(dict(CACHE_POINT))
        copied[i] = {**copied[i], "content": content}
    return copied

RETRIEVED_CHUNKS_PREFIX = (
    "RETRIEVED_CHUNKS (deduplicated and grouped by document):\n"
)
RANKING_MUST_BE_ALONE = (
    "submit_ranking must be alone. Retrieval and ranking cannot share a turn."
)


def ranking_review_prompt(query: str, session: EvalSession) -> str:
    """Build the evidence bundle for the scored ranking pass after retrieval stops."""
    draft = session.ranking_draft() or {}
    catalog = session.evidence_catalog()
    seen_map = session.seen_chunk_map()
    draft_ids = {
        str(item.get("docid") or "").strip()
        for item in (draft.get("documents") or [])
        if isinstance(item, dict)
    }
    draft_ids.update(str(docid).strip() for docid in (draft.get("ranked_docids") or []))
    draft_ids.discard("")
    visible: list[tuple[str, list[int]]] = []
    for docid in session.seen_docids():
        cnums = sorted(seen_map.get(docid, set()))
        if cnums:
            visible.append((docid, cnums))
    unused_slots = max(0, MAX_RANKED_DOCUMENTS - len(draft_ids))
    lines = [
        "FINAL_RANKING_REVIEW",
        "",
        "ORIGINAL_QUERY:",
        str(query).strip(),
        "",
        "DRAFT_RANKING (suggestion only; check it against every document below):",
        json.dumps(draft, ensure_ascii=False, indent=2),
        "",
        "DRAFT_SHAPE:",
        (
            f"The previous ranking used {len(draft_ids)} of {MAX_RANKED_DOCUMENTS} slots, "
            f"leaving {unused_slots} unused. {len(visible)} retrieved documents "
            "are available below. Add missed evidence, drop irrelevant ids, and "
            "rerank scores."
        ),
        (
            f"An unused slot scores nothing, so submitting {len(draft_ids)} "
            f"documents while {len(visible)} are available can only lose points: "
            f"the {unused_slots} weakest candidates that touch any constraint "
            "belong at the tail, below the evidence you are confident in. Fill "
            f"all {MAX_RANKED_DOCUMENTS} unless fewer than "
            f"{MAX_RANKED_DOCUMENTS} of the documents below touch the query at "
            "all."
            if unused_slots and len(visible) > len(draft_ids)
            else "Order by contribution and keep every document that touches a "
            "constraint."
        ),
        "",
    ]
    lines.append("RETRIEVED_DOCUMENTS (deduplicated; no benchmark labels):")
    compact = len(visible) > RANKING_REVIEW_FULL_TEXT_LIMIT
    if compact:
        lines.append(
            "Many documents remain. TITLE_INDEX lists every available id. "
            "FULL_CHUNKS follow for draft-ranked documents; "
            "others have a short snippet."
        )
        lines.append("")
        lines.append("TITLE_INDEX:")
        for docid, _cnums in visible:
            document = catalog.get(docid) or {}
            lines.append(f"- {docid}: {str(document.get('headline') or '')}")
    for docid, cnums in visible:
        document = catalog.get(docid) or {}
        chunks = document.get("chunks") if isinstance(document, dict) else {}
        chunks = chunks if isinstance(chunks, dict) else {}
        full = (not compact) or (docid in draft_ids)
        lines.append("")
        lines.append(f"DOCUMENT {docid}")
        lines.append(f"TITLE: {str(document.get('headline') or '')}")
        lines.append(f"URL: {str(document.get('url') or '')}")
        if full:
            for cnum in cnums:
                text = str(chunks.get(str(cnum)) or "")
                lines.append(f"[{docid}:{cnum}]")
                lines.append(text if text else "(text unavailable in persisted catalog)")
            continue
        first = str(chunks.get(str(cnums[0])) or "")
        snippet = first[:240].rstrip()
        if len(first) > 240:
            snippet += "…"
        lines.append(
            f"[{docid}:{cnums[0]}] "
            + (snippet or "(text unavailable in persisted catalog)")
        )
    return "\n".join(lines)


class GenerateFn(Protocol):
    def __call__(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[str],
        tool_choice: str | None,
    ) -> "ModelTurn": ...


@dataclass
class BedrockRetryWait:
    """Backoff sleeps between throttled Converse attempts. Not LLM generation."""

    seconds: float = 0.0


@dataclass
class ToolUse:
    tool_use_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelTurn:
    assistant_message: dict[str, Any]
    tool_uses: list[ToolUse] = field(default_factory=list)
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    text: str = ""


@dataclass
class LoopResult:
    status: str
    error: str = ""
    n_turns: int = 0
    forced_ranking: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    ranking_input_tokens: int = 0
    ranking_output_tokens: int = 0
    ranking_cache_read_tokens: int = 0
    ranking_cache_write_tokens: int = 0
    cost_usd: float | None = None
    ranking_cost_usd: float | None = None
    text: str = ""


def estimate_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
    cache_read_multiplier: float = 0.1,
    cache_write_multiplier: float = 1.25,
) -> float:
    """Bedrock reports uncached input separately from cache read and write."""
    mtok = 1_000_000.0
    input_rate = float(input_usd_per_mtok)
    return (
        input_tokens / mtok * input_rate
        + cache_read_tokens / mtok * input_rate * float(cache_read_multiplier)
        + cache_write_tokens / mtok * input_rate * float(cache_write_multiplier)
        + output_tokens / mtok * float(output_usd_per_mtok)
    )


def aws_session_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """boto3 Session kwargs. Access keys in the environment beat AWS_PROFILE."""
    env = os.environ
    region = str(
        cfg.get("aws_region")
        or env.get("BEDROCK_AWS_REGION_NAME")
        or env.get("AWS_REGION")
        or env.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    kwargs: dict[str, Any] = {"region_name": region}
    if env.get("AWS_ACCESS_KEY_ID"):
        kwargs["aws_access_key_id"] = env["AWS_ACCESS_KEY_ID"]
        if env.get("AWS_SECRET_ACCESS_KEY"):
            kwargs["aws_secret_access_key"] = env["AWS_SECRET_ACCESS_KEY"]
        if env.get("AWS_SESSION_TOKEN"):
            kwargs["aws_session_token"] = env["AWS_SESSION_TOKEN"]
    else:
        profile = str(cfg.get("aws_profile") or env.get("AWS_PROFILE") or "").strip()
        if profile:
            kwargs["profile_name"] = profile
    return kwargs


def make_bedrock_generate(
    cfg: dict[str, Any],
    *,
    client: Any | None = None,
    retry_wait: BedrockRetryWait | None = None,
) -> GenerateFn:
    """Bedrock Converse generate function with throttle retries."""
    if client is None:
        import boto3
        from botocore.config import Config as BotoConfig

        kwargs = aws_session_kwargs(cfg)
        session = boto3.Session(**kwargs)
        timeout = int(cfg.get("timeout_s") or 900)
        client = session.client(
            "bedrock-runtime",
            region_name=kwargs["region_name"],
            config=BotoConfig(
                read_timeout=min(timeout, 300),
                connect_timeout=20,
                retries={"max_attempts": 1},
            ),
        )
    model_id = str(cfg.get("model") or "").strip()
    if not model_id:
        raise ValueError("model is required")
    max_tokens = int(cfg.get("max_tokens") or 8192)
    manual_cache_points = model_uses_manual_cache_points(model_id)
    reasoning_effort = str(cfg.get("reasoning_effort") or "").strip().lower()
    wait = retry_wait if retry_wait is not None else BedrockRetryWait()

    def generate(
        *,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[str],
        tool_choice: str | None,
    ) -> ModelTurn:
        request: dict[str, Any] = {
            "modelId": model_id,
            "messages": (
                messages_with_cache_points(
                    [ensure_nonempty_message(msg) for msg in messages]
                )
                if manual_cache_points
                else [ensure_nonempty_message(msg) for msg in messages]
            ),
            "system": (
                converse_system_blocks(system)
                if manual_cache_points
                else [{"text": system}]
            ),
            "inferenceConfig": {"maxTokens": max_tokens},
            "toolConfig": bedrock_tool_config(
                tools,
                tool_choice=tool_choice,
            ),
        }
        if reasoning_effort:
            request["additionalModelRequestFields"] = {
                "reasoning": {"effort": reasoning_effort}
            }
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                response = client.converse(**request)
                return parse_converse_response(response)
            except Exception as exc:
                last_error = exc
                name = type(exc).__name__
                if name not in {
                    "ThrottlingException",
                    "ModelTimeoutException",
                    "ServiceUnavailableException",
                } and "Throttling" not in str(exc):
                    raise
                delay = min(2 ** attempt, 20)
                time.sleep(delay)
                wait.seconds += delay
        raise last_error or RuntimeError("Bedrock converse failed")

    generate.retry_wait = wait  # type: ignore[attr-defined]
    return generate


def parse_converse_response(response: dict[str, Any]) -> ModelTurn:
    message = ((response.get("output") or {}).get("message")) or {}
    content: list[Any] = []
    uses: list[ToolUse] = []
    texts: list[str] = []
    for block in list(message.get("content") or []):
        content.append(block)
        if not isinstance(block, dict):
            continue
        if "text" in block and block["text"]:
            texts.append(str(block["text"]))
        tool_use = block.get("toolUse")
        if isinstance(tool_use, dict):
            raw_input = tool_use.get("input") or {}
            if not isinstance(raw_input, dict):
                # The model occasionally emits a toolUse input that is not an
                # object. Echoing it back is a ValidationException on the whole
                # transcript, which killed the query outright; replaced with an
                # empty object the tool reports the missing arguments and the
                # model gets another turn.
                raw_input = {}
                content[-1] = {**block, "toolUse": {**tool_use, "input": raw_input}}
            uses.append(
                ToolUse(
                    tool_use_id=str(tool_use.get("toolUseId") or ""),
                    name=str(tool_use.get("name") or ""),
                    arguments=raw_input,
                )
            )
    usage = response.get("usage") or {}
    return ModelTurn(
        assistant_message={"role": "assistant", "content": content},
        tool_uses=uses,
        stop_reason=str(response.get("stopReason") or ""),
        input_tokens=int(usage.get("inputTokens") or 0),
        output_tokens=int(usage.get("outputTokens") or 0),
        cache_read_tokens=int(
            usage.get("cacheReadInputTokens") or usage.get("cacheReadInputTokenCount") or 0
        ),
        cache_write_tokens=int(
            usage.get("cacheWriteInputTokens") or usage.get("cacheWriteInputTokenCount") or 0
        ),
        text="\n".join(texts).strip(),
    )


def _tool_result_block(tool_use_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    ok = bool(payload.get("ok", True))
    return {
        "toolResult": {
            "toolUseId": tool_use_id,
            "status": "success" if ok else "error",
            "content": [{"json": payload}],
        }
    }


def ensure_nonempty_message(message: dict[str, Any]) -> dict[str, Any]:
    """Bedrock rejects a Message whose content field is empty."""
    out = dict(message)
    if out.get("content"):
        return out
    if out.get("role") == "assistant":
        out["content"] = [{"text": "(no output)"}]
    else:
        out["content"] = [
            {"text": "Continue. Use a tool. Do not answer in plain text."}
        ]
    return out


def prepare_retrieval_results(
    uses: list[ToolUse],
    payloads: list[dict[str, Any] | None],
    retrieval_indexes: list[int],
) -> list[dict[str, Any]]:
    """Keep per-call matches separate and return one grouped evidence bundle."""
    documents: dict[str, dict[str, Any]] = {}
    seen_chunk_keys: dict[str, set[tuple[str, Any]]] = {}
    for index in retrieval_indexes:
        payload = payloads[index]
        if not isinstance(payload, dict):
            continue
        matched_chunk_ids: list[str] = []
        for document in payload.get("documents") or []:
            if not isinstance(document, dict):
                continue
            docid = str(document.get("docid") or "")
            if not docid:
                continue
            grouped = documents.setdefault(
                docid,
                {
                    "docid": docid,
                    "title": document.get("title")
                    or document.get("headline")
                    or "",
                    "url": document.get("url") or "",
                    "chunks": [],
                },
            )
            keys = seen_chunk_keys.setdefault(docid, set())
            for chunk in document.get("chunks") or []:
                if not isinstance(chunk, dict):
                    continue
                raw_cnum = chunk.get("cnum")
                try:
                    cnum = int(raw_cnum) if raw_cnum is not None else None
                except (TypeError, ValueError):
                    cnum = None
                key: tuple[str, Any] = (
                    ("cnum", cnum)
                    if cnum is not None
                    else ("text", str(chunk.get("text") or ""))
                )
                if cnum is not None:
                    matched_chunk_ids.append(f"{docid}:{cnum}")
                if key not in keys:
                    keys.add(key)
                    grouped["chunks"].append(
                        {"cnum": cnum, "text": str(chunk.get("text") or "")}
                    )
        compact: dict[str, Any] = {
            "ok": bool(payload.get("ok", True)),
            "matched_chunk_ids": list(dict.fromkeys(matched_chunk_ids)),
        }
        if not compact["ok"]:
            compact["error"] = str(payload.get("error") or "retrieval failed")
        elif payload.get("note"):
            compact["note"] = str(payload["note"])
        if "truncated" in payload:
            compact["truncated"] = bool(payload["truncated"])
        payload.clear()
        payload.update(compact)

    def chunk_sort_key(chunk: dict[str, Any]) -> tuple[int, str]:
        raw_cnum = chunk.get("cnum")
        try:
            return int(raw_cnum), ""
        except (TypeError, ValueError):
            return 2**31 - 1, str(chunk.get("text") or "")

    for document in documents.values():
        document["chunks"].sort(key=chunk_sort_key)
    return list(documents.values())


def append_user_notice(messages: list[dict[str, Any]], text: str) -> None:
    """Append a user-side notice, skipping if that exact text is already present."""
    if messages and messages[-1].get("role") == "user":
        content = list(messages[-1].get("content") or [])
        if any(
            isinstance(block, dict) and block.get("text") == text for block in content
        ):
            return
        content.append({"text": text})
        messages[-1] = {"role": "user", "content": content}
        return
    messages.append({"role": "user", "content": [{"text": text}]})


def execute_turn_tools(
    uses: list[ToolUse],
    *,
    session: EvalSession,
    searcher: SearcherLike,
    cfg: dict[str, Any],
    allowed: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run this turn's tool calls. Retrieval in one turn runs in parallel."""
    uses = list(uses)
    names = [use.name for use in uses]
    invalid_mixed_ranking = "submit_ranking" in names and len(uses) > 1

    dispatch_kwargs = {
        "max_chunks": int(cfg.get("max_chunks") or 20),
        "document_max_chunks": int(cfg.get("document_max_chunks") or 200),
        "document_max_chars": int(cfg.get("document_max_chars") or 0),
        "auto_enrich_filters": bool(cfg.get("auto_enrich_filters", False)),
    }

    capped = over_round_cap(uses, round_caps(cfg))

    def run_one(index: int) -> dict[str, Any]:
        use = uses[index]
        skip_reason = capped.get(index, "")
        if skip_reason:
            pass
        elif use.name not in allowed:
            skip_reason = f"Tool {use.name} is not available in this phase."
        elif invalid_mixed_ranking and use.name == "submit_ranking":
            skip_reason = RANKING_MUST_BE_ALONE
        if skip_reason:
            return {"ok": False, "error": skip_reason}
        return dispatch_tool(
            session,
            searcher,
            use.name,
            use.arguments,
            **dispatch_kwargs,
        )

    payloads: list[dict[str, Any] | None] = [None] * len(uses)
    parallel_idxs = [
        i for i, use in enumerate(uses) if use.name in RETRIEVAL_TOOLS
    ]
    sequential_idxs = [
        i for i in range(len(uses)) if i not in set(parallel_idxs)
    ]
    if parallel_idxs:
        workers = max(1, len(parallel_idxs))
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_one, i): i for i in parallel_idxs}
            for future, index in futures.items():
                payloads[index] = future.result()
        session.add_retrieval_wait(time.monotonic() - started)
    for index in sequential_idxs:
        payloads[index] = run_one(index)
    evidence_documents = prepare_retrieval_results(uses, payloads, parallel_idxs)

    blocks: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    for use, payload in zip(uses, payloads):
        assert payload is not None
        blocks.append(_tool_result_block(use.tool_use_id, payload))
        log.append(
            {
                "name": use.name,
                "tool_use_id": use.tool_use_id,
                "ok": bool(payload.get("ok", True)),
                "arguments": use.arguments,
            }
        )
    if parallel_idxs:
        blocks.append(
            {
                "text": RETRIEVED_CHUNKS_PREFIX
                + json.dumps(evidence_documents, ensure_ascii=False, indent=2)
            }
        )
    return blocks, log


def run_agent_loop(
    *,
    session: EvalSession,
    searcher: SearcherLike,
    query: str,
    generate: GenerateFn,
    cfg: dict[str, Any],
    system_prompt: str,
    final_ranking_system_prompt: str | None = None,
    transcript_path: Path | None = None,
) -> LoopResult:
    max_rounds = int(cfg.get("max_rounds") or (int(cfg.get("max_searches") or 20) + 4))
    timeout_s = int(cfg.get("timeout_s") or 900)
    started = time.monotonic()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"text": f"USER_QUERY:\n{query.strip()}"}]}
    ]
    usage_in = 0
    usage_out = 0
    usage_cache_read = 0
    usage_cache_write = 0
    ranking_in = 0
    ranking_out = 0
    ranking_cache_read = 0
    ranking_cache_write = 0
    last_text = ""
    forced_ranking = False
    n_turns = 0
    final_ranking_attempts = 0
    max_final_ranking_attempts = int(cfg.get("max_final_ranking_attempts") or 2)
    trace: list[dict[str, Any]] = []
    ranking_review_messages: list[dict[str, Any]] = []
    input_usd = float(cfg.get("input_usd_per_mtok") or 0.0)
    output_usd = float(cfg.get("output_usd_per_mtok") or 0.0)
    cache_read_mult = float(cfg.get("cache_read_multiplier") or 0.1)
    cache_write_mult = float(cfg.get("cache_write_multiplier") or 1.25)

    def record_turn(kind: str, turn: ModelTurn, tool_log: list[dict[str, Any]]) -> None:
        nonlocal usage_in, usage_out, usage_cache_read, usage_cache_write
        nonlocal ranking_in, ranking_out, ranking_cache_read, ranking_cache_write
        nonlocal last_text, n_turns
        n_turns += 1
        usage_in += turn.input_tokens
        usage_out += turn.output_tokens
        usage_cache_read += turn.cache_read_tokens
        usage_cache_write += turn.cache_write_tokens
        ranking_in += turn.input_tokens
        ranking_out += turn.output_tokens
        ranking_cache_read += turn.cache_read_tokens
        ranking_cache_write += turn.cache_write_tokens
        last_text = turn.text or last_text
        trace.append(
            {
                "turn": n_turns,
                "kind": kind,
                "stop_reason": turn.stop_reason,
                "text": turn.text,
                "tools": tool_log,
                "input_tokens": turn.input_tokens,
                "output_tokens": turn.output_tokens,
                "cache_read_tokens": turn.cache_read_tokens,
                "cache_write_tokens": turn.cache_write_tokens,
            }
        )

    def one_generate(
        *,
        kind: str,
        tools: list[str],
        tool_choice: str | None = None,
        extra_user: str | None = None,
    ) -> ModelTurn:
        if extra_user:
            append_user_notice(messages, extra_user)
        if timeout_s - (time.monotonic() - started) <= 1:
            raise TimeoutError(f"timeout after {timeout_s}s")
        turn = generate(
            messages=messages,
            system=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
        )
        messages.append(ensure_nonempty_message(turn.assistant_message))
        blocks, tool_log = execute_turn_tools(
            turn.tool_uses,
            session=session,
            searcher=searcher,
            cfg=cfg,
            allowed=set(tools),
        )
        if not blocks:
            blocks = [{"text": "Continue. Use a tool. Do not answer in plain text."}]
        messages.append({"role": "user", "content": blocks})
        record_turn(kind, turn, tool_log)
        return turn

    def one_final_ranking_review() -> ModelTurn:
        if timeout_s - (time.monotonic() - started) <= 1:
            raise TimeoutError(f"timeout after {timeout_s}s")
        review_messages = [
            {
                "role": "user",
                "content": [{"text": ranking_review_prompt(query, session)}],
            }
        ]
        turn = generate(
            messages=review_messages,
            system=final_ranking_system_prompt or system_prompt,
            tools=["submit_ranking"],
            tool_choice="submit_ranking",
        )
        blocks, tool_log = execute_turn_tools(
            turn.tool_uses,
            session=session,
            searcher=searcher,
            cfg=cfg,
            allowed={"submit_ranking"},
        )
        ranking_review_messages.extend(
            [
                review_messages[0],
                ensure_nonempty_message(turn.assistant_message),
                {
                    "role": "user",
                    "content": blocks
                    or [{"text": "Final ranking was not submitted; try again."}],
                },
            ]
        )
        record_turn("final_ranking_review", turn, tool_log)
        return turn

    try:
        while n_turns < max_rounds and not session.ranking():
            if session.ranking_draft():
                if final_ranking_attempts >= max_final_ranking_attempts:
                    break
                final_ranking_attempts += 1
                one_final_ranking_review()
                continue
            if max_rounds - n_turns <= 2:
                forced_ranking = True
                one_generate(
                    kind="force_draft_ranking",
                    tools=["submit_ranking"],
                    tool_choice="submit_ranking",
                )
                continue
            seen = bool(session.seen_chunk_map())
            one_generate(
                kind="search",
                tools=phase_tool_names(include_ranking=seen),
            )
        if session.ranking():
            status = "completed"
            error = ""
        elif session.ranking_draft():
            status = "no_submit"
            error = "final submit_ranking pass was not called"
        else:
            status = "no_submit"
            error = "draft submit_ranking pass was not called"
    except TimeoutError as exc:
        status = "timeout"
        error = str(exc)
    except Exception as exc:
        status = "error"
        error = str(exc)

    if transcript_path is not None:
        atomic_write_json(
            transcript_path,
            transcript_json_safe(
                {
                    "messages": messages,
                    "ranking_review_messages": ranking_review_messages,
                    "turns": trace,
                    "status": status,
                    "error": error,
                }
            ),
        )

    cost_kwargs = {
        "input_usd_per_mtok": input_usd,
        "output_usd_per_mtok": output_usd,
        "cache_read_multiplier": cache_read_mult,
        "cache_write_multiplier": cache_write_mult,
    }
    return LoopResult(
        status=status,
        error=error,
        n_turns=n_turns,
        forced_ranking=forced_ranking,
        input_tokens=usage_in,
        output_tokens=usage_out,
        cache_read_tokens=usage_cache_read,
        cache_write_tokens=usage_cache_write,
        ranking_input_tokens=ranking_in,
        ranking_output_tokens=ranking_out,
        ranking_cache_read_tokens=ranking_cache_read,
        ranking_cache_write_tokens=ranking_cache_write,
        cost_usd=estimate_cost_usd(
            input_tokens=usage_in,
            output_tokens=usage_out,
            cache_read_tokens=usage_cache_read,
            cache_write_tokens=usage_cache_write,
            **cost_kwargs,
        ),
        ranking_cost_usd=estimate_cost_usd(
            input_tokens=ranking_in,
            output_tokens=ranking_out,
            cache_read_tokens=ranking_cache_read,
            cache_write_tokens=ranking_cache_write,
            **cost_kwargs,
        ),
        text=last_text,
    )


def check_bedrock(cfg: dict[str, Any]) -> dict[str, str]:
    """Cheap credential check: STS caller identity, not a model call."""
    import boto3

    kwargs = aws_session_kwargs(cfg)
    region = kwargs.get("region_name") or "us-east-1"
    session = boto3.Session(**kwargs)
    sts = session.client("sts", region_name=region)
    ident = sts.get_caller_identity()
    return {
        "account": str(ident.get("Account") or ""),
        "arn": str(ident.get("Arn") or ""),
        "region": region,
        "model": str(cfg.get("model") or ""),
    }
