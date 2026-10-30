"""Bedrock Converse tool schemas for the ranking agent loop."""

from __future__ import annotations

from typing import Any

from session import MAX_RANKED_DOCUMENTS

SEARCH_TOOL_DESCRIPTION = """
Semantic search over chunk text. Returns up to 20 ranked chunks.

One fact per call, as a natural question (unknown entity: put which/who/what/
where/when in the interrogative) or a plain sentence (a scene you can describe).
Not a keyword list. Quotes have no effect; an exact phrase is a grep.

Matching is one chunk at a time so you can not find multiple things in one call.
""".strip()


GREP_TOOL_DESCRIPTION = """
Exact lexical search for a known name, title, or literal.

Quoted words must be adjacent and in order. Unquoted words AND anywhere in one
chunk, so extra tokens throw away pages. Send one literal at a time; start with
the rarest string. | is OR variants. Not regex. Never send a bare stopword.
""".strip()


GET_DOCUMENT_TOOL_DESCRIPTION = """
Read more chunks of a document id already returned by search or grep. Use it
when shown chunks make the document promising. Search shows only a few chunks,
so a candidate's own page is mostly unread. Do not use it to reread shown text.
""".strip()


SUBMIT_RANKING_TOOL_DESCRIPTION = """
Submit a ranked list of retrieved documents relevant to the original question.

Use only shown document ids. Do not invent or duplicate ids. Ranking is by
relevance_score (0 to 1), not array order.

This is an evidence ranking, not permission to guess. Include every document
that supports identification or any material constraint, names a candidate,
supplied a clue value, or rules a candidate out — even if it does not name the
answer. Exclude only documents irrelevant to every constraint.

Ten slots and no penalty for filling them. Uncertain documents go last, not
out. An unresolved question is not a reason to rank one document.
""".strip()


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "name": "search",
            "description": SEARCH_TOOL_DESCRIPTION,
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text"],
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "One fact as a natural question when the entity is "
                            "unknown, or as a plain sentence when the scene is "
                            "concrete. Not the whole question, not a keyword "
                            "bundle, and quotes have no effect here."
                        ),
                    },
                },
            },
        },
        {
            "name": "grep",
            "description": GREP_TOOL_DESCRIPTION,
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["pattern"],
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Quoted name, title, or phrase to find in one "
                            "chunk. Spaces are AND; | separates OR variants."
                        ),
                    },
                },
            },
        },
        {
            "name": "get_document",
            "description": GET_DOCUMENT_TOOL_DESCRIPTION,
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["docid"],
                "properties": {
                    "docid": {
                        "type": "string",
                        "description": (
                            "Valid document id copied verbatim from a previous search "
                            "or grep result."
                        ),
                    },
                },
            },
        },
        {
            "name": "submit_ranking",
            "description": SUBMIT_RANKING_TOOL_DESCRIPTION,
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ranking_strategy", "documents"],
                "properties": {
                    "ranking_strategy": {
                        "type": "string",
                        "description": (
                            "State how the evidence identifies the candidate, "
                            "which document supports each constraint, and any "
                            "constraint that remains unsupported. Do not infer "
                            "missing support."
                        ),
                    },
                    "documents": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_RANKED_DOCUMENTS,
                        "description": (
                            "Up to 10 retrieved documents that directly support "
                            "identification or any material constraint. Put "
                            "identity and decisive evidence first, followed by "
                            "other constraint evidence and corroboration. Exclude "
                            "only documents irrelevant to every constraint. Extra "
                            "entries beyond 10 are dropped."
                        ),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["docid", "relevance_score"],
                            "properties": {
                                "docid": {
                                    "type": "string",
                                    "description": (
                                        "Relevant document id copied from a previous "
                                        "search, grep, or get_document result."
                                    ),
                                },
                                "relevance_score": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                    "description": (
                                        "Your assessed relevance from 0 to 1. This "
                                        "score determines rank and should reflect "
                                        "the document's evidentiary contribution "
                                        "to the whole question."
                                    ),
                                },
                            },
                        },
                    },
                },
            },
        },
    ]


def schemas_by_name() -> dict[str, dict[str, Any]]:
    return {tool["name"]: tool for tool in tool_schemas()}


def phase_tool_names(*, include_ranking: bool = False) -> list[str]:
    """Tools the model is allowed to see in this phase."""
    names = ["search", "grep", "get_document"]
    if include_ranking:
        names.append("submit_ranking")
    return names


def bedrock_tool_config(
    names: list[str],
    *,
    tool_choice: str | None = None,
) -> dict[str, Any]:
    """Bedrock Converse toolConfig for the named tools."""
    by_name = schemas_by_name()
    tools = []
    for name in names:
        schema = by_name[name]
        tools.append(
            {
                "toolSpec": {
                    "name": schema["name"],
                    "description": schema["description"],
                    "inputSchema": {"json": schema["inputSchema"]},
                }
            }
        )
    config: dict[str, Any] = {"tools": tools}
    if tool_choice:
        config["toolChoice"] = {"tool": {"name": tool_choice}}
    else:
        config["toolChoice"] = {"auto": {}}
    return config
