"""Shared paths for the BrowseComp-Plus eval package."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
PKG_DIR = EVAL_DIR.parent
REPO_DIR = PKG_DIR.parent

if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

DATA_DIR = PKG_DIR / "data"
EVAL_DATA_DIR = DATA_DIR / "eval"
STATE_PATH = DATA_DIR / "upload_state.sqlite"
MAPPING_JSON = DATA_DIR / "docid_content_id_map.json"
DECRYPTED_JSONL = EVAL_DATA_DIR / "browsecomp_plus_decrypted.jsonl"
AGENT_QUERIES_JSONL = EVAL_DATA_DIR / "agent_queries.jsonl"
QREL_EVIDENCE = EVAL_DATA_DIR / "qrel_evidence.txt"
QREL_GOLD = EVAL_DATA_DIR / "qrel_gold.txt"
PROMPT_PATH = EVAL_DIR / "agent_prompt.txt"
FINAL_RANKING_PROMPT_PATH = EVAL_DIR / "final_ranking_prompt.txt"
CONFIG_PATH = EVAL_DIR / "eval_config.json"
RUNS_DIR = EVAL_DATA_DIR / "runs"

EXPECTED_QUERY_COUNT = 830
EXPECTED_CORPUS_COUNT = 100_195


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or CONFIG_PATH
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid eval config: {cfg_path}")
    for key, minimum, maximum in (
        ("max_search_calls_per_round", 1, 20),
        ("max_grep_calls_per_round", 1, 20),
        ("max_fetch_calls_per_round", 1, 20),
        ("query_concurrency", 1, 20),
    ):
        if key not in data:
            continue
        value = data[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or (maximum is not None and value > maximum)
        ):
            requirement = (
                f"between {minimum} and {maximum}"
                if maximum is not None
                else f"at least {minimum}"
            )
            raise ValueError(f"{key} must be an integer {requirement}")
    if "reasoning_effort" in data:
        effort = data["reasoning_effort"]
        # Bedrock rejects an unknown effort with a ValidationException on the
        # first call, which would waste a whole run to learn about a typo.
        if effort not in ("", "low", "medium", "high"):
            raise ValueError(
                "reasoning_effort must be one of low, medium, high, or \"\" to "
                "leave the model default"
            )
    if not str(data.get("model") or "").strip():
        raise ValueError("model is required")
    for price_key in ("input_usd_per_mtok", "output_usd_per_mtok"):
        value = data.get(price_key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{price_key} must be a nonnegative number")
    return data


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def docids_from_field(field: Any) -> list[str]:
    out: list[str] = []
    if not field:
        return out
    if isinstance(field, list):
        for item in field:
            if isinstance(item, dict) and item.get("docid") is not None:
                out.append(str(item["docid"]))
            else:
                out.append(str(item))
        return out
    return [str(field)]


def load_jsonl(path: Path, *, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def default_python() -> str:
    venv = PKG_DIR / ".venv" / "bin" / "python"
    if venv.is_file():
        return str(venv)
    return sys.executable
