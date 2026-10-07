"""Retrieval metrics for BrowseComp-Plus agent eval.

nDCG uses binary gains on the first 10 ranked docids, where the ranking is the
agent's chunk ranking collapsed to documents in first-seen order. Scoring uses
the ranking submitted after retrieval stops. Official agent Recall is the
fraction of evidence documents seen anywhere in the search trace.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any


def _as_set(ids: Iterable[str] | None) -> set[str]:
    return {str(item) for item in (ids or []) if str(item)}


def recall_at(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    rel = _as_set(relevant)
    if not rel:
        return None
    return len(set(ranked[:k]) & rel) / len(rel)


def ndcg_at(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> float | None:
    rel = _as_set(relevant)
    if not rel:
        return None
    deduped = list(dict.fromkeys(str(item) for item in ranked if str(item)))[:k]
    dcg = 0.0
    for i, docid in enumerate(deduped, start=1):
        if docid in rel:
            dcg += 1.0 / math.log2(i + 1)
    ideal_n = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


def set_recall(retrieved: Iterable[str], relevant: Iterable[str]) -> float | None:
    rel = _as_set(relevant)
    if not rel:
        return None
    return len(_as_set(retrieved) & rel) / len(rel)


def trace_oracle_ndcg_at(
    seen: Iterable[str],
    relevant: Iterable[str],
    k: int = 10,
) -> float | None:
    """Best possible nDCG if every relevant document in the trace ranked first."""
    rel = _as_set(relevant)
    return ndcg_at(sorted(_as_set(seen) & rel), rel, k)


def ranking_retention_at(
    ranked: Sequence[str],
    seen: Iterable[str],
    relevant: Iterable[str],
    k: int = 10,
) -> float | None:
    """Fraction of seen relevant documents retained in the submitted top-k."""
    seen_relevant = _as_set(seen) & _as_set(relevant)
    if not seen_relevant:
        return None
    return len(set(ranked[:k]) & seen_relevant) / len(seen_relevant)


def score_query(
    *,
    gold_docids: Iterable[str],
    evidence_docids: Iterable[str],
    ranked_docids: Sequence[str],
    seen_docids: Iterable[str],
    uploaded_docids: Iterable[str] | None = None,
    n_ranked_chunks: int = 0,
    k: int = 10,
) -> dict[str, Any]:
    gold = _as_set(gold_docids)
    evidence = _as_set(evidence_docids)
    uploaded = _as_set(uploaded_docids) if uploaded_docids is not None else None
    ranked = list(dict.fromkeys(str(item) for item in ranked_docids if str(item)))
    seen = list(dict.fromkeys(str(item) for item in seen_docids if str(item)))

    gold_indexed = gold & uploaded if uploaded is not None else gold
    evidence_indexed = evidence & uploaded if uploaded is not None else evidence
    seen_set = set(seen)

    return {
        "ranked_docids": ranked,
        "n_ranked": len(ranked),
        "n_ranked_chunks": int(n_ranked_chunks),
        "n_seen": len(seen),
        "gold_n": len(gold),
        "evidence_n": len(evidence),
        "gold_n_indexed": len(gold_indexed),
        "evidence_n_indexed": len(evidence_indexed),
        "gold_ndcg@10": ndcg_at(ranked, gold, k),
        "evidence_ndcg@10": ndcg_at(ranked, evidence, k),
        "gold_recall@10": recall_at(ranked, gold, k),
        "evidence_recall@10": recall_at(ranked, evidence, k),
        "gold_ndcg@10_indexed_only": ndcg_at(ranked, gold_indexed, k),
        "evidence_ndcg@10_indexed_only": ndcg_at(ranked, evidence_indexed, k),
        "gold_trace_recall": set_recall(seen, gold),
        "evidence_trace_recall": set_recall(seen, evidence),
        "gold_trace_recall_indexed_only": set_recall(seen, gold_indexed),
        "evidence_trace_recall_indexed_only": set_recall(seen, evidence_indexed),
        "gold_trace_oracle_ndcg@10": trace_oracle_ndcg_at(seen, gold, k),
        "evidence_trace_oracle_ndcg@10": trace_oracle_ndcg_at(seen, evidence, k),
        "gold_ranking_retention@10": ranking_retention_at(ranked, seen, gold, k),
        "evidence_ranking_retention@10": ranking_retention_at(
            ranked, seen, evidence, k
        ),
        "ranked_covers_seen_gold": (gold & seen_set) <= set(ranked),
        "ranked_covers_seen_evidence": (evidence & seen_set) <= set(ranked),
        "missing_gold": sorted(gold - seen_set),
        "missing_evidence": sorted(evidence - seen_set),
    }


def mean(values: Iterable[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)
