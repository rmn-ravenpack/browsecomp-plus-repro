"""Detect whether document text is English before requesting Bigdata translation.

Uses a Unicode-script heuristic (catches Arabic, CJK, Cyrillic, etc.) plus
langdetect on a body sample. Frontmatter is ignored so English YAML headers
do not hide a non-English body.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0

FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
LATIN_NAME_RE = re.compile(r"\bLATIN\b")

# If this share of alphabetic characters is non-Latin, treat as non-English.
NON_LATIN_RATIO_THRESHOLD = 0.12
# langdetect P(en) below this (with Latin-script text) -> request translation.
# 0.85 catches the 0.714 bin (e.g. French bodies mis-scored as weak English).
ENGLISH_PROB_THRESHOLD = 0.85
SAMPLE_CHARS = 6000


@dataclass(frozen=True)
class LanguageDecision:
    language: str
    is_english: bool
    confidence: float
    reason: str

    @property
    def request_translation(self) -> bool:
        return not self.is_english


def strip_frontmatter(text: str) -> str:
    if not text:
        return ""
    return FRONTMATTER_RE.sub("", text, count=1)


def _sample_body(text: str, max_chars: int = SAMPLE_CHARS) -> str:
    body = strip_frontmatter(text).strip()
    if len(body) <= max_chars:
        return body
    chunk = max(max_chars // 3, 1)
    mid = max((len(body) - chunk) // 2, 0)
    end = max(len(body) - chunk, 0)
    return "\n".join((body[:chunk], body[mid : mid + chunk], body[end : end + chunk]))


def _script_counts(sample: str) -> tuple[int, int]:
    latin = 0
    other = 0
    for ch in sample:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if LATIN_NAME_RE.search(name):
            latin += 1
        else:
            other += 1
    return latin, other


def detect_language(text: str) -> LanguageDecision:
    sample = _sample_body(text or "")
    if not sample:
        return LanguageDecision("en", True, 1.0, "empty")

    latin, other = _script_counts(sample)
    letters = latin + other
    if letters >= 20:
        non_latin_ratio = other / letters
        if non_latin_ratio >= NON_LATIN_RATIO_THRESHOLD:
            return LanguageDecision(
                language="und",
                is_english=False,
                confidence=min(1.0, non_latin_ratio),
                reason=f"non_latin_ratio={non_latin_ratio:.2f}",
            )

    try:
        langs = detect_langs(sample)
    except LangDetectException:
        return LanguageDecision("en", True, 0.0, "langdetect_failed")

    if not langs:
        return LanguageDecision("en", True, 0.0, "langdetect_empty")

    top = langs[0]
    en_prob = 0.0
    for item in langs:
        if item.lang == "en":
            en_prob = float(item.prob)
            break

    if top.lang == "en" and en_prob >= ENGLISH_PROB_THRESHOLD:
        return LanguageDecision("en", True, en_prob, "langdetect_en")

    return LanguageDecision(
        language=top.lang,
        is_english=False,
        confidence=float(top.prob),
        reason=f"langdetect_{top.lang}",
    )
