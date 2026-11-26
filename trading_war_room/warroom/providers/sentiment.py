"""Dependency-free headline sentiment + severity scoring.

A deliberately simple lexicon scorer. It is not a language model and will
misread sarcasm, negation chains and context ("cuts losses" vs "cuts
guidance"). Its job is to be deterministic, fast and explainable so the News
agent can cite *why* it scored a headline. Phrases are matched before single
words so "beats estimates" is not double counted.
"""

from __future__ import annotations

import re

from warroom.models.schemas import Severity

_POS_PHRASES = {
    "beats estimates": 0.7, "beat estimates": 0.7, "raises guidance": 0.8, "raised guidance": 0.8,
    "record revenue": 0.6, "price target raised": 0.5, "upgraded to buy": 0.7, "strong demand": 0.4,
    "share buyback": 0.4, "fda approval": 0.8, "wins contract": 0.5, "better than expected": 0.6,
}
_NEG_PHRASES = {
    "misses estimates": -0.7, "missed estimates": -0.7, "cuts guidance": -0.8, "lowers guidance": -0.8,
    "price target cut": -0.5, "downgraded to sell": -0.7, "trading halted": -0.6, "sec investigation": -0.7,
    "worse than expected": -0.6, "recall": -0.5, "class action": -0.4, "going concern": -0.9,
}
_POS_WORDS = {
    "surge": 0.5, "surges": 0.5, "soar": 0.6, "soars": 0.6, "jumps": 0.4, "rally": 0.4, "rallies": 0.4,
    "upgrade": 0.5, "upgrades": 0.5, "beat": 0.4, "beats": 0.4, "record": 0.3, "growth": 0.2, "bullish": 0.4,
    "outperform": 0.4, "partnership": 0.3, "approval": 0.4, "profit": 0.2, "gains": 0.3,
}
_NEG_WORDS = {
    "plunge": -0.6, "plunges": -0.6, "tumble": -0.5, "tumbles": -0.5, "slump": -0.5, "falls": -0.3,
    "drops": -0.3, "downgrade": -0.5, "downgrades": -0.5, "miss": -0.4, "misses": -0.4, "lawsuit": -0.4,
    "probe": -0.4, "investigation": -0.4, "halt": -0.5, "halted": -0.5, "bearish": -0.4, "layoffs": -0.3,
    "weak": -0.3, "warning": -0.4, "loss": -0.2, "fraud": -0.8, "bankruptcy": -0.9, "delay": -0.3,
}
_HIGH_TRIGGERS = (
    "halt", "fda", "merger", "acquisition", "acquire", "bankruptcy", "fraud", "guidance", "sec ",
    "earnings", "fomc", "rate decision", "cpi", "payrolls", "going concern", "recall",
)
_MEDIUM_TRIGGERS = ("upgrade", "downgrade", "price target", "contract", "partnership", "analyst", "lawsuit", "probe")

_WORD_RE = re.compile(r"[a-z']+")


def score_headline(text: str) -> tuple[float, Severity, list[str]]:
    """Return (sentiment in [-1, 1], severity, matched terms)."""
    t = " " + text.lower() + " "
    total = 0.0
    hits: list[str] = []
    for table in (_POS_PHRASES, _NEG_PHRASES):
        for phrase, w in table.items():
            if phrase in t:
                total += w
                hits.append(phrase)
                t = t.replace(phrase, " ")
    for word in _WORD_RE.findall(t):
        w = _POS_WORDS.get(word) or _NEG_WORDS.get(word)
        if w:
            total += w
            hits.append(word)
    sentiment = max(-1.0, min(1.0, total))
    low = text.lower()
    if any(k in low for k in _HIGH_TRIGGERS):
        severity = Severity.HIGH
    elif any(k in low for k in _MEDIUM_TRIGGERS) or abs(sentiment) >= 0.6:
        severity = Severity.MEDIUM
    else:
        severity = Severity.LOW
    return round(sentiment, 3), severity, hits
