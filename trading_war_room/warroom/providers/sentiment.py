"""Dependency-free headline sentiment + severity scoring.

A deterministic lexicon scorer, measured against a labelled headline set in
``tests/test_sentiment.py`` (the test enforces a minimum direction accuracy,
so regressions are caught). It handles:

* **Phrase overrides first** - multi-word financial idioms whose meaning is
  not the sum of their words ("narrows loss" is good, "cuts guidance" bad,
  "cuts losses" good, "short squeeze" good, "beats low expectations" weak).
* **Negation** - "not", "no", "fails to", "misses on" ... flip the polarity
  of the next sentiment term within a 3-token window.
* **Contrast** - in "X but/despite/however Y" the clause after the pivot
  dominates (weights 0.4 / 1.0), matching how markets read such headlines.
* **Hedging** - "may", "could", "reportedly", "considering" damp the score.

It still cannot read sarcasm or world knowledge ("raises prices" may be good
or bad). Severity reflects *how much the headline can move the stock*, not
its direction.
"""

from __future__ import annotations

import re

from warroom.models.schemas import Severity

# Checked before word scoring; matched spans are removed so their words are not re-scored.
_PHRASES: dict[str, float] = {
    # positive
    "beats estimates": 0.7, "beat estimates": 0.7, "tops estimates": 0.7, "beats expectations": 0.7,
    "above expectations": 0.6, "better than expected": 0.6, "better-than-expected": 0.6,
    "raises guidance": 0.8, "raised guidance": 0.8, "raises outlook": 0.8, "boosts outlook": 0.8,
    "lifts forecast": 0.7, "raises forecast": 0.7, "record revenue": 0.6, "record quarter": 0.6,
    "price target raised": 0.5, "raises price target": 0.5, "upgraded to buy": 0.7, "upgraded to outperform": 0.6,
    "strong demand": 0.4, "share buyback": 0.4, "buyback program": 0.4, "fda approval": 0.8, "fda approves": 0.8,
    "wins contract": 0.5, "awarded contract": 0.5, "narrows loss": 0.4, "loss narrows": 0.4, "narrower loss": 0.4,
    "cuts losses": 0.3, "short squeeze": 0.5, "returns to profit": 0.6, "swings to profit": 0.6,
    "dividend increase": 0.4, "raises dividend": 0.4, "all-time high": 0.5, "record high": 0.5,
    "takeover bid": 0.6, "to be acquired": 0.7, "agrees to be acquired": 0.7, "buyout offer": 0.6,
    # negative
    "misses estimates": -0.7, "missed estimates": -0.7, "below expectations": -0.6, "misses expectations": -0.7,
    "worse than expected": -0.6, "worse-than-expected": -0.6, "cuts guidance": -0.8, "lowers guidance": -0.8,
    "cuts outlook": -0.8, "lowers outlook": -0.8, "slashes forecast": -0.8, "cuts forecast": -0.7,
    "withdraws guidance": -0.8, "price target cut": -0.5, "cuts price target": -0.5, "lowers price target": -0.5,
    "downgraded to sell": -0.7, "downgraded to underperform": -0.6, "downgraded to neutral": -0.4,
    "trading halted": -0.6, "sec investigation": -0.7, "doj probe": -0.7, "class action": -0.4,
    "going concern": -0.9, "chapter 11": -0.9, "files for bankruptcy": -0.9, "widens loss": -0.5,
    "loss widens": -0.5, "wider loss": -0.5, "swings to loss": -0.6, "dividend cut": -0.6, "cuts dividend": -0.6,
    "suspends dividend": -0.7, "profit warning": -0.7, "accounting irregularities": -0.8, "short report": -0.6,
    "secondary offering": -0.4, "stock offering": -0.4, "deal collapses": -0.6, "drops bid": -0.5,
    # weak / ambiguous constructions
    "beats low expectations": 0.15, "in line with estimates": 0.0, "in-line": 0.0, "meets estimates": 0.05,
}
# verb ... object with up to two words between ("cuts full-year outlook", "raises 2026 revenue guidance")
_PATTERNS: list[tuple[re.Pattern[str], float, str]] = [
    (re.compile(r"\b(cuts|cut|lowers|lowered|slashes|withdraws|trims|reduces)\s+(?:[\w\-$%.]+\s+){0,2}"
                r"(guidance|outlook|forecasts?)\b"), -0.8, "cuts outlook"),
    (re.compile(r"\b(raises|raised|boosts|lifts|hikes|ups|increases)\s+(?:[\w\-$%.]+\s+){0,2}"
                r"(guidance|outlook|forecasts?)\b"), 0.8, "raises outlook"),
]

_POS_WORDS: dict[str, float] = {
    "surge": 0.5, "surges": 0.5, "soar": 0.6, "soars": 0.6, "jumps": 0.4, "jump": 0.4, "rally": 0.4,
    "rallies": 0.4, "climbs": 0.3, "rises": 0.25, "gains": 0.3, "upgrade": 0.5, "upgrades": 0.5,
    "upgraded": 0.5, "beat": 0.4, "beats": 0.4, "tops": 0.3, "record": 0.3, "growth": 0.2, "bullish": 0.4,
    "outperform": 0.4, "partnership": 0.3, "approval": 0.4, "approves": 0.4, "approved": 0.4, "profit": 0.2,
    "strong": 0.25, "win": 0.3, "won": 0.3, "approve": 0.4, "higher": 0.2, "boost": 0.3, "boosts": 0.3, "accelerates": 0.3, "expands": 0.2, "wins": 0.3, "breakthrough": 0.5,
}
_NEG_WORDS: dict[str, float] = {
    "plunge": -0.6, "plunges": -0.6, "plummets": -0.7, "tumble": -0.5, "tumbles": -0.5, "slump": -0.5,
    "slumps": -0.5, "sinks": -0.5, "falls": -0.3, "drops": -0.3, "slides": -0.3, "declines": -0.3,
    "downgrade": -0.5, "downgrades": -0.5, "downgraded": -0.5, "miss": -0.4, "misses": -0.4, "missed": -0.4,
    "lawsuit": -0.4, "sued": -0.4, "probe": -0.4, "investigation": -0.4, "halt": -0.5, "halted": -0.5,
    "bearish": -0.4, "layoffs": -0.3, "weak": -0.3, "weaker": -0.3, "warning": -0.4, "warns": -0.4,
    "loss": -0.2, "fraud": -0.8, "bankruptcy": -0.9, "delay": -0.3, "delays": -0.3, "recall": -0.5,
    "recalls": -0.5, "crash": -0.6, "slashes": -0.4, "cut": -0.2, "cuts": -0.2, "resigns": -0.3, "subpoena": -0.5,
    "breach": -0.4, "rejects": -0.5, "rejected": -0.5, "losses": -0.2, "grounds": -0.4, "stalls": -0.3, "outage": -0.3, "fine": -0.3, "fined": -0.4, "ban": -0.4, "tariff": -0.2, "tariffs": -0.2,
}
_NEGATORS = {"not", "no", "never", "without", "fails", "failed", "fail", "unable"}
_DENIALS = {"denies", "denied", "refutes", "dismisses"}  # flip weakly: markets discount denials
_CONTRAST = re.compile(r"\b(but|however|despite|although|though|yet|while)\b|;|—| - ")
_HEDGES = {"may", "might", "could", "reportedly", "considering", "weighs", "explores", "rumor", "rumored", "possible"}

_HIGH_TRIGGERS = (
    "halt", "fda", "merger", "acquisition", "acquire", "acquired", "takeover", "buyout", "bankruptcy",
    "chapter 11", "fraud", "guidance", "outlook", "forecast", "sec ", "doj", "earnings", "fomc",
    "rate decision", "cpi", "payrolls", "going concern", "recall", "profit warning", "short report",
)
_MEDIUM_TRIGGERS = (
    "upgrade", "downgrade", "price target", "contract", "partnership", "analyst", "lawsuit", "probe",
    "offering", "dividend", "buyback", "layoffs", "resigns", "ceo",
)

_TOKEN_RE = re.compile(r"[a-z][a-z'\-]*")


def _score_clause(clause: str, hits: list[str]) -> float:
    text = f" {clause} "
    total = 0.0
    for pattern, w, label in _PATTERNS:
        m = pattern.search(text)
        if m:
            total += w
            hits.append(label)
            text = text[: m.start()] + " | " + text[m.end():]
    # phrases (longest first so "beats low expectations" wins over "beats")
    for phrase in sorted(_PHRASES, key=len, reverse=True):
        if f" {phrase} " in text or f" {phrase}," in text or f" {phrase}." in text:
            w = _PHRASES[phrase]
            # negated phrase: "fails to win contract" / "no fda approval"
            before = text.split(f" {phrase}", 1)[0].split()[-3:]
            if any(b in _NEGATORS for b in before):
                w = -w
                hits.append(f"not {phrase}")
            else:
                hits.append(phrase)
            total += w
            text = text.replace(f" {phrase}", " | ", 1)
    tokens = _TOKEN_RE.findall(text)
    negate_window = 0
    flip = 0.8
    for tok in tokens:
        if tok in _NEGATORS or tok in _DENIALS:
            negate_window = 3
            flip = 0.4 if tok in _DENIALS else 0.8
            continue
        w = _POS_WORDS.get(tok) or _NEG_WORDS.get(tok)
        if w:
            if negate_window > 0:
                w = -w * flip
                hits.append(f"not {tok}")
            else:
                hits.append(tok)
            total += w
            negate_window = 0
        elif negate_window:
            negate_window -= 1
    if any(t in _HEDGES for t in tokens):
        total *= 0.6
    return total


def score_headline(text: str) -> tuple[float, Severity, list[str]]:
    """Return (sentiment in [-1, 1], severity, explanation terms)."""
    low = text.lower().strip().replace(";", ",")  # semicolons juxtapose; both clauses weigh equally
    hits: list[str] = []
    parts = [p for p in _CONTRAST.split(low) if p and not _CONTRAST.fullmatch(p)]
    pivot = _CONTRAST.search(low)
    if pivot and len(parts) >= 2:
        is_despite = pivot.group(0).strip() in ("despite", "although", "though", "while")
        head, tail = parts[0], " ".join(parts[1:])
        # "X despite Y": X is the main clause. "X but Y": Y is the main clause.
        main, sub = (head, tail) if is_despite else (tail, head)
        total = _score_clause(main, hits) * 1.0 + _score_clause(sub, hits) * 0.4
    else:
        total = _score_clause(low, hits)
    sentiment = max(-1.0, min(1.0, total))

    if any(k in low for k in _HIGH_TRIGGERS):
        severity = Severity.HIGH
    elif any(k in low for k in _MEDIUM_TRIGGERS) or abs(sentiment) >= 0.6:
        severity = Severity.MEDIUM
    else:
        severity = Severity.LOW
    return round(sentiment, 3), severity, hits
