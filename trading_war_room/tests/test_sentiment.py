"""Sentiment scorer accuracy against a labelled set of realistic headlines.

Label: +1 bullish, -1 bearish, 0 neutral. A prediction counts as correct when
sign(score) matches, with |score| < 0.15 treated as neutral. The bar is a
direction-accuracy floor so changes to the lexicon cannot silently regress.
"""

import pytest

from warroom.models.schemas import Severity
from warroom.providers.sentiment import score_headline

LABELLED = [
    ("Apple beats estimates, raises guidance on iPhone demand", 1),
    ("Nvidia shares surge after record data center revenue", 1),
    ("Tesla misses estimates as margins slump", -1),
    ("AMD cuts guidance citing weak PC demand", -1),
    ("Microsoft upgraded to outperform at Morgan Stanley", 1),
    ("Intel downgraded to sell on foundry delays", -1),
    ("Rivian narrows loss, deliveries top forecasts", 1),
    ("Lucid loss widens as production stalls", -1),
    ("Moderna wins FDA approval for new vaccine", 1),
    ("FDA rejects Biogen drug application", -1),
    ("Company fails to win Pentagon contract", -1),
    ("Retailer beats estimates but cuts full-year outlook", -1),
    ("Shares rally despite revenue miss", 1),
    ("Bank misses estimates; still raises dividend", -1),  # the miss dominates
    ("Boeing plunges after FAA grounds jets", -1),
    ("Trading halted in XYZ pending news", -1),
    ("SEC investigation into accounting irregularities", -1),
    ("Firm files for Chapter 11 bankruptcy protection", -1),
    ("Chipmaker to be acquired at 40% premium", 1),
    ("Takeover talks collapse, deal collapses", -1),
    ("Stock offering of 10 million shares priced", -1),
    ("Company announces $5 billion share buyback", 1),
    ("Apple to present at investor conference next week", 0),
    ("Shares active in morning trade", 0),
    ("Earnings in line with estimates", 0),
    ("Company may explore sale of division, sources say", 0),
    ("Short report alleges fraud at EV maker", -1),
    ("Hedge funds pile in as short squeeze sends shares higher", 1),
    ("Biotech swings to profit for first time", 1),
    ("Airline issues profit warning on fuel costs", -1),
    ("Price target raised to $250 at Goldman", 1),
    ("Analyst cuts price target, keeps neutral rating", -1),
    ("CEO resigns amid probe", -1),
    ("Cloud outage hits customers for second day", -1),
    ("Semiconductor stocks climb as demand accelerates", 1),
    ("Automaker recalls 500,000 vehicles", -1),
    ("Company denies report of layoffs", 0),
    ("Not a beat: revenue misses consensus", -1),
    ("Retailer posts better-than-expected holiday sales", 1),
    ("Regulators approve merger without conditions", 1),
    ("EU fines tech giant over antitrust breach", -1),
    ("Streaming service raises dividend, boosts outlook", 1),
    ("Miner sinks on weaker copper prices", -1),
    ("Oil major meets estimates", 0),
    ("Fintech beats low expectations as losses persist", 0),
    ("Tariffs weigh on industrial outlook", -1),
]


def _direction(score: float) -> int:
    return 0 if abs(score) < 0.15 else (1 if score > 0 else -1)


def test_direction_accuracy_floor():
    wrong = []
    for text, label in LABELLED:
        s, _, hits = score_headline(text)
        if _direction(s) != label:
            wrong.append((text, label, s, hits))
    accuracy = 1 - len(wrong) / len(LABELLED)
    assert accuracy >= 0.85, f"accuracy {accuracy:.0%}; misses: {wrong}"


def test_never_confuses_strong_polarity():
    """A clearly bullish headline must never score bearish, and vice versa."""
    for text, label in LABELLED:
        s, _, _ = score_headline(text)
        if label != 0:
            assert _direction(s) != -label, (text, s)


@pytest.mark.parametrize("text,expected", [
    ("Company fails to win contract", -1),
    ("No FDA approval for drug", -1),
    ("Beats estimates but cuts guidance", -1),
    ("Shares rally despite revenue miss", 1),
    ("Narrows loss", 1),
])
def test_negation_and_contrast(text, expected):
    assert _direction(score_headline(text)[0]) == expected


def test_severity():
    assert score_headline("Trading halted in XYZ")[1] is Severity.HIGH
    assert score_headline("Analyst lifts price target")[1] is Severity.MEDIUM
    assert score_headline("Shares active in morning trade")[1] is Severity.LOW


def test_bounds():
    s, _, _ = score_headline("beats estimates raises guidance record revenue surges soars fda approval")
    assert -1 <= s <= 1
