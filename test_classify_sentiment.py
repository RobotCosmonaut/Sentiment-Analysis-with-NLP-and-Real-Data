r"""
Partition (PT) unit tests for BBCTextScraper.classify_sentiment(self, text)
Based on input-domain partitioning (Tian, Ch. 8.2 and 9.1).

Input domain: {all strings} ∪ {None}
Scoring values (computed from x by VADER): pos_score(x), neg_score(x), compound(x)

Subsets:
  S1 ≡ {x : x = "" ∨ x = None}                                                  -> Neutral
  S2 ≡ {x : x ≠ "" ∧ pos_score(x) > 0.3 ∧ neg_score(x) > 0.3}                   -> Mixed
  S3 ≡ {x : x ≠ "" ∧ ¬(pos_score(x)>0.3 ∧ neg_score(x)>0.3) ∧ compound(x) ≥ 0.05}  -> Positive
  S4 ≡ {x : x ≠ "" ∧ ¬(pos_score(x)>0.3 ∧ neg_score(x)>0.3) ∧ compound(x) ≤ -0.05} -> Negative
  S5 ≡ {x : x ≠ "" ∧ ¬(pos_score(x)>0.3 ∧ neg_score(x)>0.3) ∧ -0.05 < compound(x) < 0.05} -> Neutral

Every test case follows the same two-step procedure:
  1. Membership: verify the test point actually lies in the intended subset.
  2. Output:     verify the method returns the expected sentiment for that subset.

-----------------------------------------------------------------------------
HOW TO RUN
-----------------------------------------------------------------------------
With a results file (YYYY-MM-DD-HHMMSS_Partition_Test_Results.txt):
  python test_classify_sentiment.py                 all subsets + partition check
  python test_classify_sentiment.py S2              one subset
  python test_classify_sentiment.py S1 S3 S5        several subsets
  python test_classify_sentiment.py partition       partition (MECE) check only

With pytest only (no results file):
  pytest test_classify_sentiment.py -v                          all
  pytest test_classify_sentiment.py -v -k TestS2                one subset
  pytest test_classify_sentiment.py -v -k "TestS1 or TestS4"    several subsets
-----------------------------------------------------------------------------
"""
import argparse
import os
import platform
import sys
from datetime import datetime
from unittest.mock import patch

import pytest

from SentimentAnalysisNLP import BBCTextScraper

scraper = BBCTextScraper()

EXPECTED = {"S1": "Neutral", "S2": "Mixed", "S3": "Positive",
            "S4": "Negative", "S5": "Neutral"}

DEFINITIONS = {
    "S1": 'S1 ≡ {x : x = "" ∨ x = None}',
    "S2": 'S2 ≡ {x : x ≠ "" ∧ pos_score(x) > 0.3 ∧ neg_score(x) > 0.3}',
    "S3": 'S3 ≡ {x : x ≠ "" ∧ ¬(pos_score(x)>0.3 ∧ neg_score(x)>0.3) ∧ compound(x) ≥ 0.05}',
    "S4": 'S4 ≡ {x : x ≠ "" ∧ ¬(pos_score(x)>0.3 ∧ neg_score(x)>0.3) ∧ compound(x) ≤ -0.05}',
    "S5": 'S5 ≡ {x : x ≠ "" ∧ ¬(pos_score(x)>0.3 ∧ neg_score(x)>0.3) ∧ -0.05 < compound(x) < 0.05}',
}

# Test class for each subset (used to select subsets from the command line)
SUBSET_CLASSES = {
    "S1": "TestS1EmptyText",
    "S2": "TestS2Mixed",
    "S3": "TestS3Positive",
    "S4": "TestS4Negative",
    "S5": "TestS5Neutral",
    "partition": "TestPartition",
}


# ---------------------------------------------------------------------------
# Subset membership predicates (written directly from the subset definitions)
# ---------------------------------------------------------------------------
def not_mixed(s):
    return not (s["pos"] > 0.3 and s["neg"] > 0.3)

def in_S1(x):    return x == "" or x is None
def in_S2(x, s): return x != "" and s["pos"] > 0.3 and s["neg"] > 0.3
def in_S3(x, s): return x != "" and not_mixed(s) and s["compound"] >= 0.05
def in_S4(x, s): return x != "" and not_mixed(s) and s["compound"] <= -0.05
def in_S5(x, s): return x != "" and not_mixed(s) and -0.05 < s["compound"] < 0.05

SCORED_PREDICATES = {"S2": in_S2, "S3": in_S3, "S4": in_S4, "S5": in_S5}


def subsets_of(x, scores):
    """Return every subset (S2-S5) whose condition the test point satisfies."""
    return [name for name, pred in SCORED_PREDICATES.items() if pred(x, scores)]


# ---------------------------------------------------------------------------
# Shared test procedures (each test case calls exactly one of these)
# ---------------------------------------------------------------------------
def record_common(record_property, subset, x, rationale):
    record_property("subset", subset)
    record_property("definition", DEFINITIONS[subset])
    record_property("input", repr(x))
    record_property("rationale", rationale)
    record_property("expected", EXPECTED[subset])


def check_S1(x, rationale, record_property):
    """S1 procedure: membership is checked on x itself; no scores are computed."""
    record_common(record_property, "S1", x, rationale)

    # 1. Membership
    assert in_S1(x), f"test point {x!r} is not in S1"
    record_property("membership", "x is empty or None -> in S1")

    # 2. Output: Neutral with the fixed default scores
    result = scraper.classify_sentiment(x)
    record_property("scores", result["scores"])
    record_property("actual", result["sentiment"])
    assert result["sentiment"] == EXPECTED["S1"]
    assert result["scores"] == {"compound": 0, "pos": 0, "neu": 1, "neg": 0}


def check_real_text(subset, x, rationale, record_property):
    """S2-S5 procedure using real text scored by VADER."""
    record_common(record_property, subset, x, rationale)
    record_property("score source", "VADER (real)")

    scores = scraper.sia.polarity_scores(x)
    record_property("scores", scores)

    # 1. Membership: in exactly the intended subset
    found = subsets_of(x, scores)
    record_property("membership", f"satisfies {found}")
    assert not in_S1(x), "test point must not be in S1"
    assert found == [subset], f"expected only {subset}, found {found}; scores={scores}"

    # 2. Output: correct label; VADER's scores are passed through unchanged
    result = scraper.classify_sentiment(x)
    record_property("actual", result["sentiment"])
    assert result["sentiment"] == EXPECTED[subset]
    assert result["scores"] == scores


def check_controlled_scores(subset, scores, rationale, record_property):
    """S2-S5 procedure with VADER mocked, so the score values are chosen exactly."""
    x = "non-empty placeholder text"
    record_common(record_property, subset, x, rationale)
    record_property("score source", "mocked (chosen score values)")
    record_property("scores", scores)

    # 1. Membership: in exactly the intended subset
    found = subsets_of(x, scores)
    record_property("membership", f"satisfies {found}")
    assert found == [subset], f"expected only {subset}, found {found}; scores={scores}"

    # 2. Output
    with patch.object(scraper.sia, "polarity_scores", return_value=scores):
        result = scraper.classify_sentiment(x)
    record_property("actual", result["sentiment"])
    assert result["sentiment"] == EXPECTED[subset]


# ===========================================================================
# S1: Empty Text -> Neutral
# ===========================================================================
class TestS1EmptyText:

    def test_S1_01_empty_string(self, record_property):
        check_S1("", "Empty string: the only string in S1 (page yielded no qualifying text).",
                 record_property)

    def test_S1_02_none(self, record_property):
        check_S1(None, "None: absence of a value, handled by the same check as the empty string.",
                 record_property)


# ===========================================================================
# S2: Mixed -> pos_score > 0.3 AND neg_score > 0.3
# ===========================================================================
class TestS2Mixed:

    def test_S2_01_strong_pos_and_neg_words(self, record_property):
        check_real_text("S2", "Great joy, terrible grief.",
                        "Two strong positive and two strong negative words; no neutral words.",
                        record_property)

    def test_S2_02_short_mixed_positive_compound(self, record_property):
        check_real_text("S2", "Love and hate.",
                        "Mixed with a positive compound: S2 is checked before the compound thresholds.",
                        record_property)

    def test_S2_03_short_mixed_negative_compound(self, record_property):
        check_real_text("S2", "Happy but sad.",
                        "Mixed with a negative compound: S2 is checked before the compound thresholds.",
                        record_property)

    def test_S2_04_controlled_positive_compound(self, record_property):
        check_controlled_scores(
            "S2", {"pos": 0.50, "neg": 0.40, "neu": 0.10, "compound": 0.20},
            "Chosen scores: compound alone would place x in S3, but S2 applies first.",
            record_property)

    def test_S2_05_controlled_negative_compound(self, record_property):
        check_controlled_scores(
            "S2", {"pos": 0.35, "neg": 0.55, "neu": 0.10, "compound": -0.40},
            "Chosen scores: compound alone would place x in S4, but S2 applies first.",
            record_property)


# ===========================================================================
# S3: Positive -> not Mixed AND compound >= 0.05
# ===========================================================================
class TestS3Positive:

    def test_S3_01_strongly_positive(self, record_property):
        check_real_text("S3", "The rescue was a wonderful success.",
                        "Strongly positive words only.", record_property)

    def test_S3_02_mildly_positive(self, record_property):
        check_real_text("S3", "It was fine.",
                        "Single mildly positive word; compound just above the threshold region.",
                        record_property)

    def test_S3_03_positive_diluted_by_neutral(self, record_property):
        check_real_text(
            "S3",
            "The committee met on Tuesday to review the budget report and the "
            "regional plan, which was slightly improved.",
            "Mostly neutral words (low pos_score) but positive compound, like real news text.",
            record_property)

    def test_S3_04_controlled_pos_high_neg_low(self, record_property):
        check_controlled_scores(
            "S3", {"pos": 0.40, "neg": 0.10, "neu": 0.50, "compound": 0.60},
            "pos_score > 0.3 but neg_score <= 0.3, so not Mixed.", record_property)

    def test_S3_05_controlled_positive_only(self, record_property):
        check_controlled_scores(
            "S3", {"pos": 0.60, "neg": 0.00, "neu": 0.40, "compound": 0.80},
            "No negative content at all.", record_property)


# ===========================================================================
# S4: Negative -> not Mixed AND compound <= -0.05
# ===========================================================================
class TestS4Negative:

    def test_S4_01_strongly_negative(self, record_property):
        check_real_text("S4", "The attack killed dozens.",
                        "Strongly negative words only.", record_property)

    def test_S4_02_negative_news_sentence(self, record_property):
        check_real_text("S4", "The flood destroyed homes and left families grieving.",
                        "Negative news-style sentence.", record_property)

    def test_S4_03_positive_word_but_negative_overall(self, record_property):
        check_real_text("S4", "The food was good but the service was terrible.",
                        "Contains a positive word, but pos_score <= 0.3 and compound is negative.",
                        record_property)

    def test_S4_04_controlled_neg_high_pos_low(self, record_property):
        check_controlled_scores(
            "S4", {"pos": 0.10, "neg": 0.40, "neu": 0.50, "compound": -0.60},
            "neg_score > 0.3 but pos_score <= 0.3, so not Mixed.", record_property)

    def test_S4_05_controlled_negative_only(self, record_property):
        check_controlled_scores(
            "S4", {"pos": 0.00, "neg": 0.60, "neu": 0.40, "compound": -0.80},
            "No positive content at all.", record_property)


# ===========================================================================
# S5: Neutral -> not Mixed AND -0.05 < compound < 0.05
# ===========================================================================
class TestS5Neutral:

    def test_S5_01_factual_sentence(self, record_property):
        check_real_text("S5", "The meeting is on Tuesday.",
                        "Factual statement with no sentiment words.", record_property)

    def test_S5_02_factual_news_sentence(self, record_property):
        check_real_text("S5", "The report was published on Monday.",
                        "News-style factual statement.", record_property)

    def test_S5_03_whitespace_only(self, record_property):
        check_real_text("S5", "   ",
                        "Whitespace-only: a non-empty string, so NOT in S1; scored by VADER as zero.",
                        record_property)

    def test_S5_04_controlled_zero_compound(self, record_property):
        check_controlled_scores(
            "S5", {"pos": 0.05, "neg": 0.05, "neu": 0.90, "compound": 0.00},
            "Balanced low scores with zero compound.", record_property)

    def test_S5_05_controlled_small_nonzero_compound(self, record_property):
        check_controlled_scores(
            "S5", {"pos": 0.04, "neg": 0.02, "neu": 0.94, "compound": 0.03},
            "Small non-zero compound inside the open interval (-0.05, 0.05).",
            record_property)


# ===========================================================================
# Partition check: S2-S5 are mutually exclusive and collectively exhaustive
# over the reachable score space (S1 is separated by the empty-input check).
# ===========================================================================
class TestPartition:

    def test_partition_mece_over_score_space(self, record_property):
        x = "non-empty"
        step = 0.05
        grid = [round(i * step, 2) for i in range(21)]              # 0.00 .. 1.00
        compounds = [round(-1 + i * step, 2) for i in range(41)]    # -1.00 .. 1.00
        checked = 0
        for pos in grid:
            for neg in grid:
                if pos + neg > 1:                                   # unreachable
                    continue
                for c in compounds:
                    s = {"pos": pos, "neg": neg,
                         "neu": round(1 - pos - neg, 2), "compound": c}
                    found = subsets_of(x, s)
                    assert len(found) == 1, f"point in {found}: {s}"
                    checked += 1
        record_property("subset", "S2-S5 (partition check)")
        record_property("points checked", checked)
        record_property("result", "every point lies in exactly one subset")


# ---------------------------------------------------------------------------
# Results file writer (used when this file is run with python)
# ---------------------------------------------------------------------------
class ResultsRecorder:
    """Minimal pytest plugin that collects each test's outcome."""

    def __init__(self):
        self.results = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when == "setup" and not report.passed):
            parts = report.nodeid.split("::")
            self.results.append({
                "group": parts[1] if len(parts) > 2 else "",
                "name": parts[-1],
                "outcome": report.outcome.upper(),
                "duration": report.duration,
                "properties": dict(report.user_properties),
                "message": report.longreprtext if report.failed else "",
            })


def write_results_file(recorder, started, selected):
    here = os.path.dirname(os.path.abspath(__file__))
    filename = f"{started.strftime('%Y-%m-%d-%H%M%S')}_Partition_Test_Results.txt"
    path = os.path.join(here, filename)

    counts = {}
    for r in recorder.results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1

    line = "=" * 78
    order = ["subset", "definition", "input", "rationale", "score source", "scores",
             "membership", "expected", "actual"]

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{line}\nPARTITION TEST RESULTS\n{line}\n")
        f.write(f"Run started:     {started.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Unit under test: BBCTextScraper.classify_sentiment(self, text)\n")
        f.write("Source file:     SentimentAnalysisNLP.py\n")
        f.write(f"Subsets run:     {', '.join(selected)}\n")
        f.write(f"Python:          {platform.python_version()}\n")
        f.write(f"Pytest:          {pytest.__version__}\n\n")
        f.write("Input domain: {all strings} ∪ {None}\n\nSubsets:\n")
        for key in ("S1", "S2", "S3", "S4", "S5"):
            f.write(f"  {DEFINITIONS[key]}  -> {EXPECTED[key]}\n")

        current_group = None
        for i, r in enumerate(recorder.results, 1):
            if r["group"] != current_group:
                current_group = r["group"]
                f.write(f"\n{line}\n{current_group}\n{line}\n")
            f.write(f"\n[{i}] {r['name']}\n")
            f.write(f"    Outcome:        {r['outcome']}  ({r['duration']:.3f}s)\n")
            props = r["properties"]
            keys = [k for k in order if k in props] + [k for k in props if k not in order]
            for key in keys:
                label = key[0].upper() + key[1:] + ":"
                f.write(f"    {label:<16}{props[key]}\n")
            if r["message"]:
                f.write("    Failure details:\n")
                for msg_line in r["message"].splitlines():
                    f.write(f"      {msg_line}\n")

        f.write(f"\n{line}\nSUMMARY\n{line}\n")
        groups = {}
        for r in recorder.results:
            g = groups.setdefault(r["group"], {"PASSED": 0, "FAILED": 0, "SKIPPED": 0})
            g[r["outcome"]] = g.get(r["outcome"], 0) + 1
        for g, c in groups.items():
            f.write(f"  {g:<18} passed {c['PASSED']:>2}   failed {c['FAILED']:>2}   "
                    f"skipped {c['SKIPPED']:>2}\n")
        f.write(f"\nTotal tests: {len(recorder.results)}\n")
        for outcome in ("PASSED", "FAILED", "SKIPPED"):
            f.write(f"{outcome.capitalize() + ':':<13}{counts.get(outcome, 0)}\n")

    return path


def main():
    parser = argparse.ArgumentParser(
        description="Run partition tests for classify_sentiment and write a results file.")
    parser.add_argument(
        "subsets", nargs="*", metavar="SUBSET",
        help="Subsets to run: S1 S2 S3 S4 S5 partition (default: all)")
    args = parser.parse_args()

    requested = [s if s.lower() == "partition" else s.upper() for s in args.subsets]
    requested = ["partition" if s.lower() == "partition" else s for s in requested]
    unknown = [s for s in requested if s not in SUBSET_CLASSES]
    if unknown:
        parser.error(f"unknown subset(s): {', '.join(unknown)}. "
                     f"Choose from: {', '.join(SUBSET_CLASSES)}")
    selected = requested or list(SUBSET_CLASSES)

    pytest_args = [os.path.abspath(__file__), "-v"]
    if requested:
        pytest_args += ["-k", " or ".join(SUBSET_CLASSES[s] for s in selected)]

    started = datetime.now()
    recorder = ResultsRecorder()
    exit_code = pytest.main(pytest_args, plugins=[recorder])
    results_path = write_results_file(recorder, started, selected)
    print(f"\nResults written to: {results_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
