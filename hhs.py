"""Test cases for the lookup_search / specific_search / summarize_search
router, plus a runner that calls jev and checks it against expected
labels.

Three groups per category:
  - clear-cut: should be obvious, if these fail the category
    description itself needs fixing, not the model.
  - edge cases: deliberately ambiguous, sitting on the boundary between
    two categories. These are the ones actually worth watching — a
    classifier that nails the clear-cut cases but fails these still
    isn't reliable in production.

Fill in the `client.system_one(...)` call in run_tests() to match your
actual client import/instantiation, then run this file directly.
"""

from __future__ import annotations

# from your_client_module import client   # <- wire up your actual client
# from your_choice_module import Choice

TEST_CASES = [
    # ---------------------------------------------------------------
    # lookup_search — named person, matched by ID/email/phone/name
    # ---------------------------------------------------------------
    {"query": "what is FINEMP1042's salary", "expected": "lookup_search"},
    {"query": "find isha.chowdhury@fintechco.com", "expected": "lookup_search"},
    {"query": "who is Vihaan Desai", "expected": "lookup_search"},
    {"query": "what is the phone number for employee FINEMP1071",
        "expected": "lookup_search"},
    {"query": "show me the record for Vivaan Verma", "expected": "lookup_search"},
    # edge cases
    {
        "query": "find the employee id of the person who joined in 2020 and has a salary greater than 100000",
        "expected": "lookup_search",  # names no specific person — arguably NOT lookup_search;
        # see note below, this is the query you originally tested with jev
    },
    {
        "query": "what's my manager's email address",
        # "my" implies a specific person (the asker), even with no
        "expected": "lookup_search",
        # name/ID given explicitly — tests whether jev requires an EXPLICIT identifier
    },
    {
        "query": "list everyone in the Finance department",
        # plural/all-records, not ONE named person — should NOT
        "expected": "specific_search",
        # be lookup_search despite being about "people"
    },

    # ---------------------------------------------------------------
    # specific_search — single fact/number/date/policy detail, no identity
    # ---------------------------------------------------------------
    {"query": "what is the leave policy", "expected": "specific_search"},
    {"query": "what was Q1 2024 revenue", "expected": "specific_search"},
    {"query": "when is the reimbursement deadline", "expected": "specific_search"},
    {"query": "how many sick leave days do employees get",
        "expected": "specific_search"},
    {"query": "what is the gross margin for 2024", "expected": "specific_search"},
    # edge cases
    {
        "query": "how was the previous year company turnover",
        # given verbatim in your own criteria text — sanity check
        "expected": "specific_search",
    },
    {
        "query": "what is the dress code on Fridays",
        # policy detail, but phrased almost like a "why/how" question —
        "expected": "specific_search",
        # tests whether jev over-triggers summarize_search on question words alone
    },
    {
        "query": "summarize the reimbursement policy",
        # contains the word "summarize" but is a straight single-topic
        "expected": "specific_search",
        # lookup, not a personal/indirect question — tests literal keyword bias
    },

    # ---------------------------------------------------------------
    # summarize_search — indirect/personal "why" questions, vocabulary
    # mismatch vs. the policy doc's own wording
    # ---------------------------------------------------------------
    {"query": "why is my leave request getting rejected",
        "expected": "summarize_search"},
    {"query": "why haven't I got my travel money back yet",
        "expected": "summarize_search"},
    {"query": "why was my reimbursement denied", "expected": "summarize_search"},
    {"query": "why hasn't my salary been credited this month",
        "expected": "summarize_search"},
    {"query": "why can't I book more than 2 work from home days",
        "expected": "summarize_search"},
    # edge cases
    {
        "query": "why is the Q3 vendor cost higher than Q2",
        # "why" wording, but about company-wide FACTS/numbers,
        "expected": "specific_search",
        # not the asker's personal situation — tests whether jev over-triggers
        # summarize_search on the word "why" alone
    },
    {
        "query": "my leave request was rejected, what should I do",
        "expected": "summarize_search",  # no "why", but same personal/indirect shape —
        # tests whether jev requires the literal word "why"
    },
    {
        "query": "is there a penalty for being late to work repeatedly",
        # personal-situation flavored but no "my"/"I" pronoun at all —
        "expected": "summarize_search",
        # tests whether jev needs first-person language explicitly
    },
]


def run_tests():
    """Call jev on every test case and compare to the expected label."""
    results = []
    for case in TEST_CASES:
        # response = client.system_one(
        #     model="typesafe/jev-1.13",
        #     state={"question": case["query"]},
        #     questions={"category": Choice(instructions=case["query"], criteria={...})},
        # )
        # predicted = response["category"]  # adjust to however jev's response is shaped
        predicted = None  # <- wire this up
        correct = predicted == case["expected"]
        results.append({**case, "predicted": predicted, "correct": correct})

    total = len(results)
    correct_count = sum(r["correct"] for r in results)
    print(
        f"Overall: {correct_count}/{total} correct ({correct_count/total:.0%})\n")

    print(f"{'query':<70}{'expected':<18}{'predicted':<18}{'ok'}")
    print("-" * 110)
    for r in results:
        mark = "✓" if r["correct"] else "✗"
        print(
            f"{r['query'][:68]:<70}{r['expected']:<18}{str(r['predicted']):<18}{mark}")

    # Per-category breakdown — useful to see if one category is
    # systematically weaker than the others.
    by_category: dict[str, list[bool]] = {}
    for r in results:
        by_category.setdefault(r["expected"], []).append(r["correct"])
    print("\nPer-category accuracy:")
    for category, outcomes in by_category.items():
        acc = sum(outcomes) / len(outcomes)
        print(f"  {category:<18}{sum(outcomes)}/{len(outcomes)} ({acc:.0%})")


if __name__ == "__main__":
    run_tests()
