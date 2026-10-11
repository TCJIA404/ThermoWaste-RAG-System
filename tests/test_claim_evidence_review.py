"""Pure-stdlib OFFLINE tests: no LangGraph, no API key, no Token use."""
import json
from types import SimpleNamespace

from agent.claim_evidence_review import (
    evidence_from_messages, extract_claims, judge_prompt,
    make_report, parse_judge_output, prepare_claims, thermochemistry_flags,
)


def msg(content, name=None):
    return SimpleNamespace(content=content, name=name)


def run_tests():
    messages = [
        msg("Search query: gasification temperature\n[1] (paper07.pdf, p.7) Gasification temperature ranges 450-800 C.\n[2] (paper08.pdf, p.6) Methane steam reforming is endothermic.", name="search_papers"),
        msg("Search query: steam fuel\n[1] (paper07.pdf, p.7) Steam fuel ratio 1.2-2.4 increases hydrogen volume fraction.", name="search_papers"),
        msg("Answer contains (paper07.pdf, p.7).")
    ]
    evidence = evidence_from_messages(messages)
    assert len(evidence[("paper07.pdf", 7)]) == 2
    assert evidence[("paper08.pdf", 6)][0].startswith("Methane")
    answer = (
        "- High steam/fuel ratio raised H2 fraction (paper07.pdf, p.7).\n"
        "- Forward water-gas shift is an endothermic reaction (paper08.pdf, p.6).\n"
        "- An unsupported value is 99% (paper99.pdf, p.1).\n"
        "- Another sentence without citations."
    )
    claims = extract_claims(answer, 10)
    assert len(claims) == 3
    prepared = prepare_claims(claims, evidence)
    assert prepared[0]["evidence"][0]["has_search_excerpt"]
    assert not prepared[2]["evidence"][0]["has_search_excerpt"]
    assert claims[1]["review_flags"]
    assert not thermochemistry_flags("Forward WGS is exothermic")
    prompt = judge_prompt(prepared)
    assert "UNTRUSTED" in prompt

    good = {"reviews": [
        {"id": "C01", "verdict": "SUPPORTED", "reason": "明确提到蒸汽比", "evidence_quote": "Steam fuel ratio"},
        {"id": "C02", "verdict": "INSUFFICIENT", "reason": "检索证据未涉及 WGS 热效应", "evidence_quote": ""},
        {"id": "C03", "verdict": "INSUFFICIENT", "reason": "未检索到引用", "evidence_quote": ""}
    ]}
    reviews = parse_judge_output('```json\n' + json.dumps(good, ensure_ascii=False) + '\n```', ["C01", "C02", "C03"])
    report = make_report(claims, evidence, reviews)
    assert report["claims_count"] == 3
    assert report["all_pages_available_count"] == 2
    assert report["claims"][1]["human_review"] == "PENDING"
    assert report["claims"][2]["llm_review"]["verdict"] == "INSUFFICIENT"
    try:
        parse_judge_output('{"reviews": []}', ["C01"])
        raise AssertionError("Invalid response unexpectedly accepted")
    except ValueError:
        pass

    print("[PASS] Parse source/page excerpts returned by tools")
    print("[PASS] Extract referenced scientific assertions and missing pages")
    print("[PASS] Flag possible WGS thermochemistry confusion")
    print("[PASS] Validate provisional LLM JSON and reject malformed outputs")
    print("[PASS] Save compact review with human status PENDING")
    print("All offline claim-review tests passed.")


if __name__ == "__main__":
    run_tests()
