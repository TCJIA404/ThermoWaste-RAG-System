"""Fast offline self-test: does not need LangGraph, network, or GLM."""
from types import SimpleNamespace
from agent.citation_audit import audit_messages, citation_keys


def msg(content, name=None, tool_calls=None):
    return SimpleNamespace(content=content, name=name, tool_calls=tool_calls or [])


def test_found_and_unseen():
    messages = [
        msg("calling tools", tool_calls=[{"name": "search_papers"}, {"name": "search_papers"}]),
        msg("Search query: effect of temperature\n[1] (paper07.pdf, p.7) Some evidence.\n[2] (paper08.pdf, p.6) More evidence.", name="search_papers"),
        msg("Search query: steam/fuel\n[1] (paper07.pdf, p.10) SHAP example.", name="search_papers"),
        msg("See (paper07.pdf, p.7), (paper08.pdf, p.6), (paper55.pdf, p.2), and (paper07.pdf, p.7)."),
    ]
    r = audit_messages(messages, 1)
    assert r["tool_calls"] == 2
    assert r["cited_unique_pages"] == 3 and r["citation_occurrences"] == 4
    assert r["retrieved_citations"] == 2 and r["unseen_citations"] == 1
    assert r["retrieved_unique_pages"] == 3


def test_rag_candidates_are_not_direct_evidence():
    messages = [
        msg("RAG answer: ...\nSource candidates: (paper09.pdf, p.9)", name="ask_rag"),
        msg("(paper09.pdf, p.9)"),
    ]
    r = audit_messages(messages, 1)
    assert r["rag_only_citations"] == 1 and r["retrieved_citations"] == 0


def test_zero_citations():
    r = audit_messages([msg("No paper citations")], 0)
    assert r["cited_unique_pages"] == 0
    assert citation_keys("(paper07.pdf, p.7); (paper07.pdf，p.10)") == [("paper07.pdf", 7), ("paper07.pdf", 10)]


if __name__ == "__main__":
    test_found_and_unseen()
    test_rag_candidates_are_not_direct_evidence()
    test_zero_citations()
    print("[PASS] citation provenance parsing")
    print("[PASS] unsupported citations are flagged")
    print("[PASS] RAG source candidates not counted as direct search evidence")
    print("[PASS] no-citation and repeated-citation cases")
    print("All offline audit tests passed.")
