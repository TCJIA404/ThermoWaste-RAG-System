"""Deterministic citation provenance audit for ThermoWaste ResearchAgent.

Checks that (paperXX.pdf, p.N) citations in final output occur in the
FULL search_papers tool messages from the same LangGraph run.

This is NOT claim-level factual verification and cannot prove the text
of a retrieved page actually supports a claim.

Pure stdlib module; works with objects exposing .name, .content, .tool_calls.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Matches the filenames actually used in this project; no guessed PDF metadata.
CITATION = re.compile(
    r"(?<![\w./-])(?P<source>[A-Za-z0-9_.-]+\.pdf)\s*[,，]\s*p\.?\s*(?P<page>\d+)\b",
    flags=re.IGNORECASE,
)
EVIDENCE_LINE = re.compile(r"^\s*\[\d+\]\s*\((?P<source>[^)]+\.pdf),\s*p\.\s*(?P<page>\d+)\)\s*(?P<excerpt>.*)$", re.MULTILINE | re.IGNORECASE)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    # LangChain may occasionally return structured text blocks.
    if isinstance(value, list):
        return "\n".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in value)
    return str(value)


def _key(source: str, page: str) -> Tuple[str, int]:
    return source.strip().lower(), int(page)


def citation_keys(text: str) -> List[Tuple[str, int]]:
    """All citation keys found in a textual answer (order preserved)."""
    return [_key(m.group("source"), m.group("page")) for m in CITATION.finditer(text)]


def audit_messages(messages: Sequence[Any], tool_rounds: Optional[int] = None) -> Dict[str, Any]:
    """Audit the FULL LangGraph run messages, not the truncated terminal log."""
    if not messages:
        raise ValueError("Cannot audit empty message history")

    evidence: Dict[Tuple[str, int], Dict[str, Any]] = {}
    rag_candidates = set()
    count_calls = 0
    count_searches = 0
    count_rag = 0
    errors = []

    for message in messages:
        calls = getattr(message, "tool_calls", None) or []
        count_calls += len(calls)
        name = getattr(message, "name", None)
        if name not in ("search_papers", "ask_rag"):
            continue
        body = _text(getattr(message, "content", ""))
        if "TOOL_ERROR:" in body:
            errors.append(name)
        if name == "search_papers":
            count_searches += 1
            first_line = body.splitlines()[0] if body else ""
            query = first_line.removeprefix("Search query:").strip() if first_line.startswith("Search query:") else ""
            for match in EVIDENCE_LINE.finditer(body):
                key = _key(match.group("source"), match.group("page"))
                item = evidence.setdefault(key, {"source": match.group("source"), "page": int(match.group("page")), "queries": []})
                if query and query not in item["queries"]:
                    item["queries"].append(query)
        else:
            count_rag += 1
            if "Source candidates:" in body:
                candidates = body.split("Source candidates:", 1)[1]
                rag_candidates.update(citation_keys(candidates))

    answer = _text(getattr(messages[-1], "content", ""))
    cited = list(dict.fromkeys(citation_keys(answer)))
    statuses = []
    for key in cited:
        if key in evidence:
            status = "retrieved_passage"  # A passage exists, not proof of truth.
        elif key in rag_candidates:
            status = "rag_source_only"  # Candidate from /ask; no excerpt here.
        else:
            status = "not_seen_in_tools"
        statuses.append({"source": key[0], "page": key[1], "status": status,
                         "queries": evidence.get(key, {}).get("queries", [])})

    status_counts = Counter(record["status"] for record in statuses)
    return {
        "tool_calls": count_calls,
        "tool_rounds": tool_rounds,
        "search_tool_results": count_searches,
        "rag_tool_results": count_rag,
        "tool_errors": errors,
        "retrieved_unique_pages": len(evidence),
        "cited_unique_pages": len(statuses),
        "citation_occurrences": len(citation_keys(answer)),
        "retrieved_citations": status_counts["retrieved_passage"],
        "rag_only_citations": status_counts["rag_source_only"],
        "unseen_citations": status_counts["not_seen_in_tools"],
        "citations": statuses,
        "limitation": "Page presence only; not claim-level entailment, factual correctness, or citation-to-sentence verification."
    }


def display_report(report: Dict[str, Any]) -> None:
    print("\n===== Citation provenance audit =====")
    print("Tool calls: {} | rounds: {} | retrieved unique pages: {}".format(
        report["tool_calls"], report["tool_rounds"], report["retrieved_unique_pages"]))
    print("Citations: {} unique ({} occurrences)".format(report["cited_unique_pages"], report["citation_occurrences"]))
    labels = {"retrieved_passage": "FOUND_IN_SEARCH", "rag_source_only": "ONLY_RAG_SOURCE", "not_seen_in_tools": "NOT_RETRIEVED"}
    for row in report["citations"]:
        print("[{}] {}, p.{}".format(labels[row["status"]], row["source"], row["page"]))
    if not report["citations"]:
        print("[WARNING] No (paperXX.pdf, p.N) citations in final answer.")
    if report["tool_errors"]:
        print("[WARNING] Tool errors:", ", ".join(report["tool_errors"]))
    if report["unseen_citations"] or report["rag_only_citations"]:
        print("[ACTION] Investigate flagged citations; do not assert they were verified.")
    print("[NOTE] FOUND_IN_SEARCH only means that a cited page was returned by retrieval. ")
    print("       Scientific claims, quotations, and numerical values still need verification against the paper.")
