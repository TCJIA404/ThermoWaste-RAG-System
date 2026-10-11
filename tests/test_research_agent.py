"""Offline smoke test for research_agent.py. No GLM requests, no running FastAPI needed.
Run in the thermowaste-agent environment after installing dependencies:
    python test_research_agent.py
"""
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from research_agent import build_graph, search_papers, ask_rag


FAKE_HITS = {
    "query": "hydrogen production",
    "count": 2,
    "results": [
        {"source": "paper07.pdf", "page": 7, "text": "Hydrogen production depends on temperature.",
         "hybrid_score": 0.02, "rerank_score": 0.85},
        {"source": "paper09.pdf", "page": 9, "text": "Steam ratio also affects hydrogen production.",
         "hybrid_score": 0.01, "rerank_score": 0.75},
    ]
}


class FakeModel:
    """Issues a structured tool call, then writes an answer after observing tool output."""
    def bind_tools(self, tools):
        assert any(t.name == "search_papers" for t in tools)
        return self

    def invoke(self, messages):
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="Found evidence in (paper07.pdf, p.7) and (paper09.pdf, p.9).")
        return AIMessage(content="", tool_calls=[{
            "name": "search_papers",
            "args": {"query": "hydrogen production", "top_k": 2},
            "id": "call_offline_test",
            "type": "tool_call"
        }])


def test_tools():
    with patch("research_agent.request_api", return_value=FAKE_HITS) as mock_call:
        result = search_papers.invoke({"query": "hydrogen production", "top_k": 2})
        assert "paper07.pdf" in result and "p.7" in result, result
        assert "paper09.pdf" in result and "p.9" in result, result
        assert mock_call.call_args.args[0] == "/search"
    with patch("research_agent.request_api", return_value={
        "answer": "Temperature can affect H2 production.",
        "sources": [{"source": "paper07.pdf", "page": 7, "text": "..."}],
    }) as mock_call:
        answer = ask_rag.invoke({"query": "hydrogen production", "top_k": 1})
        assert "Temperature" in answer and "(paper07.pdf, p.7)" in answer
        assert mock_call.call_args.args[0] == "/ask"
    print("[PASS] Two tools use the expected FastAPI endpoints")


def test_graph():
    with patch("research_agent.request_api", return_value=FAKE_HITS):
        graph = build_graph(FakeModel())
        state = graph.invoke({
            "messages": [HumanMessage(content="Why does gasification temperature affect H2?")],
            "tool_rounds": 0,
        }, config={"recursion_limit": 18})
        assert state["tool_rounds"] == 1, state["tool_rounds"]
        assert any(isinstance(m, ToolMessage) for m in state["messages"])
        assert "paper07.pdf" in state["messages"][-1].content
    print("[PASS] LangGraph model -> tool -> model loop")


if __name__ == "__main__":
    test_tools()
    test_graph()
    print("All offline agent tests passed. Real API/model validation is still required.")
