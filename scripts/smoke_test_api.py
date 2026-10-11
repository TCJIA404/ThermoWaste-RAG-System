"""Day 10 smoke test for an already running ThermoWaste-RAG API.

No third-party packages needed. Run from the project root:
    python smoke_test_day10.py
Only /search is called with normal queries; this script will NOT call /ask or use LLM tokens.
"""
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8000"


def request(path, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = Request(BASE + path, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urlopen(req, timeout=180) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def main():
    try:
        status, data = request("/health")
        assert status == 200, (status, data)
        print("[PASS] /health", data.get("status"))
        cases = [
            ("normal search #1", {"query": "How does gasification temperature affect hydrogen production?", "top_k": 5}, 200),
            ("normal search #2", {"query": "How does gasification temperature affect hydrogen production?", "top_k": 5}, 200),
            ("blank query", {"query": "   ", "top_k": 5}, 422),
            ("invalid top_k", {"query": "hydrogen production", "top_k": 0}, 422),
        ]
        for label, body, expected in cases:
            status, data = request("/search", body)
            assert status == expected, (label, status, data)
            if status == 200:
                print("[PASS]", label, "hits:", data.get("count"), "timings_ms:", data.get("timings_ms"))
            else:
                print("[PASS]", label, "HTTP", status)
        print("All smoke tests passed. Next, manually test /ask in Swagger once.")
    except (URLError, ConnectionError) as exc:
        print("Cannot connect. Start the server using: python -m uvicorn api:app --reload")
        print("Error:", exc)
    except AssertionError as exc:
        print("[FAIL]", exc)


if __name__ == "__main__":
    main()
