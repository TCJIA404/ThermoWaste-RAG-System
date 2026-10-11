# FastAPI Usage Guide

[← Back to README](../README.md)

This guide documents the **implemented** endpoints in `api.py` (application version `0.3.0`). FastAPI launches the local RAG pipeline at startup using PDF files under `data/papers/` and connects to an OpenAI-compatible model API through local `.env` settings.

## 1. Start the server

From the repository root in the **RAG** environment:

```powershell
conda activate transformers
# Install missing dependencies if needed: python -m pip install -r requirements.txt
python -m uvicorn api:app --reload
```

Visit:

- Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI JSON: <http://127.0.0.1:8000/openapi.json>
- Health: <http://127.0.0.1:8000/health>

The API is intended for **local demonstrations**. `--reload` is a development option; the service has no authentication and is not designed for public Internet exposure.

## 2. Endpoints

| Method | Path | Purpose | LLM request? |
| --- | --- | --- | --- |
| `GET` | `/health` | Report service status and in-memory chunk count | No |
| `POST` | `/search` | Hybrid RRF retrieval + CrossEncoder reranking | No |
| `POST` | `/ask` | Retrieve evidence and generate an answer with cited passages | **Yes**, if evidence exists |

### `GET /health`

The server must have completed startup and loaded the PDF corpus.

**Example response (illustrative count):**

```json
{
  "status": "ok",
  "service": "ThermoWaste-RAG",
  "chunks": 1013
}
```

`chunks` is the current in-memory chunk count, not a fixed constant. A successful response does not independently test remote LLM availability.

### `POST /search`

**Request**

```json
{
  "query": "How does gasification temperature affect hydrogen production?",
  "top_k": 5
}
```

**Response shape (illustrative values; not a captured benchmark):**

```json
{
  "query": "How does gasification temperature affect hydrogen production?",
  "count": 1,
  "results": [
    {
      "source": "paper07.pdf",
      "page": 7,
      "text": "[Retrieved passage excerpt]",
      "hybrid_score": 0.03,
      "rerank_score": 0.80
    }
  ],
  "timings_ms": {
    "hybrid_ms": 18.0,
    "rerank_ms": 235.0,
    "total_ms": 255.0,
    "llm_ms": null
  }
}
```

The number of results is **up to** `top_k` and may be smaller when fewer passages are available. `hybrid_score` is an RRF ranking score; `rerank_score` is a cross-encoder score, **not** a probability that the scientific statement is true. Exact fields for the Pydantic model are `query`, `count`, `results`, `timings_ms`.

### `POST /ask`

**Request**

```json
{
  "query": "How does gasification temperature affect hydrogen production?",
  "top_k": 5
}
```

**Response shape (abbreviated illustration):**

```json
{
  "query": "How does gasification temperature affect hydrogen production?",
  "answer": "[Generated evidence-based answer, with source numbers such as [1]]",
  "sources": [
    {
      "source": "paper07.pdf",
      "page": 7,
      "text": "[Retrieved passage excerpt]",
      "hybrid_score": 0.03,
      "rerank_score": 0.80
    }
  ],
  "timings_ms": {
    "hybrid_ms": 18.0,
    "rerank_ms": 235.0,
    "total_ms": 14000.0,
    "llm_ms": 13700.0
  }
}
```

The LLM receives excerpts with numbered `Source [i]` references. In its response text, `[1]` refers to the **first item in the returned `sources` array** for that request. Reference correctness is **not automatically guaranteed**. If no excerpts are retrieved, `/ask` returns a "No supporting passages..." answer with an empty sources list and does not call the model.

## 3. Request constraints and validation

The Pydantic `QueryRequest` model applies:

| Field | Type | Constraint | Default |
| --- | --- | --- | --- |
| `query` | string | 1–2000 characters; whitespace-only is rejected by endpoint validation | Required |
| `top_k` | integer | 1–10 inclusive | 5 |

Examples of expected behavior:

| Case | Expected status |
| --- | --- |
| Valid `/search` or `/ask` | `200 OK` (on backend success) |
| Empty, blank, or missing `query` | `422 Unprocessable Entity` |
| `top_k = 0` or `top_k = 100` | `422 Unprocessable Entity` |
| Retrieval runtime error | `503 Service Unavailable` |
| LLM call failure from `/ask` | `502 Bad Gateway` |

`"string"` is syntactically valid input but often a meaningless Swagger placeholder. The current API validates **shape**, not research-question quality; it can still incur an unnecessary retrieval and LLM request for that input.

## 4. Test with PowerShell

```powershell
# Requires the server already running on 127.0.0.1:8000
Invoke-RestMethod http://127.0.0.1:8000/health

$payload = @{
  query = 'How does gasification temperature affect hydrogen production?'
  top_k = 5
} | ConvertTo-Json

# Search only: does not call the remote LLM
Invoke-RestMethod -Method Post `
  -Uri 'http://127.0.0.1:8000/search' `
  -ContentType 'application/json' -Body $payload

# Full answer: calls remote model and may incur usage charges
Invoke-RestMethod -Method Post `
  -Uri 'http://127.0.0.1:8000/ask' `
  -ContentType 'application/json' -Body $payload
```

Run the smoke test from the repository root:

```powershell
python -m scripts.smoke_test_api
```

It exercises `/health`, repeated `/search`, and validation errors; **it does not call `/ask`**. Test `/ask` manually in Swagger after confirming the provider configuration.

## 5. Timing interpretation

All values in `timings_ms` use **milliseconds**:

- `hybrid_ms`: dense/BM25 retrieval plus RRF fusion.
- `rerank_ms`: cross-encoder inference over candidates.
- `llm_ms`: time around the LLM client invocation, including remote network/model response and local lock waiting.
- `total_ms`: elapsed endpoint processing time, including overhead and waiting; it may not equal the sum of individual stages.

These timings are useful for debugging but are not P50/P95, load-test metrics, or service-level commitments. See [EVALUATION.md](EVALUATION.md) for observed single-run figures.

## 6. Troubleshooting

| Symptom | Check |
| --- | --- |
| Server fails during startup | Ensure `data/papers/` contains readable, text-extractable PDFs; check model downloads, packages, and logs |
| `/search` returns 503 | Check server exception log, PDF/index/model initialization, and query/model errors |
| `/ask` returns 502 | Check `.env` values, provider URL, model ID, API key, network, and provider account limits |
| `422` for a test request | Check required `query`, `top_k` range, and blank strings |
| Swagger shows `"string"` | Replace the placeholder with a meaningful research question |
| Agent cannot contact `/search` | Keep the RAG server running and verify `RAG_API_URL=http://127.0.0.1:8000` in local `.env.agent` |

## 7. Security and deployment status

- `.env` and `.env.agent` must remain local; publish only the placeholder `.example` files.
- Private or copyrighted PDFs, local report outputs, model artifacts, and old Git history must not be copied into a new public release.
- No authentication, rate limiting, streaming, cloud deployment, or security audit is implemented.
- Do not expose the development server to the public Internet without additional controls.
