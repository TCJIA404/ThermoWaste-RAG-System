# Retrieval Evaluation and Evidence Quality

[← Back to README](../README.md)

## 1. What the current evaluation measures

`src/evaluation.py` compares four retrieval strategies against **page-level relevance labels**:

| Method | Definition |
| --- | --- |
| Dense | BGE embeddings + FAISS similarity search |
| BM25 | Keyword-based BM25 ranking |
| Hybrid | RRF fusion of Dense and BM25 |
| Hybrid + Reranker | Hybrid candidates followed by BGE cross-encoder reranking |

The checked-in `data/eval_questions.json` contains **15 question records**, each carrying nonempty `relevant` source/page pairs and the label `source_page_evidence_verified` in the supplied snapshot. These labels were prepared within the project; they **have not been independently audited in this documentation pass**. Some entries explicitly note that supporting pages may be non-exhaustive.

No original research PDF is provided in this public repository. Therefore, **a clean clone cannot reproduce reported metric values** until the user supplies the matching local PDF corpus with matching filenames and page positions.

## 2. Evaluation protocol (as implemented)

1. Load `data/eval_questions.json`; include a question only if `relevant` is nonempty and `label_status` is `source_page_evidence_verified` (the loader also treats an absent status field as verified for backward compatibility).
2. Load local PDFs from `data/papers/`, extract text page by page, and split each page into 800-character chunks with 150 overlap.
3. Encode documents with `BAAI/bge-small-en-v1.5`, build FAISS `IndexFlatIP`, construct a BM25 index, and instantiate the cross-encoder.
4. For every question, compute Dense top 5, BM25 top 5, Hybrid top 5 (drawn from Hybrid top 10 candidates), and Reranked top 5 (rerank those Hybrid top 10 candidates).
5. Treat a returned chunk as relevant if **its source filename and PDF page number** match any labeled `(source, page)` pair; text passage content is not independently scored.
6. Average Hit@5 and reciprocal rank over the included question set. Print the resulting table to the terminal.

The script performs **no train/test split**, since this is a retrieval comparison over a fixed local PDF corpus rather than a trained supervised estimator. It does not perform statistical significance testing or aggregate multiple experimental seeds.

## 3. Metric definitions

For each query `q`, with top-5 results and set of labeled relevant pages `R(q)`:

\[
\operatorname{Hit@5}(q) =
\begin{cases}
1 & \text{if a retrieved top-5 page is in }R(q),\\
0 & \text{otherwise.}
\end{cases}
\]

\[
\operatorname{RR@5}(q) =
\begin{cases}
1/r_q & \text{if the first relevant retrieved page is at rank }r_q\le 5,\\
0 & \text{if none is in the top 5.}
\end{cases}
\]

`Hit@5` and `MRR@5` are averages across the labeled questions. A Hit@5 of 1 does **not** imply every retrieved result was correct; MRR only considers the **first** relevant hit. Neither metric by itself evaluates answer accuracy or scientific correctness.

## 4. Earlier exploratory scores — not a current benchmark

Prior local project logs recorded the following **preliminary** values:

| Strategy | Hit@5 | MRR@5 |
| --- | ---: | ---: |
| Dense | 1.000 | 0.600 |
| BM25 | 0.500 | 0.250 |
| Hybrid | 1.000 | 0.625 |
| Hybrid + Reranker | 1.000 | 0.292 |

**Do not attribute this table to the current 15-question file.** These numbers came from an earlier, small exploratory evaluation; the precise sample count, corpus snapshot and run conditions are not fully captured in the public documentation. They are presented only as historical development observations, **not** a verified head-to-head benchmark. In particular, the drop in MRR after reranking illustrates why adding a reranker does not guarantee a better outcome.

To generate metrics for the **current** annotations on your **current** local corpus, run:

```powershell
conda activate transformers
python -m src.evaluation
```

For a defensible reported benchmark, preserve the full command, package versions, PDF corpus manifest/checksums (without distributing copyrighted content), question IDs, included/excluded counts, per-question rank outputs, results table, and timestamp. Independently inspect at least a sample of relevance labels.

## 5. Observed API latency (illustrative single runs)

The following times came from **individual interactive executions**, not a standardized load test or a promise of typical speed:

| Interactive request | Hybrid (ms) | Reranker (ms) | LLM (ms) | Total (ms) |
| --- | ---: | ---: | ---: | ---: |
| Research question: gasification temperature / H₂ | 17.11 | 238.30 | 26,442.40 | 26,697.90 |
| Earlier `/ask` run (different response context) | 21.64 | 265.19 | 28,474.28 | 28,761.23 |

The LLM request dominates these particular runs, but latency varies with network conditions, provider/model response, output tokens, local device, and caching. These data do **not** estimate P50/P95, throughput, concurrent capacity, or effects of a controlled performance optimization.

## 6. Citation provenance vs. claim correctness

The optional Agent audit reports `FOUND_IN_SEARCH` when a cited `(source, page)` appeared in the **same run's** tool output. In a recorded example, all 6 unique citation pages were found among 7 retrieved unique pages; this is a **provenance-matching result**, not a correctness score.

There is a known distinction between:

- **Source-page provenance:** Did the cited filename and PDF page appear in the retrieved evidence?
- **Evidence support:** Does the cited passage actually support the whole statement, including numerical values, conditions, and qualifiers?
- **Scientific validity:** Is the statement consistent with established mechanisms and valid for the cited study context?

The current `agent/claim_evidence_review.py` may assess whole citation-bearing lines that each contain multiple scientific subclaims. A `SUPPORTED` label from an LLM is **provisional**, not human ground truth. Local quality-gate rules flag some risky patterns but do not constitute exhaustive fact-checking.

A concrete recorded failure mode is characterizing the **forward water-gas shift reaction as endothermic**. Forward WGS is exothermic; a source-page match does not correct an incorrect thermochemical mechanism. Another risk is treating H₂ volume fraction as interchangeable with H₂ yield or total syngas yield.

## 7. Recommended next evaluation milestones (not implemented)

- Build a frozen evaluation split with question type, corpus ID, and independent source-page annotation checks.
- Record per-query outputs to compare changed rankings and investigate reranker bad cases.
- Add no-answer questions and score refusal quality separately from retrieval Hit@K.
- Split generated sentences into **atomic claims** and ask human annotators to grade each claim against complete original-page evidence.
- Track supported, unsupported, contradicted, and insufficient-evidence statements, with explicit inter-annotator review where feasible.
- Run repeated warm/cold API benchmarks and report P50/P95 plus LLM token usage and cost.

## 8. Reproduction commands

```powershell
# Offline tests; no PDF corpus or LLM requests needed
python -m tests.test_evaluation
python -m tests.test_citation_audit
python -m tests.test_claim_evidence_review
python -m tests.test_quality_gate

# Requires original matching local PDFs and embedding/reranking models
python -m src.evaluation

# With FastAPI running: smoke test /health + /search and request validation
python -m scripts.smoke_test_api
```

Use Python/package versions from your tested environment when comparing results; requirements are dependency names rather than locked versions.
