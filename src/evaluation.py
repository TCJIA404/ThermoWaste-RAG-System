"""Compare retrieval methods with verified (source, PDF page) relevance labels.

Run from project root: python -m src.evaluation
Unlabelled questions do not contribute to Hit@5 / MRR@5 denominators.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def evaluate(results, relevant, k=5):
    """Return Hit@k and reciprocal rank; expects non-empty verified labels."""
    if not relevant:
        raise ValueError("Empty relevance labels cannot be scored")
    relevant_set = {(Path(row["source"]).name, int(row["page"])) for row in relevant}
    for rank, item in enumerate(results[:k], start=1):
        key = (Path(item["source"]).name, int(item["page"]))
        if key in relevant_set:
            return 1.0, 1.0 / rank
    return 0.0, 0.0

def verified_questions(rows):
    """Skip candidate questions with no source-page evidence labels."""
    return [x for x in rows if x.get("relevant") and
            x.get("label_status", "source_page_evidence_verified") == "source_page_evidence_verified"]

def main():
    from src.loader import load_folder
    from src.splitter import split_pages
    from src.embedder import Embedder
    from src.vector_store import FaissVectorStore
    from src.retriever import Retriever
    from src.bm25_retriever import BM25Retriever
    from src.hybrid_retriever import HybridRetriever
    from src.reranker import Reranker

    rows = json.loads((ROOT / 'data' / 'eval_questions.json').read_text(encoding='utf-8'))
    questions = verified_questions(rows)
    print(f"Evaluation questions: {len(questions)} labeled / {len(rows)} total")
    if not questions:
        raise SystemExit('No verified questions to score. Check data/eval_questions.json')
    pages = load_folder(str(ROOT / 'data' / 'papers'))
    chunks = split_pages(pages, chunk_size=800, chunk_overlap=150)
    if not chunks:
        raise SystemExit('No PDF chunks found in data/papers')
    embedder = Embedder()
    store = FaissVectorStore()
    store.build(embedder.encode_documents([row['text'] for row in chunks]))
    dense = Retriever(embedder, store, chunks)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(dense, bm25)
    reranker = Reranker()
    scores = {key: [] for key in ('Dense', 'BM25', 'Hybrid', 'Hybrid+Reranker')}
    for row in questions:
        query = row['query']
        candidates = hybrid.retrieve(query, top_k=10, candidate_k=20)
        runs = {
            'Dense': dense.retrieve(query, top_k=5),
            'BM25': bm25.retrieve(query, top_k=5),
            'Hybrid': candidates[:5],
            'Hybrid+Reranker': reranker.rerank(query, candidates, top_k=5),
        }
        for name, results in runs.items():
            scores[name].append(evaluate(results, row['relevant']))
    print(f"{'Method':<22} {'Hit@5':>8} {'MRR@5':>8}")
    for name, values in scores.items():
        hit = sum(v[0] for v in values) / len(values)
        mrr = sum(v[1] for v in values) / len(values)
        print(f"{name:<22} {hit:>8.3f} {mrr:>8.3f}")
    print('Note: results reflect this annotated corpus, not out-of-domain generalization.')

if __name__ == '__main__':
    main()
