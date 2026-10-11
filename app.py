"""Single-question CLI using the same retrieval pipeline as FastAPI.

Run in the original 'transformers' environment from the project root.
"""
from api import build_pipeline

def main():
    query = "How does gasification temperature affect hydrogen production?"
    pipeline = build_pipeline()
    candidates = pipeline.hybrid.retrieve(query, top_k=10, candidate_k=20, rrf_k=60)
    hits = pipeline.reranker.rerank(query, candidates, top_k=5)
    print("\n===== Retrieved passages =====")
    for rank, item in enumerate(hits, 1):
        print("{}. {} | page {} | score {:.3f}".format(rank,item['source'],item['page'],item['rerank_score']))
    print("\n===== Answer =====")
    print(pipeline.llm.answer(query=query, results=hits))

if __name__ == '__main__':
    main()
