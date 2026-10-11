# src/hybrid_retriever.py


class HybridRetriever:

    def __init__(
        self,
        dense_retriever,
        bm25_retriever
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever

    def retrieve(
        self,
        query,
        top_k=10,
        candidate_k=20,
        rrf_k=60
    ):

        dense_results = (
            self.dense_retriever.retrieve(
                query,
                top_k=candidate_k
            )
        )

        bm25_results = (
            self.bm25_retriever.retrieve(
                query,
                top_k=candidate_k
            )
        )

        score_map = {}

        metadata_map = {}

        # Dense
        for rank, item in enumerate(
            dense_results,
            start=1
        ):

            key = (
                item["source"],
                item["page"],
                item["text"]
            )

            score_map[key] = (
                score_map.get(key, 0)
                + 1 / (rrf_k + rank)
            )

            metadata_map[key] = item

        # BM25
        for rank, item in enumerate(
            bm25_results,
            start=1
        ):

            key = (
                item["source"],
                item["page"],
                item["text"]
            )

            score_map[key] = (
                score_map.get(key, 0)
                + 1 / (rrf_k + rank)
            )

            metadata_map[key] = item

        sorted_items = sorted(
            score_map.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for key, score in sorted_items[:top_k]:

            item = metadata_map[key].copy()

            item["hybrid_score"] = score

            results.append(item)

        return results