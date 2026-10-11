from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
        self,
        model_name="BAAI/bge-reranker-base"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query,
        candidates,
        top_k=5
    ):
        if not candidates:
            return []

        pairs = [
            [query, item["text"]]
            for item in candidates
        ]

        scores = self.model.predict(pairs)

        reranked_results = []

        for item, score in zip(
            candidates,
            scores
        ):
            new_item = item.copy()
            new_item["rerank_score"] = float(score)

            reranked_results.append(
                new_item
            )

        reranked_results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked_results[:top_k]