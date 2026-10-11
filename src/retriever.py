# src/retriever.py


class Retriever:

    def __init__(
        self,
        embedder,
        vector_store,
        chunks
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunks = chunks

    def retrieve(
        self,
        query,
        top_k=5
    ):

        query_embedding = self.embedder.encode_query(
            query
        )

        scores, indices = self.vector_store.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(
            scores,
            indices
        ):

            item = self.chunks[idx].copy()

            item["score"] = float(score)

            results.append(item)

        return results