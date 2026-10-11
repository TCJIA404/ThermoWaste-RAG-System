# src/vector_store.py

import faiss
import numpy as np


class FaissVectorStore:

    def __init__(self):
        self.index = None

    def build(self, embeddings):

        embeddings = np.array(
            embeddings,
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

    def search(
        self,
        query_embedding,
        top_k=5
    ):

        query_embedding = np.array(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        return scores[0], indices[0]