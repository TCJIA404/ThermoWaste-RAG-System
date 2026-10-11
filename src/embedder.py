# src/embedder.py

from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(
        self,
        model_name="BAAI/bge-small-en-v1.5"
    ):
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts):

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def encode_query(self, query):

        return self.model.encode(
            [query],
            normalize_embeddings=True
        )