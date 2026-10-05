"""Sentence-transformer embeddings with asymmetric query/passage encoding."""

import numpy as np


class Embedder:
    def __init__(self, model_name: str, query_instruction: str = "", device: str = "cpu",
                 num_threads: int | None = None):
        import torch
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        if num_threads:
            # torch defaults to one thread per visible core; in containers that is often the
            # host's core count, and oversubscription makes single-query latency 50x worse.
            torch.set_num_threads(num_threads)
        self.model_name = model_name
        self.query_instruction = query_instruction
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, batch_size=32, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        # BGE models are trained with an instruction prefix on the query side only.
        vector = self._model.encode([text], prompt=self.query_instruction or None,
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False)
        return np.asarray(vector[0], dtype="float32")
