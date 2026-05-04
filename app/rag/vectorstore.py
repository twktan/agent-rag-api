import faiss
import numpy as np
import pickle
import os


class FAISSVectorStore:

    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(
            dim)  # cosine similarity (with normalized embeddings)
        self.texts = []
        self.metadata = []

    # Add embedding
    def add(self, embeddings, texts, metadatas):
        self.index.add(np.array(embeddings).astype('float32'))
        self.texts.extend(texts)
        self.metadata.extend(metadatas)

    # Search embedding(s) closest to the query embedding
    def search(self, query_embedding, k=3):
        _, top_k_indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k)

        results = []
        for idx in top_k_indices[0]:
            results.append({
                'text': self.texts[idx],
                'metadata': self.metadata[idx]
            })

        return results

    # Save database
    def save(self, path='vectorstore'):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f'{path}/index.faiss')

        with open(f'{path}/store.pkl', 'wb') as f:
            pickle.dump((self.texts, self.metadata), f)

    # Load database
    def load(self, path='vectorstore'):
        self.index = faiss.read_index(f'{path}/index.faiss')

        with open(f'{path}/store.pkl', 'rb') as f:
            self.texts, self.metadata = pickle.load(f)
