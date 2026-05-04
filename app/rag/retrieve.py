from app.rag.embed import embed_texts


def retrieve(query, store, k=3):
    q_emb = embed_texts([query])[0]
    return store.search(q_emb, k=k)
