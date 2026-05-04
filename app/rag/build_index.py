from app.rag.ingest import ingest_directory
from app.rag.embed import embed_texts
from app.rag.vectorstore import FAISSVectorStore

# Load chunks
chunks = ingest_directory('data/docs')

texts = [c['text'] for c in chunks]
metadatas = [c['metadata'] for c in chunks]

# Embed
embeddings = embed_texts(texts)

# Build FAISS
dim = len(embeddings[0])
store = FAISSVectorStore(dim)

store.add(embeddings, texts, metadatas)

# Save
store.save()

print('Index built and saved.')
