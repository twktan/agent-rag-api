def build_prompt(query, retrieved_chunks):
    context = '\n\n'.join([c['text'] for c in retrieved_chunks])

    return f'''
You are a helpful assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{query}

Answer:
'''
