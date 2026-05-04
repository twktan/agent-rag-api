from fastapi import FastAPI
from pydantic import BaseModel
from app.rag.vectorstore import FAISSVectorStore
from app.agent.agent import RAGAgent

# Load FAISS store
store = FAISSVectorStore(dim=384)
store.load()

# Initialize RAG agent
agent = RAGAgent(store)

# Initialize FastAPI app
app = FastAPI(title="Enterprise Agentic RAG Support Platform",
              description='''
This is an internal AI assistant platform for employees of hypothetical company Trevor Tan Incorporated.

Problem Statement:
- Internal company documentation is confidential and therefore not used to train AI models such as ChatGPT or similar systems.
- Employees still require access to general-purpose AI capabilities (e.g., writing assistance, summarization, and general knowledge unrelated to internal company data).

Solution Overview:
- Retrieval-Augmented Generation (RAG) system securely retrieves confidential company documentation (and minimizes hallucinations).
- AI Agent decides to route query to RAG system (for company-specific queries) or directly to LLM (for general queries).

You may try example RAG-routed queries such as:
- What is the technology stack used in the company?
- What are the benefits available to me as an employee?

You may try example direct queries (i.e., RAG not used) such as:
- Explain succinctly, what is the difference between supervised learning versus unsupervised learning?
- Explain briefly, what is the best way to write a technical report?
''')


# Define Pydantic model
class QueryRequest(BaseModel):
    question: str


# Query endpoint
@app.post('/query')
def agent_query(req: QueryRequest):
    result = agent.run(req.question)
    return result
