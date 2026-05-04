# Enterprise Agentic RAG Support Platform

**Live API Demo:**  
https://fastapi-rag-141900972389.asia-southeast1.run.app/docs

---

## Overview
This project is an agentic RAG & end-to-end machine learning (ML) system that serves as an internal AI assistant for employees of hypothetical company, Trevor Tan Incorporated.

It demonstrates end-to-end ML engineering:
- Vector DB implementation with FAISS
- Agentic AI development with LangChain
- Performance monitoring using MLflow
- API serving with FastAPI
- Containerization with Docker
- Deployment on Google Cloud Run

---

## Problem Statement

- Internal company documentation is confidential and therefore not used to train AI models such as ChatGPT or similar systems.
- Employees still require access to general-purpose AI capabilities (e.g., writing assistance, summarization, and general knowledge unrelated to internal company data).

---

## Solution Overview

- Retrieval-Augmented Generation (RAG) system securely retrieves confidential company documentation (and minimizes hallucinations).
- AI Agent decides to route query to RAG system (for company-specific queries) or directly to LLM (for general queries).

---

## Usage

### FastAPI Swagger UI
1. Go to:  
   https://fastapi-rag-141900972389.asia-southeast1.run.app/docs  
2. Use `/query`
3. You may enter a prompt specific to the hypothetical company (i.e., Trevor Tan Incorporated) or a general prompt.

---
