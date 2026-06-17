# NLP Competition — RAG AI Agent Challenge

Welcome to the NLP Competition! Your mission is to build a **production-ready AI Agent** that answers questions based on a provided knowledge base of PDF documents using Retrieval-Augmented Generation (RAG).

---

## Overview

You will build an AI agent that:

1. Receives a user question
2. Uses a **retrieval tool** to search a knowledge base and return relevant chunks from the provided PDF
3. Uses those chunks as context to generate an accurate, grounded answer

The agent must be able to reason over the retrieved chunks and produce a final response — it should **not** answer from memory alone.

---

## Knowledge Base — PDF Document

The knowledge base consists of a single PDF:

**Encyclopedia of Ancient Egypt** — Margaret Bunson

The file is available under the `/pdf` directory in the competition workspace.

Chunk it, embed it, and load it into your retrieval system before building the agent.

---

## LLM Providers

You **must** use **Gemini** and/or **Groq**. API keys are provided below — use them directly in your project.

### API Keys

| Provider | API Key |
|----------|---------|
| **Google Gemini** | `REPLACE_WITH_GEMINI_API_KEY` |
| **Groq** | `REPLACE_WITH_GROQ_API_KEY` |

### API Documentation
- Gemini: https://ai.google.dev/gemini-api/docs
- Groq: https://console.groq.com/docs/openai

---

## Technical Requirements

### What You Must Build

#### 1. Retrieval Tool
- A function/tool that takes a query string as input
- Searches your vector store / knowledge base
- Returns the top-N most relevant text chunks
- The agent must **call this tool** explicitly — do not just stuff all chunks into the prompt

#### 2. AI Agent
- An agent that receives a user question
- Decides when and how to call the retrieval tool
- Reasons over the returned chunks
- Produces a final, grounded answer based on the document

#### 3. Production-Ready API (Required)
Your project must expose a REST API with at least the following endpoints:

```
POST /ask
```
**Request body:**
```json
{
  "question": "What were the main gods of ancient Egypt?"
}
```
**Response:**
```json
{
  "answer": "The main gods of ancient Egypt included Ra, Osiris, Isis...",
  "chunks_used": ["chunk text 1...", "chunk text 2..."]
}
```

```
GET /health
```
Returns service status — used to verify the API is running.

```
GET /docs
```
Auto-generated API documentation (e.g., Swagger/OpenAPI if using FastAPI).

### Bonus — UI
Build a GUI using any tool that you want to make the user able to interact with your agent

---

## Freedom of Choice

You are free to use **any framework, library, or technology** you prefer to build this project.

> ⚠️ **Vector Database:** You must use a **cloud-hosted vector database**.

---

Good luck to all teams! 🎉

*For questions, reach out to the competition organizers.*