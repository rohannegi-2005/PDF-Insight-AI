# 🚀 RAG with AstraDB, HuggingFace & Groq LLM

A complete **Retrieval-Augmented Generation (RAG)** pipeline that combines:
- 🧩 **LangChain** for orchestration  
- 🧠 **HuggingFace sentence-transformers** for embeddings  
- 🗄️ **AstraDB Vector Store (DataStax)** for efficient vector search  
- ⚡ **Groq LLM (Llama 3.1-8B)** for fast, context-aware answers  

This project demonstrates an end-to-end **AI knowledge retrieval system** — capable of ingesting PDFs, chunking intelligently, storing them as embeddings, and performing semantic Q&A over the content.

---

## 🌟 Features

✅ Extracts and cleans text from PDFs using a custom processor  
✅ Chunks content efficiently with LangChain’s `RecursiveCharacterTextSplitter`  
✅ Stores document embeddings in **AstraDB Vector Store**  
✅ Performs **semantic similarity search** and **retrieval-based Q&A**  
✅ Powered by **Groq’s Llama 3.1-8B model** for low-latency responses    

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-------------|
| **LLM** | Groq Llama 3.1-8B-Instant |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector DB** | AstraDB (DataStax) |
| **Framework** | LangChain 0.3+ |
| **Language** | Python 3.10+ |

---
