# 🧭 ResearchPilot — Agentic RAG Research System
### *(evolved from PDF-Insight-AI)*

A Retrieval-Augmented Generation system that started as a single-file PDF Q&A script and has been rebuilt into a **modular, hybrid-retrieval, self-correcting, multi-document agentic research system**, with a measured evaluation harness and a unified Streamlit interface. It is built entirely on free and local tooling (Groq, HuggingFace, AstraDB free tier, LangGraph).

This repo is both a working system *and* a deliberately transparent engineering log: every retrieval technique, agent, and design trade-off below was built from first principles rather than wrapped behind a high-level abstraction, so every decision can be explained, measured, and defended.

---

## 🌟 What this project does

Give it a question about an indexed collection of papers, including comparison-style questions that span several documents, and instead of a single retrieve-then-answer pass, it:

1. **Plans** — breaks a broad question into focused sub-questions
2. **Rewrites** each sub-question into a clean search query
3. **Retrieves** candidates using *both* dense (embedding) and sparse (keyword/BM25) search, fused together
4. **Reranks** those candidates with a cross-encoder for precision
5. **Critiques** its own evidence — is this actually enough to answer, or only superficially related?
6. **Retries** with a genuinely different search angle if the evidence falls short (with a hard safety cap, so it never loops forever)
7. **Synthesizes** the evidence gathered across all sub-questions into one cited answer, grounded only in what it actually found

The same retrieval stack also powers a **conversational chat mode** (memory + streaming + citations) for quick, single-pass questions. Both modes live in one Streamlit app.

---

## 🏗️ Architecture

### The agentic research loop (`graph/`)

```
User Question
      │
      ▼
┌───────────────┐
│     plan      │   broad question → focused sub-questions
└───────┬───────┘
        ▼
   for each sub-question:
        │
        ▼
 ┌─────────────┐
 │   rewrite   │   sub-question → clean search query
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  retrieve   │◀──────────────┐   dense + sparse → RRF fusion → rerank
 └──────┬──────┘               │
        ▼                      │
 ┌─────────────┐        ┌──────┴──────┐
 │   critic    │───────▶│    retry    │   insufficient + retries left
 └──────┬──────┘        └─────────────┘
        │ sufficient, or out of retries
        ▼
 ┌─────────────┐
 │  collect    │   store this sub-question's evidence
 └──────┬──────┘
        ▼
┌───────────────┐
│  synthesize   │   merge all sub-question evidence → one cited answer
└───────────────┘
```

Built with **LangGraph's `StateGraph`** — a conditional edge after the Critic node decides, on every pass, whether to loop back with a rewritten query or move on. A `max_retries` cap guarantees the loop always terminates, even for questions the documents don't actually cover.

### The conversational mode (`core/rag_pipeline.py`)

```
User Question → History-Aware Retriever → Context + Chat History → Groq LLM → Streamed Answer + Citations
```

Multi-turn memory (`RunnableWithMessageHistory`), token-by-token streaming, and a source-citation expander, running on the same underlying retrieval stack.

---

## 🧩 Key Features

### Retrieval
- ✅ **Dense retrieval** — HuggingFace `all-MiniLM-L6-v2` embeddings + AstraDB vector store, with **MMR** (Maximal Marginal Relevance) to diversify results and avoid near-duplicate chunks
- ✅ **Sparse retrieval** — a **BM25** keyword index (`rank_bm25`), catching exact terms, names, and numbers that embeddings can under-weight
- ✅ **Hybrid fusion** — dense + sparse results combined via **Reciprocal Rank Fusion (RRF)**, so chunks that rank well under *both* methods are promoted
- ✅ **Cross-encoder reranking** — a second-stage `cross-encoder/ms-marco-MiniLM-L-6-v2` model jointly scores (query, passage) pairs for a more accurate final ranking than similarity search alone

### Chunking
- ✅ **Semantic chunking** — a from-scratch, embedding-based chunker that splits text where *topic* shifts (via sentence-level cosine similarity), rather than at a fixed character count
- ✅ Fixed-size chunking kept available as a baseline, so the two strategies can be directly compared in the evaluation harness

### Agentic reasoning
- ✅ **Research Agent** — decomposes broad or comparison-style questions into sub-questions, rewrites each into a retrieval-friendly query, and generates genuinely different search angles on retry
- ✅ **Critic Agent** — judges evidence sufficiency using **structured output** (a Pydantic schema, not parsed free text), so the verdict is a reliable `bool` the graph can branch on
- ✅ **LangGraph orchestration** — the full plan → rewrite → retrieve → critique → retry → synthesize workflow, with a hard retry cap for guaranteed termination
- ✅ **Multi-document research** — chunks carry `paper_id` metadata, so evidence can be gathered per paper and compared across papers

### Evaluation
- ✅ **Evaluation harness** (`eval/`) — a hand-labeled question → relevant-chunk dataset, scored with **Recall@K** and **MRR**
- ✅ **Ablation comparisons** — dense-only vs. sparse-only vs. hybrid vs. hybrid + rerank, and fixed vs. semantic chunking, so each technique's contribution is measured rather than assumed

### Interface
- ✅ **Unified Streamlit UI** — one app with a **Chat** mode (memory, streaming, citations) and a **Research** mode that shows the sub-questions, the evidence retrieved for each, every Critic verdict, the retry history, and the final cited answer
- ✅ PDF upload + indexing directly from the sidebar

### Engineering
- ✅ Fully modular (`core/`, `agents/`, `graph/`, `eval/` separated by responsibility)
- ✅ Centralized configuration (`config.py`) — one source of truth for every model name, chunk size, and retrieval parameter
- ✅ **100% free/local-model stack** — Groq for LLM inference, HuggingFace for embeddings + reranking, no paid OpenAI API calls anywhere

---

## 🏗️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM (agents + answers)** | Groq-hosted model (e.g. `openai/gpt-oss-20b`) | Planning, query rewriting, evidence judgment, answer synthesis |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Dense semantic vectors, used for both retrieval and semantic chunking |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Second-stage, query-aware relevance scoring |
| **Dense vector store** | AstraDB (DataStax) | Stores embeddings, serves MMR-based similarity search |
| **Sparse index** | `rank_bm25` (BM25Okapi) | In-memory keyword search |
| **Orchestration** | LangChain (LCEL) + **LangGraph** (`StateGraph`) | Chains for chat mode; stateful agentic loop for research mode |
| **Evaluation** | Custom Recall@K / MRR harness | Measuring retrieval quality across configurations |
| **Frontend** | Streamlit | Unified Chat + Research interface |
| **PDF processing** | `PyPDFLoader`, custom semantic/fixed chunkers | Document ingestion |
| **Environment** | `python-dotenv` | Secrets management via `.env` |
| **Language** | Python 3.11+ | Core development environment |

---

## 🗂 Folder Structure

```
PDF-Insight-AI/
│
├── app.py                      # Unified Streamlit UI: Chat mode + Research mode
├── config.py                   # Centralized settings: model names, chunk sizes, retrieval params
├── requirements.txt
├── runtime.txt
├── .env                         # API keys (not committed)
├── .env.example
│
├── data/                        # Sample PDFs for ingestion/testing
│
├── core/                        # Deterministic retrieval engine
│   ├── document_utils.py        #   SmartPDFProcessor: PDF loading + cleaning + chunking
│   ├── semantic_chunker.py      #   Embedding-based semantic chunking
│   ├── vector_store.py          #   HuggingFace embeddings + AstraDB setup
│   ├── sparse_store.py          #   BM25 keyword index
│   ├── retrieval.py             #   Hybrid fusion (RRF) + reranking pipeline
│   ├── reranker.py              #   Cross-encoder reranking
│   └── rag_pipeline.py          #   Conversational LCEL chain (memory, streaming, citations)
│
├── agents/                      # LLM-driven decision-making layer
│   ├── prompts.py                #   All agent prompt templates, centralized
│   ├── research_agent.py         #   Question decomposition + query rewriting (first pass + retry)
│   └── critic_agent.py           #   Structured evidence-sufficiency judgment
│
├── graph/                       # Agentic orchestration
│   ├── state.py                  #   Shared state schema passed between nodes
│   └── workflow.py                #   LangGraph StateGraph: plan, retry loop, synthesis
│
└── eval/                        # Evaluation harness
    ├── dataset.py                #   Hand-labeled question → relevant-chunk set
    └── metrics.py                 #   Recall@K, MRR, and configuration comparison runner
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/rohannegi-2005/PDF-Insight-AI.git
cd PDF-Insight-AI
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```
ASTRA_DB_API_ENDPOINT=your_astra_db_endpoint
ASTRA_DB_TOKEN=your_astra_db_token
GROQ_API_KEY=your_groq_api_key
```

> **Note:** if a Groq model in `config.py` ever returns `model_not_found`, check the live list of models available to your account by calling `client.models.list()` via the `groq` SDK, and update `config.LLM_MODEL_NAME` accordingly. Available models can vary by account and change over time.

---

## 🚀 Usage

### Option A — Unified Streamlit app
```bash
streamlit run app.py
```
Upload one or more PDFs from the sidebar and click **Process & Index PDF**, then pick a mode:

- **Chat** — quick, conversational Q&A with memory, word-by-word streaming, and a sources expander
- **Research** — ask a broad or comparison-style question and watch the system plan sub-questions, retrieve and critique evidence for each, retry where evidence is weak, and produce a final cited answer

### Option B — Python API
```python
from dotenv import load_dotenv
load_dotenv()

from core.vector_store import get_embedding_model, get_vector_store, ingest_documents
from core.sparse_store import SparseStore
from core.document_utils import SmartPDFProcessor
from graph.workflow import run_research

embeddings = get_embedding_model()
vector_store = get_vector_store(embeddings)
processor = SmartPDFProcessor(embeddings=embeddings)

chunks = []
for path in ["data/paper_a.pdf", "data/paper_b.pdf"]:
    chunks += processor.process_pdf(path)
ingest_documents(vector_store, chunks)

sparse_store = SparseStore()
sparse_store.build(chunks)

result = run_research(
    "Compare the retrieval approaches used in these papers.",
    vector_store,
    sparse_store,
)

print(result["final_answer"])
print(result["attempts"])      # full retry/verdict history
```

### Option C — Run the evaluation harness
```bash
python -m eval.metrics
```
Scores each retrieval configuration against the labeled dataset and prints Recall@K and MRR side by side.

---

## 📊 Evaluation

Retrieval quality is measured, not assumed. The harness runs every configuration against a hand-labeled set of questions, each mapped to the chunk(s) that actually contain the answer.

**Metrics**
- **Recall@K** — the fraction of questions for which a relevant chunk appears in the top K results
- **MRR (Mean Reciprocal Rank)** — the average of `1 / rank` of the first relevant chunk, rewarding systems that place the right chunk higher

**Configurations compared**

| Configuration | Recall@5 | MRR |
|---|---|---|
| Dense only (fixed chunks) | — | — |
| Dense only (semantic chunks) | — | — |
| Sparse only (BM25) | — | — |
| Hybrid (RRF) | — | — |
| Hybrid + cross-encoder rerank | — | — |

<!-- Fill in with your own measured results from `python -m eval.metrics` before publishing. -->

---

## 💬 Example Interaction (Research mode)

**Question:** *"Compare the retrieval approaches used in these papers and tell me which techniques improve retrieval."*

```
Plan
  1. What retrieval method does each paper use?
  2. Which papers use MMR or another diversity technique, and why?
  3. Which papers report improvements from reranking?

Sub-question 1
  query: "retrieval approach dense sparse hybrid"
  critic: sufficient
  evidence: paper_a p.4, paper_b p.2

Sub-question 3
  attempt 1 — query: "reranking improvement results"
    critic: insufficient. "Passages describe reranking but give no results."
  attempt 2 — query: "cross-encoder reranker recall gain"
    critic: sufficient
  evidence: paper_a p.5

Final answer
  A cited comparison grounded only in the evidence above, with each
  claim tagged to its paper and page.
```

*(Illustrative trace — the Research mode UI shows this structure live: the plan, each sub-question's queries and Critic verdicts, and the final answer with citations.)*

This is the behavior the Critic + retry loop is built to produce: rather than fabricating an answer, the system is transparent about what its evidence does and doesn't support.

---

## 🧠 Engineering Decisions Worth Knowing

A few choices in this codebase that are deliberate, not default:

- **Semantic chunking is hand-written**, not pulled from `langchain_experimental`, so every step of the sentence → embed → cosine-similarity → cut decision is visible and explainable.
- **Reciprocal Rank Fusion**, not score averaging, combines dense and sparse results — raw similarity scores and BM25 scores live on incomparable scales, so only *rank position* is used to merge them.
- **Reranking is a second stage, not the primary retriever** — a cross-encoder can't be precomputed (it scores query+passage jointly), so it only runs on a small shortlist that cheaper hybrid search already narrowed down.
- **The Critic returns structured output** (`sufficient: bool`, `reason: str`) via a Pydantic schema, not free text — so the retry loop branches on a real boolean instead of parsing LLM prose.
- **The retry loop has a hard cap.** Agentic systems that can decide to "try again" also need a guaranteed way to stop — `max_retries` ensures the system always terminates with an answer, even for questions the documents don't cover.
- **Nodes are closures, not globals.** Graph nodes capture the vector and sparse stores from their enclosing scope, so multiple graphs over different collections can coexist safely in one process.
- **Every technique is evaluated against a baseline.** Hybrid search, reranking, and semantic chunking each have a measured Recall@K / MRR comparison, so claims about improvement are backed by numbers.
- **One shared `config.py`** is the single source of truth for every model name and parameter — avoiding the classic bug where two files quietly disagree about which model or setting is active.
- **No paid APIs anywhere.** Every model — LLM, embeddings, reranker — runs via Groq's free tier or a locally executed HuggingFace model.

---

## 🗺️ Roadmap

- [x] Modular folder structure (`core/`, `agents/`, `graph/`, `eval/`)
- [x] Semantic chunking
- [x] Dense retrieval (AstraDB + MMR)
- [x] Sparse retrieval (BM25)
- [x] Hybrid fusion (Reciprocal Rank Fusion)
- [x] Cross-encoder reranking
- [x] Query rewriting (Research Agent)
- [x] Evidence sufficiency checking (Critic Agent)
- [x] LangGraph agentic retry loop
- [x] Multi-document research agent (decompose a question into sub-questions across several papers)
- [x] Evaluation harness (Recall@K, MRR — measuring each technique's impact)
- [x] Unified Streamlit UI for the agentic graph (sub-questions, retry history, and citations)

---

## 🧠 Learning Highlights

This project is a hands-on demonstration of:
- Hybrid retrieval system design (dense + sparse + fusion + reranking)
- Agentic reasoning with LangGraph (stateful loops, conditional routing, structured LLM output, question decomposition)
- Evaluation-driven engineering (Recall@K, MRR, ablations against baselines)
- Retrieval-quality trade-offs (chunking strategy, candidate pool sizing, retry budgets)
- Clean, modular Python architecture built for explainability, not just functionality
- Running a complete LLM application stack on entirely free/local infrastructure

---

## 👨‍💻 Author

**Rohan Negi**
AI Developer | Enthusiast in LLMs, RAG, and applied NLP
📧 [rohannnegi2005@gmail.com](mailto:rohannnegi2005@gmail.com)
🌐 [GitHub Profile](https://github.com/rohannegi-2005)