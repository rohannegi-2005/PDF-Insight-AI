# 🚀 PDF-Insight-AI — RAG with AstraDB, HuggingFace & Groq LLM

A fully functional **Retrieval-Augmented Generation (RAG)** pipeline that turns any PDF into an intelligent, queryable knowledge source.

This project combines the power of:
- 🧩 **LangChain** for orchestration  
- 🧠 **HuggingFace sentence-transformers** for text embeddings  
- 🗄️ **AstraDB Vector Store (DataStax)** for high-performance semantic search  
- ⚡ **Groq LLM (Llama-3.1-8B-Instant)** for ultra-fast, context-aware responses  

---

## 🌟 Overview

**PDF-Insight-AI** processes raw PDF documents and allows users to ask natural-language questions about their content.  
It uses embeddings and semantic retrieval to find the most relevant chunks of information and then generates **context-grounded answers** using Groq’s Llama 3.1 LLM.  

---

## 🧠 Core Architecture

This project follows the **RAG (Retrieval-Augmented Generation)** pattern, where:
1. Documents are pre-processed and split into clean, meaningful chunks.
2. Each chunk is converted into a vector embedding using HuggingFace.
3. All embeddings are stored in **AstraDB** for vector similarity search.
4. At query time, the retriever finds the most similar chunks.
5. The LLM uses these retrieved contexts to generate factual answers.

All pipeline components are chained declaratively using **LangChain’s LCEL (LangChain Expression Language)** for simplicity, transparency, and reproducibility.

---

## 🔗 LCEL Chain Explained

LCEL allows you to combine multiple steps in one clean, composable pipeline.

**Example Flow:**

```text
User Question
   ↓
Retriever → format_docs → Prompt → Groq LLM → Output Parser
```

This design ensures:
- 🧩 Reusable pipeline components
- ⚙️ Easy debugging and visualization
- 💬 Consistent, structured outputs

---

## 🌟 Features

✅ **Smart PDF Processor** – Cleans and chunks text intelligently  
✅ **Hybrid Vector Search** – Uses AstraDB for fast, semantic retrieval  
✅ **RAG via LCEL** – Combines retriever, prompt, LLM, and parser in one flow  
✅ **High Accuracy** – Context-aware responses with factual grounding  
✅ **Secure Setup** – Uses `.env` for managing API keys safely  
✅ **Fully Open-Source** – Clean, modular Python code ready for extension  

---

## 🏗️ Tech Stack

| Layer | Technology | Purpose |
|-------|-------------|----------|
| **Language Model (LLM)** | ⚡ Groq Llama-3.1-8B-Instant | Generates answers grounded in retrieved context |
| **Embeddings** | 🧠 sentence-transformers/all-MiniLM-L6-v2 | Encodes text chunks into dense semantic vectors |
| **Vector Database** | 🗄️ AstraDB (DataStax) | Stores and retrieves embeddings efficiently for similarity search |
| **Framework** | ⚙️ LangChain ≥ 0.3 with LCEL | Defines the retrieval and generation pipeline |
| **Processing** | 📄 LangChain-Community & Text-Splitters | Cleans and splits PDF text |
| **Environment Handling** | 🔐 python-dotenv | Loads secure keys from `.env` |
| **Language** | 💻 Python 3.10+ | Core development environment |

---

## 🗂 Folder Structure

```
PDF-Insight-AI/
│
├── main.py                # Main script — runs the entire RAG pipeline
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
├── .gitignore             # Ignore venv, .env, and cache files
├── .env                   # Stores API keys (not committed)
│
└── data/                  # Optional — store your PDF files here
    └── Never Let Me Go... PDF.pdf
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rohannegi-2005/PDF-Insight-AI.git
cd PDF-Insight-AI
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables
Create a file named `.env` in the root folder:

```
ASTRA_DB_API_ENDPOINT=your_astra_db_endpoint
ASTRA_DB_TOKEN=your_astra_db_token
GROQ_API_KEY=your_groq_api_key
```

### 5️⃣ Run the App
```bash
python main.py
```

---

## 💬 Example Interaction

**Question:**
```
What is the name of the girl Samar falls in love with?
```

**Output:**
```
Answer: Kanika
```

---

## 🧩 SmartPDFProcessor Class

The built-in `SmartPDFProcessor`:
- Loads PDFs via `PyPDFLoader`
- Removes noisy text (like "Scan to Download")
- Fixes ligatures (ﬁ, ﬂ)
- Splits large text into smaller contextual chunks
- Adds metadata (page number, character count, etc.)

This ensures the embeddings are contextually meaningful and searchable.

---

## 🧱 Why LCEL (LangChain Expression Language)?

LCEL makes the RAG pipeline **cleaner and declarative**:
```python
rag_chain = (
    {"context": retriever | format_docs,
     "question": RunnablePassthrough()}
    | custom_prompt
    | llm
    | StrOutputParser()
)
```
It improves readability, modularity, and composability — making the project scalable for future extensions like APIs or UI layers.

---

## 🚀 Future Enhancements

- 🌐 Add a **Streamlit or FastAPI** interface for live Q&A   

---

## 🧠 Learning Highlights

This project demonstrates hands-on expertise in:
- RAG architecture design  
- Vector databases & semantic retrieval  
- Prompt engineering and LCEL pipelines  
- LLM integration (Groq / LangChain)  
- Clean, secure, reproducible AI project structure  

---

## 👨‍💻 Author

**Rohan Negi**  
💼 AI Developer | Enthusiast in LLMs, RAG, and applied NLP  
📧 [rohannnegi2005@gmail.com](mailto:rohannnegi2005@gmail.com)  
🌐 [GitHub Profile](https://github.com/rohannegi-2005)  

---


### 💬 Note

> This repository demonstrates practical expertise in **Retrieval-Augmented Generation (RAG)** pipelines — including embeddings, vector stores, LangChain LCEL chaining, and Groq LLM integration.  
> The project reflects strong foundations in **applied AI engineering**, **data pipelines**, and **modern GenAI tooling**, making it an excellent showcase of both **software engineering discipline** and **AI problem-solving ability**.
