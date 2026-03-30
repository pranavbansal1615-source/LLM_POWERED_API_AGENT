# 🤖 LLM-Powered Smart API Agent

## 📌 About the Project

The **LLM-Powered Smart API Agent** is an intelligent system designed to make working with API documentation easier and faster for developers.

API documentation is often messy, unstructured, and difficult to navigate. Developers usually spend a lot of time searching for correct endpoints, parameters, and usage examples. Our project addresses this problem by allowing users to **ask questions in plain English** and receive **accurate, structured, and executable API information**.

---

## ❓ Problem Statement

- API documentation is often scattered across large HTML or Markdown files.
- Finding the correct endpoint and usage example is time-consuming.
- Existing tools lack conversational, AI-powered assistance.
- Developers are forced to manually read and interpret documentation.

---

## 💡 Proposed Solution

The **Smart API Agent** uses **Large Language Models (LLMs) combined with a Retrieval-Augmented Generation (RAG) pipeline** to understand API documentation and respond intelligently to user queries.

Users can ask questions like:
> *“How do I get a list of users?”*

And the system responds with:
- The correct API endpoint
- A structured JSON representation
- Ready-to-run code snippets (cURL / Python)

---

## ⚙️ How the System Works (High Level)

1. API documentation is ingested from HTML or Markdown sources.
2. The content is cleaned and split into meaningful chunks.
3. Each chunk is converted into vector embeddings.
4. Embeddings are stored in a vector database for fast retrieval.
5. When a user asks a question:
   - The query is embedded
   - Relevant documentation is retrieved
   - The LLM generates a structured and accurate response

This approach avoids hallucinations and ensures answers are grounded in real documentation.

---

## 🧠 Tech Stack

### 🔹 AI & Retrieval (RAG Pipeline)
- **Sentence Transformers**: Generate embeddings for semantic search
- **ChromaDB**: Vector database for storing and retrieving embeddings
- **LangChain**: Text splitting, document loading, and orchestration
- **PyMuPDF**: PDF extraction and parsing
- **Transformers & PyTorch**: State-of-the-art language models

### 🔹 Backend
- **FastAPI**: High-performance REST API
- **Uvicorn**: ASGI server
- **Python-dotenv**: Environment variable management

### 🔹 Frontend
- **Streamlit**: Interactive web interface for rapid prototyping
- **Gradio**: Alternative UI option

### 🔹 DevOps
- **Docker**: Containerization for easy deployment

---

## � Prerequisites

Before running the application, ensure you have:
- Python 3.11 or higher
- Docker (optional, for containerized deployment)
- 4GB+ RAM (recommended for LLM operations)
- Internet connection (for downloading models)

---

## 🚀 Running the Application

### Streamlit
```bash
streamlit run app.py
```

### FastAPI
```bash
python main.py
```

---

## � License & Support

Open source for educational purposes. For issues or questions, please refer to the repository documentation.


