# ⚡ LLM-Powered Smart API Agent

> Upload API documentation → Ask questions in plain English → Get accurate answers with executable code snippets

---

## 📌 About the Project

The **LLM-Powered Smart API Agent** is an intelligent document question-answering system designed for developers who work with API documentation. Instead of manually combing through large PDFs, users can **upload any API documentation** and **ask questions in natural language** to get structured, accurate answers — including ready-to-run code snippets.

The system uses **Retrieval-Augmented Generation (RAG)** combined with **Large Language Models (LLMs)** to ensure answers are grounded in the actual documentation, minimizing hallucinations and maximizing accuracy.

---

## ❓ Problem Statement

- API documentation is often large, unstructured, and difficult to navigate.
- Developers waste time manually searching for correct endpoints, parameters, and usage examples.
- Existing tools lack **conversational, AI-powered assistance** for documentation.
- Code examples in docs are hard to extract and test quickly.

---

## 💡 Solution

Upload a PDF → The system **parses, chunks, and indexes** the document using vector embeddings → Ask any question → The **LLM retrieves relevant context** and generates a precise answer with code snippets.

**Key Features:**
- 📄 **PDF Upload & Processing** — Supports both text-based and scanned (OCR) PDFs
- 🔍 **Semantic Search** — Finds the most relevant documentation chunks using embeddings
- 🤖 **LLM-Powered Answers** — Generates structured responses with code blocks, endpoint details, and explanations
- 🐍 **In-Browser Python Sandbox** — Run Python code snippets directly in the browser (Pyodide-powered with `requests` pre-installed)
- 💬 **Multi-Conversation Support** — Maintain separate chat threads per document
- 🔐 **User Authentication** — Email-based login with persistent sessions

---

## 🧠 Tech Stack

### AI & RAG Pipeline
| Technology | Purpose |
|---|---|
| **Sentence Transformers** (`all-MiniLM-L6-v2`) | Generate embeddings for semantic search |
| **ChromaDB** | Vector database for storing and retrieving embeddings |
| **LangChain** | Text splitting, prompt templates, and orchestration |
| **Groq API** (`llama-3.3-70b-versatile`) | LLM inference for generating answers |
| **PyMuPDF (fitz)** | PDF text extraction |
| **Tesseract OCR** | Fallback OCR for scanned/image-based PDFs |

### Backend
| Technology | Purpose |
|---|---|
| **FastAPI** | REST API server |
| **SQLAlchemy + MySQL** | Database ORM and storage |
| **Uvicorn** | ASGI server |
| **python-dotenv** | Environment variable management |

### Frontend
| Technology | Purpose |
|---|---|
| **React 19** (Vite) | UI framework |
| **Pyodide** | In-browser Python runtime for the sandbox |
| **react-markdown** | Render markdown responses |
| **react-syntax-highlighter** | Code block syntax highlighting |

### DevOps
| Technology | Purpose |
|---|---|
| **Docker** | Containerized deployment |

---

## ⚙️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  React UI   │────▶│  FastAPI      │────▶│  ChromaDB      │
│  (Vite)     │     │  Backend      │     │  Vector Store   │
│             │     │               │     └────────────────┘
│  - Chat     │     │  - Auth       │
│  - Sidebar  │     │  - PDF Upload │     ┌────────────────┐
│  - Sandbox  │     │  - RAG Query  │────▶│  Groq LLM API  │
└─────────────┘     │  - Messages   │     │  (Llama 3.3)   │
                    └──────────────┘     └────────────────┘
                           │
                    ┌──────────────┐
                    │  MySQL DB    │
                    │  (Users,     │
                    │   Documents, │
                    │   Messages)  │
                    └──────────────┘
```

---

## 📋 Prerequisites

Before running the application, ensure you have:

- **Python 3.11+**
- **Node.js 18+** and **npm**
- **MySQL** server running locally
- **Tesseract OCR** installed ([Download](https://github.com/tesseract-ocr/tesseract))
- **Groq API Key** ([Get one free](https://console.groq.com/))
- **4GB+ RAM** (recommended for embedding model)
- Internet connection (for Groq API and downloading models on first run)

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/LLM_POWERED_API_AGENT.git
cd LLM_POWERED_API_AGENT
```

### 2. Set Up the Backend

#### a) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

#### b) Install Python dependencies

```bash
pip install -r requirements.txt
```

#### c) Set up MySQL database

Create a MySQL database called `llm_app`:

```sql
CREATE DATABASE llm_app;
```

#### d) Configure environment variables

Create a `.env` file inside the `connector/` directory:

```env
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=llm_app
GROQ_API_KEY=your_groq_api_key_here
```

#### e) Start the FastAPI backend

```bash
cd connector
uvicorn fastApiConnector:app --reload --port 8000
```

The backend will be running at `http://127.0.0.1:8000`

You can verify it by visiting `http://127.0.0.1:8000/docs` for the Swagger UI.

---

### 3. Set Up the Frontend

Open a **new terminal** and run:

```bash
cd LLM_project
npm install
npm run dev
```

The frontend will be running at `http://localhost:5173`

---

### 4. Using the Application

1. Open `http://localhost:5173` in your browser
2. Enter your email to log in (creates account automatically)
3. **Upload a PDF** — Click "Upload PDF" in the sidebar
4. **Start a chat** — Click "+ New Chat" to create a conversation for the uploaded document
5. **Ask questions** — Type your question and press Enter
6. **Run code** — Use the Python sandbox on the right to test any code snippets from the responses

---

## 🐳 Docker Deployment

```bash
# Build the Docker image
docker build -t llm-api-agent .

# Run the container
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_groq_api_key \
  -e DB_USER=root \
  -e DB_PASSWORD=your_password \
  -e DB_HOST=host.docker.internal \
  -e DB_NAME=llm_app \
  llm-api-agent
```

> **Note:** The Docker image runs the backend only. Run the frontend separately using `npm run dev`.

---

## 📁 Project Structure

```
LLM_POWERED_API_AGENT/
│
├── connector/                  # Backend (FastAPI)
│   ├── fastApiConnector.py     # API routes (auth, upload, chat, ask)
│   ├── temp.py                 # RAG pipeline (embeddings, retrieval, LLM)
│   ├── database.py             # SQLAlchemy database connection
│   ├── databasemodels.py       # ORM models (Users, Documents, etc.)
│   └── .env                    # Environment variables
│
├── LLM_project/                # Frontend (React + Vite)
│   ├── src/
│   │   ├── App.jsx             # Root component (auth routing)
│   │   ├── chatbot.jsx         # Main chat interface
│   │   ├── sandbox.jsx         # In-browser Python sandbox
│   │   ├── side_bar.jsx        # Sidebar (PDFs, chats)
│   │   ├── login.jsx           # Login page
│   │   └── index.css           # Global styles
│   ├── index.html              # Entry HTML (loads Pyodide)
│   └── package.json            # Frontend dependencies
│
├── rag_split_fncs/             # Modular RAG utilities
│   ├── process_pdfs.py         # PDF processing
│   ├── chunking_cleaning.py    # Text cleaning & splitting
│   ├── embeddings_manager.py   # Embedding generation
│   ├── vector_store.py         # ChromaDB vector store
│   └── retrieval_prompt.py     # Retrieval & prompt building
│
├── chroma_store/               # ChromaDB persistent storage
├── main.py                     # Standalone Streamlit app
├── sandbox.py                  # PDF processing sandbox
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
└── readme.md                   # This file
```

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api` | User login/register (email) |
| `POST` | `/api/documents` | Upload a PDF document |
| `POST` | `/api/conversations` | Create a new conversation |
| `POST` | `/api/ask` | Ask a question about a document |
| `POST` | `/api/messages` | Save a message |
| `GET` | `/api/messages/{conversation_id}` | Get messages for a conversation |
| `GET` | `/api/documents/{user_id}` | Get user's documents |
| `GET` | `/api/conversations/{document_id}` | Get conversations for a document |
| `GET` | `/api/user-data/{user_id}` | Get all user data (docs + chats) |

---

## 🔮 How It Works (Technical)

1. **PDF Upload** → PyMuPDF extracts text; falls back to Tesseract OCR for scanned pages
2. **Text Cleaning** → Removes page numbers, headers/footers, and formatting artifacts
3. **Chunking** → RecursiveCharacterTextSplitter with code-aware separators (2000 chars, 200 overlap)
4. **Embedding** → Sentence Transformers (`all-MiniLM-L6-v2`) generates 384-dim embeddings
5. **Indexing** → Embeddings stored in ChromaDB with document/user metadata
6. **Query** → User question is embedded → top-5 similar chunks retrieved → injected as context
7. **Answer** → Groq LLM (Llama 3.3 70B) generates a structured answer with code blocks

---

## 📜 License

Open source for educational and research purposes. Feel free to fork, modify, and extend.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
