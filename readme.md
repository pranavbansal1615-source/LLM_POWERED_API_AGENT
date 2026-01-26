# 📄 PDF Question Answering System (RAG-Based)

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that enables users to upload PDF documents and ask natural language questions based on their content. The system extracts text from both digital and scanned PDFs, converts the text into embeddings, stores them in a vector database, and retrieves relevant information to generate accurate, document-grounded answers using a Large Language Model (LLM).

The current version uses **Streamlit** for rapid prototyping and validation. The backend logic is designed to be reused and extended into a full-scale web application.

---

## 🎯 Objectives

- Extract text from digital and scanned PDFs
- Apply OCR for image-based documents
- Split documents into meaningful chunks
- Generate semantic embeddings
- Store and retrieve embeddings efficiently
- Generate accurate answers grounded in document context
- Ensure system stability using sandboxed preprocessing

---

## 🧠 Core Concepts Used

### Document Chunking & Text Splitting
Documents are divided into overlapping chunks to improve retrieval quality and avoid incomplete responses.

### Sentence Transformers
SentenceTransformer models are used to convert text into dense vector embeddings for semantic similarity search.

### Vector Database (ChromaDB)
ChromaDB stores document embeddings persistently and supports fast similarity-based retrieval.

### Retrieval-Augmented Generation (RAG)
Relevant document chunks are retrieved and passed to the LLM as context, ensuring answers are based on source documents.

### OCR for Scanned PDFs
Tesseract OCR is used to extract text from scanned PDFs, with quality checks to avoid indexing empty or noisy content.

### Sandboxing
PDF preprocessing and OCR are executed in a separate subprocess to isolate failures and maintain application stability.

---


---

## 🧪 Features Implemented

- PDF upload and indexing
- Hybrid text extraction with OCR fallback
- Validation to prevent empty embeddings
- Persistent vector storage
- Multiple queries on the same document
- Context-aware answer generation
- Defensive pipeline design

---

## ⚠️ Current Limitations

- Prototype-level Streamlit UI
- Single-user workflow
- Process-level sandboxing only
- No authentication or user isolation
- Optimized for correctness rather than scale

---

## 🚀 Future Enhancements

- Backend migration to FastAPI
- Full frontend using React or Next.js
- User authentication and multi-document support
- Container-based sandboxing (Docker or Deno)
- Web scraping integration using Scrapy
- Improved OCR preprocessing
- Retrieval quality evaluation metrics

---

## 🛠️ Tech Stack

- Python
- Streamlit
- SentenceTransformers
- ChromaDB
- LangChain
- Groq LLM API
- PyMuPDF (fitz)
- Tesseract OCR
- PyTorch

---


