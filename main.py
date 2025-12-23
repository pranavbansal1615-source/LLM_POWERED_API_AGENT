import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide TF INFO + WARNING logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # also disables oneDNN (and its info message)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*tf.losses.sparse_softmax_cross_entropy.*"
)

from sentence_transformers import SentenceTransformer
import uuid
from typing import List,Any
import numpy as np
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from langchain_core.documents import Document
import re
import torch
import subprocess
import sys
import json

##processing all the pdf files into text 

def run_pdf_in_sandbox(pdf_path):
    try:
        completed = subprocess.run(
            [sys.executable, "sandbox.py", pdf_path],
            capture_output=True,
            text=True,
            env={}  
        )

        if completed.returncode != 0:
            raise RuntimeError(completed.stderr)

        return json.loads(completed.stdout)

    except subprocess.TimeoutExpired:
        st.error("PDF processing timed out in sandbox.")
        return []


def clean_page_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        s = line.strip()

        # keep empty lines but compress later
        if not s:
            cleaned.append("")
            continue

        # plain page numbers: "3"
        if re.fullmatch(r"\d{1,3}", s):
            continue

        # "Page 12", "Page 12 of 123", "p. 3/10"
        if re.fullmatch(r"(page|p\.)\s*\d+(\s*(/|of)\s*\d+)?",
                        s, flags=re.IGNORECASE):
            continue

        # tiny non-text junk like "---", "•", "1/3"
        if len(s) <= 4 and not re.search(r"[A-Za-z]", s):
            continue

        cleaned.append(line)

    # collapse many blank lines
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

##split the text documents into chunks so that we can further creat embeddings

def split_docs(documents,chunk_size,chunk_overlap):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = [
            "\n```",      
            "\n#include", 
            "\ndef ",       
            "\nclass ",     
            "\n## ",
            "\n### ",
            "\n\n",
            "\n- ",
            "\n* ",
            ". ",      
            "\n",
            " ",
            ""
        ]
    )

    splitted_text = text_splitter.split_documents(documents)
    return splitted_text

##created the embedding manager

class EmbeddingManager:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        
        self.model_name = model_name
        self.model = None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_name, device = device)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        
        embeddings = self.model.encode(
                                    texts,
                                    batch_size=128,
                                    normalize_embeddings=True,
                                    )
        return embeddings
    

class VectorStore:

    def __init__(
        self,
        collection_name = "pdf_documents",
        persist_directory = r"C:\Users\Pranav Bansal\Documents\LLM_POWERED_API_AGENT\chroma_store"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Text document embeddings for RAG"}
        )

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text
        )
        
def load_embedding_manager():
    return EmbeddingManager()

def load_vectorstore():
    return VectorStore()

embedding_manager = load_embedding_manager()
vectorstore = load_vectorstore()

def retrieve_top_docs(query: str, top_k: int = 3):
    q_emb = embedding_manager.generate_embeddings([query])[0].tolist()
    results = vectorstore.collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    dists = results.get('distances', [[]])[0]
    return list(zip(docs, metas, dists))


from langchain_groq import ChatGroq

llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1000
    )
    
def build_context(top_docs):
    context_parts = []
    for i, (doc, meta, dist) in enumerate(top_docs):
        context_parts.append(
            f"[Source {i+1} | Page {meta.get('page', 'N/A')}]\n{doc}"
        )
    return "\n\n".join(context_parts)

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        """
        You are a helpful assistant.
        Use the following extracted parts of a document to answer the question.
        If the answer spans multiple parts, combine them logically.
        If the answer is not fully contained, say so clearly.
        If the answer is not found in the context then output that no such data is available.

        Context:
        {context}

        Question:
        {question}

        Answer:
    """
    )
)

def answer_query(query):
    top_docs = retrieve_top_docs(query)
    context = build_context(top_docs)
    formatted_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(formatted_prompt)  
    return response.content.strip()


import streamlit as st

if "pdf_indexed" not in st.session_state:
    st.session_state.pdf_indexed = False

st.set_page_config(page_title="PDF RAG System", layout="wide")

st.title("📄 PDF Question Answering System")
st.write("Upload a PDF and ask questions based on its content.")

uploaded_file = st.file_uploader(
    "UPLOAD A PDF FILE",
    type=["pdf"]
) 

if uploaded_file and st.button("Process & Index PDF"):
    with st.spinner("Processing PDF..."):

        if embedding_manager is None:
            embedding_manager = load_embedding_manager()

        if vectorstore is None:
            vectorstore = load_vectorstore()

        temp_path = "temp_uploaded.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        raw_docs = run_pdf_in_sandbox(temp_path)

        docs = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in raw_docs
        ]
 
        for d in docs:
            d.page_content = clean_page_text(d.page_content)

        chunks = split_docs(docs, 2000, 200)
        
        # 1. Filter chunks properly
        filtered_chunks = [
            doc for doc in chunks
            if doc.page_content and doc.page_content.strip()
        ]

        if not filtered_chunks:
            st.error("❌ No valid text found in PDF after cleaning.")
            st.stop()

        # 2. Extract text
        texts = [doc.page_content.strip() for doc in filtered_chunks]

        # 3. Generate embeddings
        embeddings = embedding_manager.generate_embeddings(texts)

        if embeddings is None or len(embeddings) == 0:
            st.error("❌ Embedding model returned empty embeddings.")
            st.stop()

        # 4. Add to vector store (ALIGNED!)
        vectorstore.add_documents(filtered_chunks, embeddings)


    st.success("PDF indexed successfully!")

st.markdown("---")
st.subheader("Ask a Question")

query = st.text_input("Enter your question")

if query:
    with st.spinner("Generating answer..."):
        answer = answer_query(query)

    st.markdown("### 🧠 Answer")
    st.markdown(answer)










