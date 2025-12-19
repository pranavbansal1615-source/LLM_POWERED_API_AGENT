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
from typing import List
import numpy as np
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

##processing all the pdf files into text 

def process_all_texts(path):
    all_documents = []
    pdf_dir = Path(path)
    pdf_docs = list(pdf_dir.glob("**/*.pdf"))

    for pdf_file in pdf_docs:
        new_data = PyMuPDFLoader(str(pdf_file))
        docs = new_data.load()

        for doc in docs:
            doc.metadata["file_name"] = pdf_file.name
            doc.metadata["file_type"] = "pdf"

        all_documents.extend(docs)
    
    return all_documents

pdf_files_path = "C:\\Users\\Pranav Bansal\\Documents\\LLM_POWERED_API_AGENT\\pdf_files"
docs = process_all_texts(pdf_files_path)

import re

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
        separators=["\n\n","\n"," ",""]
    )

    splitted_text = text_splitter.split_documents(documents)
    return splitted_text

for d in docs:
    d.page_content = clean_page_text(d.page_content)

chunks = split_docs(docs,2000,200)

class EmbeddingManager:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        
        self.model_name = model_name
        self.model = None
        self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        
        embeddings = self.model.encode(texts)
        return embeddings
    
texts = [doc.page_content for doc in chunks]

embedding_manager=EmbeddingManager()
embeddings = embedding_manager.generate_embeddings(texts)

import os
import uuid
from typing import List, Any
import numpy as np
import chromadb


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
        

vectorstore=VectorStore()
vectorstore.add_documents(chunks,embeddings)

def retrieve_top_docs(query: str, top_k: int = 3):
    q_emb = embedding_manager.generate_embeddings([query])[0].tolist()
    results = vectorstore.collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    dists = results.get('distances', [[]])[0]
    return list(zip(docs, metas, dists))

def build_context(top_docs):
    
    context = "\n\n".join([doc for doc, meta, dist in top_docs])
    return context

from langchain_groq import ChatGroq

llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=500
    )
    
def build_context(top_docs):
    
    context = "\n\n".join([doc for doc, meta, dist in top_docs])
    return context

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant.\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    )
)

def answer_query(query):
    top_docs = retrieve_top_docs(query)
    context = build_context(top_docs)
    formatted_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(formatted_prompt)  
    return response.content.strip()

query = input("Enter the question you want to ask : ")
answer = answer_query(query)
# print("\n🧠 Answer:\n", answer)


from IPython.display import display, Markdown
display(Markdown(answer))




