import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide TF INFO + WARNING logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # also disables oneDNN (and its info message)

from dotenv import load_dotenv
load_dotenv()

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
import fitz 
import pytesseract
from PIL import Image
import io
from langchain_core.documents import Document
import re
import torch
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

##processing all the pdf files into text 


def process_pdf_hybrid(pdf_path: str, text_threshold: int = 50):
    doc = fitz.open(pdf_path)
    docs = []

    for i, page in enumerate(doc):
        text = page.get_text()

        if len(text.strip()) < text_threshold:
            # Fallback to OCR
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            text = pytesseract.image_to_string(img)

        docs.append(
            Document(
                page_content=text.strip(),
                metadata={
                    "source_file": pdf_path.split("\\")[-1],
                    "page_number": i + 1,
                    "file_type": "pdf"
                }
            )
        )

    return docs

# pdf_files_path = "C:\\Users\\Pranav Bansal\\Documents\\LLM_POWERED_API_AGENT\\pdf_files"

# from pathlib import Path
# pdf_dir = Path(pdf_files_path)
# pdf_files = list(pdf_dir.glob("*.pdf"))

# docs = []

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
                                    normalize_embeddings=True
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


embedding_manager=EmbeddingManager()
vectorstore=VectorStore()

def generate_pdf_emb(file_path:str, user_id:str, document_id:str):

    #extracting the text from pdf
    docs = process_pdf_hybrid(file_path)


    for d in docs:
        d.page_content = clean_page_text(d.page_content)

    chunks = split_docs(docs,2000,200)

    texts = [doc.page_content for doc in chunks]

    embeddings = embedding_manager.generate_embeddings(texts)

    for chunk in chunks:
        chunk.metadata["document_id"] = document_id
        chunk.metadata["user_id"] = user_id

    vectorstore.add_documents(chunks,embeddings)      

# generate_pdf_emb()

def retrieve_top_docs(query: str, document_id:str, top_k: int = 5):
    q_emb = embedding_manager.generate_embeddings([query])[0].tolist()
    results = vectorstore.collection.query(query_embeddings=[q_emb], 
                                           n_results=top_k,
                                            where={"document_id":document_id}
                                           )
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    dists = results.get('distances', [[]])[0]
    return list(zip(docs, metas, dists))


from langchain_groq import ChatGroq

llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=2000
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
        You are a highly precise and reliable assistant designed to answer questions using retrieved document context.
        
        Instructions:
        
        * Carefully read all the provided context before answering.
        * Extract only the relevant information needed to answer the question.
        * If the answer spans multiple context chunks, combine them logically into a single coherent response.
        * Remove any irrelevant elements such as page numbers, headers, footers, tags, or formatting artifacts.
        * Pay attention to small but important details to ensure accuracy and completeness.
        * If the available context is very small, sparse, or lacks clarity, you are allowed to enhance and expand the response to make it more useful and complete, while clearly distinguishing what is inferred or supplemented.
        
        Code Handling Rules (HIGH PRIORITY BUT CONDITIONAL):
        
        * Detect if the context contains code snippets, commands, or configurations.
        * Only include code in the final answer **if it is directly relevant and necessary to answer the question**.
        * Do NOT generate or include code if the question is conceptual, explanatory, or does not require code.
        * Preserve code exactly as it appears — do NOT modify variable names, syntax, indentation, or structure unless correcting obvious formatting issues.
        * Always present code in properly formatted code blocks when included.
        * Clearly separate code from explanation.
        * If multiple code snippets exist, organize them logically with short explanations.
        * If code is incomplete in the context:
        
          * Clearly mention that it is incomplete.
          * Optionally complete it using general knowledge, but label it as: "Completed/Extended version (inferred)".
        * When both code and text are present, prioritize whichever is more relevant to the question.
        
        Context Handling Rules:
        
        1. If the answer is fully or partially present in the context:
        
           * Provide a clear, structured, and concise answer based strictly on the context.
           * Do not introduce assumptions unless necessary for coherence.
        
        2. If the relevant context is NOT present:
        
           * Clearly state: "The provided context does not contain sufficient information to answer this question."
           * Then, provide a best-effort answer using your general knowledge of the topic.
           * Clearly separate this section by saying: "Based on general knowledge:"
        
        3. If the context is partially relevant:
        
           * Use the available context first.
           * Then supplement missing details using general knowledge, clearly indicating the transition.
        
        Document-Type Specific Instructions:
        
        A. For API Documentation:
        
        * Focus on endpoints, request/response structure, parameters, authentication, and error handling.
        * Include code examples **only if they help clarify usage (e.g., request/response examples)**.
        * Present the answer in a structured format (e.g., Endpoint, Method, Headers, Parameters, Request Example, Response Example).
        * Be precise and technical; avoid unnecessary explanations.
        
        B. For General / Non-API Documents:
        
        * Provide a clear, well-explained, and easy-to-understand answer.
        * Include code **only if it is explicitly relevant or necessary** (e.g., examples, commands, scripts).
        * Clearly explain what the code does when included.
        * Maintain logical flow and readability while preserving key details.
        
        Response Formatting:
        
        * Use structured sections where appropriate.
        * Avoid excessive bullet points; prefer clear paragraphs and natural flow for better readability.
        * Use bullet points only when they significantly improve clarity (e.g., lists, steps, parameters).
        * Include code blocks only when needed.
        * Ensure the final answer is clean, readable, and well-organized without being overly fragmented.
        
        Question:
        {question}
        
        Context:
        {context}
        
        Answer:
        
        
    """
    )
)

def answer_query(query, document_id):
    top_docs = retrieve_top_docs(query,document_id)
    context = build_context(top_docs)
    formatted_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(formatted_prompt)  
    return response.content.strip()









