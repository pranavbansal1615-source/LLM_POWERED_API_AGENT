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

from retrieval_prompt import retrieve_top_docs, build_context,llm
from vector_store import VectorStore
from process_pdfs import Path, process_pdf_hybrid
from chunking_cleaning import clean_page_text,split_docs
from embeddings_manager import EmbeddingManager
from langchain_core.prompts import PromptTemplate
from rich.console import Console
from rich.markdown import Markdown

pdf_files_path = "C:\\Users\\Pranav Bansal\\Documents\\LLM_POWERED_API_AGENT\\pdf_files"
docs = []

pdf_dir = Path(pdf_files_path)
pdf_files = list(pdf_dir.glob("*.pdf"))

for pdf in pdf_files:
    doc = process_pdf_hybrid(str(pdf))
    docs.extend(doc)


for d in docs:
    d.page_content = clean_page_text(d.page_content)

chunks = split_docs(docs,2000,200)

texts = [doc.page_content for doc in chunks]
embedding_manager=EmbeddingManager()
embeddings = embedding_manager.generate_embeddings(texts)
vectorstore=VectorStore()
vectorstore.add_documents(chunks,embeddings)

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


console = Console()

while True:
    query = input("Enter the question you want to ask : ")
    if str.lower(query) == "exit":
        break
    answer = answer_query(query)

    console.print(Markdown(answer))


