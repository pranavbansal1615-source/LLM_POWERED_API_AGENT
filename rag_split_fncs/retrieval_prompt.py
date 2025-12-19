from langchain_groq import ChatGroq
from embeddings_manager import EmbeddingManager
from vector_store import VectorStore
import os

embedding_manager = EmbeddingManager()
vectorstore=VectorStore()

def retrieve_top_docs(query: str, top_k: int = 3):
    q_emb = embedding_manager.generate_embeddings([query])[0].tolist()
    results = vectorstore.collection.query(query_embeddings=[q_emb], n_results=top_k)
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    dists = results.get('distances', [[]])[0]
    return list(zip(docs, metas, dists))

llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1500
    )
    
def build_context(top_docs):
    context_parts = []
    for i, (doc, meta, dist) in enumerate(top_docs):
        context_parts.append(
            f"[Source {i+1} | Page {meta.get('page', 'N/A')}]\n{doc}"
        )
    return "\n\n".join(context_parts)


