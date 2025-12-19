from sentence_transformers import SentenceTransformer
from typing import List,Any
import numpy as np
##created the embedding manager

class EmbeddingManager:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        
        self.model_name = model_name
        self.model = None
        self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        
        embeddings = embeddings = embeddings = self.model.encode(
                                    texts,
                                    batch_size=64,
                                    normalize_embeddings=True
                                    )
        return embeddings
    