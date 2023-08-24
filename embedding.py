
import sentence_transformers
import os
from config import *
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


class Embedding:
    embeddings: object = None

    def __init__(self):
        model_path = os.path.join(MODEL_CACHE_PATH, embedding_model)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.embeddings.client = sentence_transformers.SentenceTransformer(
            self.embeddings.model_name,
            device=EMBEDDING_DEVICE,
            cache_folder=model_path)
