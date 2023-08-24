import torch
import os

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
llm_model = "jumpstart-dft-meta-textgeneration-llama-2-13b-f"

EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'model_cache')
