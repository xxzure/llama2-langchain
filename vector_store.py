from langchain.document_loaders import JSONLoader
from langchain.vectorstores import FAISS
from embedding import Embedding
import os


class VectorStore:
    def __init__(self, file_path):
        vector_path = os.path.join('faiss_index')
        if not os.path.exists(vector_path):
            embedding = Embedding().embeddings
            docs = self.load_file(file_path)
            vector_store = FAISS.from_documents(docs, embedding)
            vector_store.save_local('faiss_index', 'advertising')

    def load_file(self, file_path):
        loader = JSONLoader(file_path=file_path,
                            jq_schema='.docs[] | {name, status, advertiser_id, id, start_date, end_date}', text_content=False)
        docs = loader.load()
        return docs


def get_vector_store(file_path):
    VectorStore(file_path)
    vector_store = FAISS.load_local(
        'faiss_index', Embedding().embeddings, 'advertising')
    return vector_store
