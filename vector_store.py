from langchain.document_loaders import JSONLoader
from langchain.vectorstores import FAISS
from embedding import Embedding
import os


class VectorStore:
    def __init__(self, file_path, index_name):
        vector_path = os.path.join('faiss_index', index_name + '.faiss')
        if not os.path.exists(vector_path):
            embedding = Embedding().embeddings
            if index_name == 'advertising':
                docs = self.load_advertising(file_path)
            elif index_name == 'api':
                docs = self.load_api(file_path)
            vector_store = FAISS.from_documents(docs, embedding)
            vector_store.save_local('faiss_index', index_name)

    def load_advertising(self, file_path):
        loader = JSONLoader(file_path=file_path,
                            jq_schema='.docs[] | {name, status, advertiser_id, id,campaign_name, index_type, start_date, end_date, created_at,updated_at}', text_content=False)
        docs = loader.load()
        return docs

    def load_api(self, file_path):
        loader = JSONLoader(file_path=file_path,
                            jq_schema='.', text_content=False)
        docs = loader.load()
        return docs


def get_vector_store(file_path, index_name):
    VectorStore(file_path, index_name)
    vector_store = FAISS.load_local(
        'faiss_index', Embedding().embeddings, index_name)
    return vector_store
