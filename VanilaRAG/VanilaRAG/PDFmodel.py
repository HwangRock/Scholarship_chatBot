# model.py
import os
import re
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class ScholarshipModel:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.retriever = None

    def custom_split(self, data):
        pattern = r'(?=\n?[^\n]+\(은\(는\))' 
        return re.split(pattern, data)

    def limit_chunk_size(self, chunks, max_length=3000):
        limited_chunks = []
        for chunk in chunks:
            while len(chunk) > max_length:
                limited_chunks.append(chunk[:max_length])
                chunk = chunk[max_length:]
            limited_chunks.append(chunk)
        return limited_chunks

    def load_and_prepare_data(self):
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        data = "\n".join([page.page_content for page in pages])
        
        chunks = self.custom_split(data)
        chunks = self.limit_chunk_size(chunks)
        
        embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        db = Chroma.from_texts(chunks, embeddings_model)
        self.retriever = db.as_retriever()

    def get_retriever(self):
        if not self.retriever:
            raise ValueError("Data not prepared. Call `load_and_prepare_data()` first.")
        return self.retriever
