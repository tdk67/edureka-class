from langchain_community.document_loaders import PyPDFLoader
from typing import List

class PDFLoader:
    """
    Loads text and metadata from PDF files using LangChain's PyPDFLoader.
    """
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths

    def load(self):
        documents = []
        for path in self.file_paths:
            loader = PyPDFLoader(str(path))
            documents.extend(loader.load())
        return documents
