from langchain_text_splitters import RecursiveCharacterTextSplitter

class LangchainTextChunker:
    """
    Uses Langchain's RecursiveCharacterTextSplitter for production grade chunking.
    """
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk_documents(self, documents):
        return self.splitter.split_documents(documents)
