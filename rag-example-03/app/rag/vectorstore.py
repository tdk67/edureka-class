from langchain_qdrant import QdrantVectorStore
from app.core.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME

class VectorStore:
    """
    Qdrant-based vector store for document retrieval
    """
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.store = None
        self._connect()

    def _connect(self):
        """Safely bind to the existing Qdrant collection using the proven method."""
        self.store = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings.model,
            collection_name=COLLECTION_NAME,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=False, 
            https=True
        )

    def build(self, documents):
        """
        Build and upload the index from Document chunks.
        WARNING: force_recreate=True wipes existing data before upload.
        """
        # from_documents automatically creates and returns a perfectly bound store
        self.store = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings.model,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
            prefer_grpc=False,
            force_recreate=True 
        )
        # CRITICAL FIX: We do NOT call self._connect() here anymore. 
        # Overwriting the store here was breaking the LangChain wrapper.

    def search(self, query: str, k: int = 4):
        """Retrieve top-k relevant document chunks."""
        if self.store is None:
            raise ValueError("Vector Store not initialized")
        
        return self.store.similarity_search(query, k=k)