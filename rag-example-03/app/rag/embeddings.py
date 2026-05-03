from langchain_openai import OpenAIEmbeddings

class EmbeddingModel:
    """
    Wrapper around OpenAI Embeddings
    """
    def __init__(self):
        self.model = OpenAIEmbeddings()

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, query: str):
        return self.model.embed_query(query)
