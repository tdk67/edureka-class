from app.core.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from app.rag.loader import PDFLoader
from app.rag.chunker import LangchainTextChunker
from app.rag.embeddings import EmbeddingModel
from app.rag.vectorstore import VectorStore

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

class RAGEngine:
    """
    Singleton-style RAG Engine.
    Initialized once and serves all API queries and UI requests.
    """
    def __init__(self):
        self.embeddings = EmbeddingModel()
        self.vector_store = VectorStore(self.embeddings)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant.
        Answer the questions using only the context below.
        If the answer is not present in the context, then say:
        "I could not find the answer in the provided document(s)"

        Context:
        {context}

        Question:
        {question}

        Answer very clearly and concisely:
        """)

    def _format_docs(self, docs):
        """Extracts plain text from retrieved Document objects"""
        return "\n\n".join(doc.page_content for doc in docs)

    def ingest(self, file_paths: list) -> int:
        """Process PDFs and upload to Qdrant."""
        documents = PDFLoader(file_paths).load()
        chunker = LangchainTextChunker(CHUNK_SIZE, CHUNK_OVERLAP)
        chunks = chunker.chunk_documents(documents)
        
        self.vector_store.build(chunks)
        return len(chunks)

    def generate_answer(self, question: str) -> str:
        """Retrieve chunks and generate an answer via LangChain."""
        retriever = self.vector_store.store.as_retriever(search_kwargs={"k": TOP_K})
        
        rag_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )
        
        response = rag_chain.invoke(question)
        return response.content

    def debug_search(self, question: str):
        """Returns raw documents for UI debugging."""
        return self.vector_store.search(question, k=TOP_K)
