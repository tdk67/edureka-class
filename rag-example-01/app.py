import os
import gradio as gr
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- Configuration & Environment Variables ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = "my-db-01"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Global variable for the vector database instance
VECTOR_DB = None

# --- Connection Factories ---

def get_vector_db():
    """Safely connects to the existing vector store."""
    global VECTOR_DB
    if VECTOR_DB is not None:
        return VECTOR_DB
        
    embeddings = OpenAIEmbeddings()
    VECTOR_DB = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False, 
        https=True
    )
    return VECTOR_DB

def create_vectorstore(chunks):
    """Ingests new documents into Qdrant Cloud."""
    embeddings = OpenAIEmbeddings()
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        prefer_grpc=False, 
        force_recreate=True 
    )
    return vectorstore

# --- Core RAG Functions ---

def load_pdfs(files):
    documents = []
    for file in files:
        loader = PyPDFLoader(file.name)
        documents.extend(loader.load())
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(documents)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer the questions using only the context below.
If the answer is not present in the context, then say:
"I could not find the answer in the provided document(s)"

Context:
{context}

Question:
{question}

Answer very clearly and concisely
""")

# THIS IS THE MAGIC FIX: Extracts plain text from the retrieved Document objects
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Notice we pipe the retriever through format_docs before passing it to the prompt
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
    )
    return rag_chain

def ingest_pdf(files):
    global VECTOR_DB
    if not files:
        return "Please upload files."
    try:
        documents = load_pdfs(files)
        chunks = split_documents(documents)
        
        # Upload to Qdrant
        create_vectorstore(chunks)
        
        # Reset the connection so it pulls the fresh data cleanly
        VECTOR_DB = None 
        get_vector_db() 
        
        return f"✅ Successfully ingested {len(chunks)} chunks into '{COLLECTION_NAME}'."
    except Exception as e:
        return f"❌ Ingestion Error: {str(e)}"

def ask_question(question):
    try:
        db = get_vector_db()
        rag_chain = build_rag_chain(db)
        
        response = rag_chain.invoke(question)
        text = response.content
        
        partial = ""
        for char in text:
            partial += char
            yield partial
    except Exception as e:
        yield f"❌ Query Error: {str(e)}"

def debug_retrieval(question):
    try:
        db = get_vector_db()
        docs = db.similarity_search(question, k=3)
        
        if not docs:
            return f"No relevant chunks found for: '{question}'"

        output = f"DEBUG: Found {len(docs)} chunks.\n"
        for i, doc in enumerate(docs):
            output += f"\n--- RESULT {i+1} ---\n"
            output += f"Source: {doc.metadata.get('source','Unknown')}\n"
            output += f"{doc.page_content[:600]}...\n"
        return output
    except Exception as e:
        return f"Retrieval Debug Error: {str(e)}"

# --- Gradio UI Setup ---

def main():
    with gr.Blocks(title="Langchain based RAG PDF Assistant") as demo:
        gr.Markdown("# Langchain RAG PDF Assistant")
        gr.Markdown("Upload PDF Documents and ask questions grounded strictly in their content.")
        
        with gr.Row():
            pdf_files = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDF")
        
        ingest_btn = gr.Button("Step 1: Ingest PDF")
        ingest_status = gr.Textbox(label="Ingestion Status")
        ingest_btn.click(ingest_pdf, inputs=[pdf_files], outputs=[ingest_status])

        gr.Markdown("---")

        question = gr.Textbox(label="Step 2: Ask any question", placeholder="What does the document talk about?")
        ask_btn = gr.Button("Ask AI")
        answer_box = gr.Markdown(label="Answer")
        ask_btn.click(ask_question, inputs=[question], outputs=[answer_box])

        gr.Markdown("---")

        debug_btn = gr.Button("Debug: Show retrieved chunks")
        debug_box = gr.Textbox(label="Retrieved Chunks from Qdrant", lines=20)
        debug_btn.click(debug_retrieval, inputs=[question], outputs=[debug_box])

    demo.launch(debug=False, share=False)

if __name__ == "__main__":
    main()