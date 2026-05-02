import os
import openai
import gradio as gr
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from qdrant_client import QdrantClient

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- Configuration & Environment Variables ---
# Note: Ensure these are set in your environment or filled in below
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGCHAIN_API_KEY"
os.environ["LANGCHAIN_PROJECT"] = "edurekatest_observability"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

QDRANT_URL = "https://50d072d1-ed73-4cf2-b7d8-df6d633d337f.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "YOUR_QDRANT_API_KEY"

# Initialize LangSmith wrapped client
client = wrap_openai(openai.Client())

# Global variable for the vector database
VECTOR_DB = None

# --- Core RAG Functions ---

def load_pdfs(files):
    documents = []
    for file in files:
        loader = PyPDFLoader(file.name)
        documents.extend(loader.load())
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    
    # Initialize the vector store with Qdrant
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name="pdf_rag_edureka"
    )
    return vectorstore

# --- RAG Chain & Interface Logic ---

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

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
    )
    return rag_chain

def ingest_pdf(files):
    global VECTOR_DB
    if not files:
        return "Please upload files."

    documents = load_pdfs(files)
    chunks = split_documents(documents)
    VECTOR_DB = create_vectorstore(chunks)

    return f"Successfully ingested {len(chunks)} chunks from {len(files)} PDFs."

def ask_question(question):
    if VECTOR_DB is None:
        yield "Upload the files and ingest first."
        return

    rag_chain = build_rag_chain(VECTOR_DB)
    response = rag_chain.invoke(question)

    # Streaming effect for Gradio
    text = response.content
    partial = ""
    for char in text:
        partial += char
        yield partial

def debug_retrieval(question):
    global VECTOR_DB
    if VECTOR_DB is None:
        return "No DB loaded."

    docs = VECTOR_DB.similarity_search(question, k=3)
    output = ""
    for i, doc in enumerate(docs):
        output += f"\n--- RESULT {i+1} ---\n"
        output += f"Source: {doc.metadata.get('source','')}\n"
        output += f"Page: {doc.metadata.get('page','')}\n"
        output += f"Chunk ID: {doc.metadata.get('chunk_id','')}\n\n"
        output += doc.page_content[:1200]
    return output

# --- Gradio UI Setup ---

def main():
    with gr.Blocks(title="Langchain based RAG PDF Assistant") as demo:
        gr.Markdown("# Langchain RAG PDF Assistant")
        gr.Markdown("Upload PDF Documents and ask questions grounded strictly in their content.")
        
        with gr.Row():
            pdf_files = gr.File(
                file_types=[".pdf"],
                file_count="multiple",
                label="Upload PDF files"
            )
        
        ingest_btn = gr.Button("Ingest PDF")
        ingest_status = gr.Textbox(label="Ingestion Status")
        
        ingest_btn.click(
            ingest_pdf,
            inputs=[pdf_files],
            outputs=[ingest_status]
        )

        gr.Markdown("---")

        question = gr.Textbox(
            label="Ask any question",
            placeholder="What does the document talk about?"
        )

        ask_btn = gr.Button("Ask")
        answer_box = gr.Markdown(label="Answer")
        
        ask_btn.click(
            ask_question,
            inputs=[question],
            outputs=[answer_box]
        )

        gr.Markdown("---")

        debug_btn = gr.Button("Show retrieved chunks")
        debug_box = gr.Textbox(
            label="Retrieved Chunks",
            lines=20
        )

        debug_btn.click(
            debug_retrieval,
            inputs=[question],
            outputs=[debug_box]
        )

    demo.launch(debug=False, share=True)

if __name__ == "__main__":
    main()