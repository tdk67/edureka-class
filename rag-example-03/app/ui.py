import gradio as gr
from app.rag.engine import RAGEngine

def create_gradio_app(rag_engine: RAGEngine):
    """Builds and returns the Gradio Blocks interface."""
    
    def ingest_pdf_ui(files):
        if not files:
            return "Please upload files."
        try:
            file_paths = [file.name for file in files]
            num_chunks = rag_engine.ingest(file_paths)
            return f"✅ Successfully ingested {num_chunks} chunks."
        except Exception as e:
            return f"❌ Ingestion Error: {str(e)}"

    def ask_question_ui(question):
        try:
            # Yield characters to simulate streaming in the UI
            answer = rag_engine.generate_answer(question)
            partial = ""
            for char in answer:
                partial += char
                yield partial
        except Exception as e:
            yield f"❌ Query Error: {str(e)}"

    def debug_retrieval_ui(question):
        try:
            docs = rag_engine.debug_search(question)
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

    # Build the Gradio Blocks
    with gr.Blocks(title="Modular RAG Assistant") as demo:
        gr.Markdown("# Modular Langchain RAG with FastAPI & Qdrant")
        
        with gr.Row():
            pdf_files = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDF")
        
        ingest_btn = gr.Button("Step 1: Ingest PDF")
        ingest_status = gr.Textbox(label="Ingestion Status")
        ingest_btn.click(ingest_pdf_ui, inputs=[pdf_files], outputs=[ingest_status])

        gr.Markdown("---")
        question = gr.Textbox(label="Step 2: Ask any question", placeholder="Type your query...")
        ask_btn = gr.Button("Ask AI")
        answer_box = gr.Markdown(label="Answer")
        ask_btn.click(ask_question_ui, inputs=[question], outputs=[answer_box])

        gr.Markdown("---")
        debug_btn = gr.Button("Debug: Show retrieved chunks")
        debug_box = gr.Textbox(label="Retrieved Chunks from Qdrant", lines=15)
        debug_btn.click(debug_retrieval_ui, inputs=[question], outputs=[debug_box])

    return demo
