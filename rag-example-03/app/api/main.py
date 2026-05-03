from fastapi import FastAPI, Query
import gradio as gr

from app.rag.engine import RAGEngine
from app.ui import create_gradio_app

app = FastAPI(title="RAG Backend API with UI")

# Initialize the singleton RAG engine so both API and UI share it
rag_engine = RAGEngine()

# Create the Gradio interface and mount it inside FastAPI
gradio_demo = create_gradio_app(rag_engine)
app = gr.mount_gradio_app(app, gradio_demo, path="/ui")

@app.get("/query")
def query(question: str = Query(..., description="User question")):
    """
    Return an LLM generated answer, grounded using the PDF content (API endpoint).
    """
    try:
        answer = rag_engine.generate_answer(question)
        return {
            "question": question,
            "answer": answer
        }
    except Exception as e:
        return {
            "question": question,
            "answer": None,
            "error": str(e)
        }
