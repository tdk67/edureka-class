# LangChain RAG PDF Assistant with Qdrant

A sophisticated Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interact with them using an LLM (GPT-4o-mini). The application uses **Qdrant** as a high-performance vector database to store and retrieve document context, ensuring answers are grounded strictly in the provided data.

## 🚀 Features
- **Multi-PDF Ingestion**: Upload and process multiple PDF files simultaneously.
- **Strict Grounding**: The AI assistant only answers based on the uploaded context.
- **Cloud-Native Vector Search**: Leverages Qdrant Cloud for efficient similarity searches.
- **Traceability**: Integrated with LangSmith for full observability of the RAG chain.
- **Interactive UI**: Built with Gradio for a seamless web-based user experience.
- **Retrieval Debugging**: A dedicated "Show retrieved chunks" feature to inspect exactly what context was sent to the LLM.

---

## 🛠 Libraries & Dependencies

This project relies on several key libraries to function:

| Library | Purpose | Why it is used? |
| :--- | :--- | :--- |
| `langchain` | Framework | The backbone that orchestrates the flow between the PDF, the Vector Store, and the LLM. |
| `langchain-openai` | LLM & Embeddings | Provides access to GPT-4o-mini for answering and OpenAI's `text-embedding-3-small` for converting text to vectors. |
| `qdrant-client` & `langchain-qdrant` | Vector Database | Handles connection to the Qdrant Cloud instance and performs fast similarity searches. |
| `pypdf` | Document Loader | Essential for parsing and extracting text content from PDF files. |
| `gradio` | User Interface | Creates the web interface for file uploads, chat, and debugging. |
| `langsmith` | Observability | Used to trace and debug the LLM chain steps in real-time. |

---

## ⚙️ Installation

1. **Set up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 How to Run

1. **Set up your API Keys**:
   Open the Python script and fill in your credentials or set them as environment variables:
   - `OPENAI_API_KEY`
   - `QDRANT_API_KEY`
   - `QDRANT_URL`
   - `LANGCHAIN_API_KEY` (Optional, for tracing)

2. **Launch the application**:
   ```bash
   python main.py
   ```

3. **Access the UI**:
   Once the script runs, it will provide a local URL (e.g., `http://127.0.0.1:7860`). Open this in your browser.

---

## 🔍 How it Works (RAG Flow)
1. **Ingestion**: PDFs are loaded via `PyPDFLoader` and split into chunks of 800 characters with a 150-character overlap.
2. **Embedding**: Text chunks are converted into numerical vectors using OpenAI Embeddings.
3. **Storage**: Vectors are stored in a **Qdrant** collection.
4. **Retrieval**: When a user asks a question, the system finds the top 4 most relevant chunks from Qdrant.
5. **Generation**: The LLM receives the question and the chunks as context to generate a concise, grounded response.

---

## 🛠 Troubleshooting

**1. "I could not find the answer in the provided document(s)"**
- Ensure you clicked the **"Ingest PDF"** button after uploading.
- Check if the question is actually covered in the text of the PDF.
- Try asking the question using keywords found in the document.

**2. Rate Limit or Authentication Errors**
- Verify your `OPENAI_API_KEY` is active and has sufficient credits.
- Double-check that your `QDRANT_URL` and `QDRANT_API_KEY` are correct in the script.

**3. PDFs not loading properly**
- Ensure the PDF is not password-protected or corrupted.
- Check the console logs for any `pypdf` parsing errors.

**4. Gradio link not opening**
- Ensure no other application is using port `7860`.
- If running on a server, ensure the port is open in your firewall.