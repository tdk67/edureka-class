# Project Codebase Context

> Generated on: 02/05/2026, 22:58:23
> Total Files: 13

---

## File: 2.txt
**Description:** Source code file located at `2.txt`.

```txt
    6  brew install python@3.12
    7  echo 'export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"' >> ~/.zprofile\nsource ~/.zprofile
    8  python3 --version
    9  python --version
   10  python3 --version\npip3 --version
   11  echo 'alias python=python3' >> ~/.zshrc\necho 'alias pip=pip3' >> ~/.zshrc\nsource ~/.zshrc
   12  brew install pyenv\npyenv install 3.12.10\npyenv global 3.12.10
   13  python --version
   14  brew install git\nbrew install node\nbrew install uv
   15  pip3 install openai anthropic langchain crewai autogen jupyterlab pandas numpy
   16  python3 -m venv myenv\nsource myenv/bin/activate
   17  pip install -r requirements.txt
   18  clear
   19  uvicorn app.api.main:app
   20  uvicorn app.api.main:app
   21  uvicorn app.api.main:app

```

---

## File: app\api\main.py
**Description:** Source code file located at `app\api\main.py`.

```py
from fastapi import FastAPI, Query
from app.rag.engine import RAGEngine

app=FastAPI()

rag_engine=RAGEngine()

@app.get("/query")
def query(question:str =Query(..., description="User question")):
    """
    Return an LLM generated answer, grounded using the PDF content
    """
    try:
        answer=rag_engine.generate_answer(question)
        return {
            "question":question,
            "answer":answer
        }
    except Exception as e:
        return {
            "question":question,
            "answer":None,
            "error": str(e)
        }
```

---

## File: app\core\config.py
**Description:** Source code file located at `app\core\config.py`.

```py
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent.parent

DATA_DIR=BASE_DIR / "data"
PDF_PATH= DATA_DIR / "knowledge.pdf"

CHUNK_SIZE=500
CHUNK_OVERLAP=50

EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"


TOP_K=3
```

---

## File: app\rag\chunker.py
**Description:** Source code file located at `app\rag\chunker.py`.

```py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


class LangchainTextChunker:
    """
    Uses Langchain's RecursiveCharacterTextSpiiter for production grade chunking
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk(self, text: str)-> List[str]:
        return self.splitter.split_text(text)
```

---

## File: app\rag\embeddings.py
**Description:** Source code file located at `app\rag\embeddings.py`.

```py
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingModel:
    """
    Wrapper around Sententransformer embeddings
    """

    def __init__(self, model_name:str):
        self.model=HuggingFaceEmbeddings(
            model_name=model_name
        )

    def embed_documents(self,texts):
        return self.model.embed_documents(texts)

    def embed_query(self, query:str):
        return self.model.embed_query(query)
    
```

---

## File: app\rag\engine.py
**Description:** Source code file located at `app\rag\engine.py`.

```py
from app.core.config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME, TOP_K
from app.rag.loader import PDFLoader
from app.rag.chunker import LangchainTextChunker
from app.rag.embeddings import EmbeddingModel
from app.rag.vectorstore import VectorStore

from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

class RAGEngine:
    """
    Singleton-style RAG Engine.
    Initialized once and serves all queries
    """

    def __init__(self):
        self.vector_store=None
        self._initialize()

    def _initialize(self):
        load_dotenv()

        text=PDFLoader(PDF_PATH).load()

        chunks=LangchainTextChunker(CHUNK_SIZE, CHUNK_OVERLAP).chunk(text)

        embeddings=EmbeddingModel(EMBEDDING_MODEL_NAME)

        self.vector_store=VectorStore(embeddings)
        self.vector_store.build(chunks)

        self.llm=ChatGroq(model_name="llama-3.3-70b-versatile")

    def generate_answer(self, question:str):
        """
        Geneate an answer using the vectore store with a grounded prompt.
        Retrieve top -k chunls and pass them to llm with a strict prompt
        """
        contexts=self.vector_store.search(query=question, k=TOP_K)
        combined_text="\n\n".join(contexts)

        prompt_template=f"""
        You are a helpfile assistant. Use only the information provided in the context below to answer the question.
        If the answer is not present in the context, respond with "I don't know"

        Context: {combined_text}

        question: {question}

        answer:
        """

        agent=create_agent(
            model=self.llm,
            system_prompt="You are a helpful assistant"
        )

        result=agent.invoke({
            "messages":[
                {"role":"user", "content":prompt_template}
            ]
        }) 

        return result['messages'][-1].content
```

---

## File: app\rag\loader.py
**Description:** Source code file located at `app\rag\loader.py`.

```py
from pypdf import PdfReader
from pathlib import Path

class PDFLoader:
    """
    Used only for loading text from the pdf file    
    """
    def __init__(self, pdf_path: Path):
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {pdf_path}")
        self.pdf_path=pdf_path

    def load(self) -> str:
        reader=PdfReader(self.pdf_path)
        pages_text=[]

        for page in reader.pages:
            text=page.extract_text()
            if text:
                pages_text.append(text)

        return "\n".join(pages_text)
```

---

## File: app\rag\vectorstore.py
**Description:** Source code file located at `app\rag\vectorstore.py`.

```py
from langchain_community.vectorstores import FAISS

class VectorStore:
    """
    FAIIS-based vector store for document retreival
    """

    def __init__(self, embeddings):
        self.embeddings=embeddings
        self.store=None

    def build(self, texts):
        """
        Build FAISS inde from text chunks.
        """
        self.store=FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings.model
        )

    def search(self, query:str, k:int=3):
        """
        Retrive tok-k relvant chunks
        """
        if self.store is None:
            raise ValueError("Vector Store not initialized")
    
        docs=self.store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
        
```

---

## File: Docker\Dockerfile
**Description:** Source code file located at `Docker\Dockerfile`.

```txt
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY .env .env

EXPOSE 8001

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port","8001"]
```

---

## File: k8s\deployment.yaml
**Description:** Source code file located at `k8s\deployment.yaml`.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-app-livedemo

spec:
  replicas: 2

  selector:
    matchLabels:
      app: rag-app-livedemo

  template:
    metadata:
      labels:
        app: rag-app-livedemo

    spec:
      containers:
      - name: rag-app-livedemo
        image: rag-app-livedemo:latest
        imagePullPolicy: Never

        ports:
        - containerPort: 8001
```

---

## File: k8s\service.yaml
**Description:** Source code file located at `k8s\service.yaml`.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-service

spec:
  selector:
    app: rag-app-livedemo

  ports:
  - port: 80
    targetPort: 8001

  type: NodePort
```

---

## File: requirements.txt
**Description:** Source code file located at `requirements.txt`.

```txt
fastapi
uvicorn
pypdf
langchain
langchain-community
langchain-text-splitters
sentence-transformers
faiss-cpu
langchain_huggingface
langchain-groq
```

---

## File: setenv.bat
**Description:** Source code file located at `setenv.bat`.

```bat
set PYTHON_HOME=C:\Data\Apps\python-3.12.10
set PATH=%PYTHON_HOME%;%PYTHON_HOME%\Scripts;%PYTHON_HOME%\DLLs;;%PYTHON_HOME%\Lib;%PATH%

```

---

