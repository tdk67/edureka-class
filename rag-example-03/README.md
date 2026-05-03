# Modular LangChain RAG PDF Assistant (FastAPI + Qdrant + Gradio)

This project provides a complete, production-ready Retrieval-Augmented Generation (RAG) application. It allows users to upload PDF documents, processes the text into vector embeddings, stores them in Qdrant, and provides both a REST API and an interactive web UI (via Gradio) to query the documents using OpenAI's GPT models.

## 🚀 Features

*   **FastAPI Backend**: Robust, high-performance API serving the RAG engine.
*   **Gradio UI**: An intuitive, interactive web interface seamlessly mounted onto the FastAPI app.
*   **Qdrant Vector Database**: Efficient storage and retrieval of vector embeddings using Qdrant Cloud (configured for WSL/network stability via HTTPS).
*   **OpenAI Integration**: Utilizes OpenAI for state-of-the-art text embeddings and intelligent answer generation.
*   **Advanced Troubleshooting Tooling**: Includes built-in mechanisms to debug retrieval and verify the health of the entire pipeline.
*   **Container Ready**: Fully configured for Docker and Kubernetes deployments.

---

## 🛠 Prerequisites & API Keys

Before installing the project, you need active API keys for OpenAI and Qdrant.

### 1. OpenAI API Key
This key is required to use OpenAI's embedding models (`text-embedding-3-small`) and generation models (e.g., `gpt-4o-mini`).

*   **Get it here:** [OpenAI Developer Platform - API Keys](https://platform.openai.com/api-keys)
*   *Note: Ensure your account is funded or you have an active API subscription, as rate limits on the free tier can cause silent failures in LangChain.*

### 2. Qdrant Cloud Cluster URL & API Key
This project uses Qdrant Cloud for vector storage.

*   **Get it here:** [Qdrant Cloud Console](https://cloud.qdrant.io/)
*   **Instructions:**
    1.  Log in and create a free tier cluster.
    2.  Once created, click on your cluster to view its details.
    3.  Find your **Cluster URL** (e.g., `https://xxxx.cloud.qdrant.io`).
    4.  Navigate to **Data Access Control** and generate a new **API Key**.

---

## ⚙️ Installation (Local Environment)

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd rag-example-03
```

### 2. Set up a Virtual Environment
It is highly recommended to isolate your dependencies.

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### Key Libraries Used:
*   `fastapi` & `uvicorn`: Web framework and ASGI server for the backend.
*   `gradio`: Framework for building the interactive web UI.
*   `langchain`, `langchain-core`, `langchain-openai`, `langchain-community`: The core RAG orchestration framework.
*   `langchain-qdrant`, `qdrant-client`: Integration with the Qdrant vector database.
*   `pypdf`: Extracts text and metadata from PDF files.
*   `python-dotenv`: Manages environment variables.

### 4. Configure Environment Variables
1.  Rename `.env.example` to `.env`.
2.  Open `.env` and paste your API keys and URL:

```text
OPENAI_API_KEY=sk-your-openai-api-key-here
QDRANT_API_KEY=your-qdrant-api-key-here
QDRANT_URL=[https://your-cluster-url.cloud.qdrant.io](https://your-cluster-url.cloud.qdrant.io)
COLLECTION_NAME=my-db-01
```

---

## 🏃 Running the Application (Local)

Start the application using Uvicorn:
```bash
uvicorn app.api.main:app --reload
```

Once the server is running, you can access:
*   **The Gradio Web UI:** [http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui)
*   **The API Documentation (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🐳 Docker Deployment

You can package both the FastAPI backend and the Gradio UI into a single Docker image.

### Build the Image
From the root of your project:
```bash
docker build -t rag-app-livedemo -f Docker/Dockerfile .
```

### Run the Container
You must pass your environment variables into the container when running it. You can do this by pointing Docker to your `.env` file:
```bash
docker run -d -p 8000:8000 --env-file .env rag-app-livedemo
```

Access the UI at `http://localhost:8000/ui`.

---

## ☸️ Kubernetes (k8s) Deployment

The project includes basic Kubernetes manifests in the `k8s/` folder. Ensure you have built the Docker image locally (as `rag-app-livedemo:latest`) or pushed it to a registry your cluster can access.

*Important: The provided manifests do not include a ConfigMap or Secret for your `.env` variables. You must mount these credentials securely into your Pods for the application to function in a real cluster.*

### Deploy to Minikube
1.  Apply the Deployment:
    ```bash
    kubectl apply -f k8s/deployment.yaml
    ```
2.  Apply the Service:
    ```bash
    kubectl apply -f k8s/service.yaml
    ```
3.  Expose the Service (Minikube specific):
    ```bash
    minikube service rag-service
    ```

---

## 📦 Prerequisites Installation (Docker & Minikube)

### Windows
*   **Docker:** Download and install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows/). Ensure WSL2 integration is enabled in Docker settings.
*   **Minikube:** We recommend using the Windows Package Manager (winget) or Chocolatey.
    ```cmd
    winget install minikube
    ```
    *(Requires Docker Desktop to be running).*

### macOS
*   **Docker:** Download and install [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/).
*   **Minikube:** Use Homebrew:
    ```bash
    brew install minikube
    ```

### Linux (Ubuntu/Debian)
*   **Docker:**
    ```bash
    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL [https://download.docker.com/linux/ubuntu/gpg](https://download.docker.com/linux/ubuntu/gpg) -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] [https://download.docker.com/linux/ubuntu](https://download.docker.com/linux/ubuntu) \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update

    # Install Docker
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ```
*   **Minikube:**
    ```bash
    curl -LO [https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64](https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64)
    sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64
    ```

---

## 🚨 Troubleshooting

**1. "I could not find the answer in the provided document(s)"**
*   Did you click the **"Ingest PDF"** button after uploading? The system needs to process the text first.
*   Use the **"Debug: Show retrieved chunks"** button in the UI. If it returns 0 chunks, your search is failing. If it returns chunks but they don't contain the answer, the LLM is correctly stating it doesn't know based on the context.

**2. Application fails to start or crashes on `similarity_search`**
*   Check your terminal logs for `pydantic` or `ValidationError`. Ensure your `.env` file variables do not contain accidental spaces or hidden Windows line endings (`\r`).
*   Verify your `OPENAI_API_KEY`. If you hit a rate limit or ran out of credits, OpenAI's embedding API will reject requests, causing the app to fail silently or crash during queries.

**3. WSL users getting "Connection Refused" when talking to Qdrant Cloud**
*   This project is configured with `prefer_grpc=False` to bypass common WSL networking bugs. If you still encounter routing issues, ensure your VPN (e.g., Cisco AnyConnect) or Windows Firewall is not aggressively blocking port 443 traffic from the WSL virtual adapter.

**4. Docker Image is too large or fails to build**
*   Ensure you have a `.dockerignore` file preventing your `.venv/` folder and `__pycache__` directories from being copied into the container context.