import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# --- 1. Load Environment Variables ---
load_dotenv()

qdrant_url = os.getenv("QDRANT_URL", "").strip()
qdrant_key = os.getenv("QDRANT_API_KEY", "").strip()
openai_key = os.getenv("OPENAI_API_KEY", "").strip()
collection_name = "my-db-01"

print("=========================================")
print("🕵️ RAG PIPELINE QUERY TESTER (FIXED)")
print("=========================================\n")

# --- 2. Setup Clients ---
print("--- Step 1: Connecting to Services ---")
client = QdrantClient(url=qdrant_url, api_key=qdrant_key, https=True, prefer_grpc=False)
embeddings = OpenAIEmbeddings(api_key=openai_key)
print("✅ Connected to Qdrant & OpenAI.\n")

# --- 3. Check Collection ---
col_info = client.get_collection(collection_name)
print(f"--- Step 2: Collection Info ---")
print(f"📦 Name: {collection_name} | Status: {col_info.status} | Total Points: {col_info.points_count}\n")

# --- 4. Define Test Query ---
test_query = "tell me about python syntax"

# --- 5. Test Native Qdrant Search ---
print(f"--- Step 3: Native Qdrant Search for '{test_query}' ---")
try:
    # Convert text question to vector
    query_vector = embeddings.embed_query(test_query)
    
    native_results = None
    
    # Handle different versions of the Qdrant Python SDK
    if hasattr(client, 'query_points'):
        native_response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=3
        )
        native_results = native_response.points
    elif hasattr(client, 'search'):
        native_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3
        )
    else:
        print("❌ Could not find 'query_points' or 'search' in this QdrantClient version.")

    if native_results is not None:
        if not native_results:
            print("❌ Native search found 0 results.")
        else:
            for i, res in enumerate(native_results):
                score = getattr(res, 'score', 'N/A')
                print(f"  [Result {i+1}] Similarity Score: {score}")
                content = res.payload.get('page_content', 'NO PAGE CONTENT FOUND')
                print(f"  Content: {str(content)[:150]}...\n")
            
except Exception as e:
    print(f"❌ Native Query Error: {e}\n")

# --- 6. Test LangChain Search ---
print(f"--- Step 4: LangChain VectorStore Search for '{test_query}' ---")
try:
    # Initialize Langchain wrapper
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=qdrant_url,
        api_key=qdrant_key,
        prefer_grpc=False,
        https=True
    )
    
    # Perform similarity search
    docs = vectorstore.similarity_search(test_query, k=3)
    
    if not docs:
        print("❌ Langchain search found 0 results.")
    else:
        for i, doc in enumerate(docs):
            print(f"  [Result {i+1}] Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"  Content: {doc.page_content[:150]}...\n")
            
except Exception as e:
    print(f"❌ LangChain Query Error: {e}\n")

print("=========================================")
print("🏁 QUERY TEST COMPLETE")
print("=========================================")