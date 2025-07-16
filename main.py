# main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# --- Configuration ---
load_dotenv()

CHROMA_DATA_PATH = "./chroma_data"
CHROMA_COLLECTION_NAME = "jasper_data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_NAME = "llama-3.1-8b-instant"
SIMILARITY_THRESHOLD = 1.0
NUM_RELEVANT_CHUNKS = 3

# <--- CHANGE: A more intelligent and robust system prompt --->
SYSTEM_PROMPT = """You are Sarah, a friendly and professional customer support representative for Jasper IT Solutions LLC.
Your goal is to have a natural conversation and answer user questions accurately.

Here's how you should behave:
1.  **Use the Knowledge Base:** Refer to the "Provided Context from Knowledge Base" to answer the user's most recent question. This is your primary source of truth.
2.  **Use the Conversation History:** Refer to the "Conversation History" to understand the flow of the conversation and to answer follow-up questions.
3.  **Be Concise and Friendly:** Provide clear and helpful answers.
4.  **If the Answer Isn't in the Knowledge Base:** If you cannot find the answer in the "Provided Context", say "I'm sorry, I don't have that information right now, but I can connect you with someone who does. You can reach us at (866) 771-6669 or info@jasperitinc.com." Do not make up answers.
"""

# (The rest of the file is identical to the previous version)
resources = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading resources for FastAPI...")
    resources["embedding_model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ Embedding model loaded.")
    try:
        persistent_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
        collection = persistent_client.get_collection(CHROMA_COLLECTION_NAME)
        resources["chroma_collection"] = collection
        print(f"✅ Connected to ChromaDB collection: '{CHROMA_COLLECTION_NAME}'")
    except Exception as e:
        print(f"❌ ERROR: Could not connect to ChromaDB: {e}")
        resources["chroma_collection"] = None
    if not GROQ_API_KEY:
        print("❌ ERROR: GROQ_API_KEY not found.")
        resources["groq_client"] = None
    else:
        resources["groq_client"] = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client initialized.")
    print("👍 FastAPI resources loaded.")
    yield
    print("💧 Cleaning up resources...")
    resources.clear()

app = FastAPI(title="Jasper IT Chatbot API", lifespan=lifespan)

allowed_origins = [
    "https://axiom-web.netlify.app",
    "http://localhost", "http://localhost:8080",
    "http://127.0.0.1", "http://127.0.0.1:5500",
    "http://localhost:5500"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str; content: str
class QueryRequest(BaseModel):
    messages: List[ChatMessage]
class SourceDocument(BaseModel):
    content: str; source: str | None = None; distance: float | None = None
class QueryResponse(BaseModel):
    answer: str; sources: List[SourceDocument] | None = None

@app.post("/api/chat", response_model=QueryResponse)
async def chat_with_bot(request: QueryRequest):
    embedding_model = resources.get("embedding_model")
    chroma_collection = resources.get("chroma_collection")
    groq_client = resources.get("groq_client")
    if not all([embedding_model, chroma_collection, groq_client]):
        raise HTTPException(status_code=503, detail="Backend services not initialized.")
    if not request.messages or request.messages[-1].role != 'user':
        raise HTTPException(status_code=400, detail="Invalid request.")
    latest_user_message = request.messages[-1].content
    question_embedding = embedding_model.encode(latest_user_message).tolist()
    try:
        results = chroma_collection.query(
            query_embeddings=[question_embedding], n_results=NUM_RELEVANT_CHUNKS,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying DB: {e}")
    
    confident_sources, context_for_llm = [], ""
    distances = results.get('distances', [[]])[0]
    for i, doc in enumerate(results.get('documents', [[]])[0]):
        if distances[i] < SIMILARITY_THRESHOLD:
            source = results['metadatas'][0][i].get("source", "Unknown")
            confident_sources.append(SourceDocument(content=doc, source=source, distance=distances[i]))
            context_for_llm += f"--- Source: {source} ---\n{doc}\n\n"
    
    if not confident_sources:
        return QueryResponse(answer="I'm sorry, I don't have that information right now, but I can connect you with someone who does. You can reach us at (866) 771-6669 or info@jasperitinc.com.", sources=[])

    # <--- CHANGE: Renaming the context for clarity in the prompt --->
    final_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"**Provided Context from Knowledge Base:**\n{context_for_llm}"},
        {"role": "system", "content": f"**Conversation History:**"}
    ]
    final_messages.extend([{"role": msg.role, "content": msg.content} for msg in request.messages])

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=final_messages, model=LLM_MODEL_NAME, temperature=0.2
        )
        llm_answer = chat_completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with LLM API: {e}")

    return QueryResponse(answer=llm_answer, sources=confident_sources)

@app.get("/")
async def root():
    return {"message": "Jasper IT Chatbot Backend is running!"}