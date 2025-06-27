# main.py
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware # For frontend interaction
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv() # Load environment variables from .env

CHROMA_DATA_PATH = "./chroma_data"
CHROMA_COLLECTION_NAME = "hr_policies"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" # Or 'sentence-transformers/all-MiniLM-L6-v2'
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_NAME = "llama-3.1-8b-instant" # Or "llama3-8b-8192" or "gemma-7b-it" - check Groq for available models

# Confidence threshold for RAG (adjust based on testing)
# This is for ChromaDB's distance. Lower is better (more similar).
# For cosine similarity, higher is better. Sentence Transformers typically use cosine.
# ChromaDB by default uses L2 distance for SentenceTransformer embeddings if not specified.
# Let's assume L2, so lower is better. A value around 0.5-1.0 might be a starting point.
# If using cosine similarity (e.g. collection.query(..., include=['distances'], query_embeddings=..., metadata={"hnsw:space": "cosine"}))
# then a threshold like 0.7-0.8 would mean high similarity.
# For simplicity with default L2:
SIMILARITY_THRESHOLD = 1.0 # Adjust this! Lower means stricter.
NUM_RELEVANT_CHUNKS = 3 # Number of chunks to retrieve

# --- Initialize FastAPI App ---
app = FastAPI()

# --- CORS Middleware (for local frontend development) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for simplicity in local dev
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- Global Variables / Load Models on Startup ---
# These will be loaded once when the FastAPI app starts.
embedding_model = None
chroma_collection = None
groq_client = None

@app.on_event("startup")
async def startup_event():
    global embedding_model, chroma_collection, groq_client
    print("Loading resources for FastAPI...")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")

    print(f"Connecting to ChromaDB at: {CHROMA_DATA_PATH}...")
    try:
        persistent_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
        chroma_collection = persistent_client.get_collection(CHROMA_COLLECTION_NAME)
        # Verify collection is not empty
        if chroma_collection.count() == 0:
            print("WARNING: ChromaDB collection is empty. Did you run doc_ingestor.py?")
        else:
            print(f"Connected to ChromaDB collection: {CHROMA_COLLECTION_NAME} with {chroma_collection.count()} items.")
    except Exception as e:
        print(f"ERROR: Could not connect to ChromaDB or collection '{CHROMA_COLLECTION_NAME}' not found: {e}")
        print("Please ensure 'doc_ingestor.py' has been run successfully.")
        # You might want to raise an exception here or handle it more gracefully
        # For now, the app will run but /api/ask will likely fail.
        chroma_collection = None # Ensure it's None if connection failed

    print("Initializing Groq client...")
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not found in environment variables.")
        # Raise an error or exit if you want to be strict
        groq_client = None
    else:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq client initialized.")
    print("FastAPI resources loaded.")


# --- Pydantic Models for Request and Response ---
class QueryRequest(BaseModel):
    question: str

class SourceDocument(BaseModel):
    content: str
    source_document: str | None = None # Filename of the source PDF
    # distance: float | None = None # Optional: if you want to return similarity scores

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument] | None = None # List of relevant source snippets

# --- API Endpoint ---
@app.post("/api/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    global embedding_model, chroma_collection, groq_client

    if not embedding_model or not chroma_collection or not groq_client:
        raise HTTPException(status_code=503, detail="Backend services not fully initialized. Check server logs.")

    question = request.question
    print(f"\nReceived question: {question}")

    # 1. Embed the user's question
    print("Embedding user question...")
    question_embedding = embedding_model.encode(question).tolist()
    print("Question embedded.")

    # 2. Query ChromaDB for relevant document chunks
    print(f"Querying ChromaDB for {NUM_RELEVANT_CHUNKS} relevant chunks...")
    try:
        results = chroma_collection.query(
            query_embeddings=[question_embedding], # query_embeddings expects a list of embeddings
            n_results=NUM_RELEVANT_CHUNKS,
            include=["documents", "metadatas", "distances"] # Request distances for thresholding
        )
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        raise HTTPException(status_code=500, detail="Error querying vector database.")

    relevant_chunks_texts = results.get('documents', [[]])[0] # list of text chunks
    metadatas = results.get('metadatas', [[]])[0] # list of metadata dicts
    distances = results.get('distances', [[]])[0] # list of distances

    print(f"Retrieved {len(relevant_chunks_texts)} chunks from ChromaDB.")
    # for i, chunk in enumerate(relevant_chunks_texts):
    #     print(f"  Chunk {i+1} (Distance: {distances[i]:.4f}): {chunk[:100]}...") # Print first 100 chars

    # 3. Conditional LLM Query: Check if relevant content is found above threshold
    # Filter based on distance threshold. Chroma's default L2 distance: lower is better.
    confident_sources = []
    context_for_llm = ""

    if relevant_chunks_texts: # If any chunks were found
        for i in range(len(relevant_chunks_texts)):
            if distances[i] < SIMILARITY_THRESHOLD:
                source_doc_info = SourceDocument(
                    content=relevant_chunks_texts[i],
                    source_document=metadatas[i].get("source_document", "Unknown")
                    # distance=distances[i] # Optional
                )
                confident_sources.append(source_doc_info)
                context_for_llm += f"Document Snippet from {metadatas[i].get('source_document', 'Unknown')}:\n{relevant_chunks_texts[i]}\n\n"
            else:
                print(f"  Chunk with distance {distances[i]:.4f} rejected (threshold: {SIMILARITY_THRESHOLD}).")


    if not confident_sources:
        print("No relevant chunks found above confidence threshold.")
        return QueryResponse(
            answer="I'm sorry, I couldn't find information relevant to your question in the provided HR documents. Please try rephrasing or ask a different question.",
            sources=[]
        )

    print(f"Found {len(confident_sources)} confident sources. Constructing context for LLM...")
    # print(f"Context for LLM:\n{context_for_llm}") # For debugging

    # 4. Form a precise prompt for the LLM
    prompt = f"""You are an HR Policy Chatbot. Your goal is to answer employee questions based *strictly* on the provided context from company HR documents.
Do not use any external knowledge or make assumptions. If the answer cannot be found in the provided context, state that clearly.

Provided Context from HR Documents:
---
{context_for_llm}
---

Employee Question: {question}

Based *only* on the context above, please provide a concise answer to the employee's question. If the context does not contain the answer, say "The provided documents do not contain specific information about this topic."
Answer:
"""
    # print(f"\nPrompt for LLM:\n{prompt}") # For debugging

    # 5. Call Groq API
    print("Calling Groq LLM API...")
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=LLM_MODEL_NAME,
            temperature=0.2, # Lower temperature for more factual, less creative responses
            # max_tokens=250, # Optional: limit response length
        )
        llm_answer = chat_completion.choices[0].message.content
        print("LLM response received.")
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        raise HTTPException(status_code=500, detail=f"Error communicating with LLM API: {e}")

    return QueryResponse(answer=llm_answer, sources=confident_sources)


@app.get("/")
async def root():
    return {"message": "HR Policy Chatbot Backend is running!"}

# To run the app (from the hr_chatbot directory):
# uvicorn main:app --reload
