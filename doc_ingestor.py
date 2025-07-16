# doc_ingestor.py
import os
import fitz  # PyMuPDF for PDFs
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
DOCS_DIR = "./docs"
CHROMA_DATA_PATH = "./chroma_data"
CHROMA_COLLECTION_NAME = "jasper_data" # Renamed for clarity
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" # A good open-source embedding model

# Text splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def get_pdf_text(file_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def get_txt_text(file_path):
    """Reads text from a TXT file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def clean_text(text):
    """Basic text cleaning."""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split()) # Remove multiple spaces
    return text

def main():
    print("🚀 Starting document ingestion process...")

    # --- 1. Initialize ChromaDB Client ---
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

    # --- 2. Initialize Sentence Transformer Model ---
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ Embedding model loaded.")

    # --- 3. Clean and Recreate ChromaDB Collection ---
    # Delete the old collection if it exists to ensure a fresh start
    try:
        if CHROMA_COLLECTION_NAME in [c.name for c in client.list_collections()]:
            print(f"Collection '{CHROMA_COLLECTION_NAME}' already exists. Deleting it.")
            client.delete_collection(name=CHROMA_COLLECTION_NAME)
    except Exception as e:
        print(f"Error deleting collection: {e}")
        return # Exit if we can't delete the collection

    # Create a new collection
    try:
        collection = client.create_collection(name=CHROMA_COLLECTION_NAME)
        print(f"✅ Collection '{CHROMA_COLLECTION_NAME}' created successfully.")
    except Exception as e:
        print(f"Error creating collection: {e}")
        return

    # --- 4. Initialize Text Splitter ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    # --- 5. Process Documents in DOCS_DIR ---
    all_chunks = []
    all_metadatas = []
    all_chunk_ids = []

    for filename in os.listdir(DOCS_DIR):
        file_path = os.path.join(DOCS_DIR, filename)
        raw_text = ""
        
        # Check file type and use the appropriate extractor
        if filename.lower().endswith(".pdf"):
            print(f"\n📄 Processing PDF: {filename}...")
            raw_text = get_pdf_text(file_path)
        elif filename.lower().endswith(".txt"):
            print(f"\n📄 Processing TXT: {filename}...")
            raw_text = get_txt_text(file_path)
        else:
            print(f"\n⚠️ Skipping unsupported file type: {filename}")
            continue

        if not raw_text.strip():
            print(f"  - No text found in {filename}. Skipping.")
            continue
            
        # Clean and split the text
        cleaned_text = clean_text(raw_text)
        chunks = text_splitter.split_text(cleaned_text)
        print(f"  - Split into {len(chunks)} chunks.")

        # Prepare chunks, metadata, and IDs for this file
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"
            all_chunks.append(chunk)
            all_chunk_ids.append(chunk_id)
            all_metadatas.append({"source": filename})

    # --- 6. Generate Embeddings and Add to ChromaDB (in a single batch) ---
    if all_chunks:
        print(f"\n🧠 Generating embeddings for {len(all_chunks)} total chunks...")
        chunk_embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)
        print("  - Embeddings generated.")

        # Add all data to the collection at once
        try:
            collection.add(
                ids=all_chunk_ids,
                embeddings=chunk_embeddings.tolist(),
                documents=all_chunks,
                metadatas=all_metadatas
            )
            print(f"✅ Successfully added {len(all_chunks)} chunks to ChromaDB.")
        except Exception as e:
            print(f"❌ Error adding chunks to ChromaDB: {e}")
    else:
        print("\nNo text chunks were generated from any documents.")

    print("\n--- Ingestion Complete ---")
    print(f"Total documents processed: {len(os.listdir(DOCS_DIR))}")
    print(f"Total chunks in collection: {collection.count()}")
    print(f"ChromaDB data stored in: {CHROMA_DATA_PATH}")

if __name__ == "__main__":
    main()