# doc_ingestor.py
import os
import fitz
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter # Or your custom chunking

# --- Configuration ---
DOCS_DIR = "./docs"
CHROMA_DATA_PATH = "./chroma_data"
CHROMA_COLLECTION_NAME = "hr_policies"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" # Or 'sentence-transformers/all-MiniLM-L6-v2'

# Text splitting parameters (tune as needed)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def get_pdf_text(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def clean_text(text):
    """Basic text cleaning."""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split()) # Remove multiple spaces
    return text

def main():
    print("Starting document ingestion...")

    # --- Initialize ChromaDB Client ---
    # Using persistent client. Data will be stored in CHROMA_DATA_PATH
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

    # --- Initialize Sentence Transformer Model ---
    # We'll use HuggingFaceEmbeddingFunction for on-the-fly embedding during collection creation/query
    # For ingestion, we can also pre-compute embeddings if preferred for more control.
    # Here, let's pre-compute.
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")

    # --- Get or Create ChromaDB Collection ---
    # Note: If you use default embedding function of Chroma, you don't need to pass embeddings explicitly.
    # But since we are using specific sentence-transformer locally, we generate embeddings ourselves.
    # For a persistent client, you might want to delete the collection if you're re-ingesting
    # or handle it more gracefully. For MVP, let's try get_or_create.
    try:
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
            # embedding_function is not directly used here if we provide embeddings ourselves,
            # but good to be aware of if ChromaDB were to generate them.
        )
        print(f"Using collection: {CHROMA_COLLECTION_NAME}")
    except Exception as e:
        print(f"Error getting or creating collection: {e}")
        return

    # --- Initialize Text Splitter ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    processed_files = 0
    total_chunks = 0

    # --- Process PDF Documents ---
    for filename in os.listdir(DOCS_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(DOCS_DIR, filename)
            print(f"\nProcessing document: {pdf_path}...")

            # 1. Extract Text
            raw_text = get_pdf_text(pdf_path)
            # print(f"  Extracted {len(raw_text)} characters.")

            # 2. Clean Text (optional, but good practice)
            cleaned_text = clean_text(raw_text)
            # print(f"  Cleaned text length: {len(cleaned_text)} characters.")

            # 3. Chunk Text
            chunks = text_splitter.split_text(cleaned_text)
            print(f"  Split into {len(chunks)} chunks.")

            if not chunks:
                print(f"  No text chunks extracted from {filename}. Skipping.")
                continue

            # 4. Generate Embeddings for Chunks
            print(f"  Generating embeddings for {len(chunks)} chunks...")
            chunk_embeddings = embedding_model.encode(chunks, show_progress_bar=True)
            print("  Embeddings generated.")

            # 5. Store in ChromaDB
            # Create unique IDs for each chunk
            chunk_ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

            # Prepare metadata
            metadatas = [{"source_document": filename, "chunk_id": chunk_id}
                         for chunk_id, i in zip(chunk_ids, range(len(chunks)))]


            # Add to collection (can take a few seconds for many chunks)
            try:
                collection.add(
                    ids=chunk_ids,
                    embeddings=chunk_embeddings.tolist(), # Convert numpy array to list
                    documents=chunks, # The actual text content of the chunk
                    metadatas=metadatas
                )
                print(f"  Successfully added {len(chunks)} chunks from {filename} to ChromaDB.")
                total_chunks += len(chunks)
            except Exception as e:
                print(f"  Error adding chunks to ChromaDB for {filename}: {e}")

            processed_files += 1

    if processed_files == 0:
        print("No PDF files found in the docs directory.")
    else:
        print(f"\n--- Ingestion Complete ---")
        print(f"Processed {processed_files} PDF files.")
        print(f"Total chunks added to ChromaDB: {total_chunks}")
        print(f"ChromaDB data stored in: {CHROMA_DATA_PATH}")
        print(f"Collection count: {collection.count()}")

if __name__ == "__main__":
    main()