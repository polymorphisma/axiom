**Project Title**: *HR Policy Chatbot – Powered by Internal Company Documents*

**Objective**:
Build a low-cost, document-grounded chatbot that answers employee questions strictly from the company’s HR policy documentation. The chatbot will feature a simple chat interface, ensure responses are derived only from provided documents, and operate under a budget of < \$4/month (aiming for \$0 backend hosting costs on Fly.io within free allowances).

---

### ✅ **Core Project Goals**

1.  **Document Ingestion (MVP)**: Parse and embed static HR **PDF documents** (max 5–10 MB total for initial MVP).
2.  **Local Vector Storage**: Store document embeddings in a local **ChromaDB** instance.
3.  **Chat Interface**: Accept user questions via a simple web-based chat interface (plain HTML/CSS/JS).
4.  **RAG Core**: Implement similarity search against ChromaDB to find relevant document chunks.
5.  **Conditional LLM Query**: Only query the LLM if relevant content is found above a defined confidence threshold; otherwise, return a pre-defined fallback message.
6.  **Ultra-Low Cost Operation**:
    *   Host the frontend on Netlify (free tier).
    *   Host the backend on Fly.io, aiming for \$0 cost by staying within their free resource allowances. A credit card will be required on file by Fly.io.
    *   Utilize a cost-effective LLM API (Groq).
    *   The total monthly operational cost (primarily LLM API) should be < \$4.
7.  **Grounded Responses**: Ensure responses derive strictly from provided documents through RAG and meticulous prompt engineering.
8.  **Deployability**: Structure the project for easy document swap and potential reuse by other teams.

---

### 🛠️ **Finalized Tech Stack & Services**

*   **Frontend**:
    *   **Technology**: Plain HTML, CSS, and JavaScript.
    *   **Hosting**: **Netlify** (Free tier).
*   **Backend**:
    *   **Framework**: **FastAPI** (Python).
    *   **Hosting**: **Fly.io**
        *   Utilizes Docker for deployment.
        *   Aims to operate within Fly.io's free resource allowances (e.g., for compute, basic egress, and potentially a small persistent volume).
        *   **Note**: Fly.io requires a credit card on file to deploy applications, even if usage stays within free allowances.
*   **LLM API**:
    *   **Primary Choice**: **Groq API** with **Mistral 7B Instruct** (or a similar cost-effective model like `Llama 3 8B Instruct` if pricing is comparable).
*   **Embeddings**:
    *   **Model**: Local HuggingFace Sentence Transformer (e.g., `BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2`).
    *   **Process**: Generated locally by an admin script; embeddings stored in ChromaDB.
*   **Vector Store**:
    *   **Database**: **ChromaDB** (Local/embedded mode). The database files will be persisted within the Fly.io environment (either baked into the Docker image for MVP or using a small, free-tier Fly Volume).
*   **Document Parsing & Utilities**:
    *   **PDFs**: `PyMuPDF` (Fitz).
    *   **Text Chunking**: `langchain-text-splitters` (e.g., `RecursiveCharacterTextSplitter`) or custom Python logic.
    *   **Deployment**: Docker.

---

### 🚀 **MVP - Phased Implementation Plan**

**Phase 0: Setup & Prerequisites**

1.  **Account Sign-ups**:
    *   GitHub account.
    *   Netlify account (link to GitHub).
    *   Groq API account (obtain API key).
    *   Fly.io account (add credit card as required; install `flyctl` CLI).
2.  **Local Development Environment**: Python, VS Code, Git, Docker Desktop.

**Phase 1: Core Backend & RAG Logic (Local First)**

3.  **Document Ingestion Script (`doc_ingestor.py`)**:
    *   Target: 1-2 sample HR PDF documents.
    *   Functionality:
        *   Extract text using `PyMuPDF`.
        *   Clean text.
        *   Chunk text into segments.
        *   Load the chosen local Sentence Transformer model.
        *   Generate embeddings for each chunk.
        *   Initialize ChromaDB in a persistent local directory (e.g., `./chroma_data`).
        *   Store chunks, embeddings, and basic metadata (e.g., chunk ID, source document name if multiple) in ChromaDB.
    *   *Execution: Run locally by an admin to populate/update the `./chroma_data` directory.*
4.  **FastAPI Application (`main.py`)**:
    *   Load the Sentence Transformer model and ChromaDB (from `./chroma_data`) on startup.
    *   Implement an API endpoint (e.g., `/api/ask`):
        *   Accepts: `{ "question": "user's question" }`.
        *   Embeds the user's question.
        *   Queries ChromaDB for top N relevant document chunks.
        *   **Initial Fallback**: If no relevant chunks (or below a preliminary score threshold), return a "not found" message.
        *   Constructs context from retrieved chunks.
        *   Forms a precise prompt for the LLM (instructing it to use *only* the provided context).
        *   Calls Groq API (Mistral 7B) with the context and question.
        *   Returns: `{ "answer": "LLM's answer", "sources": ["snippet1_text", "snippet2_text"] }`.
    *   Test rigorously locally (e.g., using Postman or `curl`).

**Phase 2: Dockerization & Backend Deployment to Fly.io**

5.  **Dockerfile for FastAPI App**:
    *   Set up a Python environment.
    *   Copy application code, `requirements.txt`.
    *   **Crucially, copy the populated `./chroma_data` directory into the Docker image.** This makes the vector store part of the deployed application.
    *   Define `CMD` to run Uvicorn with the FastAPI app.
6.  **Fly.io Configuration (`fly.toml`)**:
    *   Define the application name, primary region.
    *   Specify build strategy (Docker).
    *   Configure services, internal port, health checks.
    *   Set environment variables (e.g., `GROQ_API_KEY`).
7.  **Deploy to Fly.io**:
    *   Use `flyctl launch` (for initial setup) and `flyctl deploy` for subsequent deployments.
    *   Ensure the app deploys and the API endpoint is accessible via the provided Fly.io URL.
    *   Monitor initial logs for any errors.

**Phase 3: Simple Frontend & Connection**

8.  **Basic Chat UI (`index.html`, `style.css`, `script.js`)**:
    *   Input field for questions, submit button, display area for conversation.
    *   JavaScript in `script.js` to:
        *   On submit, take the question from the input field.
        *   Use the `fetch` API to send a POST request to your deployed Fly.io backend URL (`/api/ask`).
        *   Handle the JSON response and display the answer and source snippets.
        *   Manage basic loading states and error display.
9.  **Frontend Deployment to Netlify**:
    *   Push frontend code to a GitHub repository.
    *   Connect the GitHub repository to Netlify for continuous deployment.
    *   Ensure the frontend correctly calls the Fly.io backend.

**Phase 4: Testing, Refinement & Iteration (Post-MVP)**

10. **End-to-End Testing**: Test the full flow with various HR policy questions.
11. **Refine Fallback & Confidence**: Improve logic for when to return the "cannot find" message based on similarity scores.
12. **Prompt Engineering**: Iterate on the LLM prompt to improve answer quality, conciseness, and adherence to instructions.
13. **UI Enhancements**: Minor improvements to the chat interface if time permits.
14. **Documentation**: Basic README explaining setup, how to update documents (rerun ingestion script, redeploy backend), and environment variables.

---

### Key Considerations for Fly.io

*   **ChromaDB Data Persistence**: For the MVP, baking the `./chroma_data` folder into the Docker image is simplest. If document updates are frequent or DB size becomes an issue, explore using Fly Volumes (which has a free allowance) to persist ChromaDB data outside the ephemeral container filesystem.
*   **Resource Monitoring**: Keep an eye on resource usage in the Fly.io dashboard to ensure you remain within free allowances.
*   **`fly.toml` Configuration**: This file is key to defining how your application runs on Fly.io.

